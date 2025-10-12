from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError
from playwright.sync_api import sync_playwright

BASE = "https://www.colamanga.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

CookieDict = Dict[str, Any]

RESOLVER_JS_PATH = Path(__file__).with_name("resolver_js.js")

def get_chapter_images(
    chapter_url: str, timeout_ms: int = 15000
) -> Tuple[List[str], List[CookieDict], str]:
    if not chapter_url.startswith("http"):
        raise ValueError("chapter_url must be an absolute https:// URL")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=UA,
            extra_http_headers={"Referer": BASE},
            ignore_https_errors=True,
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(chapter_url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle")
        except Exception:
            pass
        try:
            page.wait_for_selector("img[data-src], img[srcset], img[src]", timeout=timeout_ms)
        except Exception:
            page.wait_for_timeout(1000)
        script = RESOLVER_JS_PATH.read_text(encoding="utf-8")
        result = page.evaluate(script)
        if not result or not result.get("ok"):
            browser.close()
            raise RuntimeError(f"ColaManga resolver failed: {result}")
        cookies: List[CookieDict] = ctx.cookies()
        page_url = page.url
        browser.close()
    return result["urls"], cookies, page_url

def download_images(
    urls: List[str],
    dest_dir: Path,
    cookies: Optional[List[CookieDict]] = None,
    referer: Optional[str] = None,
) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    referer_header, origin_header, referer_host = _referer_and_origin(referer)
    sess.headers.update({"User-Agent": UA})
    base_headers = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Dest": "image",
        "User-Agent": UA,
    }
    if referer_header:
        base_headers["Referer"] = referer_header
    if origin_header:
        base_headers["Origin"] = origin_header
    if cookies:
        for cookie in cookies:
            name = cookie.get("name")
            if not name:
                continue
            value = cookie.get("value", "")
            domain = cookie.get("domain") or ".colamanga.com"
            path = cookie.get("path") or "/"
            secure = bool(cookie.get("secure"))
            created = requests.cookies.create_cookie(
                domain=domain,
                name=name,
                value=value,
                path=path,
                secure=secure,
            )
            sess.cookies.set_cookie(created)
    saved_paths: List[Path] = []
    for i, raw_url in enumerate(urls, 1):
        url = _normalize_image_url(raw_url)
        headers = _per_request_headers(base_headers, referer_host, url)
        with sess.get(url, stream=True, timeout=45, headers=headers) as response:
            content_type = response.headers.get("Content-Type")
            if response.status_code >= 400 and not _is_image_content_type(content_type):
                response.raise_for_status()
            if not _is_image_content_type(content_type):
                raise RuntimeError(
                    f"ColaManga image request returned a non-image payload: {url}"
                )

            ext = _extension_from_response(url, content_type)
            data = bytearray()
            for chunk in response.iter_content(1 << 14):
                if chunk:
                    data.extend(chunk)

        data_bytes = bytes(data)
        if _should_convert_from_webp(ext, content_type):
            try:
                data_bytes, ext = _convert_webp_bytes(data_bytes)
            except UnidentifiedImageError as exc:
                raise RuntimeError(
                    f"Failed to decode ColaManga WEBP image from {url}"
                ) from exc

        path = dest_dir / f"page_{i:03d}{ext}"
        path.write_bytes(data_bytes)
        saved_paths.append(path)
    return saved_paths

def _referer_and_origin(chapter_referer: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    if chapter_referer:
        try:
            parsed = urlparse(chapter_referer)
        except Exception:
            parsed = None
        if parsed and parsed.scheme in {"http", "https"} and parsed.netloc:
            origin = f"{parsed.scheme}://{parsed.netloc}"
            return chapter_referer, origin, parsed.netloc
    origin = BASE.rstrip("/")
    parsed_origin = urlparse(origin)
    host = parsed_origin.netloc if parsed_origin.netloc else None
    return origin + "/", origin, host


def _per_request_headers(
    base_headers: Dict[str, str],
    referer_host: Optional[str],
    url: str,
) -> Dict[str, str]:
    headers = dict(base_headers)
    target = urlparse(url)
    target_host = target.netloc
    sec_fetch_site = _sec_fetch_site(referer_host, target_host)
    headers["Sec-Fetch-Site"] = sec_fetch_site
    if sec_fetch_site != "same-origin":
        headers.pop("Origin", None)
    return headers


def _sec_fetch_site(
    referer_host: Optional[str],
    target_host: Optional[str],
) -> str:
    if not referer_host or not target_host:
        return "cross-site"
    ref = referer_host.lower()
    tgt = target_host.lower()
    if ref == tgt:
        return "same-origin"
    if _is_same_site(ref, tgt):
        return "same-site"
    return "cross-site"


def _is_same_site(a: str, b: str) -> bool:
    a_parts = a.split(".")
    b_parts = b.split(".")
    if len(a_parts) < 2 or len(b_parts) < 2:
        return False
    return a_parts[-2:] == b_parts[-2:]

def _ext_from_url(url: str) -> str:
    q = _normalize_image_url(url).split("?", 1)[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
        if q.endswith(ext):
            return ext
    return ".jpg"


def _normalize_image_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("//"):
        normalized = "https:" + normalized
    if ".enc." in normalized:
        normalized = normalized.replace(".enc.", ".")
    return normalized


def _is_image_response(response: requests.Response) -> bool:
    return _is_image_content_type(response.headers.get("Content-Type"))


def _is_image_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    return "image" in content_type.lower()


def _extension_from_response(url: str, content_type: Optional[str]) -> str:
    if content_type:
        lower = content_type.lower()
        if "jpeg" in lower or "jpg" in lower:
            return ".jpg"
        if "png" in lower:
            return ".png"
        if "webp" in lower:
            return ".webp"
        if "gif" in lower:
            return ".gif"
        if "avif" in lower:
            return ".avif"
    return _ext_from_url(url)


def _should_convert_from_webp(ext: str, content_type: Optional[str]) -> bool:
    lower = (content_type or "").lower()
    return ext == ".webp" or "webp" in lower


def _convert_webp_bytes(data: bytes) -> Tuple[bytes, str]:
    with Image.open(BytesIO(data)) as image:
        has_alpha = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        output = BytesIO()
        if has_alpha:
            image.convert("RGBA").save(output, format="PNG")
            return output.getvalue(), ".png"
        image.convert("RGB").save(output, format="JPEG", quality=95)
        return output.getvalue(), ".jpg"
