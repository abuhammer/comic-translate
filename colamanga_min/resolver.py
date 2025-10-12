import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import requests
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
    referer_header = _normalize_referer(referer)
    sess.headers.update({
        "User-Agent": UA,
        "Referer": referer_header,
        "Origin": referer_header.rstrip("/"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
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
        with sess.get(url, stream=True, timeout=45) as response:
            if response.status_code >= 400 and not _is_image_response(response):
                response.raise_for_status()
            ext = _extension_from_response(url, response.headers.get("Content-Type"))
            path = dest_dir / f"page_{i:03d}{ext}"
            with open(path, "wb") as f:
                for chunk in response.iter_content(1 << 14):
                    if chunk:
                        f.write(chunk)
        saved_paths.append(path)
    return saved_paths

def _normalize_referer(chapter_referer: Optional[str]) -> str:
    if chapter_referer:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(chapter_referer)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                origin = f"{parsed.scheme}://{parsed.netloc}"
                return origin.rstrip("/") + "/"
        except Exception:
            pass
    return BASE.rstrip("/") + "/"

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
    content_type = response.headers.get("Content-Type", "")
    if content_type and "image" in content_type.lower():
        return True
    return False


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
