import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, UnidentifiedImageError
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

BASE = "https://www.colamanga.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

CookieDict = Dict[str, Any]

RESOLVER_JS_PATH = Path(__file__).with_name("resolver_js.js")

_BLOB_CAPTURE_SCRIPT = """
(() => {
  const originalCreate = URL.createObjectURL.bind(URL);
  const store = [];
  URL.createObjectURL = (blob) => {
    store.push({ blob, type: blob.type || '', size: blob.size, created: Date.now() });
    return originalCreate(blob);
  };
  window.__COLA_CAPTURED_BLOBS__ = store;
})();
"""

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
    if not urls:
        return []

    saved_paths: List[Path] = []
    target_url = referer or BASE
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=UA,
                extra_http_headers={"Referer": BASE},
                ignore_https_errors=True,
            )
            context.add_init_script(_BLOB_CAPTURE_SCRIPT)
            if cookies:
                context.add_cookies(_prepare_playwright_cookies(cookies))
            page = context.new_page()
            page.goto(target_url, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            _ensure_reader_ready(page)

            expected = len(urls)
            _ = _preload_chapter(page, expected)
            for page_number in range(1, expected + 1):
                if not _ensure_blob_for_index(page, page_number):
                    if page_number == 1:
                        raise RuntimeError(
                            "Unable to load the first ColaManga page from the browser"
                        )
                    break

                entry = _extract_blob_entry(page, page_number - 1)
                if not entry:
                    raise RuntimeError(
                        f"ColaManga failed to capture image data for page {page_number}"
                    )
                data_bytes = entry["bytes"]
                mime = entry.get("type") or _guess_mime_from_bytes(data_bytes)
                ext = _extension_from_blob(mime, data_bytes)
                if _should_convert_from_webp(ext, mime):
                    try:
                        data_bytes, ext = _convert_webp_bytes(data_bytes)
                    except UnidentifiedImageError as exc:
                        raise RuntimeError(
                            "Failed to decode ColaManga WEBP image payload"
                        ) from exc
                path = dest_dir / f"page_{page_number:03d}{ext}"
                path.write_bytes(data_bytes)
                saved_paths.append(path)
        finally:
            browser.close()
    return saved_paths

def _prepare_playwright_cookies(cookies: List[CookieDict]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        if not name or value is None:
            continue
        prepared.append(
            {
                "name": name,
                "value": value,
                "domain": cookie.get("domain") or "www.colamanga.com",
                "path": cookie.get("path") or "/",
                "secure": bool(cookie.get("secure")),
                "httpOnly": bool(cookie.get("httpOnly")),
                "sameSite": cookie.get("sameSite", "Lax"),
                "expires": cookie.get("expires"),
            }
        )
    return prepared


def _trigger_page_load(page, index: int) -> None:
    page.evaluate(
        """
        (idx) => {
            const cr = window.__cr;
            if (!cr || typeof cr.showPic !== 'function') {
                throw new Error('ColaManga reader script unavailable');
            }
            try {
                cr.showPic(idx);
            } catch (error) {
                console.warn('ColaManga showPic failed', error);
            }
        }
        """,
        index,
    )


def _ensure_reader_ready(page) -> None:
    page.evaluate(
        """
        () => {
            if (window.__cr) {
                window.__cr.isfromMangaRead = 1;
            }
            if (window.__cad && typeof window.__cad.setCookieValue === 'function') {
                window.__cad.setCookieValue();
            }
        }
        """
    )


def _preload_chapter(page, expected: int) -> int:
    max_attempts = max(expected * 2, 60)
    last_count = -1
    stagnant_steps = 0
    for _ in range(max_attempts):
        current = page.evaluate(
            "(window.__COLA_CAPTURED_BLOBS__ || []).length"
        )
        if current >= expected:
            return current
        if current == last_count:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
            last_count = current
        page.mouse.wheel(0, 1600)
        page.wait_for_timeout(300)
        if stagnant_steps >= 5:
            next_index = current + 1
            page.evaluate(
                "(idx) => { try { window.__cr && window.__cr.showPic(idx); } catch (e) {} }",
                next_index,
            )
            stagnant_steps = 0
    return page.evaluate("(window.__COLA_CAPTURED_BLOBS__ || []).length")


def _ensure_blob_for_index(page, index: int) -> bool:
    exists = page.evaluate(
        """
        (idx) => {
            const store = window.__COLA_CAPTURED_BLOBS__ || [];
            return Boolean(store[idx - 1]);
        }
        """,
        index,
    )
    if exists:
        return True
    _trigger_page_load(page, index)
    try:
        page.wait_for_function(
            """
            (idx) => {
                const store = window.__COLA_CAPTURED_BLOBS__ || [];
                return Boolean(store[idx - 1]);
            }
            """,
            arg=index,
            timeout=45000,
        )
        return True
    except PlaywrightTimeout:
        return False


def _extract_blob_entry(page, zero_index: int) -> Optional[Dict[str, Any]]:
    entry = page.evaluate(
        """
        async (idx) => {
            const store = window.__COLA_CAPTURED_BLOBS__ || [];
            const item = store[idx];
            if (!item) {
                return null;
            }
            const base64Data = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onerror = () => reject(reader.error || new Error('Failed to read ColaManga blob'));
                reader.onload = () => {
                    const result = reader.result;
                    if (typeof result === 'string') {
                        const comma = result.indexOf(',');
                        resolve(comma >= 0 ? result.slice(comma + 1) : result);
                    } else {
                        resolve('');
                    }
                };
                reader.readAsDataURL(item.blob);
            });
            store[idx] = null;
            return {
                type: item.type || '',
                b64: base64Data,
            };
        }
        """,
        zero_index,
    )
    if not entry:
        return None
    try:
        data_bytes = base64.b64decode(entry["b64"], validate=True)
    except Exception:
        data_bytes = base64.b64decode(entry.get("b64", ""))
    return {"type": entry.get("type", ""), "bytes": data_bytes}


def _guess_mime_from_bytes(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"GIF":
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/jpeg"


def _extension_from_blob(mime: Optional[str], data: bytes) -> str:
    if mime:
        lower = mime.lower()
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
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"GIF":
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


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
