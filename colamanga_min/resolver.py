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

def get_chapter_images(chapter_url: str, timeout_ms: int = 15000) -> Tuple[List[str], List[CookieDict]]:
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
        browser.close()
    return result["urls"], cookies

def download_images(urls: List[str], dest_dir: Path, cookies: Optional[List[CookieDict]] = None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": UA,
        "Referer": BASE,
        "Accept": "image/avif,image/webp,image/apng,*/*;q=0.8"
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
    for i, url in enumerate(urls, 1):
        if url.startswith("//"):
            url = "https:" + url
        ext = _ext_from_url(url)
        path = dest_dir / f"page_{i:03d}{ext}"
        with sess.get(url, stream=True, timeout=45) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 14):
                    if chunk:
                        f.write(chunk)

def _ext_from_url(url: str) -> str:
    q = url.split("?", 1)[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
        if q.endswith(ext):
            return ext
    return ".jpg"
