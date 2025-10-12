import json
from pathlib import Path
from typing import List, Tuple, Optional
import requests
from playwright.sync_api import sync_playwright

BASE = "https://www.colamanga.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

RESOLVER_JS_PATH = Path(__file__).with_name("resolver_js.js")

def get_chapter_images(chapter_url: str, timeout_ms: int = 15000) -> Tuple[List[str], dict]:
    if not chapter_url.startswith("http"):
        raise ValueError("chapter_url must be an absolute https:// URL")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=UA,
            extra_http_headers={"Referer": BASE},
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(chapter_url, wait_until="domcontentloaded")
        page.wait_for_timeout(600)
        script = RESOLVER_JS_PATH.read_text(encoding="utf-8")
        result = page.evaluate(script)
        if not result or not result.get("ok"):
            browser.close()
            raise RuntimeError(f"ColaManga resolver failed: {result}")
        cookies = {c["name"]: c["value"] for c in ctx.cookies()}
        browser.close()
    return result["urls"], cookies

def download_images(urls: List[str], dest_dir: Path, cookies: Optional[dict] = None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": UA,
        "Referer": BASE,
        "Accept": "image/avif,image/webp,image/apng,*/*;q=0.8"
    })
    if cookies:
        for k, v in cookies.items():
            c = requests.cookies.create_cookie(domain="www.colamanga.com", name=k, value=v, path="/")
            sess.cookies.set_cookie(c)
    for i, url in enumerate(urls, 1):
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
