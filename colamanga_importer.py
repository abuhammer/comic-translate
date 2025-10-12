from pathlib import Path
from typing import List

from colamanga_min import download_images, get_chapter_images


def _ext_from_url(url: str) -> str:
    sanitized = url.split("?", 1)[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
        if sanitized.endswith(ext):
            return ext
    return ".jpg"


def import_colamanga_chapter(chapter_url: str, out_dir: str) -> List[str]:
    dest_dir = Path(out_dir)
    urls, cookies, referer = get_chapter_images(chapter_url)
    download_images(urls, dest_dir, cookies, referer=referer)
    return [
        str((dest_dir / f"page_{index:03d}{_ext_from_url(url)}").resolve())
        for index, url in enumerate(urls, start=1)
    ]
