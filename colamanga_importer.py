from pathlib import Path
from typing import List

from colamanga_min import download_images, get_chapter_images


def import_colamanga_chapter(chapter_url: str, out_dir: str) -> List[str]:
    dest_dir = Path(out_dir)
    urls, cookies, referer = get_chapter_images(chapter_url)
    saved_paths = download_images(urls, dest_dir, cookies, referer=referer)
    return [str(path.resolve()) for path in saved_paths]
