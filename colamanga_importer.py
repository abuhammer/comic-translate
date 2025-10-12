from pathlib import Path
from colamanga_min import get_chapter_images, download_images

def import_colamanga_chapter(chapter_url: str, out_dir: str):
    urls, cookies = get_chapter_images(chapter_url)
    download_images(urls, Path(out_dir), cookies)
    return [str((Path(out_dir) / f"page_{i+1:03d}.jpg").absolute()) for i in range(len(urls))]
