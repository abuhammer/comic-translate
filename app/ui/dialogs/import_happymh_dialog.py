from __future__ import annotations

import json
import mimetypes
import os
import random
import re
import shutil
import tempfile
import time
from urllib.parse import urljoin, urlparse

import requests
from PySide6 import QtCore, QtGui, QtWidgets


class ImportHappymhDownloadWorker(QtCore.QThread):
    """Worker that downloads images from a HappyMH reading or manga URL."""

    progress = QtCore.Signal(int, int)
    error = QtCore.Signal(str)
    completed = QtCore.Signal(list)

    _DEFAULT_VERSION = "v3.1818134"

    def __init__(self, url: str, temp_dir: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.url = url
        self.temp_dir = temp_dir
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Linux; Android 13; Pixel 5) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.6613.120 Mobile Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,"
                    "application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="128", "Google Chrome";v="128"',
                "sec-ch-ua-mobile": "?1",
                "sec-ch-ua-platform": '"Android"',
                "sec-ch-ua-arch": '"arm"',
                "sec-ch-ua-bitness": '"64"',
                "sec-ch-ua-full-version": '"128.0.6613.120"',
                "sec-ch-ua-full-version-list": (
                    '"Not.A/Brand";v="8.0.0.0", "Chromium";v="128.0.6613.120", '
                    '"Google Chrome";v="128.0.6613.120"'
                ),
                "sec-ch-ua-model": '"Pixel 5"',
                "sec-ch-ua-platform-version": '"14.0.0"',
                "Upgrade-Insecure-Requests": "1",
                "TE": "trailers",
            }
        )
        self._bootstrapped = False

    def _sleep_before_request(self) -> None:
        """Sleep for a randomised delay to avoid rate-limiting."""

        distribution = random.choice(("uniform", "normal", "exponential"))
        if distribution == "uniform":
            delay = random.uniform(0.75, 2.5)
        elif distribution == "normal":
            delay = max(0.25, random.gauss(1.5, 0.5))
        else:  # exponential
            delay = min(3.0, random.expovariate(1.0))
        time.sleep(delay)

    def _session_get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: int = 30,
    ) -> requests.Response:
        self._sleep_before_request()
        response = self._session.get(url, params=params, headers=headers, timeout=timeout)
        return response

    def _navigation_headers(self, *, referer: str | None = None) -> dict[str, str]:
        headers = {
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
        }
        if referer:
            headers["Referer"] = referer
            headers["Sec-Fetch-Site"] = "same-origin"
        else:
            headers["Sec-Fetch-Site"] = "none"
        return headers

    def _bootstrap_session(self, *, force: bool = False) -> None:
        if self._bootstrapped and not force:
            return

        try:
            response = self._session_get(
                "https://m.happymh.com/",
                headers=self._navigation_headers(),
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException:
            # The bootstrap request only exists to collect cookies/JS challenges.
            if force:
                # Allow callers to try another bootstrap attempt later.
                self._bootstrapped = False
            return

        self._bootstrapped = True

    def _normalise_url(self, url: str) -> str:
        stripped = url.strip()
        if not stripped:
            raise ValueError("The provided URL is empty.")
        parsed = urlparse(stripped)
        if not parsed.scheme:
            stripped = "https://" + stripped
            parsed = urlparse(stripped)
        if not parsed.netloc:
            raise ValueError("The provided URL is not valid.")
        return stripped

    def _extract_reading_url(self, url: str) -> tuple[str, str, str | None]:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain.endswith("happymh.com"):
            raise ValueError("The provided URL is not part of happymh.com.")

        path = parsed.path.rstrip("/")
        if path.startswith("/mangaread/"):
            code = path.split("/", 2)[2]
            reading_url = parsed._replace(path=f"/mangaread/{code}", query="", fragment="").geturl()
            return reading_url, code, "https://m.happymh.com/"

        if not path.startswith("/manga/"):
            raise ValueError("Please provide a HappyMH reading or manga URL.")

        # Fetch the manga page to locate the reading link
        response = self._session_get(
            url,
            timeout=30,
            headers=self._navigation_headers(referer="https://m.happymh.com/"),
        )
        response.raise_for_status()
        html = response.text

        match = re.search(r"\"chapterUrl\"\s*:\s*\"(/mangaread/[^\"]+)\"", html)
        if not match:
            match = re.search(r"href=\"(/mangaread/[^\"#?]+)\"", html)
        if not match:
            raise ValueError("Could not locate a chapter link on the provided manga page.")

        reading_path = match.group(1)
        reading_url = urljoin(url, reading_path)
        return reading_url, reading_path.rsplit("/", 1)[-1], url

    def _extract_version(self, html: str, code: str) -> str | None:
        patterns = [
            r"/v2\.0/apis/manga/reading\?code=" + re.escape(code) + r"(?:&|&amp;)v=([^\"'&<]+)",
            r"/v2\.0/apis/manga/reading\?code=" + re.escape(code) + r"[^\"]*?\\u0026v=([^\"\\]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None

    def _guess_extension(self, img_url: str, response: requests.Response) -> str:
        parsed = urlparse(img_url)
        ext = os.path.splitext(parsed.path)[1].lower()
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        if ext in valid_exts:
            return ".jpg" if ext == ".jpeg" else ext
        mime_ext = mimetypes.guess_extension(response.headers.get("Content-Type", ""))
        if mime_ext:
            mime_ext = ".jpg" if mime_ext == ".jpe" else mime_ext
            if mime_ext in valid_exts:
                return mime_ext
        return ".jpg"

    def _request_chapter_data(self, reading_url: str, code: str, version: str | None) -> dict:
        params = {"code": code}
        if version:
            params["v"] = version
        headers = {
            "Referer": reading_url,
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://m.happymh.com",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        }
        response = self._session_get(
            "https://m.happymh.com/v2.0/apis/manga/reading",
            params=params,
            headers=headers,
            timeout=30,
        )
        if response.status_code == 403 and version:
            raise RuntimeError("forbidden")
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("Unexpected response from HappyMH API.") from exc
        if payload.get("status") != 0 or "data" not in payload:
            raise ValueError(payload.get("msg") or "Failed to retrieve chapter data.")
        return payload["data"]

    def _fetch_reading_page(
        self, reading_url: str, referer: str | None
    ) -> requests.Response:
        headers = self._navigation_headers(referer=referer)
        response = self._session_get(
            reading_url,
            timeout=30,
            headers=headers,
        )
        if response.status_code == 403:
            self._bootstrap_session(force=True)
            headers = self._navigation_headers(referer=referer)
            response = self._session_get(
                reading_url,
                timeout=30,
                headers=headers,
            )
        return response

    def run(self) -> None:  # noqa: D401
        try:
            normalised = self._normalise_url(self.url)
            self._bootstrap_session()
            reading_url, code, referer = self._extract_reading_url(normalised)
        except (requests.RequestException, ValueError) as exc:
            self.error.emit(str(exc))
            return

        try:
            html_response = self._fetch_reading_page(reading_url, referer)
            html_response.raise_for_status()
        except requests.RequestException as exc:
            self.error.emit(str(exc))
            return

        version = self._extract_version(html_response.text, code) or self._DEFAULT_VERSION

        try:
            chapter_data = self._request_chapter_data(reading_url, code, version)
        except RuntimeError:
            fallbacks: list[str | None] = []
            if version != self._DEFAULT_VERSION:
                fallbacks.append(self._DEFAULT_VERSION)
            fallbacks.append(None)

            chapter_data = None
            for fallback in fallbacks:
                try:
                    chapter_data = self._request_chapter_data(reading_url, code, fallback)
                    break
                except RuntimeError:
                    continue
                except requests.RequestException as exc:
                    self.error.emit(str(exc))
                    return
                except ValueError as exc:
                    self.error.emit(str(exc))
                    return

            if chapter_data is None:
                self.error.emit("Failed to retrieve chapter data from HappyMH.")
                return
        except requests.RequestException as exc:
            self.error.emit(str(exc))
            return
        except ValueError as exc:
            self.error.emit(str(exc))
            return

        scans = chapter_data.get("scans", [])
        image_urls = [scan.get("url") for scan in scans if scan.get("url")]

        if not image_urls:
            self.error.emit("No downloadable images were found for the provided chapter.")
            return

        total = len(image_urls)
        downloaded_files: list[str] = []

        for index, img_url in enumerate(image_urls, start=1):
            try:
                img_response = self._session_get(
                    img_url,
                    headers={"Referer": reading_url},
                    timeout=60,
                )
                img_response.raise_for_status()
            except requests.RequestException as exc:
                self.error.emit(str(exc))
                return

            extension = self._guess_extension(img_url, img_response)
            filename = f"{index:04d}{extension}"
            img_path = os.path.join(self.temp_dir, filename)

            try:
                with open(img_path, "wb") as handle:
                    handle.write(img_response.content)
            except OSError as exc:
                self.error.emit(str(exc))
                return

            downloaded_files.append(img_path)
            self.progress.emit(index, total)

        self.completed.emit(downloaded_files)


class ImportHappymhDialog(QtWidgets.QDialog):
    """Dialog that downloads images from HappyMH chapters."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Import from HappyMH"))
        self.setModal(True)

        self.temp_dir = tempfile.mkdtemp(prefix="happymh_import_")
        self._cleanup_on_close = True
        self.download_worker: ImportHappymhDownloadWorker | None = None
        self._downloaded_files: list[str] = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        description = QtWidgets.QLabel(
            self.tr(
                "Paste a HappyMH chapter URL. The application will download and order "
                "all images found in the chapter."
            )
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.url_input = QtWidgets.QLineEdit(self)
        self.url_input.setPlaceholderText(self.tr("https://m.happymh.com/mangaread/..."))
        self.url_input.returnPressed.connect(self.start_download)
        layout.addWidget(self.url_input)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.import_button = self.button_box.addButton(
            self.tr("Import"), QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.cancel_button = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.rejected.connect(self.reject)
        self.import_button.clicked.connect(self.start_download)
        layout.addWidget(self.button_box)

    def start_download(self) -> None:
        url = self.get_url()
        if not url:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Missing URL"),
                self.tr("Please enter a valid HappyMH URL."),
            )
            return

        if self.download_worker and self.download_worker.isRunning():
            return

        self._toggle_ui_for_download(True)
        self.status_label.setText(self.tr("Fetching chapter information..."))
        self.status_label.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.download_worker = ImportHappymhDownloadWorker(url, self.temp_dir, self)
        self.download_worker.progress.connect(self._on_progress)
        self.download_worker.error.connect(self._on_error)
        self.download_worker.completed.connect(self._on_completed)
        self.download_worker.start()

    def _toggle_ui_for_download(self, downloading: bool) -> None:
        self.url_input.setEnabled(not downloading)
        self.import_button.setEnabled(not downloading)
        self.cancel_button.setEnabled(not downloading)

    def _on_progress(self, current: int, total: int) -> None:
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(
            self.tr("Downloading image {current} of {total}...").format(
                current=current, total=total
            )
        )

    def _on_error(self, message: str) -> None:
        self._toggle_ui_for_download(False)
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.download_worker = None
        QtWidgets.QMessageBox.critical(
            self,
            self.tr("Download failed"),
            message,
        )

    def _on_completed(self, files: list[str]) -> None:
        self._downloaded_files = files
        self._cleanup_on_close = False
        self.download_worker = None
        self.accept()

    def get_url(self) -> str:
        return self.url_input.text().strip()

    def get_temp_dir(self) -> str:
        return self.temp_dir

    def get_downloaded_files(self) -> list[str]:
        return list(self._downloaded_files)

    def cleanup(self) -> None:
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def reject(self) -> None:
        super().reject()
        if self._cleanup_on_close:
            self.cleanup()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            if self._cleanup_on_close:
                self.cleanup()
        finally:
            super().closeEvent(event)
