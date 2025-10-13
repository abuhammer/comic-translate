from __future__ import annotations

import os
import shutil
import tempfile

from PySide6 import QtCore, QtGui, QtWidgets

from colamanga_importer import import_colamanga_chapter


class ColaMangaDownloadWorker(QtCore.QThread):
    """Worker thread that downloads ColaManga chapter images."""

    error = QtCore.Signal(str)
    completed = QtCore.Signal(list)

    def __init__(self, url: str, output_dir: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.url = url
        self.output_dir = output_dir

    def run(self) -> None:  # noqa: D401
        try:
            files = import_colamanga_chapter(self.url, self.output_dir)
        except Exception as exc:  # noqa: BLE001 - surface the Playwright/network error to UI
            self.error.emit(str(exc))
            return

        if not files:
            self.error.emit("No images were downloaded from the provided ColaManga URL.")
            return

        self.completed.emit(files)


class ImportColaMangaDialog(QtWidgets.QDialog):
    """Dialog allowing the user to import a ColaManga chapter by URL."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Import from ColaManga"))
        self.setModal(True)

        self.temp_dir = tempfile.mkdtemp(prefix="colamanga_import_")
        self._cleanup_on_close = True
        self.download_worker: ColaMangaDownloadWorker | None = None
        self._downloaded_files: list[str] = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        description = QtWidgets.QLabel(
            self.tr(
                "Paste the ColaManga chapter URL. The application will launch a headless browser "
                "to gather the image list and download the chapter into a temporary project."
            )
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.url_input = QtWidgets.QLineEdit(self)
        self.url_input.setPlaceholderText(self.tr("https://www.colamanga.com/..."))
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
                self.tr("Please enter a valid ColaManga chapter URL."),
            )
            return

        if self.download_worker and self.download_worker.isRunning():
            return

        self._toggle_ui_for_download(True)
        self.status_label.setText(self.tr("Fetching chapter details..."))
        self.status_label.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.download_worker = ColaMangaDownloadWorker(url, self.temp_dir, self)
        self.download_worker.error.connect(self._on_error)
        self.download_worker.completed.connect(self._on_completed)
        self.download_worker.start()

    def _toggle_ui_for_download(self, downloading: bool) -> None:
        self.url_input.setEnabled(not downloading)
        self.import_button.setEnabled(not downloading)
        self.cancel_button.setEnabled(not downloading)

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
