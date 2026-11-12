"""Keluaran transkrip: terminal (stdout) dan file transcript_YYYY-MM-DD.txt (append)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


def status(message: str) -> None:
    """Pesan status/peringatan ke stderr, supaya stdout hanya berisi transkrip."""
    print(message, file=sys.stderr, flush=True)


class TranscriptWriter:
    def __init__(self, directory: Path, session_label: str) -> None:
        self._directory = directory
        self._session_label = session_label
        self._file: TextIO | None = None
        self._path: Path | None = None
        self.last_path: Path | None = None

    def write(self, stamp: str, text: str) -> None:
        line = f"[{stamp}] {text}"
        print(line, flush=True)
        f = self._file_for_today()
        f.write(line + "\n")
        f.flush()

    def _file_for_today(self) -> TextIO:
        now = datetime.now()
        path = self._directory / f"transcript_{now:%Y-%m-%d}.txt"
        if path != self._path:  # run pertama, atau melewati tengah malam
            self.close()
            self._file = path.open("a", encoding="utf-8")
            self._path = self.last_path = path
            self._file.write(f"--- Sesi {now:%Y-%m-%d %H:%M:%S} | {self._session_label} ---\n")
        assert self._file is not None
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._path = None
