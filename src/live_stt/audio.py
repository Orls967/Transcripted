"""Sumber audio (producer): mikrofon lewat sounddevice, atau file lewat soundfile.

Keduanya mengisi queue yang sama dengan blok float32 mono pada sample rate sumber;
`None` menandai akhir stream.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from .vad import SAMPLE_RATE

BLOCK_SEC = 0.032
FILE_QUEUE_SEC = 30  # backpressure mode file: paling banyak segini audio menunggu di queue


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    samplerate: int
    is_default: bool


def list_input_devices() -> list[InputDevice]:
    default_input = sd.default.device[0]
    return [
        InputDevice(
            index=i,
            name=d["name"],
            channels=d["max_input_channels"],
            samplerate=int(d["default_samplerate"]),
            is_default=(i == default_input),
        )
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


class MicSource:
    """Merekam dari input device. Callback PortAudio hanya menyalin blok ke queue,
    jadi perekaman tidak pernah menunggu transkripsi."""

    def __init__(self, device: InputDevice, out: queue.Queue) -> None:
        self._queue = out
        self.overflows = 0
        self.started_at = 0.0
        channels = min(2, device.channels)
        # Minta 16 kHz langsung; kalau device menolak, pakai rate aslinya (di-resample konsumen).
        last_error: Exception | None = None
        for rate in dict.fromkeys([SAMPLE_RATE, device.samplerate]):
            try:
                self._stream = sd.InputStream(
                    device=device.index,
                    channels=channels,
                    samplerate=rate,
                    blocksize=round(rate * BLOCK_SEC),
                    dtype="float32",
                    callback=self._callback,
                )
            except sd.PortAudioError as exc:
                last_error = exc
                continue
            self.samplerate = rate
            self.block_sec = round(rate * BLOCK_SEC) / rate
            return
        raise RuntimeError(f"Tidak bisa membuka device '{device.name}': {last_error}")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0].copy()
        self._queue.put_nowait(mono)

    def start(self) -> None:
        self.started_at = time.time()
        self._stream.start()

    def stop(self) -> None:
        """Hentikan perekaman dan tandai akhir stream."""
        self._stream.stop()
        self._stream.close()
        self._queue.put(None)


class FileSource(threading.Thread):
    """Membaca file audio (WAV/MP3/FLAC/…) blok demi blok ke queue yang sama."""

    def __init__(self, path: Path, out: queue.Queue) -> None:
        super().__init__(name="file-reader", daemon=True)
        self._path = path
        self._queue = out
        self._stop_requested = threading.Event()
        self.error: Exception | None = None
        info = sf.info(str(path))
        self.samplerate = info.samplerate
        self.duration_sec = info.duration
        self.block_sec = round(info.samplerate * BLOCK_SEC) / info.samplerate

    @staticmethod
    def queue_maxsize() -> int:
        return ceil(FILE_QUEUE_SEC / BLOCK_SEC)

    def run(self) -> None:
        try:
            for block in self._blocks():
                if not self._put(block):
                    break
        except Exception as exc:  # dilaporkan oleh main thread
            self.error = exc
        finally:
            self._queue.put(None)

    def _blocks(self):
        with sf.SoundFile(str(self._path)) as f:
            blocksize = round(f.samplerate * BLOCK_SEC)
            if f.format == "MP3":
                # Dekoder MP3 libsndfile mengeluarkan error bila MP3 ber-sample-rate
                # rendah dibaca sepotong-sepotong, jadi MP3 didekode sekaligus.
                data = f.read(dtype="float32", always_2d=True).mean(axis=1)
                for i in range(0, len(data), blocksize):
                    yield data[i : i + blocksize]
            else:
                for block in f.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
                    yield block.mean(axis=1)

    def _put(self, block: np.ndarray) -> bool:
        while not self._stop_requested.is_set():
            try:
                self._queue.put(block, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def stop(self) -> None:
        self._stop_requested.set()


class StreamResampler:
    """Resample streaming ke 16 kHz tanpa artefak di batas blok.

    Tiap potongan di-resample bersama konteks di kiri-kanannya, lalu hanya bagian
    tengah yang dipakai; keluaran tertunda sebesar konteks (~10 ms).
    """

    def __init__(self, source_rate: int, target_rate: int = SAMPLE_RATE) -> None:
        g = gcd(source_rate, target_rate)
        self._up = target_rate // g
        self._down = source_rate // g
        self._margin = self._down * ceil(source_rate * 0.01 / self._down)
        self._buffer = np.zeros(self._margin, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        ready = (len(self._buffer) - 2 * self._margin) // self._down * self._down
        if ready <= 0:
            return np.zeros(0, dtype=np.float32)
        chunk = self._buffer[: ready + 2 * self._margin]
        resampled = resample_poly(chunk, self._up, self._down)
        skip = self._margin // self._down * self._up
        out = resampled[skip : skip + ready // self._down * self._up]
        self._buffer = self._buffer[ready:]
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """Keluarkan sisa audio di akhir stream (konteks kanan diisi nol)."""
        remaining = len(self._buffer) - self._margin
        pad = self._margin + (-remaining) % self._down
        out = self.process(np.zeros(pad, dtype=np.float32))
        self._buffer = np.zeros(self._margin, dtype=np.float32)
        return out
