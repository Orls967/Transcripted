"""Silero VAD (ONNX) dan pemotong segmen berbasis jeda hening."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from importlib import resources

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms; ukuran frame yang diwajibkan Silero VAD v5 pada 16 kHz
CONTEXT_SAMPLES = 64  # ekor frame sebelumnya yang ditempel di depan tiap frame
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE

MAX_SEGMENT_SEC = 10.0
MIN_SPEECH_MS = 250  # segmen dengan suara lebih pendek dari ini dibuang
TAIL_MS = 200  # hening yang disisakan di akhir segmen supaya kata terakhir tidak terpotong
FORCED_CUT_WINDOW_SEC = 2.0  # saat potong paksa, cari titik hening di rentang akhir ini
NEG_THRESHOLD_OFFSET = 0.15  # histeresis akhir-bicara (konvensi Silero)


class SileroVAD:
    """Probabilitas suara per frame 512 sampel, dengan state rekuren antar frame."""

    def __init__(self) -> None:
        model = resources.files("live_stt") / "assets" / "silero_vad.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with resources.as_file(model) as path:
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame])[np.newaxis, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(out[0, 0])


@dataclass
class Segment:
    audio: np.ndarray  # float32, 16 kHz mono
    start_sample: int  # posisi awal segmen dihitung dari awal stream

    @property
    def start_sec(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Segmenter:
    """Mengumpulkan audio dan memotongnya menjadi segmen ucapan.

    Segmen dimulai saat VAD mendeteksi suara (diawali pre-roll), dan diakhiri saat
    hening berlangsung `min_silence_ms` atau panjangnya mencapai `max_segment_sec`.
    Audio tanpa suara tidak pernah menghasilkan segmen.
    """

    def __init__(
        self,
        vad: SileroVAD,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        pre_roll_ms: int = 300,
        max_segment_sec: float = MAX_SEGMENT_SEC,
    ) -> None:
        self._vad = vad
        self._threshold = threshold
        self._neg_threshold = max(threshold - NEG_THRESHOLD_OFFSET, 0.01)
        self._min_silence_frames = max(1, round(min_silence_ms / FRAME_MS))
        self._max_frames = max(1, int(max_segment_sec * SAMPLE_RATE // FRAME_SAMPLES))
        self._min_speech_frames = max(1, round(MIN_SPEECH_MS / FRAME_MS))
        self._tail_frames = round(TAIL_MS / FRAME_MS)
        self._cut_window_frames = int(FORCED_CUT_WINDOW_SEC * SAMPLE_RATE // FRAME_SAMPLES)

        self._preroll: deque[np.ndarray] = deque(maxlen=round(pre_roll_ms / FRAME_MS))
        self._pending = np.zeros(0, dtype=np.float32)  # sisa sampel yang belum genap 1 frame
        self._position = 0  # jumlah sampel yang sudah diproses sebagai frame
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._voiced: list[bool] = []
        self._segment_start = 0
        self._silence_run = 0
        self.dropped_short = 0

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Masukkan audio 16 kHz mono (panjang bebas); kembalikan segmen yang selesai."""
        data = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n_frames = len(data) // FRAME_SAMPLES
        segments: list[Segment] = []
        for i in range(n_frames):
            frame = data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            segments.extend(self._process_frame(frame))
        self._pending = data[n_frames * FRAME_SAMPLES :]
        return segments

    def flush(self) -> list[Segment]:
        """Akhir stream: keluarkan segmen yang sedang berjalan (bila ada)."""
        segments: list[Segment] = []
        if self._in_speech:
            if len(self._pending):
                self._frames.append(self._pending)
                self._voiced.append(False)
                self._silence_run += 1
            segments.extend(self._emit(self._trailing_silence_cut()))
        self._pending = np.zeros(0, dtype=np.float32)
        self._reset_segment()
        self._preroll.clear()
        self._vad.reset()
        return segments

    def _process_frame(self, frame: np.ndarray) -> list[Segment]:
        prob = self._vad(frame)
        self._position += FRAME_SAMPLES

        if not self._in_speech:
            if prob < self._threshold:
                self._preroll.append(frame)
                return []
            self._in_speech = True
            self._frames = [*self._preroll, frame]
            self._voiced = [False] * len(self._preroll) + [True]
            self._segment_start = self._position - len(self._frames) * FRAME_SAMPLES
            self._silence_run = 0
            self._preroll.clear()
            return []

        voiced = prob >= self._neg_threshold
        self._frames.append(frame)
        self._voiced.append(voiced)
        self._silence_run = 0 if voiced else self._silence_run + 1

        if self._silence_run >= self._min_silence_frames:
            segments = self._emit(self._trailing_silence_cut())
            self._to_idle()
            return segments
        if len(self._frames) >= self._max_frames:
            return self._forced_cut()
        return []

    def _trailing_silence_cut(self) -> int:
        """Indeks potong yang membuang hening di ekor, menyisakan sedikit tail."""
        keep = min(self._silence_run, self._tail_frames)
        return len(self._frames) - self._silence_run + keep

    def _forced_cut(self) -> list[Segment]:
        """Segmen mencapai panjang maksimum: potong di titik hening terakhir pada
        rentang akhir; bila tidak ada hening, di frame berenergi terendah (paling
        mungkin celah antar kata)."""
        n = len(self._frames)
        window = range(max(n - self._cut_window_frames, 0), n)
        silent = [i for i in window if not self._voiced[i]]
        if silent:
            cut = silent[-1] + 1
        else:
            cut = min(window, key=lambda i: float(np.dot(self._frames[i], self._frames[i]))) + 1
        segments = self._emit(cut)
        rest = slice(cut, None)
        if any(self._voiced[rest]):
            # Ucapan masih berlanjut: sisanya menjadi awal segmen berikutnya.
            self._segment_start += cut * FRAME_SAMPLES
            self._frames = self._frames[rest]
            self._voiced = self._voiced[rest]
        else:
            self._frames = self._frames[rest]
            self._to_idle()
        return segments

    def _emit(self, cut: int) -> list[Segment]:
        # Potongan dengan suara terlalu singkat (bunyi klik, atau ekor kata sisa potong
        # paksa) hanya akan memancing halusinasi Whisper, jadi tidak dikirim.
        voiced = sum(self._voiced[:cut])
        if voiced < self._min_speech_frames:
            if voiced:
                self.dropped_short += 1
            return []
        audio = np.concatenate(self._frames[:cut])
        return [Segment(audio=audio, start_sample=self._segment_start)]

    def _to_idle(self) -> None:
        # Frame terakhir tetap disimpan sebagai pre-roll untuk segmen berikutnya.
        self._preroll.extend(self._frames[-self._preroll.maxlen :] if self._preroll.maxlen else [])
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._frames = []
        self._voiced = []
        self._silence_run = 0
