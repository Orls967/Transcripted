"""Thread konsumen: queue audio → VAD → segmen → mlx-whisper → filter → keluaran."""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .audio import StreamResampler
from .output import TranscriptWriter, status
from .vad import SAMPLE_RATE, Segment, Segmenter

INITIAL_PROMPT_ID = (
    "Berikut adalah percakapan dalam bahasa Indonesia yang sering bercampur dengan "
    "istilah English, misalnya meeting, deadline, feedback, dan update project."
)

# Konvensi Whisper: segmen dianggap bukan ucapan hanya bila no_speech_prob tinggi
# DAN avg_logprob rendah. no_speech_prob tinggi saja tidak cukup, supaya ucapan
# pelan yang valid tidak ikut terbuang.
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4  # di atas ini teks berulang-ulang (loop)
MAX_CONSECUTIVE_REPEATS = 2  # teks identik ke-3 berturut-turut dibuang
# Whisper mengisi audio pendek dengan hening sampai 30 dtk dan kadang "mendengar"
# kalimat di hening itu (mis. "Terima kasih."). Cirinya: teks dimulai di detik terakhir
# audio asli, padahal sisa audionya terlalu singkat untuk mengucapkan teks sepanjang itu.
PADDING_WINDOW_SEC = 1.0
MAX_CHARS_PER_SEC = 30.0

BACKLOG_WARN_SEC = 5.0
BACKLOG_WARN_INTERVAL_SEC = 10.0
SILENT_INPUT_CHECK_SEC = 3.0

# Kalimat yang kerap dihalusinasikan Whisper (berasal dari subtitle video di data latih).
HALLUCINATIONS = [
    "terima kasih telah menonton",
    "terima kasih sudah menonton",
    "terima kasih telah menyaksikan",
    "terima kasih sudah menyaksikan",
    "sampai jumpa di video selanjutnya",
    "sampai jumpa di video berikutnya",
    "jangan lupa subscribe",
    "jangan lupa like",
    "like share dan subscribe",
    "selamat menikmati",
    "sub indo oleh",
    "subtitle oleh",
    "diterjemahkan oleh",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitle by",
    "subtitles by",
    "translated by",
    "amara org",
]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def is_known_hallucination(norm: str, prompt_norm: str) -> bool:
    words = len(norm.split())
    for phrase in HALLUCINATIONS:
        # Dibuang hanya bila segmen nyaris seluruhnya berisi frasa tersebut.
        if phrase in norm and words <= len(phrase.split()) + 3:
            return True
    if prompt_norm and words >= 3:  # gema dari initial prompt
        if norm in prompt_norm or SequenceMatcher(None, norm, prompt_norm).ratio() > 0.8:
            return True
    return False


def resolve_model(model: str) -> str:
    """Kembalikan path lokal model. Bila sudah ada di cache Hugging Face, tidak ada
    akses jaringan sama sekali; bila belum, model diunduh sekali."""
    if Path(model).exists():
        return model
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id=model, local_files_only=True))
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.npz"))
        if (path / "config.json").exists() and has_weights:
            return str(path)
    except Exception:
        pass
    status(f"Model {model} belum ada di cache, mengunduh dari Hugging Face…")
    return snapshot_download(repo_id=model)


class TranscriberThread(threading.Thread):
    def __init__(
        self,
        audio_queue: queue.Queue,
        writer: TranscriptWriter,
        *,
        model: str,
        language: str | None,
        vad_threshold: float,
        min_silence_ms: int,
        pre_roll_ms: int,
    ) -> None:
        super().__init__(name="transcriber", daemon=True)
        self._queue = audio_queue
        self._writer = writer
        self._model = model
        self._language = language
        self._prompt = INITIAL_PROMPT_ID if language == "id" else None
        self._prompt_norm = normalize(self._prompt) if self._prompt else ""
        self._vad_options = dict(
            threshold=vad_threshold, min_silence_ms=min_silence_ms, pre_roll_ms=pre_roll_ms
        )

        # Diisi lewat configure_source() sebelum audio mulai mengalir.
        self._source_rate = SAMPLE_RATE
        self._block_sec = 0.0
        self._stamp: Callable[[float], str] = lambda offset: f"{offset:.1f}"
        self._live = False

        self.ready = threading.Event()  # model sudah dimuat (atau gagal)
        self.done = threading.Event()  # semua audio sudah diproses (atau gagal)
        self.error: BaseException | None = None
        self.segments_sent = 0
        self.lines_written = 0
        self.filtered = 0
        self.dropped_short = 0
        self.audio_sec = 0.0
        self.transcribe_sec = 0.0
        self._last_norm = ""
        self._repeats = 0
        self._last_backlog_warning = 0.0

    def configure_source(
        self, *, samplerate: int, block_sec: float, stamp: Callable[[float], str], live: bool
    ) -> None:
        """`stamp` mengubah posisi segmen (detik sejak awal stream) menjadi HH:MM:SS."""
        self._source_rate = samplerate
        self._block_sec = block_sec
        self._stamp = stamp
        self._live = live

    def run(self) -> None:
        try:
            self._load_model()
            self.ready.set()
            self._consume()
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()
            self.done.set()

    def _load_model(self) -> None:
        # Model dimuat dan dipakai di thread yang sama (MLX sensitif lintas thread).
        import tqdm

        # tqdm (dipakai mlx-whisper) secara default membuat semaphore multiprocessing,
        # yang memicu peringatan "leaked semaphore" bila program keluar paksa.
        tqdm.tqdm.set_lock(threading.RLock())

        import mlx_whisper

        from .vad import SileroVAD

        self._transcribe_fn = mlx_whisper.transcribe
        self._model = resolve_model(self._model)
        self._segmenter = Segmenter(SileroVAD(), **self._vad_options)
        # Pemanasan: memuat bobot dan meng-compile graph, supaya segmen pertama tidak lambat.
        self._run_whisper(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def _run_whisper(self, audio: np.ndarray) -> dict:
        return self._transcribe_fn(
            audio,
            path_or_hf_repo=self._model,
            language=self._language,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            temperature=0.0,
            # Penyaringan dilakukan sendiri di _keep() supaya alasannya eksplisit.
            no_speech_threshold=None,
            logprob_threshold=None,
            compression_ratio_threshold=None,
            verbose=None,
        )

    def _consume(self) -> None:
        resampler = (
            StreamResampler(self._source_rate) if self._source_rate != SAMPLE_RATE else None
        )
        peak = 0.0
        checked_silence = not self._live
        while True:
            block = self._queue.get()
            if block is None:
                break
            if not checked_silence:
                peak = max(peak, float(np.abs(block).max(initial=0.0)))
                if self.audio_sec >= SILENT_INPUT_CHECK_SEC:
                    checked_silence = True
                    if peak == 0.0:
                        status(
                            "⚠ Audio dari device ini nol total. Kemungkinan izin mikrofon belum "
                            "diberikan ke aplikasi terminal (lihat README), atau sumbernya memang diam."
                        )
            if resampler is not None:
                block = resampler.process(block)
            self.audio_sec += len(block) / SAMPLE_RATE
            for segment in self._segmenter.feed(block):
                self._handle(segment)

        tail = [resampler.flush()] if resampler is not None else []
        for chunk in tail:
            self.audio_sec += len(chunk) / SAMPLE_RATE
            for segment in self._segmenter.feed(chunk):
                self._handle(segment)
        for segment in self._segmenter.flush():
            self._handle(segment)
        self.dropped_short = self._segmenter.dropped_short

    def _handle(self, segment: Segment) -> None:
        started = time.perf_counter()
        result = self._run_whisper(segment.audio)
        self.transcribe_sec += time.perf_counter() - started
        self.segments_sent += 1

        texts = [
            s["text"].strip()
            for s in result.get("segments", [])
            if self._keep(s, segment.duration_sec)
        ]
        text = " ".join(t for t in texts if t)
        if text:
            self._writer.write(self._stamp(segment.start_sec), text)
            self.lines_written += 1
        if self._live:
            self._warn_if_backlogged()

    def _keep(self, seg: dict, audio_sec: float) -> bool:
        norm = normalize(seg.get("text", ""))
        if not norm:
            return False
        remaining = audio_sec - seg.get("start", 0.0)
        in_padding = (
            remaining < PADDING_WINDOW_SEC
            and len(norm) / max(remaining, 0.05) > MAX_CHARS_PER_SEC
        )
        no_speech = (
            seg.get("no_speech_prob", 0.0) > NO_SPEECH_THRESHOLD
            and seg.get("avg_logprob", 0.0) < LOGPROB_THRESHOLD
        )
        looping = seg.get("compression_ratio", 0.0) > COMPRESSION_RATIO_THRESHOLD
        if no_speech or looping or in_padding or is_known_hallucination(norm, self._prompt_norm):
            self.filtered += 1
            return False
        # Teks identik yang terus berulang antar segmen adalah ciri khas halusinasi.
        if norm == self._last_norm:
            self._repeats += 1
        else:
            self._last_norm, self._repeats = norm, 0
        if self._repeats >= MAX_CONSECUTIVE_REPEATS:
            self.filtered += 1
            return False
        return True

    def _warn_if_backlogged(self) -> None:
        backlog_sec = self._queue.qsize() * self._block_sec
        now = time.monotonic()
        if (
            backlog_sec > BACKLOG_WARN_SEC
            and now - self._last_backlog_warning > BACKLOG_WARN_INTERVAL_SEC
        ):
            self._last_backlog_warning = now
            status(
                f"⚠ Antrean menumpuk: {backlog_sec:.1f} dtk audio belum diproses "
                "(transkripsi lebih lambat dari audio masuk). Coba model yang lebih kecil."
            )
