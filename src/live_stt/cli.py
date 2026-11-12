"""CLI live speech-to-text lokal: mic/file → Silero VAD → mlx-whisper."""

from __future__ import annotations

import argparse
import os
import queue
import sys
from datetime import datetime
from pathlib import Path

from .output import TranscriptWriter, status

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="live-stt",
        description="Live speech-to-text 100% lokal (mlx-whisper + Silero VAD).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="repo Hugging Face atau path model MLX Whisper (default: %(default)s)")
    parser.add_argument("--language", default="id", help="kode bahasa Whisper, atau 'auto' untuk deteksi otomatis (default: %(default)s)")
    parser.add_argument("--device", help="input device: nomor indeks atau potongan nama (melewati menu)")
    parser.add_argument("--file", type=Path, help="transkrip file audio (WAV/MP3) lewat pipeline yang sama, tanpa mic")
    vad = parser.add_argument_group("VAD")
    vad.add_argument("--vad-threshold", type=float, default=0.5, help="ambang probabilitas suara, 0–1 (default: %(default)s)")
    vad.add_argument("--min-silence-ms", type=int, default=600, help="durasi hening yang memotong segmen (default: %(default)s)")
    vad.add_argument("--pre-roll-ms", type=int, default=300, help="audio sebelum suara terdeteksi yang ikut disertakan (default: %(default)s)")
    vad.add_argument("--no-enhance", action="store_true", help="matikan pra-proses suara jauh/bising (high-pass, peredam noise, penguatan otomatis)")
    args = parser.parse_args(argv)

    if not 0.0 < args.vad_threshold < 1.0:
        parser.error("--vad-threshold harus di antara 0 dan 1")
    if args.min_silence_ms <= 0:
        parser.error("--min-silence-ms harus lebih besar dari 0")
    if args.pre_roll_ms < 0:
        parser.error("--pre-roll-ms tidak boleh negatif")
    if args.file is not None and args.device is not None:
        parser.error("--file dan --device tidak bisa dipakai bersamaan")
    if args.file is not None and not args.file.is_file():
        parser.error(f"file tidak ditemukan: {args.file}")
    return args


def choose_device(requested: str | None):
    from .audio import list_input_devices

    devices = list_input_devices()
    if not devices:
        sys.exit("Tidak ada input device audio yang ditemukan.")

    if requested is not None:
        if requested.isdigit():
            matches = [d for d in devices if d.index == int(requested)]
        else:
            matches = [d for d in devices if requested.lower() in d.name.lower()]
        if not matches:
            names = "\n".join(f"  [{d.index}] {d.name}" for d in devices)
            sys.exit(f"Input device '{requested}' tidak ditemukan. Yang tersedia:\n{names}")
        return matches[0]

    default = next((d for d in devices if d.is_default), devices[0])
    status("Input device tersedia:")
    for d in devices:
        mark = "  ← default" if d is default else ""
        status(f"  [{d.index}] {d.name} ({d.channels} ch, {d.samplerate} Hz){mark}")
    while True:
        try:
            answer = input(f"Pilih nomor device [Enter = {default.index}]: ").strip()
        except EOFError:
            return default
        if not answer:
            return default
        chosen = [d for d in devices if answer.isdigit() and d.index == int(answer)]
        if chosen:
            return chosen[0]
        status("Nomor tidak valid, coba lagi.")


def format_offset(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        code = run(args)
    except KeyboardInterrupt:
        status("\nDibatalkan.")
        force_exit(130)
    sys.exit(code)


def force_exit(code: int) -> None:
    """Keluar seketika tanpa menunggu thread transkripsi yang masih bekerja."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run(args: argparse.Namespace) -> int:
    from .audio import FileSource, MicSource
    from .transcriber import TranscriberThread

    language = None if args.language.lower() == "auto" else args.language.lower()

    if args.file is not None:
        audio_queue: queue.Queue = queue.Queue(maxsize=FileSource.queue_maxsize())
        try:
            source = FileSource(args.file, audio_queue)
        except Exception as exc:
            status(f"Tidak bisa membaca file audio: {exc}")
            return 1
        label = f"file: {args.file.name}"
        stamp = format_offset
    else:
        device = choose_device(args.device)
        audio_queue = queue.Queue()
        try:
            source = MicSource(device, audio_queue)
        except RuntimeError as exc:
            status(str(exc))
            return 1
        label = f"mic: {device.name}"

        def stamp(offset: float) -> str:
            return f"{datetime.fromtimestamp(source.started_at + offset):%H:%M:%S}"

    writer = TranscriptWriter(Path.cwd(), f"{label} | model: {args.model}")
    worker = TranscriberThread(
        audio_queue,
        writer,
        model=args.model,
        language=language,
        vad_threshold=args.vad_threshold,
        min_silence_ms=args.min_silence_ms,
        pre_roll_ms=args.pre_roll_ms,
        enhance=not args.no_enhance,
    )
    worker.configure_source(
        samplerate=source.samplerate,
        block_sec=source.block_sec,
        stamp=stamp,
        live=args.file is None,
    )

    status(f"Memuat model {args.model}…")
    worker.start()
    while not worker.ready.wait(0.2):
        pass
    if worker.error is not None:
        status(f"Gagal memuat model: {worker.error}")
        return 1

    if args.file is not None:
        status(f"Mentranskrip {args.file.name} ({format_offset(source.duration_sec)})…")
    else:
        status(f"Merekam dari '{label[5:]}' @ {source.samplerate} Hz. Ctrl+C untuk berhenti.")
    source.start()

    # Menunggu lewat Event, bukan Thread.join(): di Python <= 3.12, join() yang
    # terinterupsi Ctrl+C bisa keliru menandai thread sudah selesai.
    interrupted = False
    try:
        while not worker.done.wait(0.2):
            pass
    except KeyboardInterrupt:
        interrupted = True
        status("\nMenghentikan… memproses audio tersisa (Ctrl+C lagi untuk keluar paksa).")
        source.stop()
        try:
            while not worker.done.wait(0.2):
                pass
        except KeyboardInterrupt:
            writer.close()
            status("Keluar paksa.")
            force_exit(130)

    saved = writer.last_path
    writer.close()

    source_error = getattr(source, "error", None)
    if worker.error is not None or source_error is not None:
        status(f"Terjadi kesalahan: {worker.error or source_error}")
        return 1

    overflows = getattr(source, "overflows", 0)
    if overflows:
        status(f"⚠ {overflows} blok audio hilang di sisi driver (input overflow).")
    speed = worker.audio_sec / worker.transcribe_sec if worker.transcribe_sec else 0.0
    status(
        f"Selesai: {worker.audio_sec:.1f} dtk audio, {worker.segments_sent} segmen dikirim ke "
        f"Whisper ({worker.transcribe_sec:.1f} dtk"
        + (f", {speed:.0f}x realtime" if speed else "")
        + f"), {worker.lines_written} baris ditulis, {worker.filtered} dibuang filter, "
        f"{worker.dropped_short} terlalu pendek."
    )
    if saved is not None:
        status(f"Transkrip tersimpan di {saved.name}")
    return 130 if interrupted else 0
