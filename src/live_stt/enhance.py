"""Pra-proses audio mic supaya suara jauh/pelan dan ruangan bising tetap terdeteksi.

Mic laptop menangkap gemuruh frekuensi rendah (getaran meja, ketikan, kipas) yang
bisa 20-30 dB lebih keras dari ucapan jarak jauh, dan Silero VAD tidak mengenali
ucapan yang levelnya terlalu dekat dengan lantai noise. `Enhancer` menghasilkan dua
sinyal yang sejajar sampel-demi-sampel:

- `audio`: hanya di-high-pass. Inilah yang dikirim ke Whisper (Whisper sendiri
  tahan terhadap level rendah dan noise, jadi tidak diubah lebih jauh).
- `vad`: high-pass -> peredam noise spektral -> penguatan otomatis (AGC). Hanya
  dipakai untuk keputusan "ada suara atau tidak".
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .vad import SAMPLE_RATE

HIGHPASS_HZ = 100.0

FFT_SIZE = 512
HOP = FFT_SIZE // 2  # overlap 50%; keluaran tertunda satu hop (16 ms)
HOPS_PER_SEC = SAMPLE_RATE / HOP

# Peredam noise (Wiener) dengan estimasi noise "pengikut minimum" per bin frekuensi:
# langsung turun mengikuti energi terendah, naik pelan-pelan saat energi lebih tinggi.
PSD_SMOOTHING = 0.8
NOISE_RISE_DB_PER_SEC = 3.0
OVERSUBTRACTION = 3.0
GAIN_FLOOR = 0.15

# AGC: level ucapan diikuti dengan serangan seketika dan pelepasan lambat, lalu
# dikuatkan ke level target. Penguatan dibatasi supaya noise sisa tidak ikut terangkat
# sampai dianggap suara.
TARGET_RMS = 0.05
LEVEL_RELEASE_DB_PER_SEC = 6.0
MAX_GAIN = 100.0  # +40 dB
MAX_NOISE_RMS = 0.003  # noise sisa setelah penguatan dijaga di bawah ~-50 dBFS
FLOOR_RISE_DB_PER_SEC = 1.0
NOISE_WINDOW_SEC = 15.0  # lantai noise ruangan = persentil 10 level dalam jendela ini
NOISE_UPDATE_HOPS = 16


class Enhancer:
    def __init__(self) -> None:
        self._sos = butter(4, HIGHPASS_HZ, "highpass", fs=SAMPLE_RATE, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2))
        # sqrt-Hann periodik: analisis x sintesis berjumlah 1 pada overlap 50%.
        self._window = np.sqrt(np.hanning(FFT_SIZE + 1)[:-1]).astype(np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._previous = np.zeros(HOP, dtype=np.float32)
        self._overlap = np.zeros(HOP, dtype=np.float32)
        self._psd: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._noise_rise = 10 ** (NOISE_RISE_DB_PER_SEC / 10 / HOPS_PER_SEC)
        self._level = 0.0
        self._level_release = 10 ** (-LEVEL_RELEASE_DB_PER_SEC / 20 / HOPS_PER_SEC)
        self._floor = 0.0
        self._floor_rise = 10 ** (FLOOR_RISE_DB_PER_SEC / 20 / HOPS_PER_SEC)
        self._gain = 1.0
        # Lantai noise audio (setelah high-pass, sebelum peredam), untuk menilai seberapa
        # samar sebuah segmen dibanding ruangannya.
        self.noise_rms = 0.0
        self._levels = np.zeros(int(NOISE_WINDOW_SEC * HOPS_PER_SEC), dtype=np.float32)
        self._hops_seen = 0

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kembalikan (audio, vad) yang sama panjang; panjangnya kelipatan HOP."""
        filtered, self._zi = sosfilt(self._sos, samples, zi=self._zi)
        data = np.concatenate([self._pending, filtered.astype(np.float32)])
        n_hops = len(data) // HOP
        audio = np.empty(n_hops * HOP, dtype=np.float32)
        vad = np.empty(n_hops * HOP, dtype=np.float32)
        for i in range(n_hops):
            hop = data[i * HOP : (i + 1) * HOP]
            out = slice(i * HOP, (i + 1) * HOP)
            # Hasil peredam untuk iterasi ini sejajar dengan hop sebelumnya.
            audio[out] = self._previous
            vad[out] = self._auto_gain(self._denoise(hop))
            self._track_noise(hop)
            self._previous = hop
        self._pending = data[n_hops * HOP :]
        return audio, vad

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        """Keluarkan sisa audio di akhir stream."""
        pad = HOP + (-len(self._pending)) % HOP
        return self.process(np.zeros(pad, dtype=np.float32))

    def _track_noise(self, hop: np.ndarray) -> None:
        self._levels[self._hops_seen % len(self._levels)] = np.sqrt(np.mean(hop * hop))
        self._hops_seen += 1
        if self._hops_seen % NOISE_UPDATE_HOPS == 0:
            seen = self._levels[: min(self._hops_seen, len(self._levels))]
            self.noise_rms = float(np.percentile(seen, 10))

    def _denoise(self, hop: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(np.concatenate([self._previous, hop]) * self._window)
        power = spectrum.real**2 + spectrum.imag**2
        if self._psd is None or self._noise is None:
            self._psd = power.copy()
            self._noise = power.copy()
        else:
            self._psd = PSD_SMOOTHING * self._psd + (1 - PSD_SMOOTHING) * power
            self._noise = np.minimum(self._psd, self._noise * self._noise_rise)
        gain = np.maximum(
            1.0 - OVERSUBTRACTION * self._noise / np.maximum(power, 1e-20), GAIN_FLOOR
        )
        frame = np.fft.irfft(spectrum * gain, FFT_SIZE).astype(np.float32) * self._window
        result = self._overlap + frame[:HOP]
        self._overlap = frame[HOP:]
        return result

    def _auto_gain(self, hop: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(hop * hop)))
        self._level = max(rms, self._level * self._level_release)
        self._floor = rms if self._floor == 0.0 else min(rms, self._floor * self._floor_rise)
        target = min(
            TARGET_RMS / max(self._level, 1e-9),
            MAX_NOISE_RMS / max(self._floor, 1e-9),
            MAX_GAIN,
        )
        target = max(target, 1.0)
        ramp = np.linspace(self._gain, target, HOP, endpoint=False, dtype=np.float32)
        self._gain = target
        return np.clip(hop * ramp, -1.0, 1.0)
