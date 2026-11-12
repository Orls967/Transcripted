# live-stt

Live speech-to-text di terminal yang berjalan 100% lokal di Mac Apple Silicon:
mikrofon → Silero VAD (ONNX) → [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (akselerasi Metal).
Default-nya bahasa Indonesia dengan prompt bilingual Indonesia–Inggris.

## Install

Butuh macOS di Apple Silicon dan [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Tidak perlu PyTorch maupun ffmpeg. Saat pertama dijalankan, model Whisper
(`mlx-community/whisper-large-v3-turbo`, ±1,6 GB) diunduh sekali ke cache Hugging Face;
setelah itu program tidak mengakses jaringan sama sekali.

## Menjalankan

```bash
uv run live-stt
```

Program menampilkan daftar input device, lalu mulai merekam. Hasil muncul per segmen
ucapan dengan timestamp `[HH:MM:SS]` dan otomatis ditambahkan (append) ke
`transcript_YYYY-MM-DD.txt` di folder tempat program dijalankan. Tekan `Ctrl+C` untuk
berhenti: audio yang tersisa diproses dulu, lalu file ditutup (`Ctrl+C` kedua = keluar paksa).

| Opsi | Default | Keterangan |
| --- | --- | --- |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Repo Hugging Face atau path model MLX Whisper |
| `--language` | `id` | Kode bahasa Whisper, atau `auto` untuk deteksi otomatis |
| `--device` | (menu) | Nomor indeks atau potongan nama input device, melewati menu |
| `--file` | – | Transkrip file audio (WAV/MP3) lewat pipeline yang sama, tanpa mic |
| `--vad-threshold` | `0.5` | Ambang probabilitas suara (0–1); naikkan bila ruangan bising |
| `--min-silence-ms` | `600` | Durasi hening yang memotong segmen |
| `--pre-roll-ms` | `300` | Audio sebelum suara terdeteksi yang ikut disertakan |
| `--no-enhance` | – | Matikan pra-proses suara jauh/bising (lihat di bawah) |

Contoh:

```bash
uv run live-stt --device BlackHole --language en
```

```bash
uv run live-stt --file rekaman.mp3
```

Pada mode `--file`, timestamp menunjukkan posisi di dalam file.

## Izin mikrofon macOS

macOS memberi izin mikrofon ke **aplikasi terminal** yang menjalankan program (Terminal,
iTerm, VS Code, dst.), bukan ke `live-stt` sendiri.

- Saat pertama merekam, macOS menampilkan dialog izin. Pilih **Allow**.
- Kalau dialognya tidak muncul atau pernah ditolak, buka **System Settings → Privacy &
  Security → Microphone**, aktifkan aplikasi terminal Anda, lalu tutup dan buka lagi
  terminalnya.
- Tanpa izin, macOS tidak memberi error; audio yang masuk hanya hening total. Program
  akan memperingatkan bila audio yang diterima nol semua.

## Suara jauh dan ruangan bising

Mic laptop menangkap banyak gemuruh frekuensi rendah (getaran meja, ketikan, kipas), dan
VAD tidak mengenali ucapan yang levelnya dekat dengan lantai noise, padahal Whisper masih
sanggup membacanya. Karena itu, sebelum VAD audio melewati high-pass 100 Hz, peredam noise
spektral, dan penguatan otomatis. Whisper sendiri menerima audio yang hanya di-high-pass.

- Suara latar yang terlalu samar tetap bisa memicu segmen. Hasilnya dibuang bila Whisper
  ragu (keyakinan rendah) atau hanya mengeluarkan frasa basa-basi seperti "Terima kasih.".
- Peredam noise efektif untuk noise yang ajek (kipas, AC, dengung). Musik atau orang lain
  yang ikut berbicara tidak bisa dipisahkan.
- Volume input mic di **System Settings → Sound → Input** ikut berpengaruh; naikkan bila
  pembicara jauh.
- `--no-enhance` mematikan pra-proses ini, mis. untuk membandingkan hasil.

## Merekam audio sistem (BlackHole)

Untuk mentranskrip suara dari aplikasi lain (rapat online, video), pasang
[BlackHole](https://github.com/ExistentialAudio/BlackHole), arahkan output audio macOS
ke BlackHole (atau ke Multi-Output Device supaya tetap terdengar di speaker), lalu pilih
BlackHole di menu device atau lewat `--device BlackHole`.

## Catatan

- Segmen dipotong saat ada jeda hening dan dibatasi 10 detik. Ucapan yang lebih panjang
  dari itu tanpa jeda dipotong paksa di titik paling sunyi, jadi sesekali ada kata yang
  kurang tepat di sambungannya.
- Bila muncul peringatan "Antrean menumpuk", transkripsi lebih lambat dari audio yang
  masuk; pakai model lebih kecil, mis. `--model mlx-community/whisper-small-mlx`.
- Model VAD (`src/live_stt/assets/silero_vad.onnx`, lisensi MIT) berasal dari
  [snakers4/silero-vad](https://github.com/snakers4/silero-vad) v5.1.2.
