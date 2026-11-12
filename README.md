# live-stt

Live speech-to-text di terminal yang berjalan 100% lokal di Mac Apple Silicon:
mikrofon → Silero VAD (ONNX) → [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (akselerasi Metal).
Default-nya bahasa Indonesia dengan prompt bilingual Indonesia–Inggris.

```
[17:04:12] Besok kita ada meeting dengan klien jam 10 pagi.
[17:04:16] Jangan lupa update progress project dan kirim feedback sebelum deadline.
```

## Daftar isi

- [Install](#install)
- [Menjalankan](#menjalankan)
- [Opsi](#opsi)
- [Keluaran](#keluaran)
- [Izin mikrofon macOS](#izin-mikrofon-macos)
- [Merekam audio sistem (BlackHole)](#merekam-audio-sistem-blackhole)
- [Cara kerja](#cara-kerja)
- [Suara jauh dan ruangan bising](#suara-jauh-dan-ruangan-bising)
- [Filter anti-halusinasi](#filter-anti-halusinasi)
- [Hasil pengukuran](#hasil-pengukuran)
- [Menguji tanpa mikrofon](#menguji-tanpa-mikrofon)
- [Pemecahan masalah](#pemecahan-masalah)
- [Keterbatasan](#keterbatasan)
- [Struktur project](#struktur-project)

## Install

Butuh macOS di Apple Silicon dan [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Orls967/Transcripted.git
```

```bash
cd Transcripted
```

```bash
uv sync
```

- Python 3.12 dipasang otomatis oleh uv (project mendukung 3.11–3.13).
- Tidak perlu PyTorch maupun ffmpeg.
- Saat pertama dijalankan, model Whisper (`mlx-community/whisper-large-v3-turbo`, ±1,6 GB)
  diunduh sekali ke cache Hugging Face (`~/.cache/huggingface/`). Setelah itu program
  tidak mengakses jaringan sama sekali.

## Menjalankan

Jalankan dari dalam folder project:

```bash
uv run live-stt
```

Program menampilkan daftar input device, lalu mulai merekam setelah Anda memilih (Enter =
device default). Tekan `Ctrl+C` untuk berhenti: audio yang tersisa diproses dulu, lalu
file transkrip ditutup. `Ctrl+C` kedua = keluar paksa.

Untuk menjalankan dari folder mana pun:

```bash
uv run --directory ~/Documents/Transcripted live-stt
```

Contoh lain:

```bash
uv run live-stt --device BlackHole --language en
```

```bash
uv run live-stt --file rekaman.mp3
```

```bash
uv run live-stt --model mlx-community/whisper-small-mlx
```

## Opsi

| Opsi | Default | Keterangan |
| --- | --- | --- |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Repo Hugging Face atau path model MLX Whisper |
| `--language` | `id` | Kode bahasa Whisper, atau `auto` untuk deteksi otomatis |
| `--device` | (menu) | Nomor indeks atau potongan nama input device, melewati menu |
| `--file` | – | Transkrip file audio (WAV/MP3) lewat pipeline yang sama, tanpa mic |
| `--vad-threshold` | `0.5` | Ambang probabilitas suara (0–1); naikkan bila terlalu banyak pemicu palsu |
| `--min-silence-ms` | `600` | Durasi hening yang memotong segmen |
| `--pre-roll-ms` | `300` | Audio sebelum suara terdeteksi yang ikut disertakan |
| `--no-enhance` | – | Matikan pra-proses suara jauh/bising |

Prompt bilingual hanya dipakai saat `--language id`. `--file` dan `--device` tidak bisa
dipakai bersamaan.

## Keluaran

- **Terminal (stdout)**: satu baris per segmen ucapan, `[HH:MM:SS] teks`. Pada mode mic
  timestamp adalah jam saat segmen dimulai; pada mode `--file` posisi di dalam file.
- **File**: baris yang sama ditambahkan (append) ke `transcript_YYYY-MM-DD.txt` di folder
  tempat program dijalankan, diawali satu baris pembuka per sesi. File berganti otomatis
  saat melewati tengah malam.
- **Status (stderr)**: pesan muat model, peringatan, dan ringkasan di akhir, misalnya:

```
Selesai: 13.0 dtk audio, 2 segmen dikirim ke Whisper (1.9 dtk, 7x realtime), 2 baris ditulis, 0 dibuang filter, 0 terlalu pendek.
```

Karena transkrip hanya ke stdout, `uv run live-stt > hasil.txt` menyimpan teksnya saja.

## Izin mikrofon macOS

macOS memberi izin mikrofon ke **aplikasi terminal** yang menjalankan program (Terminal,
iTerm, VS Code, dst.), bukan ke `live-stt` sendiri.

- Saat pertama merekam, macOS menampilkan dialog izin. Pilih **Allow**.
- Kalau dialognya tidak muncul atau pernah ditolak, buka **System Settings → Privacy &
  Security → Microphone**, aktifkan aplikasi terminal Anda, lalu tutup dan buka lagi
  terminalnya.
- Tanpa izin, macOS tidak memberi error; audio yang masuk hanya hening total. Program
  akan memperingatkan bila audio yang diterima nol semua.

## Merekam audio sistem (BlackHole)

Untuk mentranskrip suara dari aplikasi lain (rapat online, video), pasang
[BlackHole](https://github.com/ExistentialAudio/BlackHole), arahkan output audio macOS
ke BlackHole (atau ke Multi-Output Device supaya tetap terdengar di speaker), lalu pilih
BlackHole di menu device atau lewat `--device BlackHole`.

## Cara kerja

```
mic (sounddevice) ─┐
                   ├─► queue.Queue ─► resample 16 kHz ─► pra-proses ─► Silero VAD ─► segmen
file (soundfile) ──┘                                                                   │
                                                                                        ▼
            terminal + transcript_YYYY-MM-DD.txt ◄─ filter anti-halusinasi ◄─ mlx-whisper
```

- **Producer–consumer.** Callback perekam hanya menyalin blok audio ke `queue.Queue`,
  jadi perekaman tidak pernah menunggu transkripsi. Satu thread lain mengambil dari queue
  dan menjalankan VAD, Whisper, dan filter.
- **Segmentasi.** Segmen dimulai saat VAD mendeteksi suara (diawali pre-roll) dan dipotong
  saat hening berlangsung `--min-silence-ms`. Panjang maksimum 10 detik; ucapan yang lebih
  panjang dipotong di titik hening terakhir, atau di frame paling sunyi bila tidak ada jeda.
  Audio tanpa suara tidak pernah dikirim ke Whisper, dan potongan dengan suara di bawah
  250 ms dibuang.
- **Whisper.** Tiap segmen ditranskrip sendiri dengan `condition_on_previous_text=False`
  dan suhu 0. Model dimuat dan dipanaskan sekali saat start supaya segmen pertama tidak lambat.
- **Peringatan antrean.** Bila audio yang menunggu melebihi 5 detik, program memperingatkan
  bahwa transkripsi lebih lambat dari audio yang masuk.
- **Device.** Program meminta 16 kHz ke device; bila ditolak, merekam di sample rate
  aslinya lalu di-resample. Input stereo dicampur ke mono.

## Suara jauh dan ruangan bising

Mic laptop menangkap banyak gemuruh frekuensi rendah (getaran meja, ketikan, kipas), dan
VAD tidak mengenali ucapan yang levelnya dekat dengan lantai noise, padahal Whisper masih
sanggup membacanya. Karena itu audio diproses menjadi dua jalur yang sejajar:

| Jalur | Pemrosesan | Dipakai untuk |
| --- | --- | --- |
| VAD | high-pass 100 Hz → peredam noise spektral → penguatan otomatis | Keputusan "ada suara atau tidak" |
| Whisper | high-pass 100 Hz saja | Audio yang ditranskrip |

- Peredam noise efektif untuk noise yang ajek (kipas, AC, dengung). Musik atau orang lain
  yang ikut berbicara tidak bisa dipisahkan.
- Volume input mic di **System Settings → Sound → Input** ikut berpengaruh; naikkan bila
  pembicara jauh.
- `--no-enhance` mematikan pra-proses ini, mis. untuk membandingkan hasil.

## Filter anti-halusinasi

Whisper cenderung mengarang teks saat audionya hening atau tidak terbaca. Sebuah hasil
dibuang bila salah satu kondisi ini terpenuhi:

| Kondisi | Alasan |
| --- | --- |
| `no_speech_prob > 0,6` **dan** `avg_logprob < -1,0` | Konvensi Whisper untuk "bukan ucapan"; `no_speech_prob` tinggi saja tidak cukup, supaya ucapan pelan yang valid tidak hilang |
| `compression_ratio > 2,4` | Teks berulang-ulang (loop) |
| Cocok dengan daftar frasa halusinasi umum | Mis. "Terima kasih telah menonton", "Subtitle by …", atau gema dari initial prompt |
| Teks identik untuk ketiga kalinya berturut-turut | Ciri halusinasi berulang |
| Teks dimulai di detik terakhir segmen padahal terlalu panjang untuk sisa audionya | Berasal dari hening yang ditambahkan Whisper sampai 30 detik |
| "Terima kasih." (dan sejenisnya) menutupi ≥ 3 detik audio | Ucapan aslinya hanya sekitar 1 detik |
| Segmen samar (< 12 dB di atas lantai noise) **dan** `avg_logprob < -0,6`, atau berupa frasa basa-basi | Suara latar yang tidak terbaca; bacaan yang benar tetap berkeyakinan tinggi |

Jumlah yang dibuang tampil di ringkasan akhir ("dibuang filter").

## Hasil pengukuran

Skala kecil: satu MacBook Pro M4, satu ruangan, model `whisper-large-v3-turbo`. Angka ini
gambaran, bukan benchmark.

**Kecepatan.** 5–10x realtime (segmen 10 detik selesai dalam ±1–2 detik).

**Audio bersih (suara TTS, lewat `--file`).** WAV 16 kHz, MP3, WAV 44,1 kHz, dan WAV 48 kHz
stereo semuanya terbaca tepat, termasuk kalimat campur Indonesia–Inggris. File hening:
0 segmen dikirim ke Whisper.

**Suara pelan lewat mic.** Audio TTS dua kalimat diputar dari speaker laptop makin pelan
dan direkam mic bawaan:

| Level ucapan di atas noise | Tanpa pra-proses | Dengan pra-proses |
| --- | --- | --- |
| ±25 dB | 2 kalimat benar | 2 kalimat benar |
| ±10 dB | terpecah dan salah | 2 kalimat benar |
| ±8 dB | hanya 2× "Terima kasih." palsu | kalimat 1 benar, kalimat 2 hilang |
| ±7–10 dB (rekaman lain) | tidak ada apa-apa | 2 kalimat benar |
| Ruangan saja, 45 detik | tidak ada baris | tidak ada baris (4 segmen samar dibuang) |

**Obrolan santai dari jauh.** Pada dua rekaman 25 detik berisi obrolan beberapa orang,
deteksi suaranya bekerja (ucapan ±30 dB di atas lantai noise), tetapi keyakinan Whisper
hanya −0,6 sampai −0,9, dibanding −0,05 sampai −0,15 pada ucapan yang jelas. Temperature
fallback, prompt yang berbeda, dan segmen yang lebih panjang tidak memperbaikinya.

## Menguji tanpa mikrofon

`--file` menjalankan file audio lewat pipeline yang sama (VAD → segmentasi → Whisper).
Contoh membuat audio uji dengan suara bawaan macOS:

```bash
mkdir -p tests/audio
```

```bash
say -v Damayanti -o tests/audio/uji.aiff "Selamat pagi semuanya. [[slnc 3000]] Tolong siapkan laporan sebelum hari Jumat."
```

```bash
afconvert -f WAVE -d LEI16@16000 tests/audio/uji.aiff tests/audio/uji.wav
```

```bash
uv run live-stt --file tests/audio/uji.wav
```

Folder `tests/audio/` dan file `transcript_*.txt` di-ignore git.

## Pemecahan masalah

| Gejala | Penyebab dan solusi |
| --- | --- |
| `error: Failed to spawn: live-stt` | Perintah dijalankan di luar folder project. `cd` ke folder project dulu, atau pakai `uv run --directory <folder> live-stt` |
| Tidak ada teks sama sekali, ada peringatan "audio nol total" | Izin mikrofon belum diberikan ke aplikasi terminal (lihat [Izin mikrofon macOS](#izin-mikrofon-macos)) |
| Suara dari jauh tidak tertangkap | Naikkan volume input mic, dekatkan mic, atau turunkan `--vad-threshold` (mis. `0.35`) |
| Muncul baris yang tidak diucapkan | Naikkan `--vad-threshold` (mis. `0.6`) supaya suara latar tidak memicu segmen |
| Kalimat terpotong di tengah | Naikkan `--min-silence-ms` (mis. `900`) |
| Peringatan "Antrean menumpuk" | Transkripsi lebih lambat dari audio masuk; pakai model lebih kecil, mis. `--model mlx-community/whisper-small-mlx` |
| BlackHole tidak muncul di menu | BlackHole belum terpasang, atau belum dipilih sebagai output audio macOS |
| Gagal memuat model saat offline | Model belum pernah diunduh; jalankan sekali dengan koneksi internet |

## Keterbatasan

- Ucapan lebih dari 10 detik tanpa jeda dipotong paksa, jadi sesekali ada kata yang kurang
  tepat di sambungannya.
- Obrolan santai beberapa orang dari jarak jauh (bahasa gaul, tawa, bicara bersamaan)
  masih sering salah dibaca oleh model turbo.
- Potongan suara yang sangat pendek dan tidak jelas bisa terbaca sebagai frasa basa-basi
  seperti "Terima kasih.".
- Tidak ada pemisahan pembicara (diarization).
- File MP3 didekode sekaligus ke memori, jadi file berjam-jam memakan banyak RAM.
- Ambang pra-proses dan filter dikalibrasi dari belasan rekaman di satu ruangan; kondisi
  lain mungkin perlu penyetelan.
- Belum diuji: BlackHole, device yang menolak 16 kHz, dan peringatan antrean pada beban nyata.

## Struktur project

```
pyproject.toml            dependency (torch dikecualikan) dan entry point live-stt
src/live_stt/
  cli.py                  argumen, menu device, perakitan thread, Ctrl+C
  audio.py                sumber audio: mic (sounddevice), file (soundfile), resampler
  enhance.py              pra-proses: high-pass, peredam noise, penguatan otomatis
  vad.py                  Silero VAD (onnxruntime) dan pemotong segmen
  transcriber.py          thread konsumen: VAD → mlx-whisper → filter anti-halusinasi
  output.py               cetak ke terminal dan append ke file transkrip
  assets/silero_vad.onnx  model VAD
```

Model VAD (`silero_vad.onnx`, lisensi MIT) berasal dari
[snakers4/silero-vad](https://github.com/snakers4/silero-vad) v5.1.2 dan sengaja ikut di
repo supaya clone baru langsung jalan. `mlx-whisper` mendeklarasikan `torch` sebagai
dependency, tetapi hanya skrip konversi modelnya yang memakainya, jadi `torch`
dikecualikan lewat `[tool.uv] exclude-dependencies`.
