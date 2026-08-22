# WL-BISINDO Hand134 Transformer — Kaggle Training + Local Realtime

Prototype pengenalan **32 kosakata isolated BISINDO** menggunakan fitur dua tangan (*Hand134*) dan *Dual-Hand Temporal Transformer*.

Repository/paket ini sengaja memisahkan dua lingkungan:

- **Kaggle** → preprocessing dataset dan training model.
- **Local PC** → pengujian webcam/video, output teks, dan Indonesian Neural TTS.

> Preprocessing dataset sudah dilakukan terpisah. Notebook training di paket ini **tidak membaca video mentah dan tidak menjalankan preprocessing ulang**.

---

## Pipeline

```text
KAGGLE
Raw WL-BISINDO videos
        ↓
Preprocessing V2
MediaPipe Pose (helper) + MediaPipe Hands
        ↓
48 frame × 134 fitur tangan
        ↓
Kaggle Dataset:
preprocessing-output-bisindo
        ↓
Training V4
Dual-Hand Temporal Transformer
        ↓
TorchScript + feature mean/std + class mapping

LOCAL PC
Webcam / video
        ↓
Preprocessing realtime yang sama
        ↓
Rolling 48-frame Hand134 sequence
        ↓
TorchScript model
        ↓
Prediction voting
        ↓
Teks
        ↓
Indonesian Neural TTS
```

---

## Struktur paket

```text
WL_BISINDO_FINAL_KAGGLE_LOCAL/
│
├── README.md
│
├── kaggle/
│   └── 02_WL_BISINDO_TRAIN_V4_HAND134_TRANSFORMER_FINAL_KAGGLE.ipynb
│
└── local/
    ├── realtime_bisindo.py
    ├── check_setup.py
    ├── requirements.txt
    ├── setup_env.bat
    ├── run.bat
    │
    └── model/
        └── COPY_MODEL_FILES_HERE.txt
```

---

# A. Kaggle — Training

## Input

Notebook training langsung membaca hasil preprocessing dari:

```text
/kaggle/input/datasets/loliwibu/preprocessing-output-bisindo
```

File preprocessing yang dipakai:

```text
X_hands134.npy
hand_observed_mask.npy
hand_valid_mask.npy
pose_anchor_valid.npy
labels.npy
signer_ids.npy
sample_ids.npy
metadata.csv
class_mapping.json
```

Input model:

```text
48 timestep × 134 fitur
```

Per tangan:

```text
21 landmark × (x,y,z) lokal terhadap wrist = 63
wrist global terhadap body anchor          = 3
presence                                    = 1
------------------------------------------------
per hand                                   = 67
dua tangan                                 = 134
```

Pose **tidak masuk sebagai fitur model**. Pose hanya membantu memperoleh *shoulder center*, *body scale*, dan referensi wrist anatomis.

## Menjalankan

1. Attach Kaggle Dataset `preprocessing-output-bisindo`.
2. Aktifkan GPU.
3. Buka notebook:
   ```text
   kaggle/02_WL_BISINDO_TRAIN_V4_HAND134_TRANSFORMER_FINAL_KAGGLE.ipynb
   ```
4. Jalankan semua cell.
5. Untuk run panjang gunakan **Save Version → Save & Run All**.

Di bagian awal harus muncul:

```text
Using: /kaggle/input/datasets/loliwibu/preprocessing-output-bisindo
✅ All preprocessing V2 files found
```

dan integrity check:

```text
✅ Input integrity PASSED
```

## Output runtime yang wajib diambil

Setelah training selesai, ambil minimal:

```text
wl_bisindo_hand134_transformer_traced.pt
feature_mean.npy
feature_std.npy
class_mapping.json
```

Kemudian copy ke:

```text
local/model/
```

Struktur akhirnya:

```text
local/
├── realtime_bisindo.py
├── check_setup.py
├── requirements.txt
├── setup_env.bat
├── run.bat
│
└── model/
    ├── wl_bisindo_hand134_transformer_traced.pt
    ├── feature_mean.npy
    ├── feature_std.npy
    └── class_mapping.json
```

---

# B. Local PC — Setup

Runtime lokal direkomendasikan menggunakan **Python 3.11 x64**.

## Setup otomatis Windows

Buka terminal pada folder `local/`, lalu:

```bat
setup_env.bat
```

Script akan:

```text
1. membuat .venv
2. mengaktifkan environment
3. upgrade pip
4. install dependency
5. membuat folder model
6. memeriksa environment
```

Setelah 4 file hasil training sudah berada di `local/model/`, jalankan:

```bat
run.bat
```

`run.bat` otomatis mengaktifkan `.venv`, menjalankan pengecekan model, kemudian membuka realtime webcam.

Argumen juga dapat diteruskan melalui `run.bat`, misalnya:

```bat
run.bat --camera 1
```

atau:

```bat
run.bat --threshold 0.80 --infer-every 2
```

---

# C. Local PC — Menjalankan Manual

Aktifkan environment:

### Command Prompt

```bat
.venv\Scripts\activate.bat
```

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Kemudian:

```bash
python check_setup.py
python realtime_bisindo.py
```

---

# D. Fast Continuous Recognition

Versi realtime ini memperbaiki perilaku lama yang membuat pengguna harus menurunkan/mengeluarkan tangan sebelum gesture berikutnya.

Sekarang:

```text
Camera
  ↓
MediaPipe Pose + Hands
  ↓
rolling raw landmark window 48 frame
  ↓
interpolation + EMA
  ↓
Hand134
  ↓
feature normalization
  ↓
Transformer
  ↓
inferensi setiap 2 frame
  ↓
confidence + top1/top2 margin
  ↓
temporal voting 2 dari 3
  ↓
kata diterima
```

Yang penting:

```text
raw rolling window TIDAK dihapus setelah kata diterima
```

dan **tidak ada syarat `hand_disappeared == True`**.

Jadi alurnya dapat berupa:

```text
gesture A
   ↓
A diterima
   ↓
tangan tetap berada di frame
   ↓
langsung gesture B
   ↓
rolling window terus bergerak
   ↓
B diterima
```

Untuk kelas yang **berbeda**, default *debounce* hanya:

```text
change_cooldown = 0.20 detik
```

Untuk mengulang kelas yang sama tanpa mengeluarkan tangan:

```text
repeat_cooldown = 1.60 detik
```

Cooldown kelas yang sama lebih panjang untuk mengurangi kata yang terus terulang ketika satu gesture ditahan.

---

# E. Parameter Realtime

Default:

```text
sequence length  = 48
feature dim      = 134
threshold        = 0.75
min margin       = 0.10
minimum valid    = 0.25
vote window      = 3
vote hits        = 2
infer every      = 2 frame
change cooldown  = 0.20 s
repeat cooldown  = 1.60 s
```

Lebih responsif:

```bash
python realtime_bisindo.py --infer-every 1
```

Kalau terlalu banyak *false positive*:

```bash
python realtime_bisindo.py --threshold 0.82 --min-margin 0.15
```

Kalau prediksi terlalu susah keluar:

```bash
python realtime_bisindo.py --threshold 0.68 --min-margin 0.08
```

Webcam lain:

```bash
python realtime_bisindo.py --camera 1
```

Paksa CPU:

```bash
python realtime_bisindo.py --cpu
```

Tanpa mirror preview:

```bash
python realtime_bisindo.py --no-mirror
```

Test video:

```bash
python realtime_bisindo.py --video "path/to/video.mp4"
```

Semua contoh path di README bersifat relatif/generik dan tidak bergantung pada struktur folder komputer tertentu.

---

# F. Indonesian Neural TTS

Realtime menggunakan **Microsoft Edge Neural TTS**.

Default:

```text
id-ID-ArdiNeural
```

Voice alternatif:

```text
id-ID-GadisNeural
```

Jalankan voice perempuan:

```bash
python realtime_bisindo.py --voice id-ID-GadisNeural
```

TTS dibuat asynchronous sehingga audio tidak menghentikan kamera atau inferensi.

Kata yang sudah dibuat disimpan di:

```text
local/.tts_cache/
```

Jika suatu kata sudah ada di cache, audio berikutnya dapat diputar tanpa membuat file baru.

> Edge TTS memerlukan internet ketika audio suatu kata belum tersedia di cache. Jika TTS gagal, recognition tetap berjalan.

Matikan TTS:

```bash
python realtime_bisindo.py --no-tts
```

Matikan auto-speak tetapi tetap izinkan tombol `S`:

```bash
python realtime_bisindo.py --no-auto-speak
```

---

# G. Keyboard

Saat realtime:

```text
Q / ESC = keluar
C       = hapus seluruh text buffer
X / B   = hapus kata terakhir
S       = bacakan seluruh text buffer
T       = toggle TTS
R       = reset temporal state
```

---

# H. Kenapa Preview Bisa Mirror?

Inference selalu memakai **frame asli** supaya konsisten dengan preprocessing dataset.

Mirror hanya dilakukan pada tampilan:

```text
model input = original frame
preview     = mirrored frame
```

Ini disengaja.

---

# I. Mapping 32 Kelas WL-BISINDO

```text
0  Air
1  Belajar
2  Cari
3  Hari
4  Ingat
5  Lagi
6  Maaf
7  Makan
8  Motor
9  Saya
10 Terima kasih
11 Tuli
12 Apa
13 Siapa
14 Kapan
15 Di mana
16 Mengapa
17 Bagaimana
18 Merah
19 Kuning
20 Hijau
21 Hitam
22 Dengar
23 Berangkat
24 Datang
25 Teman
26 Keluarga
27 Rumah
28 Pagi
29 Siang
30 Sore
31 Malam
```

---

# J. Keterbatasan

Model saat ini adalah **isolated BISINDO word recognition**, bukan penerjemah BISINDO continuous penuh.

Saat ini:

```text
gesture → recognized word → text → TTS
```

Belum:

```text
continuous sign stream
→ sign segmentation
→ sentence grammar translation
→ natural language generation
```

Dataset/model juga belum memiliki kelas eksplisit:

```text
no_sign
background
transition
```

Karena itu threshold dan voting tetap diperlukan untuk mengurangi prediksi saat pengguna tidak sedang membuat salah satu dari 32 gesture.

Untuk pengembangan berikutnya, kelas `no_sign/background/transition` dan dataset gesture berurutan akan membantu continuous recognition menjadi lebih stabil.

---

# K. Ringkasan Environment

```text
Kaggle:
- preprocessing
- training
- evaluation
- export model

Local PC:
- webcam/video test
- Hand134 preprocessing realtime
- TorchScript inference
- text buffer
- Indonesian Neural TTS
```

Dengan pemisahan ini, preprocessing/training Kaggle tidak tercampur dengan kode implementasi lokal.
