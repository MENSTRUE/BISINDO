# WL-BISINDO Hand134 Transformer

Real-time **isolated BISINDO word recognition** prototype using dual-hand landmark features (*Hand134*) and a **Dual-Hand Temporal Transformer**.

The project separates model development from local deployment:

- **Kaggle** — preprocessing, training, evaluation, and model export.
- **Local PC** — webcam/video inference, text output, and Indonesian Neural TTS.

> The current model recognizes **32 isolated BISINDO vocabulary classes**. It is not yet a full continuous sign-language translation system.

---

## Project Overview

```text
KAGGLE
Raw WL-BISINDO videos
        ↓
Preprocessing V2
MediaPipe Pose (helper) + MediaPipe Hands
        ↓
48 frames × 134 hand features
        ↓
Preprocessed Kaggle Dataset
        ↓
Dual-Hand Temporal Transformer
        ↓
Evaluation + checkpoint + TorchScript export

LOCAL PC
Webcam / video
        ↓
Realtime Hand134 preprocessing
        ↓
Rolling 48-frame temporal window
        ↓
Feature normalization
        ↓
TorchScript inference
        ↓
Confidence + margin + temporal voting
        ↓
Recognized text
        ↓
Indonesian Neural TTS
```

---

## Model Performance

Final signer-independent evaluation:

| Metric | Result |
|---|---:|
| Best development epoch | 18 |
| Best development Macro-F1 | 94.93% |
| Final unseen-signer accuracy | 86.56% |
| Final unseen-signer Macro-F1 | 85.73% |
| Zero-F1 classes | 0 |
| Final test signer | Signer 4 |

The final evaluation uses **Signer 4 as an unseen signer**, while Signers 0–3 are used for model development and final retraining.

> Dataset-level accuracy does not guarantee identical webcam performance. Lighting, camera quality, distance, background, signer style, and landmark quality can affect real-world inference.

---

## Architecture

```text
Left Hand (67 features) ──→ Left Hand Encoder ──┐
                                                ├─→ Fusion
Right Hand (67 features) → Right Hand Encoder ──┘
                                                     ↓
                                              Temporal Conv
                                                     ↓
                                             Transformer Encoder
                                                     ↓
                                               Classification Head
                                                     ↓
                                                  32 classes
```

Input shape:

```text
[B, 48, 134]
```

Per hand:

```text
21 landmarks × (x, y, z) local to wrist = 63
global wrist relative to body anchor     = 3
hand presence flag                       = 1
------------------------------------------------
per hand                                 = 67
two hands                                = 134
```

**MediaPipe Pose is not part of the model input.** It is used only as a helper for body center, body scale, and anatomical wrist references.

---

## Repository Structure

```text
BISINDO/
│
├── README.md
├── .gitignore
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
        ├── class_mapping.json
        ├── feature_mean.npy
        ├── feature_std.npy
        └── wl_bisindo_hand134_transformer_traced.pt
```

Large model weights, datasets, virtual environments, caches, and raw videos should not be committed directly to Git.

---

# Kaggle Workflow

## 1. Preprocessed Input

The training notebook reads the preprocessed Hand134 dataset from:

```text
/kaggle/input/datasets/loliwibu/preprocessing-output-bisindo
```

Expected files:

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

Expected core shapes:

```text
X_hands134.npy         : (1600, 48, 134)
hand_observed_mask.npy : (1600, 48, 2)
hand_valid_mask.npy    : (1600, 48, 2)
labels.npy             : (1600,)
signer_ids.npy         : (1600,)
```

The training notebook does **not** re-run raw-video preprocessing.

---

## 2. Signer-Independent Protocol

Development protocol:

```text
Signer 0–3 → development pool
Signer 4   → final unseen test
```

The notebook attempts to use one complete development signer as unseen-signer validation. After selecting the best epoch, the model is reinitialized and trained again on all Signers 0–3 before evaluating Signer 4 once.

---

## 3. Training Configuration

Main configuration:

```text
Sequence length       : 48
Feature dimension     : 134
Classes               : 32
Hand embedding        : 96
Transformer dimension : 192
Attention heads       : 6
Transformer layers    : 3
Feed-forward dimension: 384
Dropout               : 0.25

Batch size            : 64
Learning rate         : 3e-4
Weight decay          : 1e-3
Label smoothing       : 0.03
Gradient clipping     : 1.0
Max development epoch : 100
Early stopping        : 15
```

Training augmentation includes:

- temporal speed resampling,
- temporal shift,
- coordinate noise,
- short frame masking,
- occasional single-hand dropout.

---

## 4. Run on Kaggle

1. Attach the preprocessed dataset.
2. Enable a GPU accelerator.
3. Open:

```text
kaggle/02_WL_BISINDO_TRAIN_V4_HAND134_TRANSFORMER_FINAL_KAGGLE.ipynb
```

4. Run all cells.
5. For long runs, use **Save Version → Save & Run All**.

Expected input checks:

```text
✅ All preprocessing V2 files found
✅ Input integrity PASSED
✅ DataLoader PASSED
```

---

## 5. Deployment Artifacts

For local inference, copy these files from Kaggle output:

```text
wl_bisindo_hand134_transformer_traced.pt
feature_mean.npy
feature_std.npy
class_mapping.json
```

Place them in:

```text
local/model/
```

For future fine-tuning, also keep the regular PyTorch checkpoint:

```text
hand134_transformer_final.pt
```

> Use the regular PyTorch checkpoint for further training. The TorchScript file is intended for inference/deployment.

---

# Local Realtime

## 1. Requirements

Recommended environment:

```text
Python 3.11 x64
```

Main runtime dependencies:

```text
numpy
opencv-contrib-python
mediapipe
torch
edge-tts
pygame
```

---

## 2. Windows Setup

Open a terminal inside:

```text
local/
```

Run once:

```bat
setup_env.bat
```

The script will:

```text
1. create .venv
2. activate the environment
3. upgrade pip tooling
4. install dependencies
5. verify runtime files
6. test model input/output shape
```

After setup:

```bat
run.bat
```

---

## 3. Manual Run

Command Prompt:

```bat
.venv\Scripts\activate.bat
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
python check_setup.py
python realtime_bisindo.py
```

---

# Fast Continuous Realtime Logic

The local runtime uses a rolling temporal window. It does **not** require the user's hands to leave the frame before recognizing the next gesture.

```text
Webcam
  ↓
Pose + Hands
  ↓
rolling 48-frame landmark window
  ↓
short-gap interpolation
  ↓
EMA smoothing
  ↓
Hand134 feature construction
  ↓
feature normalization
  ↓
Transformer inference
  ↓
confidence + top1/top2 margin
  ↓
temporal voting
  ↓
accepted word
```

Important behavior:

- The 48-frame rolling buffer is **not cleared** after a word is accepted.
- A different gesture can follow while the hands remain visible.
- A held gesture is accepted only once.
- The same class is re-armed only after a genuine transition/uncertain period.
- Low-confidence raw predictions are not presented as accepted words.
- TTS keeps only the newest pending utterance to avoid stale audio queues.

---

## Realtime Parameters

Default behavior is tuned for responsive local inference:

```text
sequence length       = 48
feature dimension     = 134
threshold             = 0.75
minimum margin        = 0.10
minimum valid ratio   = 0.25
vote window           = 3
vote hits             = 2
infer every           = 2 frames
change cooldown       = 0.20 s
neutral reset hits    = 3
```

More responsive:

```bash
python realtime_bisindo.py --infer-every 1
```

More conservative:

```bash
python realtime_bisindo.py --threshold 0.82 --min-margin 0.15
```

More permissive:

```bash
python realtime_bisindo.py --threshold 0.68 --min-margin 0.08
```

Use another camera:

```bash
python realtime_bisindo.py --camera 1
```

Force CPU:

```bash
python realtime_bisindo.py --cpu
```

Disable mirrored preview:

```bash
python realtime_bisindo.py --no-mirror
```

Test a video:

```bash
python realtime_bisindo.py --video "path/to/video.mp4"
```

---

# Indonesian Neural TTS

The local runtime uses **Microsoft Edge Neural TTS**.

Default voice:

```text
id-ID-ArdiNeural
```

Alternative voice:

```text
id-ID-GadisNeural
```

Example:

```bash
python realtime_bisindo.py --voice id-ID-GadisNeural
```

Disable TTS:

```bash
python realtime_bisindo.py --no-tts
```

Disable automatic speech while keeping manual speech available:

```bash
python realtime_bisindo.py --no-auto-speak
```

Generated audio is cached locally in:

```text
local/.tts_cache/
```

Edge TTS may require internet access when a requested utterance is not yet cached.

---

# Keyboard Controls

```text
Q / ESC = exit
C       = clear text buffer
X / B   = remove last word
S       = speak current text buffer
T       = toggle TTS
R       = reset temporal state
```

---

# Fine-Tuning with New Data

The model can be fine-tuned later when additional BISINDO data becomes available.

## Same 32 classes

Recommended flow:

```text
new videos
   ↓
run the SAME Hand134 preprocessing
   ↓
48 × 134 feature sequences
   ↓
load hand134_transformer_final.pt
   ↓
fine-tune with a lower learning rate
   ↓
re-evaluate signer-independently
   ↓
export a new TorchScript model
```

Suggested starting learning rate:

```text
1e-5 to 5e-5
```

Do not fine-tune only on the new dataset if it is very small. Mix old and new training samples to reduce **catastrophic forgetting**.

## Adding new classes

If new vocabulary classes are added:

```text
32 classes → N classes
```

the final classification head and `class_mapping.json` must be expanded.

A common strategy is:

1. load the previous feature encoders and Transformer weights,
2. replace the final classification layer,
3. initialize the new output head,
4. train the head first,
5. unfreeze the full network with a small learning rate,
6. evaluate on unseen signers.

## Local vs Kaggle fine-tuning

Fine-tuning is technically possible on a local PC, but:

- **CUDA GPU available** → local fine-tuning is practical.
- **CPU only** → possible, but significantly slower.
- **Kaggle GPU** → usually more convenient for repeated experiments.

For deployment, keep using TorchScript. For training/fine-tuning, keep the normal PyTorch checkpoint.

---

# Vocabulary

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

# Current Limitations

The current system is an **isolated-word recognition prototype**:

```text
gesture → recognized word → text → TTS
```

It is not yet:

```text
continuous sign stream
→ automatic sign segmentation
→ BISINDO grammar translation
→ natural Indonesian sentence generation
```

The current model also does not include explicit:

```text
no_sign
background
transition
```

classes.

Therefore, thresholding, confidence margin, temporal voting, and transition logic are still required to reduce false positives.

Future work may include:

- explicit `no_sign/background/transition` samples,
- more signers,
- broader lighting/background variation,
- additional regional BISINDO data,
- continuous sign segmentation,
- sequence-to-text translation,
- language-model-based sentence refinement,
- mobile or edge deployment.

---

# Notes

- Inference uses the **original camera frame** for consistency with dataset preprocessing.
- Mirroring is applied only to the preview.
- Model weights and datasets are intentionally excluded from normal Git commits when they are too large.
- Keep the original checkpoint separately if future fine-tuning is planned.

---

## Status

```text
Preprocessing        ✅
Signer-independent training ✅
Final unseen-signer evaluation ✅
TorchScript export   ✅
Local webcam inference ✅
Fast sequential gesture handling ✅
Indonesian Neural TTS ✅
Full continuous BISINDO translation 🚧
```
