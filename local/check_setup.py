#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--allow-missing-model",
    action="store_true",
)
args = parser.parse_args()

print("=" * 72)
print("WL-BISINDO LOCAL REALTIME - SETUP CHECK")
print("=" * 72)
print("Python    :", sys.version.split()[0])

errors = []
warnings = []

try:
    import numpy as np
    print("NumPy     :", np.__version__)
except Exception as exc:
    errors.append(f"NumPy: {exc}")

try:
    import cv2
    print("OpenCV    :", cv2.__version__)
except Exception as exc:
    errors.append(f"OpenCV: {exc}")

try:
    import mediapipe as mp
    print("MediaPipe :", mp.__version__)
except Exception as exc:
    errors.append(f"MediaPipe: {exc}")

try:
    import torch
    print("PyTorch   :", torch.__version__)
    print("CUDA      :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU       :", torch.cuda.get_device_name(0))
except Exception as exc:
    errors.append(f"PyTorch: {exc}")

try:
    import edge_tts
    print("edge-tts  : OK")
except Exception as exc:
    errors.append(f"edge-tts: {exc}")

try:
    import pygame
    print("pygame    :", pygame.version.ver)
except Exception as exc:
    errors.append(f"pygame: {exc}")

if sys.version_info[:2] != (3, 11):
    warnings.append(
        "Python 3.11 x64 direkomendasikan untuk runtime ini."
    )

required = [
    MODEL_DIR / "wl_bisindo_hand134_transformer_traced.pt",
    MODEL_DIR / "feature_mean.npy",
    MODEL_DIR / "feature_std.npy",
    MODEL_DIR / "class_mapping.json",
]

print()
print("MODEL FILES")

missing_model = []

for path in required:
    ok = path.exists()
    print(f"{'[OK]' if ok else '[MISSING]':<10} {path.name}")
    if not ok:
        missing_model.append(path)

if missing_model and not args.allow_missing_model:
    errors.extend(
        f"Missing: {path}"
        for path in missing_model
    )

if not missing_model:
    import numpy as np
    import torch

    mean = np.load(
        MODEL_DIR / "feature_mean.npy"
    )
    std = np.load(
        MODEL_DIR / "feature_std.npy"
    )

    print()
    print("feature_mean:", mean.shape)
    print("feature_std :", std.shape)

    if mean.shape != (134,):
        errors.append(
            f"feature_mean harus (134,), found {mean.shape}"
        )

    if std.shape != (134,):
        errors.append(
            f"feature_std harus (134,), found {std.shape}"
        )

    with open(
        MODEL_DIR / "class_mapping.json",
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    print("Classes     :", len(mapping))

    if len(mapping) != 32:
        errors.append(
            f"class mapping harus 32 class, found {len(mapping)}"
        )

    if not errors:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = torch.jit.load(
            str(
                MODEL_DIR
                / "wl_bisindo_hand134_transformer_traced.pt"
            ),
            map_location=device,
        ).eval()

        dummy = torch.zeros(
            1,
            48,
            134,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            out = model(dummy)

        print("Model output:", tuple(out.shape))

        if tuple(out.shape) != (1, 32):
            errors.append(
                "Output model harus (1,32), "
                f"found {tuple(out.shape)}"
            )

print()

for item in warnings:
    print("[WARN]", item)

if missing_model and args.allow_missing_model:
    print()
    print(
        "[INFO] Dependency siap, tetapi model belum dicopy."
    )
    print(
        "[INFO] Setelah training Kaggle selesai, copy 4 file "
        "runtime ke folder local/model/."
    )

if errors:
    print()
    print("❌ SETUP BELUM SIAP")
    for err in errors:
        print(" -", err)
    raise SystemExit(1)

print()
print("✅ SETUP DEPENDENCY SIAP")

if not missing_model:
    print("✅ MODEL RUNTIME SIAP")
    print("Jalankan: run.bat")
