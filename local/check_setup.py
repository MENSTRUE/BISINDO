#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
ACTIVE_MODEL_FILE = ROOT / "active_model.txt"
MODEL_ROOT = ROOT / "models"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--allow-missing-model",
    action="store_true",
)
args = parser.parse_args()


def get_active_version():
    if not ACTIVE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"active_model.txt tidak ditemukan: {ACTIVE_MODEL_FILE}"
        )

    version = ACTIVE_MODEL_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not version:
        raise ValueError(
            "active_model.txt kosong. Isi misalnya: v1"
        )

    if any(ch in version for ch in ("/", "\\", "..")):
        raise ValueError(
            "active_model.txt hanya boleh berisi nama folder, "
            "misalnya v1 atau v2."
        )

    return version


active_version = get_active_version()
model_dir = MODEL_ROOT / active_version

print("=" * 72)
print("WL-BISINDO LOCAL REALTIME - SETUP CHECK")
print("=" * 72)
print("Python       :", sys.version.split()[0])
print("Active model :", active_version)
print("Model folder :", model_dir)

errors = []
warnings = []

try:
    import numpy as np
    print("NumPy        :", np.__version__)
except Exception as exc:
    errors.append(f"NumPy: {exc}")

try:
    import cv2
    print("OpenCV       :", cv2.__version__)
except Exception as exc:
    errors.append(f"OpenCV: {exc}")

try:
    import mediapipe as mp
    print("MediaPipe    :", mp.__version__)
except Exception as exc:
    errors.append(f"MediaPipe: {exc}")

try:
    import torch
    print("PyTorch      :", torch.__version__)
    print("CUDA         :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU          :", torch.cuda.get_device_name(0))
except Exception as exc:
    errors.append(f"PyTorch: {exc}")

try:
    import edge_tts
    print("edge-tts     : OK")
except Exception as exc:
    errors.append(f"edge-tts: {exc}")

try:
    import pygame
    print("pygame       :", pygame.version.ver)
except Exception as exc:
    errors.append(f"pygame: {exc}")

if sys.version_info[:2] != (3, 11):
    warnings.append(
        "Python 3.11 x64 direkomendasikan."
    )

required = [
    model_dir / "wl_bisindo_hand134_transformer_traced.pt",
    model_dir / "feature_mean.npy",
    model_dir / "feature_std.npy",
    model_dir / "class_mapping.json",
]

print()
print("ACTIVE MODEL FILES")

missing = []

for path in required:
    ok = path.exists()
    print(
        f"{'[OK]' if ok else '[MISSING]':<10} {path.name}"
    )
    if not ok:
        missing.append(path)

if missing and not args.allow_missing_model:
    errors.extend(
        f"Missing: {path}"
        for path in missing
    )

if not missing:
    import numpy as np
    import torch

    mean = np.load(
        model_dir / "feature_mean.npy"
    )
    std = np.load(
        model_dir / "feature_std.npy"
    )

    with open(
        model_dir / "class_mapping.json",
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    print()
    print("feature_mean :", mean.shape)
    print("feature_std  :", std.shape)
    print("Classes      :", len(mapping))

    if mean.shape != (134,):
        errors.append(
            f"feature_mean harus (134,), found {mean.shape}"
        )

    if std.shape != (134,):
        errors.append(
            f"feature_std harus (134,), found {std.shape}"
        )

    if len(mapping) != 32:
        errors.append(
            f"class_mapping harus 32 kelas, found {len(mapping)}"
        )

    if not errors:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = torch.jit.load(
            str(
                model_dir
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

        print("Model output :", tuple(out.shape))

        if tuple(out.shape) != (1, 32):
            errors.append(
                f"Output model harus (1, 32), found {tuple(out.shape)}"
            )

for warning in warnings:
    print("[WARN]", warning)

if missing and args.allow_missing_model:
    print()
    print(
        "[INFO] Dependency siap, model aktif belum lengkap."
    )
    print(
        f"[INFO] Taruh file model di: models/{active_version}/"
    )

if errors:
    print()
    print("❌ SETUP BELUM SIAP")
    for err in errors:
        print(" -", err)
    raise SystemExit(1)

print()
print("✅ SETUP SIAP")
print(f"✅ ACTIVE MODEL: {active_version}")
