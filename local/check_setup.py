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
parser.add_argument("--allow-missing-model", action="store_true")
args = parser.parse_args()


def get_active_version():
    if not ACTIVE_MODEL_FILE.exists():
        raise FileNotFoundError(f"active_model.txt tidak ditemukan: {ACTIVE_MODEL_FILE}")
    version = ACTIVE_MODEL_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("active_model.txt kosong. Isi misalnya: v1")
    if any(ch in version for ch in ("/", "\\", "..")):
        raise ValueError("active_model.txt hanya boleh berisi v1, v2, dan seterusnya.")
    return version


def default_config(version):
    if version == "v1":
        return {
            "name": "WL-BISINDO Hand134 Transformer V4",
            "runtime": "torchscript", "inference_mode": "sequence",
            "model_file": "wl_bisindo_hand134_transformer_traced.pt",
            "mean_file": "feature_mean.npy", "std_file": "feature_std.npy",
            "mapping_file": "class_mapping.json", "sequence_length": 48,
            "feature_dim": 134, "num_classes": 32,
        }
    if version == "v2":
        return {
            "name": "BISINDO Alphabet Dual-Hand MLP",
            "runtime": "onnxruntime", "inference_mode": "single_frame",
            "model_file": "alphabet_model_v7.onnx",
            "mean_file": "feature_mean_v7.npy", "std_file": "feature_std_v7.npy",
            "mapping_file": "class_mapping_v7.json", "sequence_length": 1,
            "feature_dim": 134, "num_classes": 26,
        }
    raise FileNotFoundError(
        f"models/{version}/model_config.json wajib ada untuk versi model baru."
    )


active_version = get_active_version()
model_dir = MODEL_ROOT / active_version
config_path = model_dir / "model_config.json"
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = default_config(active_version)

runtime = str(config["runtime"]).lower()
mode = str(config["inference_mode"]).lower()
feature_dim = int(config["feature_dim"])
sequence_length = int(config["sequence_length"])
num_classes = int(config["num_classes"])
paths = {
    "model": model_dir / config["model_file"],
    "mean": model_dir / config["mean_file"],
    "std": model_dir / config["std_file"],
    "mapping": model_dir / config["mapping_file"],
}

print("=" * 72)
print("WL-BISINDO LOCAL REALTIME - DYNAMIC SETUP CHECK")
print("=" * 72)
print("Python         :", sys.version.split()[0])
print("Active model   :", active_version)
print("Model name     :", config["name"])
print("Model folder   :", model_dir)
print("Runtime        :", runtime)
print("Inference mode :", mode)
print("Expected input :", f"1 x {sequence_length} x {feature_dim}" if mode == "sequence" else f"1 x {feature_dim}")
print("Expected output:", f"1 x {num_classes}")

errors = []
warnings = []

try:
    import numpy as np
    print("NumPy          :", np.__version__)
except Exception as exc:
    errors.append(f"NumPy: {exc}")
try:
    import cv2
    print("OpenCV         :", cv2.__version__)
except Exception as exc:
    errors.append(f"OpenCV: {exc}")
try:
    import mediapipe as mp
    print("MediaPipe      :", mp.__version__)
except Exception as exc:
    errors.append(f"MediaPipe: {exc}")
try:
    import torch
    print("PyTorch        :", torch.__version__)
    print("CUDA           :", torch.cuda.is_available())
except Exception as exc:
    errors.append(f"PyTorch: {exc}")
if runtime == "onnxruntime":
    try:
        import onnxruntime as ort
        print("ONNX Runtime   :", ort.__version__)
    except Exception as exc:
        errors.append(f"ONNX Runtime: {exc}")
try:
    import edge_tts
    print("edge-tts       : OK")
except Exception as exc:
    errors.append(f"edge-tts: {exc}")
try:
    import pygame
    print("pygame         :", pygame.version.ver)
except Exception as exc:
    errors.append(f"pygame: {exc}")

if sys.version_info[:2] != (3, 11):
    warnings.append("Python 3.11 x64 direkomendasikan.")

print("\nACTIVE MODEL FILES")
missing = []
for label, path in paths.items():
    ok = path.exists()
    print(f"{'[OK]' if ok else '[MISSING]':<10} {label:<8} {path.name}")
    if not ok:
        missing.append(path)
if missing and not args.allow_missing_model:
    errors.extend(f"Missing: {path}" for path in missing)

if not missing:
    import numpy as np
    mean = np.load(paths["mean"]).astype(np.float32)
    std = np.load(paths["std"]).astype(np.float32)
    with paths["mapping"].open("r", encoding="utf-8") as f:
        mapping = {int(k): str(v) for k, v in json.load(f).items()}

    print("\nfeature_mean   :", mean.shape)
    print("feature_std    :", std.shape)
    print("Classes        :", len(mapping))
    if mean.shape != (feature_dim,):
        errors.append(f"feature_mean harus ({feature_dim},), found {mean.shape}")
    if std.shape != (feature_dim,):
        errors.append(f"feature_std harus ({feature_dim},), found {std.shape}")
    if len(mapping) != num_classes:
        errors.append(f"class_mapping harus {num_classes} kelas, found {len(mapping)}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        errors.append("Normalisasi mengandung NaN/Inf")

    if not errors:
        dummy_shape = (1, sequence_length, feature_dim) if mode == "sequence" else (1, feature_dim)
        if runtime == "torchscript":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.jit.load(str(paths["model"]), map_location=device).eval()
            with torch.inference_mode():
                output = model(torch.zeros(*dummy_shape, dtype=torch.float32, device=device))
            output_shape = tuple(output.shape)
        elif runtime == "onnxruntime":
            import onnxruntime as ort
            session = ort.InferenceSession(str(paths["model"]), providers=["CPUExecutionProvider"])
            input_meta = session.get_inputs()[0]
            output = session.run(None, {input_meta.name: np.zeros(dummy_shape, dtype=np.float32)})[0]
            output_shape = tuple(output.shape)
            print("ONNX input name :", input_meta.name)
        else:
            errors.append(f"Runtime tidak didukung: {runtime}")
            output_shape = None

        if output_shape is not None:
            print("Model output   :", output_shape)
            if output_shape != (1, num_classes):
                errors.append(f"Output model harus (1,{num_classes}), found {output_shape}")

for warning in warnings:
    print("[WARN]", warning)
if missing and args.allow_missing_model:
    print(f"\n[INFO] Dependency siap; lengkapi folder models/{active_version}/")
if errors:
    print("\n❌ SETUP BELUM SIAP")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print("\n✅ SETUP SIAP")
print(f"✅ ACTIVE MODEL: {active_version} ({runtime}/{mode})")
