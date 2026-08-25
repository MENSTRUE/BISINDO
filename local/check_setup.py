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


def discover_config(model_dir: Path, version: str):
    deployment = model_dir / "deployment_config.json"
    model_config = model_dir / "model_config.json"

    # NEW v1 multimodal package from Kaggle A/B/C/D training.
    if deployment.exists():
        with deployment.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "kind": "multimodal",
            "name": str(cfg.get("name", "WL-BISINDO Multimodal Temporal Transformer")),
            "runtime": str(cfg.get("runtime", "torchscript")).lower(),
            "inference_mode": "sequence",
            "model_file": str(cfg.get("model_file", "wl_bisindo_multimodal_traced.pt")),
            "mapping_file": "class_mapping.json",
            "sequence_length": int(cfg.get("sequence_length", 48)),
            "num_classes": int(cfg.get("num_classes", 32)),
            "winner_mode": str(cfg.get("winner_mode", "C")),
            "inputs": cfg.get("inputs", {
                "hand": [1, 48, 134],
                "pose": [1, 48, 36],
                "facehead": [1, 48, 52],
                "facecrop": [1, 48, 48, 48],
            }),
        }

    # Existing dynamic model packages, e.g. alphabet V8.4.
    if model_config.exists():
        with model_config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "kind": "hand134",
            "name": str(cfg["name"]),
            "runtime": str(cfg["runtime"]).lower(),
            "inference_mode": str(cfg["inference_mode"]).lower(),
            "model_file": str(cfg["model_file"]),
            "mean_file": str(cfg["mean_file"]),
            "std_file": str(cfg["std_file"]),
            "mapping_file": str(cfg["mapping_file"]),
            "sequence_length": int(cfg["sequence_length"]),
            "feature_dim": int(cfg["feature_dim"]),
            "num_classes": int(cfg["num_classes"]),
            "winner_mode": None,
        }

    # Legacy fallback.
    if version == "v1":
        return {
            "kind": "hand134",
            "name": "WL-BISINDO Hand134 Transformer V4",
            "runtime": "torchscript",
            "inference_mode": "sequence",
            "model_file": "wl_bisindo_hand134_transformer_traced.pt",
            "mean_file": "feature_mean.npy",
            "std_file": "feature_std.npy",
            "mapping_file": "class_mapping.json",
            "sequence_length": 48,
            "feature_dim": 134,
            "num_classes": 32,
            "winner_mode": None,
        }

    if version == "v2":
        return {
            "kind": "hand134",
            "name": "BISINDO Alphabet V8.4 Temporal Transformer",
            "runtime": "torchscript",
            "inference_mode": "sequence",
            "model_file": "alphabet_temporal_v8_4_traced.pt",
            "mean_file": "feature_mean_v8_4.npy",
            "std_file": "feature_std_v8_4.npy",
            "mapping_file": "class_mapping_v8_4.json",
            "sequence_length": 48,
            "feature_dim": 134,
            "num_classes": 26,
            "winner_mode": None,
        }

    raise FileNotFoundError(
        f"Tidak ada deployment_config.json / model_config.json di {model_dir}"
    )


active_version = get_active_version()
model_dir = MODEL_ROOT / active_version
config = discover_config(model_dir, active_version)

runtime = config["runtime"]
mode = config["inference_mode"]
kind = config["kind"]
sequence_length = config["sequence_length"]
num_classes = config["num_classes"]

print("=" * 72)
print("WL-BISINDO LOCAL REALTIME - DYNAMIC SETUP CHECK")
print("=" * 72)
print("Python         :", sys.version.split()[0])
print("Active model   :", active_version)
print("Model name     :", config["name"])
print("Model folder   :", model_dir)
print("Runtime        :", runtime)
print("Inference mode :", mode)
print("Model kind     :", kind)
if config.get("winner_mode"):
    print("Winner mode    :", config["winner_mode"])

if kind == "multimodal":
    print("Expected inputs:")
    for name in ["hand", "pose", "facehead", "facecrop"]:
        shape = config["inputs"].get(name)
        print(f"  {name:<10} : " + " x ".join(map(str, shape)))
    print("Expected output:", f"1 x {num_classes}")
else:
    feature_dim = config["feature_dim"]
    expected = (
        f"1 x {sequence_length} x {feature_dim}"
        if mode == "sequence"
        else f"1 x {feature_dim}"
    )
    print("Expected input :", expected)
    print("Expected output:", f"1 x {num_classes}")

errors = []
warnings = []

try:
    import numpy as np
    print("NumPy          :", np.__version__)
except Exception as exc:
    np = None
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
    torch = None
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

if kind == "multimodal":
    paths = {
        "model": model_dir / config["model_file"],
        "hand_mean": model_dir / "hand_mean.npy",
        "hand_std": model_dir / "hand_std.npy",
        "pose_mean": model_dir / "pose_mean.npy",
        "pose_std": model_dir / "pose_std.npy",
        "face_mean": model_dir / "facehead_mean.npy",
        "face_std": model_dir / "facehead_std.npy",
        "crop_stats": model_dir / "facecrop_stats.json",
        "mapping": model_dir / config["mapping_file"],
        "deployment": model_dir / "deployment_config.json",
    }
else:
    paths = {
        "model": model_dir / config["model_file"],
        "mean": model_dir / config["mean_file"],
        "std": model_dir / config["std_file"],
        "mapping": model_dir / config["mapping_file"],
    }

missing = []
for label, path in paths.items():
    ok = path.exists()
    print(f"{'[OK]' if ok else '[MISSING]':<10} {label:<12} {path.name}")
    if not ok:
        missing.append(path)

if missing and not args.allow_missing_model:
    errors.extend(f"Missing: {path}" for path in missing)

if not missing and np is not None:
    with paths["mapping"].open("r", encoding="utf-8") as f:
        mapping = {int(k): str(v) for k, v in json.load(f).items()}
    print("\nClasses        :", len(mapping))
    if len(mapping) != num_classes:
        errors.append(f"class_mapping harus {num_classes} kelas, found {len(mapping)}")

    if kind == "multimodal":
        expected_stats = {
            "hand_mean": (134,), "hand_std": (134,),
            "pose_mean": (36,), "pose_std": (36,),
            "face_mean": (52,), "face_std": (52,),
        }
        arrays = {}
        for key, shape in expected_stats.items():
            arr = np.load(paths[key]).astype(np.float32)
            arrays[key] = arr
            print(f"{key:<15}:", arr.shape)
            if arr.shape != shape:
                errors.append(f"{key} harus {shape}, found {arr.shape}")
            if not np.isfinite(arr).all():
                errors.append(f"{key} mengandung NaN/Inf")

        with paths["crop_stats"].open("r", encoding="utf-8") as f:
            crop_stats = json.load(f)
        print("facecrop mean  :", crop_stats.get("mean"))
        print("facecrop std   :", crop_stats.get("std"))

        expected_inputs = {
            "hand": [1, 48, 134],
            "pose": [1, 48, 36],
            "facehead": [1, 48, 52],
            "facecrop": [1, 48, 48, 48],
        }
        for key, expected_shape in expected_inputs.items():
            got = list(config["inputs"].get(key, []))
            if got != expected_shape:
                errors.append(f"deployment input {key} harus {expected_shape}, found {got}")

        if not errors and torch is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.jit.load(str(paths["model"]), map_location=device).eval()
            dummy = (
                torch.zeros(1, 48, 134, dtype=torch.float32, device=device),
                torch.zeros(1, 48, 36, dtype=torch.float32, device=device),
                torch.zeros(1, 48, 52, dtype=torch.float32, device=device),
                torch.zeros(1, 48, 48, 48, dtype=torch.float32, device=device),
            )
            with torch.inference_mode():
                output = model(*dummy)
            print("Model output   :", tuple(output.shape))
            if tuple(output.shape) != (1, num_classes):
                errors.append(
                    f"Output model harus (1,{num_classes}), found {tuple(output.shape)}"
                )

    else:
        mean = np.load(paths["mean"]).astype(np.float32)
        std = np.load(paths["std"]).astype(np.float32)
        feature_dim = config["feature_dim"]
        print("feature_mean   :", mean.shape)
        print("feature_std    :", std.shape)
        if mean.shape != (feature_dim,):
            errors.append(f"feature_mean harus ({feature_dim},), found {mean.shape}")
        if std.shape != (feature_dim,):
            errors.append(f"feature_std harus ({feature_dim},), found {std.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            errors.append("Normalisasi mengandung NaN/Inf")

        if not errors and runtime == "torchscript" and torch is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.jit.load(str(paths["model"]), map_location=device).eval()
            dummy_shape = (
                (1, sequence_length, feature_dim)
                if mode == "sequence"
                else (1, feature_dim)
            )
            with torch.inference_mode():
                output = model(torch.zeros(*dummy_shape, dtype=torch.float32, device=device))
            print("Model output   :", tuple(output.shape))
            if tuple(output.shape) != (1, num_classes):
                errors.append(
                    f"Output model harus (1,{num_classes}), found {tuple(output.shape)}"
                )

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
print(f"✅ ACTIVE MODEL: {active_version} ({runtime}/{mode}/{kind})")
