#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WL-BISINDO Hand134 Transformer V4
Fast Continuous Local Realtime + Indonesian Neural TTS

Runtime pipeline:
Webcam / video
  -> MediaPipe Pose (helper only)
  -> MediaPipe Hands
  -> anatomical left/right assignment
  -> ROI recovery for missing hand
  -> rolling 48-frame raw landmark window
  -> short-gap interpolation + EMA smoothing
  -> Hand134 features (48 x 134)
  -> same feature_mean / feature_std normalization as training
  -> TorchScript Hand134 Transformer
  -> confidence + top1/top2 margin + temporal voting
  -> text
  -> Indonesian Edge Neural TTS

Important:
- NO hand_disappeared requirement.
- The rolling sequence is NOT cleared after a word is accepted.
- A different next gesture can be recognized while hands remain visible.
- The same held gesture is emitted only once until a real transition occurs.
- Low-confidence raw classes are hidden from the main UI.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# Prevent MediaPipe from trying optional TensorFlow paths in some installs.
import sys
sys.modules.setdefault("tensorflow", None)

import mediapipe as mp


# ============================================================
# Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent

ACTIVE_MODEL_FILE = APP_DIR / "active_model.txt"
MODEL_ROOT = APP_DIR / "models"


def get_active_model_version() -> str:
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

    # Keep selector simple and safe: v1, v2, v3, model_a, etc.
    if any(ch in version for ch in ("/", "\\", "..")):
        raise ValueError(
            "active_model.txt hanya boleh berisi nama folder model, "
            "misalnya v1 atau v2."
        )

    return version


ACTIVE_MODEL_VERSION = get_active_model_version()
DEFAULT_MODEL_DIR = MODEL_ROOT / ACTIVE_MODEL_VERSION


@dataclass(frozen=True)
class RuntimeSpec:
    version: str
    name: str
    runtime: str
    inference_mode: str
    kind: str
    model_file: str
    mapping_file: str
    sequence_length: int
    num_classes: int
    feature_dim: int = 134
    winner_mode: str | None = None


def load_runtime_spec(model_dir: Path, version: str) -> RuntimeSpec:
    """Auto-detect current deployment package.

    Priority:
    1) deployment_config.json from Multimodal A/B/C/D training
    2) model_config.json from legacy / alphabet deployments
    3) conservative legacy fallback
    """
    deployment_path = model_dir / "deployment_config.json"
    model_config_path = model_dir / "model_config.json"

    if deployment_path.exists():
        with deployment_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        inputs = cfg.get("inputs", {})
        hand_shape = inputs.get("hand", [1, 48, 134])
        pose_shape = inputs.get("pose", [1, 48, 36])
        face_shape = inputs.get("facehead", [1, 48, 52])
        crop_shape = inputs.get("facecrop", [1, 48, 48, 48])

        if list(hand_shape) != [1, 48, 134]:
            raise ValueError(f"deployment_config hand input tidak sesuai: {hand_shape}")
        if list(pose_shape) != [1, 48, 36]:
            raise ValueError(f"deployment_config pose input tidak sesuai: {pose_shape}")
        if list(face_shape) != [1, 48, 52]:
            raise ValueError(f"deployment_config facehead input tidak sesuai: {face_shape}")
        if list(crop_shape) != [1, 48, 48, 48]:
            raise ValueError(f"deployment_config facecrop input tidak sesuai: {crop_shape}")

        return RuntimeSpec(
            version=version,
            name=str(cfg.get("name", "WL-BISINDO Multimodal Temporal Transformer")),
            runtime=str(cfg.get("runtime", "torchscript")).lower(),
            inference_mode="sequence",
            kind="multimodal",
            model_file=str(cfg.get("model_file", "wl_bisindo_multimodal_traced.pt")),
            mapping_file="class_mapping.json",
            sequence_length=int(cfg.get("sequence_length", 48)),
            num_classes=int(cfg.get("num_classes", 32)),
            feature_dim=134,
            winner_mode=str(cfg.get("winner_mode", "C")),
        )

    if model_config_path.exists():
        with model_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        runtime = str(cfg["runtime"]).lower()
        mode = str(cfg["inference_mode"]).lower()
        if runtime not in {"torchscript", "onnxruntime"}:
            raise ValueError(f"Runtime tidak didukung: {runtime}")
        if mode not in {"sequence", "single_frame"}:
            raise ValueError(f"inference_mode tidak didukung: {mode}")

        return RuntimeSpec(
            version=version,
            name=str(cfg["name"]),
            runtime=runtime,
            inference_mode=mode,
            kind="hand134",
            model_file=str(cfg["model_file"]),
            mapping_file=str(cfg["mapping_file"]),
            sequence_length=int(cfg["sequence_length"]),
            num_classes=int(cfg["num_classes"]),
            feature_dim=int(cfg["feature_dim"]),
            winner_mode=None,
        )

    # Legacy fallback kept so older folders are still readable.
    if version == "v1":
        return RuntimeSpec(
            version=version,
            name="WL-BISINDO Hand134 Transformer V4",
            runtime="torchscript",
            inference_mode="sequence",
            kind="hand134",
            model_file="wl_bisindo_hand134_transformer_traced.pt",
            mapping_file="class_mapping.json",
            sequence_length=48,
            num_classes=32,
            feature_dim=134,
        )
    if version == "v2":
        return RuntimeSpec(
            version=version,
            name="BISINDO Alphabet V8.4 Temporal Transformer",
            runtime="torchscript",
            inference_mode="sequence",
            kind="hand134",
            model_file="alphabet_temporal_v8_4_traced.pt",
            mapping_file="class_mapping_v8_4.json",
            sequence_length=48,
            num_classes=26,
            feature_dim=134,
        )

    raise FileNotFoundError(
        f"Tidak ada deployment_config.json / model_config.json di {model_dir}"
    )


RUNTIME_SPEC = load_runtime_spec(DEFAULT_MODEL_DIR, ACTIVE_MODEL_VERSION)

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / RUNTIME_SPEC.model_file
DEFAULT_MAPPING_PATH = DEFAULT_MODEL_DIR / RUNTIME_SPEC.mapping_file

# Legacy hand-only stats. Multimodal uses separate branch stats.
if RUNTIME_SPEC.kind == "hand134":
    if (DEFAULT_MODEL_DIR / "model_config.json").exists():
        with (DEFAULT_MODEL_DIR / "model_config.json").open("r", encoding="utf-8") as f:
            _legacy_cfg = json.load(f)
        DEFAULT_MEAN_PATH = DEFAULT_MODEL_DIR / _legacy_cfg["mean_file"]
        DEFAULT_STD_PATH = DEFAULT_MODEL_DIR / _legacy_cfg["std_file"]
    elif ACTIVE_MODEL_VERSION == "v2":
        DEFAULT_MEAN_PATH = DEFAULT_MODEL_DIR / "feature_mean_v8_4.npy"
        DEFAULT_STD_PATH = DEFAULT_MODEL_DIR / "feature_std_v8_4.npy"
    else:
        DEFAULT_MEAN_PATH = DEFAULT_MODEL_DIR / "feature_mean.npy"
        DEFAULT_STD_PATH = DEFAULT_MODEL_DIR / "feature_std.npy"
else:
    DEFAULT_MEAN_PATH = DEFAULT_MODEL_DIR / "hand_mean.npy"
    DEFAULT_STD_PATH = DEFAULT_MODEL_DIR / "hand_std.npy"

# ============================================================
# Preprocessing V2 constants — MUST match Kaggle preprocessing
# ============================================================

SEQ_LEN = RUNTIME_SPEC.sequence_length
NUM_HANDS = 2
NUM_HAND_LANDMARKS = 21

HAND_FEATURES = 67
FEATURE_DIM = RUNTIME_SPEC.feature_dim
NUM_CLASSES = RUNTIME_SPEC.num_classes

LEFT = 0
RIGHT = 1

LEFT_PRESENCE_IDX = 66
RIGHT_PRESENCE_IDX = 133

# Recovery / smoothing.
MAX_INTERP_GAP = 6
EDGE_FILL = 2
EMA_ALPHA = 0.65
FACE_BOX_EMA_ALPHA = 0.65

# Pose quality.
POSE_VIS_THRESHOLD = 0.25
POSE_FEATURE_VIS_THRESHOLD = 0.20
FACE_POINT_VIS_THRESHOLD = 0.20

POSE_SELECTED = [
    (0, "nose"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
]
POSE_DIM = 36
POSE_VIS_IDXS = [3, 7, 11, 15, 19, 23, 27, 31, 35]

FACE_POSE_SELECTED = [
    (0, "nose"),
    (1, "left_eye_inner"),
    (2, "left_eye"),
    (3, "left_eye_outer"),
    (4, "right_eye_inner"),
    (5, "right_eye"),
    (6, "right_eye_outer"),
    (7, "left_ear"),
    (8, "right_ear"),
    (9, "mouth_left"),
    (10, "mouth_right"),
]
FACEHEAD_DIM = 52
FACE_HEAD_DIM = FACEHEAD_DIM
FACE_VIS_IDXS = list(range(33, 44))
FACE_PRESENCE_IDX = 51
FACE_CROP_SIZE = 48

# Hand detector.
FULL_HAND_DET_CONF = 0.30
FULL_HAND_TRACK_CONF = 0.30
RECOVERY_DET_CONF = 0.20

# Low-light / backlight fallback. This only changes pixels seen by
# MediaPipe; landmark coordinates and Hand134 math stay unchanged.
FALLBACK_HAND_DET_CONF = 0.15
DETECTOR_CLAHE_CLIP = 2.0
DETECTOR_DARK_CENTER_MEAN = 92.0
DETECTOR_LOW_P10 = 34.0


# ============================================================
# Detector-only photometric robustness
# ============================================================

def prepare_detector_frame(frame, enabled=True):
    """
    Improve dark/backlit webcam frames for MediaPipe only.

    IMPORTANT:
    - Display still uses the original camera frame.
    - No geometry is changed (no resize/crop/flip here).
    - Therefore Hand134 coordinates keep the same coordinate system.
    """
    if not enabled:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        cy0, cy1 = int(h * 0.18), int(h * 0.88)
        cx0, cx1 = int(w * 0.18), int(w * 0.82)
        center = gray[cy0:cy1, cx0:cx1]
        stats = {
            "center_mean": float(center.mean()) if center.size else float(gray.mean()),
            "p10": float(np.percentile(gray, 10)),
            "p90": float(np.percentile(gray, 90)),
            "enhanced": False,
        }
        return frame, False, stats

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cy0, cy1 = int(h * 0.18), int(h * 0.88)
    cx0, cx1 = int(w * 0.18), int(w * 0.82)
    center = gray[cy0:cy1, cx0:cx1]

    center_mean = float(center.mean()) if center.size else float(gray.mean())
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))

    # Strong window/backlight usually produces a dark person in the center
    # even when the global image mean looks acceptable.
    should_enhance = (
        center_mean < DETECTOR_DARK_CENTER_MEAN
        or p10 < DETECTOR_LOW_P10
        or (p90 - p10) > 155.0
    )

    if not should_enhance:
        return frame, False, {
            "center_mean": center_mean,
            "p10": p10,
            "p90": p90,
            "enhanced": False,
        }

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=DETECTOR_CLAHE_CLIP,
        tileGridSize=(8, 8),
    )
    l_chan = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(
        cv2.merge((l_chan, a_chan, b_chan)),
        cv2.COLOR_LAB2BGR,
    )

    # Mild gamma lift only when the central subject is very dark.
    if center_mean < 70.0:
        gamma = 0.72
        lut = np.array(
            [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(enhanced, lut)

    return enhanced, True, {
        "center_mean": center_mean,
        "p10": p10,
        "p90": p90,
        "enhanced": True,
    }


# ============================================================
# Small helpers
# ============================================================

def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping = {int(k): str(v) for k, v in raw.items()}

    if len(mapping) != NUM_CLASSES:
        raise ValueError(
            f"class_mapping harus {NUM_CLASSES} kelas, "
            f"found {len(mapping)}"
        )

    return mapping


def load_runtime_files(
    model_path: Path,
    mean_path: Path,
    std_path: Path,
    mapping_path: Path,
    device: torch.device,
):
    mapping = load_mapping(mapping_path)

    if RUNTIME_SPEC.kind == "multimodal":
        paths = {
            "model": model_path,
            "hand_mean": DEFAULT_MODEL_DIR / "hand_mean.npy",
            "hand_std": DEFAULT_MODEL_DIR / "hand_std.npy",
            "pose_mean": DEFAULT_MODEL_DIR / "pose_mean.npy",
            "pose_std": DEFAULT_MODEL_DIR / "pose_std.npy",
            "face_mean": DEFAULT_MODEL_DIR / "facehead_mean.npy",
            "face_std": DEFAULT_MODEL_DIR / "facehead_std.npy",
            "crop_stats": DEFAULT_MODEL_DIR / "facecrop_stats.json",
            "mapping": mapping_path,
        }
        missing = [p for p in paths.values() if not p.exists()]
        if missing:
            lines = "\n".join(f" - {p}" for p in missing)
            raise FileNotFoundError(
                "File runtime multimodal belum lengkap:\n"
                f"{lines}"
            )

        stats = {
            "hand_mean": np.load(paths["hand_mean"]).astype(np.float32),
            "hand_std": np.load(paths["hand_std"]).astype(np.float32),
            "pose_mean": np.load(paths["pose_mean"]).astype(np.float32),
            "pose_std": np.load(paths["pose_std"]).astype(np.float32),
            "face_mean": np.load(paths["face_mean"]).astype(np.float32),
            "face_std": np.load(paths["face_std"]).astype(np.float32),
        }
        with paths["crop_stats"].open("r", encoding="utf-8") as f:
            crop_stats = json.load(f)
        stats["crop_mean"] = float(crop_stats["mean"])
        stats["crop_std"] = max(float(crop_stats["std"]), 1e-4)

        expected = {
            "hand_mean": (134,), "hand_std": (134,),
            "pose_mean": (36,), "pose_std": (36,),
            "face_mean": (52,), "face_std": (52,),
        }
        for key, shape in expected.items():
            if stats[key].shape != shape:
                raise ValueError(f"{key} harus {shape}, found {stats[key].shape}")
            if not np.isfinite(stats[key]).all():
                raise ValueError(f"{key} mengandung NaN/Inf")
        for key in ["hand_std", "pose_std", "face_std"]:
            stats[key] = np.where(stats[key] < 1e-5, 1.0, stats[key]).astype(np.float32)

        if RUNTIME_SPEC.runtime != "torchscript":
            raise ValueError("Multimodal deployment saat ini harus TorchScript")

        model = torch.jit.load(str(model_path), map_location=device).eval()
        dummy_hand = torch.zeros(1, SEQ_LEN, 134, dtype=torch.float32, device=device)
        dummy_pose = torch.zeros(1, SEQ_LEN, 36, dtype=torch.float32, device=device)
        dummy_face = torch.zeros(1, SEQ_LEN, 52, dtype=torch.float32, device=device)
        dummy_crop = torch.zeros(1, SEQ_LEN, 48, 48, dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model(dummy_hand, dummy_pose, dummy_face, dummy_crop)
        if tuple(output.shape) != (1, NUM_CLASSES):
            raise RuntimeError(
                f"Output multimodal harus (1,{NUM_CLASSES}), found {tuple(output.shape)}"
            )
        return model, stats, mapping

    # Legacy / alphabet Hand134 single-input path.
    required = [model_path, mean_path, std_path, mapping_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        lines = "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError("File runtime belum lengkap:\n" + lines)

    feature_mean = np.load(mean_path).astype(np.float32)
    feature_std = np.load(std_path).astype(np.float32)
    if feature_mean.shape != (FEATURE_DIM,):
        raise ValueError(f"feature_mean harus ({FEATURE_DIM},), found {feature_mean.shape}")
    if feature_std.shape != (FEATURE_DIM,):
        raise ValueError(f"feature_std harus ({FEATURE_DIM},), found {feature_std.shape}")
    feature_std = np.where(feature_std < 1e-5, 1.0, feature_std).astype(np.float32)

    if RUNTIME_SPEC.runtime == "torchscript":
        model = torch.jit.load(str(model_path), map_location=device).eval()
        dummy_shape = (1, SEQ_LEN, FEATURE_DIM) if RUNTIME_SPEC.inference_mode == "sequence" else (1, FEATURE_DIM)
        dummy = torch.zeros(*dummy_shape, dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model(dummy)
        output_shape = tuple(output.shape)
    else:
        import onnxruntime as ort
        model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = model.get_inputs()[0]
        dummy_shape = (1, SEQ_LEN, FEATURE_DIM) if RUNTIME_SPEC.inference_mode == "sequence" else (1, FEATURE_DIM)
        output = model.run(None, {input_meta.name: np.zeros(dummy_shape, dtype=np.float32)})[0]
        output_shape = tuple(output.shape)

    if output_shape != (1, NUM_CLASSES):
        raise RuntimeError(f"Output model harus (1,{NUM_CLASSES}), found {output_shape}")

    return model, {"feature_mean": feature_mean, "feature_std": feature_std}, mapping


def landmark_list_to_array(hand_landmarks) -> np.ndarray:
    return np.asarray(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )


# ============================================================
# Preprocessing V2: pose anchor
# ============================================================

def pose_anchor(pose_result):
    body_center = np.array(
        [0.5, 0.5, 0.0],
        dtype=np.float32,
    )
    body_scale = 0.35

    left_wrist = None
    right_wrist = None
    valid = 0.0

    if pose_result.pose_landmarks is None:
        return (
            body_center,
            body_scale,
            left_wrist,
            right_wrist,
            valid,
        )

    lms = pose_result.pose_landmarks.landmark

    left_shoulder = np.array(
        [lms[11].x, lms[11].y, lms[11].z],
        dtype=np.float32,
    )

    right_shoulder = np.array(
        [lms[12].x, lms[12].y, lms[12].z],
        dtype=np.float32,
    )

    body_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    body_scale = float(
        max(
            np.linalg.norm(
                left_shoulder[:2] - right_shoulder[:2]
            ),
            0.08,
        )
    )

    if lms[15].visibility >= POSE_VIS_THRESHOLD:
        left_wrist = np.array(
            [lms[15].x, lms[15].y, lms[15].z],
            dtype=np.float32,
        )

    if lms[16].visibility >= POSE_VIS_THRESHOLD:
        right_wrist = np.array(
            [lms[16].x, lms[16].y, lms[16].z],
            dtype=np.float32,
        )

    valid = 1.0

    return (
        body_center,
        body_scale,
        left_wrist,
        right_wrist,
        valid,
    )


def extract_selected_pose_raw(
    pose_result,
):
    coords = np.full(
        (
            len(POSE_SELECTED),
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid = np.zeros(
        len(POSE_SELECTED),
        dtype=np.uint8,
    )

    visibility = np.zeros(
        len(POSE_SELECTED),
        dtype=np.float32,
    )

    if (
        pose_result.pose_landmarks
        is None
    ):
        return (
            coords,
            valid,
            visibility,
        )

    lms = (
        pose_result
        .pose_landmarks
        .landmark
    )

    for j, (
        index,
        _,
    ) in enumerate(
        POSE_SELECTED
    ):
        lm = lms[index]

        vis = float(
            np.clip(
                lm.visibility,
                0.0,
                1.0,
            )
        )

        visibility[j] = vis

        if (
            vis
            >= POSE_FEATURE_VIS_THRESHOLD
        ):
            coords[j] = np.array(
                [
                    lm.x,
                    lm.y,
                    lm.z,
                ],
                dtype=np.float32,
            )

            valid[j] = 1

    return (
        coords,
        valid,
        visibility,
    )


def pose_to_feature(
    pose_coords,
    pose_valid,
    pose_visibility,
    body_center,
    body_scale,
):
    feat = np.zeros(
        POSE_DIM,
        dtype=np.float32,
    )

    scale = max(
        float(body_scale),
        0.08,
    )

    for j in range(
        len(POSE_SELECTED)
    ):
        base = j * 4

        if (
            pose_valid[j] > 0
            and np.isfinite(
                pose_coords[j]
            ).all()
        ):
            local = (
                pose_coords[j]
                - body_center
            ) / scale

            feat[
                base:base + 3
            ] = np.clip(
                local,
                -10.0,
                10.0,
            )

            feat[
                base + 3
            ] = float(
                np.clip(
                    pose_visibility[j],
                    0.0,
                    1.0,
                )
            )

    return feat


def extract_face_pose_raw(
    pose_result,
):
    coords = np.full(
        (
            len(
                FACE_POSE_SELECTED
            ),
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    visibility = np.zeros(
        len(
            FACE_POSE_SELECTED
        ),
        dtype=np.float32,
    )

    valid = np.zeros(
        len(
            FACE_POSE_SELECTED
        ),
        dtype=np.uint8,
    )

    if (
        pose_result.pose_landmarks
        is None
    ):
        return (
            coords,
            visibility,
            valid,
        )

    lms = (
        pose_result
        .pose_landmarks
        .landmark
    )

    for j, (
        index,
        _,
    ) in enumerate(
        FACE_POSE_SELECTED
    ):
        lm = lms[index]

        vis = float(
            np.clip(
                lm.visibility,
                0.0,
                1.0,
            )
        )

        visibility[j] = vis

        if (
            vis
            >= FACE_POINT_VIS_THRESHOLD
        ):
            coords[j] = np.array(
                [
                    lm.x,
                    lm.y,
                    lm.z,
                ],
                dtype=np.float32,
            )

            valid[j] = 1

    return (
        coords,
        visibility,
        valid,
    )


def safe_distance(
    coords,
    valid,
    a,
    b,
):
    if (
        valid[a] > 0
        and valid[b] > 0
        and np.isfinite(
            coords[a]
        ).all()
        and np.isfinite(
            coords[b]
        ).all()
    ):
        return float(
            np.linalg.norm(
                coords[
                    a,
                    :2,
                ]
                - coords[
                    b,
                    :2,
                ]
            )
        )

    return 0.0


def facehead_to_feature(
    coords,
    visibility,
    valid,
    body_center,
    body_scale,
):
    feat = np.zeros(
        FACE_HEAD_DIM,
        dtype=np.float32,
    )

    # Need at least a small face constellation.
    num_valid = int(
        valid.sum()
    )

    if num_valid < 3:
        return (
            feat,
            0,
        )

    # Array positions:
    # 0 nose
    # 2 left eye center
    # 5 right eye center
    # 7 left ear
    # 8 right ear
    # 9 mouth left
    # 10 mouth right

    if (
        valid[2] > 0
        and valid[5] > 0
    ):
        face_center = (
            coords[2]
            + coords[5]
        ) / 2.0

    elif (
        valid[0] > 0
    ):
        face_center = (
            coords[0]
            .copy()
        )

    else:
        valid_points = (
            coords[
                valid > 0
            ]
        )

        face_center = np.mean(
            valid_points,
            axis=0,
        )

    ear_distance = safe_distance(
        coords,
        valid,
        7,
        8,
    )

    eye_distance = safe_distance(
        coords,
        valid,
        2,
        5,
    )

    face_scale = max(
        ear_distance,
        eye_distance,
        float(body_scale)
        * 0.35,
        0.04,
    )

    # 11 xyz = 33
    for j in range(
        len(
            FACE_POSE_SELECTED
        )
    ):
        if (
            valid[j] > 0
            and np.isfinite(
                coords[j]
            ).all()
        ):
            local = (
                coords[j]
                - face_center
            ) / face_scale

            start = (
                j * 3
            )

            feat[
                start:start + 3
            ] = np.clip(
                local,
                -6.0,
                6.0,
            )

    # 11 visibility = 11 -> indices 33:44
    feat[
        33:44
    ] = np.where(
        valid > 0,
        np.clip(
            visibility,
            0.0,
            1.0,
        ),
        0.0,
    ).astype(
        np.float32
    )

    mouth_width = safe_distance(
        coords,
        valid,
        9,
        10,
    )

    # Eye-line orientation.
    eye_sin = 0.0
    eye_cos = 0.0

    if (
        valid[2] > 0
        and valid[5] > 0
    ):
        delta = (
            coords[
                5,
                :2,
            ]
            - coords[
                2,
                :2,
            ]
        )

        norm = float(
            np.linalg.norm(
                delta
            )
        )

        if norm > 1e-6:
            eye_cos = float(
                delta[0]
                / norm
            )

            eye_sin = float(
                delta[1]
                / norm
            )

    # Nose location relative to body.
    nose_body_x = 0.0
    nose_body_y = 0.0

    if (
        valid[0] > 0
        and np.isfinite(
            coords[0]
        ).all()
    ):
        nose_body = (
            coords[0]
            - body_center
        ) / max(
            float(body_scale),
            0.08,
        )

        nose_body_x = float(
            np.clip(
                nose_body[0],
                -10.0,
                10.0,
            )
        )

        nose_body_y = float(
            np.clip(
                nose_body[1],
                -10.0,
                10.0,
            )
        )

    # 7 geometry/global -> 44:51
    feat[
        44:51
    ] = np.asarray(
        [
            eye_distance
            / face_scale,
            ear_distance
            / face_scale,
            mouth_width
            / face_scale,
            eye_sin,
            eye_cos,
            nose_body_x,
            nose_body_y,
        ],
        dtype=np.float32,
    )

    # Presence -> index 51
    feat[51] = 1.0

    return (
        feat,
        1,
    )


def face_bbox_from_pose(
    coords,
    valid,
    frame_shape,
    body_scale,
):
    h, w = frame_shape[:2]

    valid_xy = (
        coords[
            valid > 0,
            :2,
        ]
    )

    if len(valid_xy) < 3:
        return None

    x_min = float(
        np.min(
            valid_xy[:, 0]
        )
    )

    x_max = float(
        np.max(
            valid_xy[:, 0]
        )
    )

    y_min = float(
        np.min(
            valid_xy[:, 1]
        )
    )

    y_max = float(
        np.max(
            valid_xy[:, 1]
        )
    )

    cx = (
        x_min
        + x_max
    ) / 2.0

    cy = (
        y_min
        + y_max
    ) / 2.0

    raw_w = max(
        x_max - x_min,
        float(body_scale)
        * 0.45,
        0.06,
    )

    raw_h = max(
        y_max - y_min,
        float(body_scale)
        * 0.55,
        0.08,
    )

    # Include forehead/chin area beyond sparse pose face points.
    side = max(
        raw_w * 1.80,
        raw_h * 2.00,
    )

    side = float(
        np.clip(
            side,
            0.10,
            0.55,
        )
    )

    # Shift slightly upward because pose mouth/nose points make
    # the raw center a little low.
    cy = (
        cy
        - 0.10
        * side
    )

    x0 = int(
        np.clip(
            (
                cx
                - side / 2.0
            )
            * w,
            0,
            w - 1,
        )
    )

    y0 = int(
        np.clip(
            (
                cy
                - side / 2.0
            )
            * h,
            0,
            h - 1,
        )
    )

    x1 = int(
        np.clip(
            (
                cx
                + side / 2.0
            )
            * w,
            1,
            w,
        )
    )

    y1 = int(
        np.clip(
            (
                cy
                + side / 2.0
            )
            * h,
            1,
            h,
        )
    )

    if (
        x1 - x0 < 12
        or y1 - y0 < 12
    ):
        return None

    return np.asarray(
        [
            x0,
            y0,
            x1,
            y1,
        ],
        dtype=np.float32,
    )


def smooth_bbox(
    bbox,
    prev_bbox,
    alpha=FACE_BOX_EMA_ALPHA,
):
    if bbox is None:
        return prev_bbox

    if prev_bbox is None:
        return bbox.copy()

    return (
        alpha
        * bbox
        + (
            1.0
            - alpha
        )
        * prev_bbox
    ).astype(
        np.float32
    )


def crop_face_gray(
    frame,
    bbox,
):
    if bbox is None:
        return (
            np.zeros(
                (
                    FACE_CROP_SIZE,
                    FACE_CROP_SIZE,
                ),
                dtype=np.uint8,
            ),
            0,
        )

    h, w = (
        frame.shape[:2]
    )

    x0, y0, x1, y1 = [
        int(
            round(v)
        )
        for v in bbox
    ]

    x0 = int(
        np.clip(
            x0,
            0,
            w - 1,
        )
    )

    y0 = int(
        np.clip(
            y0,
            0,
            h - 1,
        )
    )

    x1 = int(
        np.clip(
            x1,
            x0 + 1,
            w,
        )
    )

    y1 = int(
        np.clip(
            y1,
            y0 + 1,
            h,
        )
    )

    crop = frame[
        y0:y1,
        x0:x1,
    ]

    if crop.size == 0:
        return (
            np.zeros(
                (
                    FACE_CROP_SIZE,
                    FACE_CROP_SIZE,
                ),
                dtype=np.uint8,
            ),
            0,
        )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    resized = cv2.resize(
        gray,
        (
            FACE_CROP_SIZE,
            FACE_CROP_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return (
        resized.astype(
            np.uint8
        ),
        1,
    )


# ============================================================
# Preprocessing V2: hand candidates + anatomical assignment
# ============================================================

def collect_full_hand_candidates(hand_result):
    candidates = []

    if hand_result.multi_hand_landmarks is None:
        return candidates

    handedness_list = hand_result.multi_handedness or []

    for i, landmarks in enumerate(
        hand_result.multi_hand_landmarks
    ):
        arr = landmark_list_to_array(landmarks)

        handedness_label = None
        handedness_score = 0.0

        if i < len(handedness_list):
            cls = handedness_list[i].classification[0]
            handedness_label = cls.label
            handedness_score = float(cls.score)

        candidates.append(
            {
                "arr": arr,
                "wrist": arr[0],
                "handedness_label": handedness_label,
                "handedness_score": handedness_score,
            }
        )

    return candidates


def xy_distance(a, b) -> float:
    return float(
        np.linalg.norm(
            a[:2] - b[:2]
        )
    )


def assign_candidates_to_body_sides(
    candidates,
    left_pose_wrist,
    right_pose_wrist,
    prev_left_wrist,
    prev_right_wrist,
):
    assigned = {
        LEFT: None,
        RIGHT: None,
    }

    if len(candidates) == 0:
        return assigned

    anchors = {
        LEFT: (
            left_pose_wrist
            if left_pose_wrist is not None
            else prev_left_wrist
        ),
        RIGHT: (
            right_pose_wrist
            if right_pose_wrist is not None
            else prev_right_wrist
        ),
    }

    if len(candidates) == 1:
        cand = candidates[0]
        costs = {}

        for side in [LEFT, RIGHT]:
            anchor = anchors[side]
            if anchor is not None:
                costs[side] = xy_distance(
                    cand["wrist"],
                    anchor,
                )

        if costs:
            chosen_side = min(
                costs,
                key=costs.get,
            )
            assigned[chosen_side] = cand["arr"]
            return assigned

        # Same fallback convention as preprocessing V2.
        # Dataset input is not mirrored, so MediaPipe label is flipped.
        label = cand["handedness_label"]

        if label == "Left":
            assigned[RIGHT] = cand["arr"]
        elif label == "Right":
            assigned[LEFT] = cand["arr"]
        else:
            assigned[LEFT] = cand["arr"]

        return assigned

    candidates = candidates[:2]

    c0 = candidates[0]
    c1 = candidates[1]

    if (
        anchors[LEFT] is not None
        and anchors[RIGHT] is not None
    ):
        cost_a = (
            xy_distance(
                c0["wrist"],
                anchors[LEFT],
            )
            + xy_distance(
                c1["wrist"],
                anchors[RIGHT],
            )
        )

        cost_b = (
            xy_distance(
                c0["wrist"],
                anchors[RIGHT],
            )
            + xy_distance(
                c1["wrist"],
                anchors[LEFT],
            )
        )

        if cost_a <= cost_b:
            assigned[LEFT] = c0["arr"]
            assigned[RIGHT] = c1["arr"]
        else:
            assigned[LEFT] = c1["arr"]
            assigned[RIGHT] = c0["arr"]

        return assigned

    remaining = [0, 1]

    for side in [LEFT, RIGHT]:
        anchor = anchors[side]

        if anchor is None or not remaining:
            continue

        best_idx = min(
            remaining,
            key=lambda j: xy_distance(
                candidates[j]["wrist"],
                anchor,
            ),
        )

        assigned[side] = candidates[best_idx]["arr"]
        remaining.remove(best_idx)

    if remaining:
        if assigned[LEFT] is None:
            assigned[LEFT] = candidates[
                remaining[0]
            ]["arr"]
        elif assigned[RIGHT] is None:
            assigned[RIGHT] = candidates[
                remaining[0]
            ]["arr"]

    return assigned


# ============================================================
# Preprocessing V2: ROI recovery
# ============================================================

def make_wrist_roi(
    frame,
    target_wrist,
    body_scale,
):
    if target_wrist is None:
        return None

    h, w = frame.shape[:2]

    cx = int(
        np.clip(
            target_wrist[0],
            0.0,
            1.0,
        )
        * w
    )

    cy = int(
        np.clip(
            target_wrist[1],
            0.0,
            1.0,
        )
        * h
    )

    shoulder_px = (
        body_scale
        * max(w, h)
    )

    half = int(
        np.clip(
            1.10 * shoulder_px,
            72,
            0.30 * max(w, h),
        )
    )

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)

    if x1 - x0 < 48 or y1 - y0 < 48:
        return None

    crop = frame[y0:y1, x0:x1]

    return (
        crop,
        x0,
        y0,
        x1,
        y1,
    )


def map_crop_hand_to_full(
    crop_hand_landmarks,
    x0,
    y0,
    x1,
    y1,
    full_w,
    full_h,
):
    crop_w = x1 - x0
    crop_h = y1 - y0

    arr = np.zeros(
        (
            NUM_HAND_LANDMARKS,
            3,
        ),
        dtype=np.float32,
    )

    for i, lm in enumerate(
        crop_hand_landmarks.landmark
    ):
        arr[i, 0] = (
            x0 + lm.x * crop_w
        ) / full_w

        arr[i, 1] = (
            y0 + lm.y * crop_h
        ) / full_h

        arr[i, 2] = (
            lm.z * crop_w / full_w
        )

    return arr


def recover_hand_from_roi(
    frame,
    target_wrist,
    body_scale,
    recovery_hands,
):
    roi = make_wrist_roi(
        frame,
        target_wrist,
        body_scale,
    )

    if roi is None:
        return None

    (
        crop,
        x0,
        y0,
        x1,
        y1,
    ) = roi

    detector_crop, _, _ = prepare_detector_frame(
        crop,
        enabled=True,
    )

    crop_rgb = cv2.cvtColor(
        detector_crop,
        cv2.COLOR_BGR2RGB,
    )

    result = recovery_hands.process(
        crop_rgb
    )

    if (
        result.multi_hand_landmarks is None
        or len(
            result.multi_hand_landmarks
        ) == 0
    ):
        return None

    full_h, full_w = frame.shape[:2]

    arrays = []

    for landmarks in (
        result.multi_hand_landmarks
    ):
        arr = map_crop_hand_to_full(
            landmarks,
            x0,
            y0,
            x1,
            y1,
            full_w,
            full_h,
        )
        arrays.append(arr)

    return min(
        arrays,
        key=lambda arr: xy_distance(
            arr[0],
            target_wrist,
        ),
    )


# ============================================================
# Preprocessing V2: temporal recovery + smoothing
# ============================================================

def interpolate_short_gaps(
    track,
    max_gap=MAX_INTERP_GAP,
    edge_fill=EDGE_FILL,
):
    out = track.copy()

    valid = np.isfinite(
        out
    ).all(
        axis=(1, 2)
    )

    valid_idx = np.where(
        valid
    )[0]

    if len(valid_idx) == 0:
        return out, valid.copy()

    for a, b in zip(
        valid_idx[:-1],
        valid_idx[1:],
    ):
        gap = b - a - 1

        if (
            gap <= 0
            or gap > max_gap
        ):
            continue

        start = out[a]
        end = out[b]

        for k in range(
            1,
            gap + 1,
        ):
            ratio = (
                k / (gap + 1)
            )

            out[a + k] = (
                (1.0 - ratio) * start
                + ratio * end
            )

    first = int(valid_idx[0])
    last = int(valid_idx[-1])

    for t in range(
        max(0, first - edge_fill),
        first,
    ):
        out[t] = out[first]

    for t in range(
        last + 1,
        min(
            len(out),
            last + edge_fill + 1,
        ),
    ):
        out[t] = out[last]

    new_valid = np.isfinite(
        out
    ).all(
        axis=(1, 2)
    )

    return out, new_valid


def ema_smooth_track(
    track,
    valid_mask,
    alpha=EMA_ALPHA,
):
    out = track.copy()
    prev = None

    for t in range(len(out)):
        if not valid_mask[t]:
            prev = None
            continue

        if prev is None:
            prev = out[t].copy()
        else:
            prev = (
                alpha * out[t]
                + (1.0 - alpha) * prev
            )
            out[t] = prev

    return out


def fill_pose_anchors(
    centers,
    scales,
    valid,
):
    out_centers = centers.copy()
    out_scales = scales.copy()

    valid_idx = np.where(
        valid > 0.5
    )[0]

    if len(valid_idx) == 0:
        out_centers[:] = np.array(
            [0.5, 0.5, 0.0],
            dtype=np.float32,
        )
        out_scales[:] = 0.35
        return out_centers, out_scales

    timeline = np.arange(
        len(valid)
    )

    for d in range(3):
        out_centers[:, d] = np.interp(
            timeline,
            valid_idx,
            centers[
                valid_idx,
                d,
            ],
        )

    out_scales[:] = np.interp(
        timeline,
        valid_idx,
        scales[valid_idx],
    )

    return out_centers, out_scales


# ============================================================
# Preprocessing V2: Hand134
# ============================================================

def hand_to_feature(
    hand_arr,
    body_center,
    body_scale,
    valid,
):
    feat = np.zeros(
        HAND_FEATURES,
        dtype=np.float32,
    )

    if not valid:
        return feat

    wrist = hand_arr[0].copy()

    local = (
        hand_arr - wrist
    )

    hand_scale = float(
        max(
            np.linalg.norm(
                hand_arr[9, :2]
                - hand_arr[0, :2]
            ),
            np.linalg.norm(
                hand_arr[5, :2]
                - hand_arr[17, :2]
            ),
            np.linalg.norm(
                hand_arr[12, :2]
                - hand_arr[0, :2]
            ),
            0.025,
        )
    )

    local = (
        local / hand_scale
    )

    wrist_global = (
        wrist - body_center
    ) / max(
        body_scale,
        0.08,
    )

    feat[:63] = local.reshape(-1)
    feat[63:66] = wrist_global
    feat[66] = 1.0

    feat[:66] = np.clip(
        feat[:66],
        -10.0,
        10.0,
    )

    return feat


@dataclass
class FrameState:
    tracks: np.ndarray              # [2,21,3], NaN for missing
    observed: np.ndarray            # [2]
    body_center: np.ndarray         # [3]
    body_scale: float
    pose_valid: int
    pose_coords: np.ndarray         # [9,3]
    pose_point_valid: np.ndarray    # [9]
    pose_visibility: np.ndarray     # [9]
    face_coords: np.ndarray         # [11,3]
    face_visibility: np.ndarray     # [11]
    face_point_valid: np.ndarray    # [11]
    face_crop: np.ndarray           # [48,48] uint8
    face_crop_valid: int


def _recover_hand_tracks(states):
    """Apply the same short-gap interpolation + EMA as preprocessing."""
    tracks = np.stack(
        [s.tracks for s in states],
        axis=0,
    ).astype(np.float32)

    observed = np.stack(
        [s.observed for s in states],
        axis=0,
    ).astype(np.uint8)

    final_valid = np.zeros(
        (SEQ_LEN, NUM_HANDS),
        dtype=np.uint8,
    )

    for side in [LEFT, RIGHT]:
        track, valid_mask = interpolate_short_gaps(
            tracks[:, side]
        )
        track = ema_smooth_track(
            track,
            valid_mask,
        )
        tracks[:, side] = track
        final_valid[:, side] = valid_mask.astype(np.uint8)

    return tracks, observed, final_valid


def normalize_hand_branch(sequence, mean, std):
    left_presence = sequence[:, LEFT_PRESENCE_IDX].copy()
    right_presence = sequence[:, RIGHT_PRESENCE_IDX].copy()

    x = (sequence - mean) / std

    left_absent = left_presence < 0.5
    right_absent = right_presence < 0.5

    x[left_absent, 0:66] = 0.0
    x[right_absent, 67:133] = 0.0
    x[:, LEFT_PRESENCE_IDX] = left_presence
    x[:, RIGHT_PRESENCE_IDX] = right_presence

    return x.astype(np.float32)


def normalize_pose_branch(sequence, mean, std):
    vis = sequence[:, POSE_VIS_IDXS].copy()
    x = (sequence - mean) / std

    for j, vis_idx in enumerate(POSE_VIS_IDXS):
        base = j * 4
        absent = vis[:, j] < 0.2
        x[absent, base:base + 3] = 0.0
        x[:, vis_idx] = vis[:, j]

    return x.astype(np.float32)


def normalize_facehead_branch(sequence, mean, std):
    vis = sequence[:, 33:44].copy()
    presence = sequence[:, FACE_PRESENCE_IDX].copy()
    x = (sequence - mean) / std

    for j in range(11):
        base = j * 3
        absent = vis[:, j] < 0.2
        x[absent, base:base + 3] = 0.0

    x[:, 33:44] = vis
    x[:, FACE_PRESENCE_IDX] = presence

    face_absent = presence < 0.5
    x[face_absent, 0:33] = 0.0
    x[face_absent, 44:51] = 0.0

    return x.astype(np.float32)


def build_hand134_sequence(states, feature_mean, feature_std):
    if len(states) < SEQ_LEN:
        return None, None

    states = list(states)[-SEQ_LEN:]

    tracks, observed, final_valid = _recover_hand_tracks(states)

    body_centers = np.stack(
        [s.body_center for s in states],
        axis=0,
    ).astype(np.float32)

    body_scales = np.asarray(
        [s.body_scale for s in states],
        dtype=np.float32,
    )

    pose_valid = np.asarray(
        [s.pose_valid for s in states],
        dtype=np.uint8,
    )

    body_centers, body_scales = fill_pose_anchors(
        body_centers,
        body_scales,
        pose_valid,
    )

    sequence = np.zeros(
        (SEQ_LEN, 134),
        dtype=np.float32,
    )

    for t in range(SEQ_LEN):
        left_feat = hand_to_feature(
            tracks[t, LEFT],
            body_centers[t],
            body_scales[t],
            bool(final_valid[t, LEFT]),
        )
        right_feat = hand_to_feature(
            tracks[t, RIGHT],
            body_centers[t],
            body_scales[t],
            bool(final_valid[t, RIGHT]),
        )
        sequence[t] = np.concatenate(
            [left_feat, right_feat]
        ).astype(np.float32)

    x = normalize_hand_branch(
        sequence,
        feature_mean,
        feature_std,
    )

    quality = {
        "observed_any": float((observed.sum(axis=1) > 0).mean()),
        "valid_any": float((final_valid.sum(axis=1) > 0).mean()),
        "left_valid": float(final_valid[:, LEFT].mean()),
        "right_valid": float(final_valid[:, RIGHT].mean()),
        "pose_valid": float(pose_valid.mean()),
        "face_valid": 0.0,
        "face_crop_valid": 0.0,
    }

    return x, quality


def build_multimodal_inputs(states, stats):
    """Build exact runtime inputs for the winning multimodal model.

    Returns a 4-input tuple because the traced model keeps the common
    A/B/C/D forward signature even when winner C does not use FaceCropCNN.
    """
    if len(states) < SEQ_LEN:
        return None, None

    states = list(states)[-SEQ_LEN:]

    tracks, observed, final_valid = _recover_hand_tracks(states)

    body_centers = np.stack(
        [s.body_center for s in states],
        axis=0,
    ).astype(np.float32)
    body_scales = np.asarray(
        [s.body_scale for s in states],
        dtype=np.float32,
    )
    pose_frame_valid = np.asarray(
        [s.pose_valid for s in states],
        dtype=np.uint8,
    )

    body_centers, body_scales = fill_pose_anchors(
        body_centers,
        body_scales,
        pose_frame_valid,
    )

    hand_sequence = np.zeros((SEQ_LEN, 134), dtype=np.float32)
    pose_sequence = np.zeros((SEQ_LEN, POSE_DIM), dtype=np.float32)
    face_sequence = np.zeros((SEQ_LEN, FACEHEAD_DIM), dtype=np.float32)
    facehead_valid = np.zeros(SEQ_LEN, dtype=np.uint8)
    face_crop_valid = np.asarray(
        [s.face_crop_valid for s in states],
        dtype=np.uint8,
    )
    face_crops = np.stack(
        [s.face_crop for s in states],
        axis=0,
    ).astype(np.float32)

    for t, state in enumerate(states):
        left_feat = hand_to_feature(
            tracks[t, LEFT],
            body_centers[t],
            body_scales[t],
            bool(final_valid[t, LEFT]),
        )
        right_feat = hand_to_feature(
            tracks[t, RIGHT],
            body_centers[t],
            body_scales[t],
            bool(final_valid[t, RIGHT]),
        )
        hand_sequence[t] = np.concatenate(
            [left_feat, right_feat]
        ).astype(np.float32)

        pose_sequence[t] = pose_to_feature(
            state.pose_coords,
            state.pose_point_valid,
            state.pose_visibility,
            body_centers[t],
            body_scales[t],
        )

        face_feat, face_ok = facehead_to_feature(
            state.face_coords,
            state.face_visibility,
            state.face_point_valid,
            body_centers[t],
            body_scales[t],
        )
        face_sequence[t] = face_feat
        facehead_valid[t] = int(face_ok)

    hand_x = normalize_hand_branch(
        hand_sequence,
        stats["hand_mean"],
        stats["hand_std"],
    )
    pose_x = normalize_pose_branch(
        pose_sequence,
        stats["pose_mean"],
        stats["pose_std"],
    )
    face_x = normalize_facehead_branch(
        face_sequence,
        stats["face_mean"],
        stats["face_std"],
    )

    crop_x = face_crops / 255.0
    crop_x = (
        crop_x - stats["crop_mean"]
    ) / stats["crop_std"]
    crop_x[face_crop_valid < 0.5] = 0.0
    crop_x = crop_x.astype(np.float32)

    quality = {
        "observed_any": float((observed.sum(axis=1) > 0).mean()),
        "valid_any": float((final_valid.sum(axis=1) > 0).mean()),
        "left_valid": float(final_valid[:, LEFT].mean()),
        "right_valid": float(final_valid[:, RIGHT].mean()),
        "pose_valid": float(pose_frame_valid.mean()),
        "face_valid": float(facehead_valid.mean()),
        "face_crop_valid": float(face_crop_valid.mean()),
    }

    return (
        hand_x,
        pose_x,
        face_x,
        crop_x,
    ), quality


def assign_candidates_for_static_model(candidates):
    """Match Training A V7: use MediaPipe's raw handedness slots."""
    assigned = {LEFT: None, RIGHT: None}
    scores = {LEFT: -1.0, RIGHT: -1.0}

    for candidate in candidates:
        label = str(candidate.get("handedness_label") or "").lower()
        score = float(candidate.get("handedness_score", 0.0))
        if label == "left":
            side = LEFT
        elif label == "right":
            side = RIGHT
        else:
            continue

        if score > scores[side]:
            assigned[side] = candidate["arr"]
            scores[side] = score

    # Rare fallback when handedness is unavailable.
    unassigned = [c for c in candidates if str(c.get("handedness_label") or "").lower() not in {"left", "right"}]
    for candidate in unassigned:
        free = LEFT if assigned[LEFT] is None else RIGHT
        if assigned[free] is None:
            assigned[free] = candidate["arr"]

    return assigned


def static_v2_hand_feature(hand_arr):
    """Exact 67-D hand feature used by BISINDO Training A V7."""
    if hand_arr is None or not np.isfinite(hand_arr).all():
        return np.zeros(HAND_FEATURES, dtype=np.float32)

    points = np.asarray(hand_arr, dtype=np.float32)
    points = points - points[0:1]

    palm_scale = float(np.linalg.norm(points[9, :2]))
    if palm_scale < 1e-5:
        palm_scale = float(
            np.max(np.linalg.norm(points[:, :2], axis=1))
        )
    palm_scale = max(palm_scale, 1e-3)
    normalized = np.clip(points / palm_scale, -5.0, 5.0)

    palm_span = np.linalg.norm(
        normalized[5, :2] - normalized[17, :2]
    )
    hand_length = np.linalg.norm(normalized[12, :2])
    fingertip_spread = np.mean([
        np.linalg.norm(normalized[4, :2] - normalized[8, :2]),
        np.linalg.norm(normalized[8, :2] - normalized[12, :2]),
        np.linalg.norm(normalized[12, :2] - normalized[16, :2]),
        np.linalg.norm(normalized[16, :2] - normalized[20, :2]),
    ])

    return np.concatenate([
        normalized.reshape(-1),
        np.asarray(
            [palm_span, hand_length, fingertip_spread],
            dtype=np.float32,
        ),
        np.ones(1, dtype=np.float32),
    ]).astype(np.float32)


def build_static_hand134(candidates, feature_mean, feature_std):
    assigned = assign_candidates_for_static_model(candidates)
    left = static_v2_hand_feature(assigned[LEFT])
    right = static_v2_hand_feature(assigned[RIGHT])
    feature = np.concatenate([left, right]).astype(np.float32)
    normalized = ((feature - feature_mean) / feature_std).astype(np.float32)

    detected = int(left[LEFT_PRESENCE_IDX] > 0.5) + int(
        right[LEFT_PRESENCE_IDX] > 0.5
    )
    quality = {
        "observed_any": float(detected > 0),
        "valid_any": float(detected > 0),
        "left_valid": float(left[LEFT_PRESENCE_IDX] > 0.5),
        "right_valid": float(right[LEFT_PRESENCE_IDX] > 0.5),
    }
    return normalized, quality


# ============================================================
# Model inference
# ============================================================

def predict_sequence(
    model,
    model_input,
    device,
):
    if RUNTIME_SPEC.kind == "multimodal":
        hand_x, pose_x, face_x, crop_x = model_input
        tensors = [
            torch.from_numpy(hand_x).unsqueeze(0).to(device, non_blocking=True),
            torch.from_numpy(pose_x).unsqueeze(0).to(device, non_blocking=True),
            torch.from_numpy(face_x).unsqueeze(0).to(device, non_blocking=True),
            torch.from_numpy(crop_x).unsqueeze(0).to(device, non_blocking=True),
        ]
        with torch.inference_mode():
            logits = model(*tensors)
            probs_np = torch.softmax(logits, dim=1)[0].cpu().numpy()

    elif RUNTIME_SPEC.runtime == "torchscript":
        x = torch.from_numpy(
            model_input
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )
        with torch.inference_mode():
            logits = model(x)
            probs_np = torch.softmax(logits, dim=1)[0].cpu().numpy()

    else:
        input_meta = model.get_inputs()[0]
        logits_np = model.run(
            None,
            {input_meta.name: model_input[None, ...].astype(np.float32)},
        )[0][0]
        shifted = logits_np - np.max(logits_np)
        exp_values = np.exp(shifted)
        probs_np = exp_values / np.sum(exp_values)

    probs = torch.from_numpy(
        np.asarray(probs_np, dtype=np.float32)
    )

    top_k = min(2, probs.numel())
    values, indices = torch.topk(probs, k=top_k)

    pred_id = int(indices[0].item())
    confidence = float(values[0].item())
    second = float(values[1].item()) if top_k > 1 else 0.0
    margin = confidence - second

    return pred_id, confidence, margin


# ============================================================
# Indonesian Neural TTS
# ============================================================

class IndonesianTTS:
    def __init__(
        self,
        enabled=True,
        voice="id-ID-ArdiNeural",
        cache_dir=None,
    ):
        self.enabled = bool(enabled)
        self.voice = str(voice)

        if cache_dir is None:
            cache_dir = APP_DIR / ".tts_cache"

        self.cache_dir = Path(
            cache_dir
        )
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Only keep the newest pending utterance.
        # This prevents a backlog such as "Maaf, Maaf, Maaf"
        # from continuing to play after the visual prediction changed.
        self._queue = queue.Queue(
            maxsize=1
        )
        self._stop = threading.Event()

        self._last_enqueued_text = None
        self._last_enqueued_at = 0.0

        self._edge_tts = None
        self._pygame = None
        self._ready = False

        if not self.enabled:
            print("[TTS] OFF")
            return

        try:
            import edge_tts
            import pygame

            self._edge_tts = edge_tts
            self._pygame = pygame

            pygame.mixer.init()

            self._ready = True

            print(
                "[TTS] Indonesian Neural Voice:",
                self.voice,
            )

            self._thread = threading.Thread(
                target=self._worker,
                daemon=True,
            )
            self._thread.start()

        except Exception as exc:
            self._ready = False
            print(
                "[WARN] TTS tidak aktif:",
                exc,
            )
            print(
                "[INFO] Recognition tetap berjalan. "
                "TTS memerlukan edge-tts + pygame."
            )

    @property
    def ready(self):
        return (
            self.enabled
            and self._ready
        )

    def toggle(self):
        self.enabled = (
            not self.enabled
        )
        print(
            "[TTS]",
            "ON" if self.enabled else "OFF",
        )

    def _cache_file(
        self,
        text,
    ):
        key = hashlib.sha1(
            (
                self.voice
                + "|"
                + text
            ).encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        return (
            self.cache_dir
            / f"{key}.mp3"
        )

    async def _generate(
        self,
        text,
        path,
    ):
        communicator = (
            self._edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate="+0%",
                volume="+0%",
                pitch="+0Hz",
            )
        )

        await communicator.save(
            str(path)
        )

    def _play(
        self,
        path,
    ):
        self._pygame.mixer.music.load(
            str(path)
        )
        self._pygame.mixer.music.play()

        while (
            self._pygame.mixer.music
            .get_busy()
        ):
            if self._stop.is_set():
                break

            time.sleep(
                0.02
            )

    def _worker(self):
        while not self._stop.is_set():
            try:
                text = self._queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue

            try:
                path = self._cache_file(
                    text
                )

                if not path.exists():
                    asyncio.run(
                        self._generate(
                            text,
                            path,
                        )
                    )

                if path.exists():
                    self._play(
                        path
                    )

            except Exception as exc:
                print(
                    f"[WARN] TTS gagal '{text}':",
                    exc,
                )

            finally:
                self._queue.task_done()

    def speak(
        self,
        text,
    ):
        if not self.ready:
            return

        text = str(
            text
        ).strip()

        if not text:
            return

        now = time.monotonic()

        # Extra TTS-level duplicate guard.
        # Recognition already has a stronger event gate, but this
        # prevents accidental duplicate queueing as a second safety net.
        if (
            text == self._last_enqueued_text
            and now - self._last_enqueued_at < 1.25
        ):
            return

        self._last_enqueued_text = text
        self._last_enqueued_at = now

        # Drop stale pending speech and keep only the newest one.
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(
                text
            )
        except queue.Full:
            pass

    def close(self):
        self._stop.set()

        try:
            if self._pygame is not None:
                self._pygame.mixer.music.stop()
        except Exception:
            pass


# ============================================================
# Display helpers
# ============================================================

def draw_hand_points(
    frame,
    hand_arr,
    color,
):
    if hand_arr is None:
        return

    if not np.isfinite(
        hand_arr
    ).all():
        return

    h, w = frame.shape[:2]

    for p in hand_arr:
        x = int(
            np.clip(
                p[0],
                0.0,
                1.0,
            )
            * w
        )
        y = int(
            np.clip(
                p[1],
                0.0,
                1.0,
            )
            * h
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            color,
            -1,
            cv2.LINE_AA,
        )


def put_line(
    frame,
    text,
    y,
    scale=0.55,
    thickness=1,
    x=20,
):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_translucent_panel(
    frame,
    x0,
    y0,
    x1,
    y1,
    alpha=0.58,
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1, y1),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )


# ============================================================
# Main realtime
# ============================================================

def run(args):
    device = select_device(
        force_cpu=args.cpu
    )

    model_path = Path(args.model)
    mean_path = Path(args.mean)
    std_path = Path(args.std)
    mapping_path = Path(args.mapping)

    model, runtime_data, labels = load_runtime_files(
        model_path,
        mean_path,
        std_path,
        mapping_path,
        device,
    )

    print("=" * 72)
    print(f"{RUNTIME_SPEC.name} — FAST CONTINUOUS")
    print("=" * 72)
    print("Active model   :", ACTIVE_MODEL_VERSION)
    print("Model folder   :", DEFAULT_MODEL_DIR)
    print(
        "Device         :",
        device if RUNTIME_SPEC.runtime == "torchscript" else "onnxruntime-cpu",
    )
    print("Runtime        :", RUNTIME_SPEC.runtime)
    print("Inference mode :", RUNTIME_SPEC.inference_mode)
    print("Model kind     :", RUNTIME_SPEC.kind)
    if RUNTIME_SPEC.winner_mode is not None:
        print("Winner mode    :", RUNTIME_SPEC.winner_mode)

    if device.type == "cuda":
        print("GPU            :", torch.cuda.get_device_name(0))

    if RUNTIME_SPEC.kind == "multimodal":
        print("Input model    : 4 inputs")
        print("  hand         : 1 x 48 x 134")
        print("  pose         : 1 x 48 x 36")
        print("  facehead     : 1 x 48 x 52")
        print("  facecrop     : 1 x 48 x 48 x 48")
    else:
        print(
            "Input model    :",
            f"{SEQ_LEN} x {FEATURE_DIM}"
            if RUNTIME_SPEC.inference_mode == "sequence"
            else f"{FEATURE_DIM} (single frame)",
        )
    print(
        "Threshold      :",
        args.threshold,
    )
    print(
        "Margin         :",
        args.min_margin,
    )
    print(
        "Vote           :",
        f"{args.vote_hits}/{args.vote_window}",
    )
    print(
        "Infer every    :",
        args.infer_every,
        "frame",
    )
    print(
        "Detector enhance:",
        "ON" if args.detector_enhance else "OFF",
    )
    print(
        "Detector fallback:",
        "ON" if args.detector_fallback else "OFF",
    )
    print(
        "IMPORTANT      :",
        "tidak perlu tangan keluar frame",
    )
    print("=" * 72)

    tts = IndonesianTTS(
        enabled=not args.no_tts,
        voice=args.voice,
        cache_dir=APP_DIR / ".tts_cache",
    )

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    if args.video:
        source = str(
            Path(args.video)
        )
        cap = cv2.VideoCapture(
            source
        )
    else:
        source = int(
            args.camera
        )

        cap = None
        camera_attempts = []
        backends = (
            [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            if os.name == "nt"
            else [cv2.CAP_ANY]
        )
        camera_ids = [source] + [i for i in range(3) if i != source]
        for camera_id in camera_ids:
            for backend in backends:
                candidate_cap = cv2.VideoCapture(camera_id, backend)
                camera_attempts.append((camera_id, backend))
                if candidate_cap.isOpened():
                    ok_probe, _ = candidate_cap.read()
                    if ok_probe:
                        cap = candidate_cap
                        source = camera_id
                        print(f"[CAMERA] Opened index {camera_id}, backend {backend}")
                        break
                candidate_cap.release()
            if cap is not None:
                break

        if cap is None:
            raise RuntimeError(
                "Tidak bisa membuka kamera. Sudah mencoba indeks 0-2 "
                "dengan DirectShow/MSMF/Default. Pastikan kamera terpasang, "
                "izin Windows aktif, dan tidak dipakai aplikasi lain."
            )

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Tidak bisa membuka input: {source}"
        )

    if not args.video:
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            args.width,
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            args.height,
        )
        cap.set(
            cv2.CAP_PROP_FPS,
            args.camera_fps,
        )

    raw_window = deque(
        maxlen=SEQ_LEN
    )

    prediction_history = deque(
        maxlen=args.vote_window
    )

    sentence = []

    prev_left_wrist = None
    prev_right_wrist = None
    prev_face_bbox = None

    frame_id = 0

    current_pred = "-"
    current_conf = 0.0
    current_margin = 0.0
    stable_text = "-"
    accepted_text = "-"

    last_emit_label = None
    last_emit_time = 0.0

    # After a word is emitted, the same class is LOCKED.
    # It is only re-armed after a genuine uncertain/transition period.
    # This avoids "Maaf Maaf Maaf" while one sign is still being held.
    same_label_rearmed = True
    neutral_streak = 0

    quality = {
        "observed_any": 0.0,
        "valid_any": 0.0,
        "left_valid": 0.0,
        "right_valid": 0.0,
    }

    detector_info = {
        "pose": False,
        "primary_hands": 0,
        "fallback_hands": 0,
        "left_now": False,
        "right_now": False,
        "enhanced": False,
        "center_mean": 0.0,
    }
    no_hand_streak = 0

    fps_samples = deque(
        maxlen=30
    )
    last_loop = time.perf_counter()

    WINDOW_NAME = f"BISINDO Realtime — {ACTIVE_MODEL_VERSION}"

    # Keep the OpenCV window attached to the actual camera frame size.
    # WINDOW_AUTOSIZE prevents a maximized/resized window from leaving
    # a large gray unused area around the image.
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE,
    )

    print()
    print("KEYBOARD")
    print(" Q / ESC : keluar")
    print(" C       : clear semua kata")
    print(" X / B   : hapus kata terakhir")
    print(" S       : speak kalimat saat ini")
    print(" T       : toggle auto TTS")
    print(" R       : reset temporal state")
    print()

    static_single_frame = (
        RUNTIME_SPEC.inference_mode == "single_frame"
    )

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.30,
        min_tracking_confidence=0.30,
    ) as pose, mp_hands.Hands(
        static_image_mode=static_single_frame,
        max_num_hands=2,
        model_complexity=(1 if static_single_frame else 0),
        min_detection_confidence=FULL_HAND_DET_CONF,
        min_tracking_confidence=FULL_HAND_TRACK_CONF,
    ) as hands, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=FALLBACK_HAND_DET_CONF,
    ) as fallback_hands, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=RECOVERY_DET_CONF,
    ) as recovery_hands:

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    if args.video:
                        print(
                            "[INFO] Video selesai."
                        )
                        break

                    print(
                        "[WARN] Frame kamera gagal dibaca."
                    )
                    continue

                frame_id += 1

                # Detector input may be photometrically enhanced for dark /
                # backlit webcams. Geometry is unchanged and display remains
                # the original frame.
                detector_frame, enhanced_used, light_stats = (
                    prepare_detector_frame(
                        frame,
                        enabled=args.detector_enhance,
                    )
                )

                rgb = cv2.cvtColor(
                    detector_frame,
                    cv2.COLOR_BGR2RGB,
                )
                rgb.flags.writeable = False

                pose_result = pose.process(
                    rgb
                )
                hand_result = hands.process(
                    rgb
                )

                (
                    body_center,
                    body_scale,
                    left_pose_wrist,
                    right_pose_wrist,
                    pvalid,
                ) = pose_anchor(
                    pose_result
                )

                pose_coords, pose_point_valid, pose_visibility = (
                    extract_selected_pose_raw(pose_result)
                )

                face_coords, face_visibility, face_point_valid = (
                    extract_face_pose_raw(pose_result)
                )

                current_face_bbox = face_bbox_from_pose(
                    face_coords,
                    face_point_valid,
                    frame.shape,
                    body_scale,
                )
                if current_face_bbox is not None:
                    prev_face_bbox = smooth_bbox(
                        current_face_bbox,
                        prev_face_bbox,
                    )
                face_crop, face_crop_ok = crop_face_gray(
                    frame,
                    prev_face_bbox,
                )

                primary_candidates = (
                    collect_full_hand_candidates(
                        hand_result
                    )
                )
                candidates = primary_candidates
                fallback_count = 0

                # A separate static-image detector is only invoked when the
                # tracking detector finds nothing. This is intentionally
                # lower-threshold and helps with first acquisition / low light.
                if (
                    len(candidates) == 0
                    and args.detector_fallback
                ):
                    fallback_result = fallback_hands.process(
                        rgb
                    )
                    fallback_candidates = (
                        collect_full_hand_candidates(
                            fallback_result
                        )
                    )
                    fallback_count = len(
                        fallback_candidates
                    )
                    if fallback_candidates:
                        candidates = fallback_candidates

                detector_info["pose"] = bool(
                    pvalid > 0.5
                )
                detector_info["primary_hands"] = len(
                    primary_candidates
                )
                detector_info["fallback_hands"] = fallback_count
                detector_info["enhanced"] = bool(
                    enhanced_used
                )
                detector_info["center_mean"] = float(
                    light_stats["center_mean"]
                )

                assigned = (
                    assign_candidates_to_body_sides(
                        candidates,
                        left_pose_wrist,
                        right_pose_wrist,
                        prev_left_wrist,
                        prev_right_wrist,
                    )
                )

                tracks = np.full(
                    (
                        NUM_HANDS,
                        NUM_HAND_LANDMARKS,
                        3,
                    ),
                    np.nan,
                    dtype=np.float32,
                )

                observed = np.zeros(
                    NUM_HANDS,
                    dtype=np.uint8,
                )

                for side in [LEFT, RIGHT]:
                    arr = assigned[side]

                    if arr is not None:
                        tracks[side] = arr
                        observed[side] = 1

                # Same ROI recovery as preprocessing V2.
                if (
                    not np.isfinite(
                        tracks[LEFT]
                    ).all()
                    and left_pose_wrist
                    is not None
                ):
                    recovered = (
                        recover_hand_from_roi(
                            frame,
                            left_pose_wrist,
                            body_scale,
                            recovery_hands,
                        )
                    )

                    if recovered is not None:
                        tracks[LEFT] = recovered

                if (
                    not np.isfinite(
                        tracks[RIGHT]
                    ).all()
                    and right_pose_wrist
                    is not None
                ):
                    recovered = (
                        recover_hand_from_roi(
                            frame,
                            right_pose_wrist,
                            body_scale,
                            recovery_hands,
                        )
                    )

                    if recovered is not None:
                        tracks[RIGHT] = recovered

                if np.isfinite(
                    tracks[LEFT]
                ).all():
                    prev_left_wrist = (
                        tracks[
                            LEFT,
                            0,
                        ].copy()
                    )

                if np.isfinite(
                    tracks[RIGHT]
                ).all():
                    prev_right_wrist = (
                        tracks[
                            RIGHT,
                            0,
                        ].copy()
                    )

                detector_info["left_now"] = bool(
                    np.isfinite(tracks[LEFT]).all()
                )
                detector_info["right_now"] = bool(
                    np.isfinite(tracks[RIGHT]).all()
                )

                if (
                    detector_info["left_now"]
                    or detector_info["right_now"]
                ):
                    no_hand_streak = 0
                else:
                    no_hand_streak += 1
                    if no_hand_streak % 30 == 0:
                        print(
                            "[DETECT] no hand | "
                            f"pose={int(detector_info['pose'])} | "
                            f"primary={detector_info['primary_hands']} | "
                            f"fallback={detector_info['fallback_hands']} | "
                            f"center_luma={detector_info['center_mean']:.1f} | "
                            f"enhanced={detector_info['enhanced']}"
                        )

                raw_window.append(
                    FrameState(
                        tracks=tracks,
                        observed=observed,
                        body_center=body_center,
                        body_scale=float(body_scale),
                        pose_valid=int(pvalid > 0.5),
                        pose_coords=pose_coords,
                        pose_point_valid=pose_point_valid,
                        pose_visibility=pose_visibility,
                        face_coords=face_coords,
                        face_visibility=face_visibility,
                        face_point_valid=face_point_valid,
                        face_crop=face_crop,
                        face_crop_valid=int(face_crop_ok),
                    )
                )

                can_infer = (
                    (
                        RUNTIME_SPEC.inference_mode == "single_frame"
                        or len(raw_window) == SEQ_LEN
                    )
                    and frame_id
                    % args.infer_every
                    == 0
                )

                if can_infer:
                    if RUNTIME_SPEC.kind == "multimodal":
                        model_input, quality = build_multimodal_inputs(
                            raw_window,
                            runtime_data,
                        )
                    elif RUNTIME_SPEC.inference_mode == "single_frame":
                        model_input, quality = build_static_hand134(
                            candidates,
                            runtime_data["feature_mean"],
                            runtime_data["feature_std"],
                        )
                    else:
                        model_input, quality = build_hand134_sequence(
                            raw_window,
                            runtime_data["feature_mean"],
                            runtime_data["feature_std"],
                        )

                    (
                        pred_id,
                        conf,
                        margin,
                    ) = predict_sequence(
                        model,
                        model_input,
                        device,
                    )

                    current_pred = labels.get(
                        pred_id,
                        str(pred_id),
                    )
                    current_conf = conf
                    current_margin = margin

                    passed = (
                        conf >= args.threshold
                        and margin
                        >= args.min_margin
                        and quality[
                            "valid_any"
                        ]
                        >= args.min_valid_ratio
                    )

                    if passed:
                        neutral_streak = 0

                        prediction_history.append(
                            (
                                pred_id,
                                conf,
                            )
                        )
                    else:
                        neutral_streak += 1

                        # One uncertain inference should not erase
                        # the rolling raw sequence.
                        prediction_history.append(
                            (
                                -1,
                                0.0,
                            )
                        )

                        # IMPORTANT:
                        # Repeating the SAME word is allowed only after
                        # a short transition/uncertain period.
                        # Hands may remain inside the camera frame.
                        if (
                            neutral_streak
                            >= args.neutral_reset_hits
                        ):
                            same_label_rearmed = True
                            stable_text = "-"

                    valid_votes = [
                        item
                        for item
                        in prediction_history
                        if item[0] >= 0
                    ]

                    stable_id = None
                    stable_count = 0
                    stable_conf = 0.0

                    if valid_votes:
                        counts = Counter(
                            p
                            for p, _
                            in valid_votes
                        )

                        (
                            stable_id,
                            stable_count,
                        ) = counts.most_common(
                            1
                        )[0]

                        stable_conf = float(
                            np.mean(
                                [
                                    c
                                    for p, c
                                    in valid_votes
                                    if p
                                    == stable_id
                                ]
                            )
                        )

                    if (
                        stable_id is not None
                        and stable_count
                        >= args.vote_hits
                        and stable_conf
                        >= args.threshold
                    ):
                        stable_text = labels.get(
                            stable_id,
                            str(stable_id),
                        )

                        now = time.monotonic()

                        # Different class:
                        # may emit immediately after the short change debounce.
                        #
                        # Same class:
                        # NEVER repeats merely because time passed.
                        # It must first be re-armed by a genuine transition
                        # (several uncertain inferences). Hands do NOT need
                        # to leave the frame.
                        is_new_class = (
                            last_emit_label is None
                            or stable_id
                            != last_emit_label
                        )

                        is_rearmed_same_class = (
                            last_emit_label is not None
                            and stable_id
                            == last_emit_label
                            and same_label_rearmed
                        )

                        debounce_ok = (
                            now
                            - last_emit_time
                            >= args.change_cooldown
                        )

                        allowed = (
                            debounce_ok
                            and (
                                is_new_class
                                or is_rearmed_same_class
                            )
                        )

                        if allowed:
                            accepted_text = (
                                stable_text
                            )

                            sentence.append(
                                stable_text
                            )

                            last_emit_label = (
                                stable_id
                            )
                            last_emit_time = now

                            # Lock the currently accepted class.
                            # Holding the same sign cannot emit it again.
                            same_label_rearmed = False
                            neutral_streak = 0

                            # Clear only vote history.
                            # DO NOT clear rolling raw_window.
                            prediction_history.clear()

                            print(
                                f"[SIGN] "
                                f"{stable_text:<15} "
                                f"conf="
                                f"{stable_conf:.3f} "
                                f"margin="
                                f"{margin:.3f}"
                            )

                            if (
                                args.auto_speak
                                and not args.no_tts
                            ):
                                tts.speak(
                                    stable_text
                                )

                # FPS
                now_perf = time.perf_counter()
                dt = max(
                    now_perf - last_loop,
                    1e-9,
                )
                last_loop = now_perf
                fps_samples.append(
                    1.0 / dt
                )
                fps = float(
                    np.mean(
                        fps_samples
                    )
                )

                # Draw on original coordinates.
                vis = frame.copy()

                draw_hand_points(
                    vis,
                    tracks[LEFT],
                    (0, 255, 0),
                )
                draw_hand_points(
                    vis,
                    tracks[RIGHT],
                    (255, 0, 255),
                )

                # Mirror DISPLAY only.
                if not args.no_mirror:
                    vis = cv2.flip(
                        vis,
                        1,
                    )

                # --------------------------------------------------
                # Compact UI + detector diagnostics
                # --------------------------------------------------
                if len(raw_window) < SEQ_LEN:
                    status_text = (
                        f"Menyiapkan buffer "
                        f"{len(raw_window)}/{SEQ_LEN}"
                    )
                    candidate_text = "-"
                elif (
                    quality["valid_any"]
                    < args.min_valid_ratio
                ):
                    status_text = "Tangan belum terbaca"
                    candidate_text = "-"
                elif (
                    current_conf < args.threshold
                    or current_margin < args.min_margin
                ):
                    status_text = "Mendeteksi..."
                    candidate_text = "-"
                else:
                    status_text = "Kandidat stabilisasi"
                    candidate_text = (
                        f"{current_pred} "
                        f"({current_conf:.0%})"
                    )

                text_line = (
                    " ".join(
                        sentence[-6:]
                    )
                    if sentence
                    else "-"
                )

                panel_x0 = 10
                panel_y0 = 10
                panel_x1 = min(
                    vis.shape[1] - 10,
                    720,
                )
                panel_y1 = min(
                    vis.shape[0] - 10,
                    232,
                )

                draw_translucent_panel(
                    vis,
                    panel_x0,
                    panel_y0,
                    panel_x1,
                    panel_y1,
                    alpha=0.60,
                )

                put_line(
                    vis,
                    f"Status: {status_text}",
                    34,
                    0.56,
                    2,
                    20,
                )

                put_line(
                    vis,
                    f"Kandidat: {candidate_text}",
                    60,
                    0.46,
                    1,
                    20,
                )

                put_line(
                    vis,
                    f"Terdeteksi: {accepted_text}",
                    86,
                    0.56,
                    2,
                    20,
                )

                put_line(
                    vis,
                    f"Teks: {text_line}",
                    112,
                    0.46,
                    1,
                    20,
                )

                put_line(
                    vis,
                    (
                        f"Model {ACTIVE_MODEL_VERSION} | "
                        f"{RUNTIME_SPEC.runtime}/{RUNTIME_SPEC.inference_mode} | "
                        f"Buffer {len(raw_window)}/{SEQ_LEN}"
                    ),
                    138,
                    0.41,
                    1,
                    20,
                )

                put_line(
                    vis,
                    (
                        f"Pose {'YES' if detector_info['pose'] else 'NO'} | "
                        f"Hand L {'YES' if detector_info['left_now'] else 'NO'} "
                        f"R {'YES' if detector_info['right_now'] else 'NO'} | "
                        f"P/F {detector_info['primary_hands']}/"
                        f"{detector_info['fallback_hands']}"
                    ),
                    164,
                    0.41,
                    1,
                    20,
                )

                put_line(
                    vis,
                    (
                        f"Valid {quality['valid_any']:.0%} "
                        f"L {quality['left_valid']:.0%} "
                        f"R {quality['right_valid']:.0%} | "
                        f"Light {detector_info['center_mean']:.0f} "
                        f"{'ENH' if detector_info['enhanced'] else 'RAW'}"
                    ),
                    190,
                    0.41,
                    1,
                    20,
                )

                put_line(
                    vis,
                    (
                        f"Conf {current_conf:.2f} | "
                        f"Margin {current_margin:.2f} | "
                        f"FPS {fps:.1f} | "
                        f"TTS {'ON' if tts.enabled else 'OFF'}"
                    ),
                    216,
                    0.41,
                    1,
                    20,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    vis,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    27,
                    ord("q"),
                    ord("Q"),
                ):
                    break

                if key in (
                    ord("c"),
                    ord("C"),
                ):
                    sentence.clear()
                    accepted_text = "-"
                    print(
                        "[INFO] Text cleared."
                    )

                if key in (
                    ord("x"),
                    ord("X"),
                    ord("b"),
                    ord("B"),
                ):
                    if sentence:
                        removed = (
                            sentence.pop()
                        )
                        print(
                            "[INFO] Removed:",
                            removed,
                        )

                if key in (
                    ord("s"),
                    ord("S"),
                ):
                    if sentence:
                        tts.speak(
                            " ".join(
                                sentence
                            )
                        )

                if key in (
                    ord("t"),
                    ord("T"),
                ):
                    tts.toggle()

                if key in (
                    ord("r"),
                    ord("R"),
                ):
                    raw_window.clear()
                    prediction_history.clear()
                    prev_left_wrist = None
                    prev_right_wrist = None
                    prev_face_bbox = None
                    last_emit_label = None
                    last_emit_time = 0.0
                    same_label_rearmed = True
                    neutral_streak = 0
                    stable_text = "-"
                    accepted_text = "-"
                    current_pred = "-"
                    current_conf = 0.0
                    current_margin = 0.0

                    print(
                        "[INFO] Temporal state reset."
                    )

        finally:
            cap.release()
            cv2.destroyAllWindows()
            tts.close()

    print()
    print("FINAL TEXT")
    print(
        " ".join(sentence)
        if sentence
        else "-"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "WL-BISINDO Hand134 Transformer V4 "
            "local realtime."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            DEFAULT_MODEL_PATH
        ),
    )
    parser.add_argument(
        "--mean",
        type=str,
        default=str(
            DEFAULT_MEAN_PATH
        ),
    )
    parser.add_argument(
        "--std",
        type=str,
        default=str(
            DEFAULT_STD_PATH
        ),
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=str(
            DEFAULT_MAPPING_PATH
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--vote-window",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--vote-hits",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--change-cooldown",
        type=float,
        default=0.20,
        help=(
            "Minimum detik antara kata berbeda. "
            "Tidak membutuhkan tangan keluar."
        ),
    )
    parser.add_argument(
        "--neutral-reset-hits",
        type=int,
        default=3,
        help=(
            "Jumlah inference transisi/uncertain untuk "
            "mengizinkan kelas yang sama diucapkan lagi. "
            "Tidak perlu tangan keluar frame."
        ),
    )

    parser.add_argument(
        "--voice",
        type=str,
        default="id-ID-ArdiNeural",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
    )
    parser.add_argument(
        "--auto-speak",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    parser.add_argument(
        "--detector-enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Aktifkan CLAHE/gamma detector-only untuk webcam gelap/backlight."
        ),
    )
    parser.add_argument(
        "--detector-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Gunakan static low-threshold MediaPipe Hands bila tracker utama "
            "tidak menemukan tangan."
        ),
    )

    parser.add_argument(
        "--no-mirror",
        action="store_true",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=960,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.vote_hits > args.vote_window:
        raise ValueError(
            "--vote-hits tidak boleh lebih besar "
            "dari --vote-window"
        )

    if args.infer_every < 1:
        raise ValueError(
            "--infer-every minimal 1"
        )

    run(args)
