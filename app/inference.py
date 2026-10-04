"""推理后端封装：有权重走 YOLO，没权重自动降级为传统 CV 基线。

权重查找顺序（按 GPU 架构选 TensorRT 引擎，见设计 7.4）：
    models/seg_{white,uv}_sm{arch}.engine   TensorRT，最快，不跨架构通用
    models/seg_{white,uv}.onnx              ONNX Runtime，换卡/CI 兜底
    models/seg_{white,uv}.pt                PyTorch 权重
    → 都没有则用 detectors/baseline.py
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import MODEL_DIR, class_by_index, grading_config
from .detectors import baseline

Detection = Tuple[np.ndarray, str, float]

_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}
_BACKENDS: Dict[str, str] = {}


def _gpu_arch() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm{major}{minor}"
    except Exception:
        pass
    return None


def _weight_path(modality: str) -> Optional[Path]:
    stem = f"seg_{modality}"
    arch = _gpu_arch()
    candidates = []
    if arch:
        candidates.append(MODEL_DIR / f"{stem}_{arch}.engine")
    candidates += [MODEL_DIR / f"{stem}.engine", MODEL_DIR / f"{stem}.onnx", MODEL_DIR / f"{stem}.pt"]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load(modality: str):
    """惰性加载模型。失败不抛异常，回落到基线。"""
    if modality in _MODELS:
        return _MODELS[modality]

    if os.environ.get("MOLD_FORCE_BASELINE") == "1":
        _MODELS[modality] = None
        _BACKENDS[modality] = "baseline (forced)"
        return None

    path = _weight_path(modality)
    if path is None:
        _MODELS[modality] = None
        _BACKENDS[modality] = "baseline (no weights)"
        return None

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        _MODELS[modality] = model
        _BACKENDS[modality] = f"ultralytics:{path.name}"
        return model
    except Exception as exc:  # pragma: no cover
        _MODELS[modality] = None
        _BACKENDS[modality] = f"baseline (load failed: {type(exc).__name__})"
        return None


def backend_name(modality: str) -> str:
    _load(modality)
    return _BACKENDS.get(modality, "baseline")


def _yolo_to_detections(result, shape: Tuple[int, int], modality: str, wall_mask: np.ndarray) -> List[Detection]:
    h, w = shape
    out: List[Detection] = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_idx = int(boxes.cls[i])
        meta = class_by_index(modality, cls_idx)

        if masks is not None and masks.data is not None and i < len(masks.data):
            m = masks.data[i].cpu().numpy().astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:  # 只有框没有掩膜时退化成矩形掩膜
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            m = np.zeros((h, w), np.uint8)
            m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1

        m = (m & (wall_mask > 0).astype(np.uint8)).astype(np.uint8)
        if m.sum() == 0:
            continue
        out.append((m, meta["name"], round(conf, 4)))
    return out


def run(bgr: np.ndarray, wall_mask: np.ndarray, modality: str) -> List[Detection]:
    """对单张图做检测，返回统一格式的检测结果。"""
    modality = "uv" if modality == "uv" else "white"
    model = _load(modality)
    if model is None:
        return baseline.detect(bgr, wall_mask, modality)

    cfg = grading_config()["preprocess"]
    with _LOCK:
        results = model.predict(
            source=bgr,
            imgsz=cfg["infer_size"],
            conf=0.25,
            iou=0.6,
            retina_masks=True,
            verbose=False,
        )
    if not results:
        return []
    return _yolo_to_detections(results[0], bgr.shape[:2], modality, wall_mask)
