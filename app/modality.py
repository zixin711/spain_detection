"""光源判别：判断一张图是白光照还是 395nm 紫外照（设计 3.1 节）。

两个域的差别极大，规则版（统计量 + 线性打分）就能做到很高准确率，且零训练成本、完全可解释。
数据量上来后可以训练 MobileNetV3-Small 替换 `classify`，接口保持不变。
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import cv2
import numpy as np

from .config import grading_config

WHITE = "white"
UV = "uv"
UNKNOWN = "unknown"


@dataclass
class ModalityResult:
    modality: str
    confidence: float
    score: float
    features: Dict[str, float]
    source: str = "auto"

    def as_dict(self) -> Dict:
        return asdict(self)


def extract_features(bgr: np.ndarray) -> Dict[str, float]:
    """提取用于判别光源的统计量。"""
    small = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
    b, g, r = (small[:, :, i].astype(np.float32) for i in range(3))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0          # OpenCV H 是 0~179，换算成 0~360 度
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    # 蓝紫色相集中度：色相落在 220~300 度且有一定饱和度的像素占比
    violet = ((hue >= 220) & (hue <= 300) & (sat > 0.15)).mean()
    # 黄绿荧光像素占比（紫外图特有）
    fluor = ((hue >= 35) & (hue <= 95) & (sat > 0.25) & (val > 0.2)).mean()
    # 灰轴像素占比：R≈G≈B，白光图里很多
    mx = np.max(small.astype(np.float32), axis=2)
    mn = np.min(small.astype(np.float32), axis=2)
    gray_like = ((mx - mn) < 22).mean()

    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    p = hist / max(hist.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return {
        "b_minus_r_mean": float((b - r).mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
        "violet_ratio": float(violet),
        "fluor_ratio": float(fluor),
        "gray_like_ratio": float(gray_like),
        "hue_entropy": entropy,
        "saturation_mean": float(sat.mean()),
    }


@functools.lru_cache(maxsize=1)
def _learned_model() -> Optional[Dict]:
    """scripts/train_modality_clf.py --sklearn --write 产出的逻辑回归权重，存在则优先使用。"""
    from .config import MODEL_DIR

    path = MODEL_DIR / "modality_clf.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def classify(bgr: np.ndarray, hint: Optional[str] = None) -> ModalityResult:
    """返回光源判别结果。hint 由用户在前端指定时优先采用。"""
    feats = extract_features(bgr)
    if hint in (WHITE, UV):
        return ModalityResult(hint, 1.0, 1.0 if hint == UV else -1.0, feats, source="user_hint")

    cfg = grading_config()["modality_classifier"]

    learned = _learned_model()
    if learned:
        x = np.array([feats[k] for k in learned["features"]], dtype=np.float64)
        z = (x - np.array(learned["mean"])) / np.array(learned["scale"])
        logit = float(np.dot(z, np.array(learned["coef"])) + learned["intercept"])
        prob_uv = 1.0 / (1.0 + math.exp(-logit))
        score = float(prob_uv * 2.0 - 1.0)
        if abs(score) < cfg["confident_margin"]:
            return ModalityResult(UNKNOWN, round(0.5 + abs(score) / 2, 4), round(score, 4), feats, source="learned")
        return ModalityResult(UV if score > 0 else WHITE,
                              round(max(prob_uv, 1 - prob_uv), 4), round(score, 4), feats, source="learned")

    # 每一项都归一化到 [-1, 1]，正值支持 uv，负值支持 white
    z_br = np.tanh((feats["b_minus_r_mean"] - cfg["b_minus_r_mean_min"]) / 20.0)
    z_violet = np.tanh((feats["violet_ratio"] - cfg["hue_concentration_min"]) / 0.25)
    z_entropy = np.tanh((cfg["color_entropy_max"] - feats["hue_entropy"]) / 0.8)
    z_gray = np.tanh((0.35 - feats["gray_like_ratio"]) / 0.25)

    # 亮度只是弱证据：一张拍得暗的白光照并不是紫外照。
    # 只有在同时存在蓝紫色偏（B>R 或紫色像素占比够高）时，"暗"才算支持紫外的证据。
    violet_evidence = feats["b_minus_r_mean"] > 0 or feats["violet_ratio"] > 0.2
    z_val = np.tanh((cfg["value_mean_max"] - feats["value_mean"]) / 40.0) if violet_evidence else -0.2

    score = float(0.50 * z_br + 0.25 * z_violet + 0.10 * z_entropy + 0.10 * z_gray + 0.05 * z_val)
    margin = cfg["confident_margin"]

    if abs(score) < margin:
        modality = UNKNOWN
    else:
        modality = UV if score > 0 else WHITE
    confidence = float(min(1.0, abs(score) / max(margin, 1e-6) * 0.5 + 0.5)) if modality != UNKNOWN else float(0.5 + abs(score))
    return ModalityResult(modality, round(confidence, 4), round(score, 4), feats)


def group_by_modality(results) -> Dict[str, list]:
    groups: Dict[str, list] = {WHITE: [], UV: [], UNKNOWN: []}
    for idx, res in enumerate(results):
        groups.setdefault(res.modality, []).append(idx)
    return groups
