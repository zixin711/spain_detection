"""三证据通道打分：H 荧光活性 / M 覆盖 / L 深度厚度（设计 2.2、3.3、3.5）。

缺哪条通道就置 None，由 grading 层按 modality_matrix 重归一化权重。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import grading_config

MOLD_GROUPS = {"mold", "damage"}
FLUOR_POSITIVE = {"fluor_point", "fluor_local", "fluor_band", "fluor_saturated", "fluor_chaotic"}


def fluorescence_baseline(bgr: np.ndarray, wall_mask: np.ndarray, fluor_mask: np.ndarray) -> float:
    """图内非荧光墙面区域的中位响应，用于消除手电亮度与距离差异（设计 3.5）。"""
    cfg = grading_config()["fluorescence"]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    resp = (hsv[:, :, 2].astype(np.float32) / 255.0) * (hsv[:, :, 1].astype(np.float32) / 255.0)
    ref = (wall_mask > 0) & (fluor_mask == 0)
    if ref.sum() < 100:
        return 0.0
    return float(np.percentile(resp[ref], cfg["baseline_percentile"]))


def compute_H(uv_images: List[Tuple[np.ndarray, np.ndarray]], uv_instances: List[Dict[str, Any]]) -> Optional[float]:
    """荧光活性分。

    ⚠️ 实测发现：395nm 下墙面霉菌通常**不发黄绿荧光**，而是比墙面更暗
    （墙面涂料的荧光增白剂发光，霉菌吸收/淬灭它），菌落前沿呈亮白蕾丝状。
    所以 H 由两部分构成：异常区覆盖率 + 相对墙面照明场的平均衰减深度。
    两者都是绝对量，不随画面里霉的多少自归一化，因此能区分 L1~L5。
    """
    if not uv_images:
        return None
    cfg = grading_config()["fluorescence"]
    cov_ref = float(cfg.get("coverage_reference", 0.60))
    att_ref = float(cfg.get("attenuation_reference", 0.40))

    best = 0.0
    for src_idx, (bgr, wall_mask) in enumerate(uv_images):
        insts = [i for i in uv_instances
                 if i["source_index"] == src_idx and i["cls"] in FLUOR_POSITIVE]
        if not insts:
            continue
        coverage = float(sum(i["area_ratio"] for i in insts))
        areas = np.array([i["area_px"] for i in insts], dtype=np.float64)
        attens = np.array([i.get("uv_attenuation") or 0.0 for i in insts], dtype=np.float64)
        depth = float((attens * areas).sum() / max(areas.sum(), 1.0))

        score = 0.65 * min(coverage / cov_ref, 1.0) + 0.35 * min(max(depth, 0.0) / att_ref, 1.0)
        best = max(best, float(np.clip(score, 0.0, 1.0)))

    # 有紫外图但没有异常区 → H=0（这是有效证据，不是缺失）
    return round(best, 4)


def compute_M(instances: List[Dict[str, Any]], modality_key: str) -> Optional[float]:
    """覆盖分：霉变区面积占比，按密度加权。白光优先；仅紫外时用荧光覆盖代理。"""
    if modality_key == "uv":
        insts = [i for i in instances if i["cls"] in FLUOR_POSITIVE]
        if not insts:
            return 0.0
        ratio = sum(i["area_ratio"] for i in insts)
        return round(float(np.clip(ratio, 0.0, 1.0)), 4)

    insts = [i for i in instances if i["group"] in MOLD_GROUPS]
    if not insts:
        return 0.0
    weighted = 0.0
    for i in insts:
        w = {"H": 1.0, "M": 0.75, "L": 0.5}.get(i.get("density") or "M", 0.75)
        weighted += i["area_ratio"] * w
    return round(float(np.clip(weighted, 0.0, 1.0)), 4)


def compute_L(instances: List[Dict[str, Any]], has_white: bool) -> Optional[float]:
    """深度厚度分。紫外单模态下几乎没有深度线索，返回 None。"""
    if not has_white:
        return None
    insts = [i for i in instances if i["group"] in MOLD_GROUPS and i.get("thickness_score") is not None]
    if not insts:
        return 0.0
    # 面积加权，大块厚霉比小点更能代表整体厚度
    tot_area = sum(i["area_px"] for i in insts) or 1
    score = sum(i["thickness_score"] * i["area_px"] for i in insts) / tot_area
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def modality_key(has_white: bool, has_uv: bool) -> str:
    if has_white and has_uv:
        return "both"
    if has_uv:
        return "uv"
    return "white"


def compute_channels(instances, uv_images, has_white: bool, has_uv: bool):
    """返回 (channels, weights, key)。channels 中 None 表示该通道本次无证据。"""
    key = modality_key(has_white, has_uv)
    matrix = grading_config()["modality_matrix"][key]

    uv_instances = [i for i in instances if i["modality"] == "uv"]
    white_instances = [i for i in instances if i["modality"] == "white"]

    H = compute_H(uv_images, uv_instances) if has_uv else None
    M = compute_M(white_instances if has_white else uv_instances, "white" if has_white else "uv")
    L = compute_L(white_instances, has_white)

    channels = {"H": H, "M": M, "L": L}
    weights = dict(matrix["weights"])
    return channels, weights, key


def i_mold(channels: Dict[str, Optional[float]], weights: Dict[str, float]) -> float:
    total = 0.0
    for name, w in weights.items():
        v = channels.get(name)
        if v is None or w <= 0:
            continue
        total += w * float(v)
    return round(float(np.clip(total, 0.0, 1.0)), 4)
