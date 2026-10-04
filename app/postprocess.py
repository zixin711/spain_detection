"""掩膜后处理与实例特征提取（设计 4 章第 5、6 步）。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import class_table, grading_config


def _polygon(mask: np.ndarray, max_points: int = 80) -> List[List[int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) > max_points:
        step = max(1, len(approx) // max_points)
        approx = approx[::step]
    return [[int(x), int(y)] for x, y in approx]


def _position_hint(bbox: List[int], shape: Tuple[int, int]) -> str:
    h, w = shape
    cx = (bbox[0] + bbox[2]) / 2.0 / w
    cy = (bbox[1] + bbox[3]) / 2.0 / h
    row = "上" if cy < 0.34 else ("中" if cy < 0.67 else "下")
    col = "左" if cx < 0.34 else ("中" if cx < 0.67 else "右")
    if row == "中" and col == "中":
        return "画面中央"
    return f"画面{row}{col}区域"


def instance_features(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """逐实例特征：颜色、致密度、纹理、厚度代理量。"""
    idx = mask.astype(bool)
    area = int(idx.sum())
    if area == 0:
        return {}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    val = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    hue = hsv[:, :, 0].astype(np.float32) * 2.0

    dark_ratio = float((val[idx] < 80).mean())
    sat_mean = float(sat[idx].mean())
    hue_median = float(np.median(hue[idx]))
    val_mean = float(val[idx].mean() / 255.0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt))) if cnt is not None else float(area)
    fill_ratio = float(area / max(hull_area, 1.0))

    # 厚度代理量（设计 3.7）
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    shadow_grad = float(np.clip(grad[idx].mean() / 60.0, 0.0, 1.0))
    contrast_var = float(np.clip(gray[idx].std() / 60.0, 0.0, 1.0))
    highlight_loss = float(1.0 - np.clip((val[idx] > 200).mean() * 6.0, 0.0, 1.0))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture_energy = float(np.clip(np.abs(lap[idx]).mean() / 30.0, 0.0, 1.0))

    return {
        "dark_ratio": round(dark_ratio, 4),
        "sat_mean": round(sat_mean, 4),
        "val_mean": round(val_mean, 4),
        "hue_median": round(hue_median, 1),
        "fill_ratio": round(fill_ratio, 4),
        "shadow_grad": round(shadow_grad, 4),
        "contrast_var": round(contrast_var, 4),
        "highlight_loss": round(highlight_loss, 4),
        "texture_energy": round(texture_energy, 4),
    }


def density_score(features: Dict[str, float], area_ratio: float) -> Tuple[str, float]:
    """密度分档 H/M/L（设计 2.2 / grading.yaml density）。"""
    cfg = grading_config()["density"]
    w = cfg["weights"]
    score = (w["dark_ratio"] * features.get("dark_ratio", 0.0)
             + w["saturation"] * features.get("sat_mean", 0.0)
             + w["fill_ratio"] * features.get("fill_ratio", 0.0)
             + w["area"] * float(np.clip(area_ratio * 12.0, 0.0, 1.0)))
    score = float(np.clip(score, 0.0, 1.0))
    if score >= cfg["high_min"]:
        return "H", round(score, 4)
    if score >= cfg["medium_min"]:
        return "M", round(score, 4)
    return "L", round(score, 4)


def thickness_score(features: Dict[str, float]) -> Tuple[str, float]:
    """厚度分与深度档位（视觉估计，非实测）。"""
    cfg = grading_config()["depth"]
    fw = cfg["feature_weights"]
    score = (fw["shadow_grad"] * features.get("shadow_grad", 0.0)
             + fw["contrast_var"] * features.get("contrast_var", 0.0)
             + fw["highlight_loss"] * features.get("highlight_loss", 0.0)
             + fw["texture_energy"] * features.get("texture_energy", 0.0))
    score = float(np.clip(score, 0.0, 1.0))
    band = cfg["bands"][-1]["name"]
    for b in cfg["bands"]:
        if score < b["max_score"]:
            band = b["name"]
            break
    return band, round(score, 4)


def merge_nearby(detections, dilate_px: int = 5):
    """把彼此非常靠近的同类实例合并，避免一片霉斑被切成十几块。"""
    if len(detections) < 2:
        return detections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    merged: List[Any] = []
    used = set()
    for i, (m_i, c_i, s_i) in enumerate(detections):
        if i in used:
            continue
        acc, score, n = m_i.copy(), s_i, 1
        di = cv2.dilate(m_i, kernel)
        for j in range(i + 1, len(detections)):
            if j in used:
                continue
            m_j, c_j, s_j = detections[j]
            if c_j != c_i:
                continue
            if int((di & m_j).sum()) > 0:
                acc = ((acc | m_j) > 0).astype(np.uint8)
                score += s_j
                n += 1
                used.add(j)
        merged.append((acc, c_i, round(score / n, 4)))
    return merged


def build_instances(bgr: np.ndarray, detections, wall_area: int, modality: str,
                    source_index: int, start_id: int = 1,
                    px_per_cm: float = None) -> List[Dict[str, Any]]:
    """把检测结果转成带完整属性的实例列表。"""
    cfg = grading_config()["preprocess"]
    table = class_table(modality)
    h, w = bgr.shape[:2]
    instances: List[Dict[str, Any]] = []
    next_id = start_id

    # 紫外下同一片菌落常被拆成几十个碎块，合并半径给大一些
    detections = merge_nearby(detections, dilate_px=14 if modality == "uv" else 5)

    from .detectors.baseline import attenuation

    atten_map = attenuation(bgr)

    for mask, cls_name, conf in detections:
        mask = mask.astype(np.uint8)
        area = int(mask.sum())
        min_area = cfg["min_instance_area_px"] * (6 if modality == "uv" else 1)
        if area < min_area:
            continue
        ys, xs = np.where(mask > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        feats = instance_features(bgr, mask)
        area_ratio = area / max(wall_area, 1)
        meta = table.get(cls_name, {})

        if modality == "uv":
            group = "fluor_negative" if cls_name == "fluor_artifact" else "fluor"
        else:
            group = meta.get("group", "mold")

        inst: Dict[str, Any] = {
            "id": next_id,
            "source_index": source_index,
            "modality": modality,
            "cls": cls_name,
            "cls_cn": meta.get("cn", cls_name),
            "group": group,
            "conf": float(conf),
            "bbox": bbox,
            "polygon": _polygon(mask),
            "area_px": area,
            "area_ratio": round(area_ratio, 6),
            "area_cm2": round(area / (px_per_cm ** 2), 2) if px_per_cm else None,
            "position_hint": _position_hint(bbox, (h, w)),
            "features": feats,
            "_mask": mask,
        }

        if modality == "uv":
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            idx = mask.astype(bool)
            resp = float(((hsv[:, :, 2][idx] / 255.0) * (hsv[:, :, 1][idx] / 255.0)).mean())
            inst["uv_response"] = round(resp, 4)
            # 相对墙面照明场的衰减深度：霉菌在 395nm 下淬灭墙面荧光，越严重越暗
            inst["uv_attenuation"] = round(float(np.clip(1.0 - atten_map[idx].mean(), -1.0, 1.0)), 4)
        else:
            inst["attenuation"] = round(
                float(np.clip(1.0 - atten_map[mask.astype(bool)].mean(), -1.0, 1.0)), 4)
            dens, dscore = density_score(feats, area_ratio)
            band, tscore = thickness_score(feats)
            inst.update({"density": dens, "density_score": dscore,
                         "depth_band": band, "thickness_score": tscore})

        instances.append(inst)
        next_id += 1

    return instances
