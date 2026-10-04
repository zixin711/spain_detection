"""传统 CV 基线检测器（M0 用，无需权重即可跑通全链路）。

核心方法：**光照场归一化后的绝对衰减**（illumination-corrected attenuation）

  1. 用灰度形态学闭运算估计"干净墙面"的照明场 L(x,y) —— 闭运算会把周围明亮的墙面值
     填进暗的霉斑区域，得到"如果这里没有霉，应该有多亮"。
  2. R = V / L，干净墙面 R≈1，霉斑 R<1。R 是绝对量，不随画面里霉的多少漂移。
  3. 霉斑 = R < 阈值。覆盖率与平均衰减深度都可以直接和分级阈值比较。

为什么不用"相对局部背景的 z-score"：z-score 的分母（标准差）本身随霉量增长，
L1 和 L5 会被归一化成差不多的数值，实测在 5 组配对样张上完全无法区分等级。
换成绝对衰减后，L1→L5 的覆盖率与衰减深度都严格单调。

紫外通道尤其重要：**395nm 下墙面霉菌通常不发黄绿荧光，而是比墙面更暗**
（墙面涂料里的荧光增白剂发光，霉菌吸收/淬灭），菌落边缘则呈亮白蕾丝状。
所以紫外检测是"双向偏离"：暗斑（菌落本体）+ 亮丝（菌落前沿）。
少数基材上确实会出现黄绿发射型荧光，作为补充通道保留。

有训练好的 YOLO 权重时由 app/inference.py 自动切换到模型推理，本模块作为兜底。
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Detection = Tuple[np.ndarray, str, float]   # (mask uint8 0/1, class_name, confidence)

_MIN_AREA_FRAC = 4e-5
_ILLUM_SMALL = 256
_ILLUM_CLOSE_FRAC = 0.22


# --------------------------------------------------------------------------- #
def illumination_field(channel: np.ndarray) -> np.ndarray:
    """估计"干净墙面"的照明场：灰度闭运算把亮的墙面值填进暗的霉斑区。

    在缩略图上做，再插值回原尺寸——核要比最大的霉斑还大，全分辨率下太慢。
    """
    h, w = channel.shape[:2]
    if w >= h:
        size = (_ILLUM_SMALL, max(1, int(_ILLUM_SMALL * h / w)))
    else:
        size = (max(1, int(_ILLUM_SMALL * w / h)), _ILLUM_SMALL)
    small = cv2.resize(channel, size, interpolation=cv2.INTER_AREA)

    k = max(9, int(max(small.shape) * _ILLUM_CLOSE_FRAC) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    closed = cv2.GaussianBlur(closed, (31, 31), 0)
    return cv2.resize(closed, (w, h), interpolation=cv2.INTER_CUBIC)


def attenuation(bgr: np.ndarray) -> np.ndarray:
    """返回 R = V / 照明场。干净墙面 ≈ 1.0，霉斑 < 1.0，亮丝 > 1.0。"""
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    return V / np.maximum(illumination_field(V), 1.0)


def _clean(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,) * 2))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k,) * 2))


def _fill_holes(comp: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return comp
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled


def _components(mask: np.ndarray, min_area: int, fill: bool = True) -> List[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        out.append(_fill_holes(comp) if fill else comp)
    return out


def _is_line_like(comp: np.ndarray) -> bool:
    """瓷砖缝隙霉：细长、贴近水平或垂直。"""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    w, h = rect[1]
    if min(w, h) < 1e-3:
        return False
    elong = max(w, h) / max(min(w, h), 1e-3)
    angle = abs(rect[2] % 90)
    return elong > 5.0 and (angle < 15 or angle > 75)


def _granularity(gray: np.ndarray, comp: np.ndarray) -> float:
    """霉斑是颗粒状（高频能量高），水渍是平滑晕染（低）。"""
    idx = comp.astype(bool)
    if idx.sum() < 20:
        return 0.0
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))[idx].mean())


# --------------------------------------------------------------------------- #
def detect_white(bgr: np.ndarray, wall_mask: np.ndarray, thr: float = 0.90) -> List[Detection]:
    """白光图：照明归一化后取衰减区，再按颜色与纹理分表型。"""
    h, w = bgr.shape[:2]
    min_area = max(40, int(_MIN_AREA_FRAC * h * w))

    R = attenuation(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    dark = ((R < thr) & (wall_mask > 0)).astype(np.uint8)
    # 明显偏色（绿/粉）即使不暗也算候选
    colored = ((((hue > 60) & (hue < 170) & (sat > 0.28)) |
                (((hue > 300) | (hue < 25)) & (sat > 0.35))) & (wall_mask > 0)).astype(np.uint8)

    candidate = _clean(((dark | colored) > 0).astype(np.uint8))

    detections: List[Detection] = []
    for comp in _components(candidate, min_area):
        idx = comp.astype(bool)
        area = int(idx.sum())
        m_hue = float(np.median(hue[idx]))
        m_sat = float(np.median(sat[idx]))
        m_val = float(np.median(val[idx]))
        atten = float(1.0 - R[idx].mean())          # 衰减深度，0~1
        gran = _granularity(gray, comp)

        edge = cv2.Canny((comp * 255).astype(np.uint8), 50, 150)
        compactness = float((edge > 0).sum()) / max(np.sqrt(area), 1.0)

        conf = float(np.clip(0.35 + atten * 1.2 + gran / 60.0, 0.30, 0.94))

        # 分类顺序：先排掉三个负类的典型特征，剩下的默认按霉菌处理。
        # 基线阶段宁可高估不可漏判——把疑似判成污渍会直接从面积统计里消失，
        # 而误把污渍算成霉只是抬高等级，后者由训练好的模型来纠正。
        if _is_line_like(comp):
            cls = "grout_mold"
        elif gran < 2.0 and compactness < 4.0 and atten < 0.15 and m_sat < 0.30:
            cls = "damp_stain"                       # 平滑晕染、浅衰减 → 水渍/返潮
            conf = float(np.clip(conf * 0.9, 0.3, 0.85))
        elif m_val > 0.85 and m_sat < 0.10 and gran < 2.5:
            cls = "efflorescence"                    # 很亮、无色、无颗粒感 → 返碱结晶
        elif m_sat > 0.35 and not (60 <= m_hue <= 170) and not (m_hue >= 300 or m_hue <= 25):
            cls = "dirt_stain"                       # 明显带色但不是霉的色系 → 污渍
            conf = float(np.clip(conf * 0.85, 0.3, 0.8))
        elif 60 <= m_hue <= 170 and m_sat > 0.22:
            cls = "mold_green"
        elif (m_hue >= 300 or m_hue <= 25) and m_sat > 0.28 and m_val > 0.35:
            cls = "mold_pink"
        elif m_val > 0.72 and gran >= 2.5:
            cls = "mold_white"                       # 浅色绒毛状 / 初期灰霉
        else:
            cls = "mold_black"

        detections.append((comp, cls, round(conf, 4)))
    return detections


def detect_uv(bgr: np.ndarray, wall_mask: np.ndarray,
              dark_thr: float = 0.88, lace_thr: float = 1.05) -> List[Detection]:
    """紫外图：暗斑（淬灭/吸收）+ 亮丝（菌落前沿）+ 可选黄绿发射。"""
    from ..config import grading_config

    cfg = grading_config()["fluorescence"]
    h, w = bgr.shape[:2]
    min_area = max(24, int(_MIN_AREA_FRAC * h * w))
    wall_area = max(int(wall_mask.sum()), 1)

    R = attenuation(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = V / 255.0

    # ① 菌落本体：比墙面暗
    dark = ((R < dark_thr) & (wall_mask > 0)).astype(np.uint8)

    # ② 菌落前沿：亮 + 高频纹理强（蕾丝状细丝）
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(V, (3, 3), 0), cv2.CV_32F, ksize=3))
    tex = cv2.GaussianBlur(lap, (15, 15), 0)
    tex_n = tex / (np.percentile(tex, 95) + 1e-6)
    lace = ((R > lace_thr) & (tex_n > 0.55) & (wall_mask > 0)).astype(np.uint8)

    # ③ 发射型黄绿荧光（部分基材上会出现，作为补充通道）
    emission = ((hue >= cfg["hue_min_deg"]) & (hue <= cfg["hue_max_deg"]) &
                (sat >= cfg["sat_min"]) & (val >= cfg["val_min"]) & (wall_mask > 0)).astype(np.uint8)

    anomaly = _clean(((dark | lace | emission) > 0).astype(np.uint8), 3, 7)

    # ④ 荧光干扰：亮、平滑、低纹理（荧光增白剂、塑料件、纸张），排除在异常之外
    artifact = ((R > 1.10) & (tex_n < 0.25) & (wall_mask > 0)).astype(np.uint8)
    artifact = _clean(artifact, 5, 9)
    anomaly[artifact > 0] = 0

    detections: List[Detection] = []
    for comp in _components(anomaly, min_area):
        idx = comp.astype(bool)
        area = int(idx.sum())
        ratio = area / wall_area
        atten = float(1.0 - R[idx].mean())
        lace_share = float(lace[idx].mean())

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt))) if cnt is not None else float(area)
        fill = area / max(hull_area, 1.0)
        perim = float(cv2.arcLength(cnt, True)) if cnt is not None else 0.0
        roughness = perim / max(np.sqrt(area), 1e-3)

        if roughness > 14.0 and fill < 0.40 and atten > 0.25:
            cls = "fluor_chaotic"          # 边缘杂乱 + 深衰减 → 疑似基材破坏
        elif ratio > 0.20 and atten > 0.22:
            cls = "fluor_saturated"
        elif ratio > 0.03:
            cls = "fluor_band"
        elif ratio > 0.002:
            cls = "fluor_local"
        else:
            cls = "fluor_point"

        conf = float(np.clip(0.35 + atten * 1.4 + lace_share * 0.3, 0.30, 0.94))
        detections.append((comp, cls, round(conf, 4)))

    for comp in _components(artifact, min_area * 4):
        detections.append((comp, "fluor_artifact", 0.55))

    return detections


def detect(bgr: np.ndarray, wall_mask: np.ndarray, modality: str) -> List[Detection]:
    if modality == "uv":
        return detect_uv(bgr, wall_mask)
    return detect_white(bgr, wall_mask)
