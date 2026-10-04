"""图像预处理：EXIF 校正、缩放、质量质检、墙面区域分割、（可选）双图配准。"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import grading_config

try:  # iPhone HEIC 直传
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - 可选依赖
    pass


@dataclass
class Quality:
    blur_var: float
    exposure: str
    overexposed_ratio: float
    underexposed_ratio: float
    passed: bool                      # 硬性不合格 → 判 UNKNOWN
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)   # 软性问题 → 只扣置信度


def load_image(data: bytes) -> np.ndarray:
    """从上传的字节流读图，按 EXIF 旋转校正，返回 BGR ndarray。"""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_long_side(bgr: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
    max_side = max_side or grading_config()["preprocess"]["max_side"]
    h, w = bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def check_quality(bgr: np.ndarray, modality: str) -> Quality:
    """模糊与曝光质检。

    清晰度分两档：低于 warn 阈值只扣置信度并提示，低于 hard 阈值才判 UNKNOWN。
    一张偏软但检出充分的照片不应该被直接作废。
    紫外照本身偏暗、对比度低，拉普拉斯方差天然比白光低一个量级，单独设阈值。
    """
    cfg = grading_config()["unknown"]
    is_uv = modality == "uv"
    warn_thr = cfg["blur_var_min_uv"] if is_uv else cfg["blur_var_min"]
    hard_thr = cfg["blur_var_hard_min_uv"] if is_uv else cfg["blur_var_hard_min"]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    over = float((gray >= 250).mean())
    under_thresh = 8 if is_uv else 15
    under = float((gray <= under_thresh).mean())

    reasons: List[str] = []
    warnings: List[str] = []

    if blur_var < hard_thr:
        reasons.append(f"图像过于模糊（拉普拉斯方差 {blur_var:.0f} < {hard_thr:.0f}），无法判读")
    elif blur_var < warn_thr:
        warnings.append(f"图像偏软（拉普拉斯方差 {blur_var:.0f}），细节可能丢失，结果置信度已下调")

    if over > cfg["over_under_exposed_max"]:
        reasons.append(f"过曝像素占比 {over:.0%} 偏高")
    if not is_uv and under > cfg["over_under_exposed_max"]:
        reasons.append(f"欠曝像素占比 {under:.0%} 偏高")

    if over > cfg["over_under_exposed_max"]:
        exposure = "overexposed"
    elif under > cfg["over_under_exposed_max"]:
        exposure = "underexposed"
    else:
        exposure = "ok"

    return Quality(round(blur_var, 2), exposure, round(over, 4), round(under, 4),
                   not reasons, reasons, warnings)


def segment_wall(bgr: np.ndarray, modality: str) -> Tuple[np.ndarray, bool]:
    """有效墙面区域分割，用作面积占比的分母（设计 3.6 节）。

    M0 用传统方法：取画面中最大的、低纹理的连通平面区域。
    有 wall_seg 权重时由 inference 层替换为模型分割。
    返回 (mask uint8 0/1, fallback_flag)
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 纹理能量低 = 平整墙面；纹理高 = 家具、植物、杂物
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    energy = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(energy), (31, 31), 0)
    if energy.max() > 1e-6:
        energy = energy / energy.max()
    smooth = (energy < 0.18).astype(np.uint8)

    # 紫外图整体偏暗，额外排除完全黑的背景（照不到的地方不算墙面）
    if modality == "uv":
        smooth &= (gray > 6).astype(np.uint8)

    kernel = np.ones((15, 15), np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, connectivity=8)
    if n <= 1:
        return np.ones((h, w), np.uint8), True

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    mask = (labels == best).astype(np.uint8)

    # 用凸包补上被霉斑/家具打断的部分，墙面本身是连续平面
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [hull], -1, 1, thickness=cv2.FILLED)
        mask = filled

    ratio = float(mask.sum()) / float(h * w)
    if ratio < grading_config()["unknown"]["wall_area_ratio_min"]:
        # 分割不可信，回退整幅画面并标记，让分级层降置信
        return np.ones((h, w), np.uint8), True
    return mask, False


def register(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """把 src 配准到 dst 的坐标系。返回 (单应矩阵, 内点比例)。失败返回 (None, 0.0)。"""
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 紫外图暗，先做 CLAHE 提特征
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g1, g2 = clahe.apply(g1), clahe.apply(g2)

    orb = cv2.ORB_create(nfeatures=4000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)[:400]
    if len(matches) < 12:
        return None, 0.0

    p1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if H is None or inliers is None:
        return None, 0.0
    return H, float(inliers.mean())
