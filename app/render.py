"""标注图渲染：半透明掩膜 + 轮廓 + 编号 + 等级角标。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import LEVEL_COLORS, LEVEL_NAMES, class_table

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _has_cjk_font() -> bool:
    return any(Path(p).exists() for p in _FONT_CANDIDATES)


# label_map 里的 color 描述的是"这类东西长什么样"（黑霉就是黑的），
# 直接拿它画叠加层会和目标本身混在一起看不见。渲染另用一套高对比的展示色。
_DISPLAY_RGB = {
    "mold_black": (255, 92, 60), "mold_green": (90, 220, 120), "mold_white": (250, 230, 120),
    "mold_pink": (255, 120, 190), "grout_mold": (255, 170, 60), "substrate_damage": (255, 60, 60),
    "damp_stain": (90, 160, 245), "efflorescence": (170, 200, 255), "dirt_stain": (150, 150, 170),
    "fluor_point": (170, 255, 90), "fluor_local": (215, 255, 60), "fluor_band": (255, 240, 40),
    "fluor_saturated": (255, 170, 30), "fluor_chaotic": (255, 70, 70), "fluor_artifact": (130, 130, 235),
}
_FALLBACK_RGB = (255, 90, 60)


def display_bgr(cls_name: str):
    r, g, b = _DISPLAY_RGB.get(cls_name, _FALLBACK_RGB)
    return (b, g, r)


def annotate(bgr: np.ndarray, instances: List[Dict[str, Any]], summary: Dict[str, Any],
             source_index: int = 0, modality: str = "white") -> np.ndarray:
    """在原图上叠加检测结果，返回 BGR 图。"""
    canvas = bgr.copy()
    overlay = canvas.copy()
    table = class_table(modality)

    picked = [i for i in instances if i["source_index"] == source_index]
    for inst in picked:
        mask = inst.get("_mask")
        if mask is None:
            continue
        overlay[mask > 0] = display_bgr(inst["cls"])

    # 覆盖率高时降低填充不透明度，否则整张图会被糊住看不出原貌
    covered = float(sum(int(i["_mask"].sum()) for i in picked if i.get("_mask") is not None))
    frac = covered / float(canvas.shape[0] * canvas.shape[1] + 1e-9)
    alpha = 0.36 if frac < 0.25 else (0.26 if frac < 0.5 else 0.18)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    for inst in picked:
        mask = inst.get("_mask")
        if mask is None:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 先描一圈深色再描亮色，保证在浅墙和深霉斑上都看得见
        cv2.drawContours(canvas, cnts, -1, (20, 20, 20), 4)
        cv2.drawContours(canvas, cnts, -1, display_bgr(inst["cls"]), 2)

    # 文字层用 PIL，OpenCV 画不了中文
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = canvas.shape[:2]
    scale = max(w, h) / 1200.0
    f_small = _font(max(14, int(20 * scale)))
    f_big = _font(max(18, int(30 * scale)))

    level = summary.get("level", "UNKNOWN")
    color = LEVEL_COLORS.get(level, (128, 128, 128))
    if _has_cjk_font():
        text = f"{level} {LEVEL_NAMES.get(level, '')}"
    else:
        text = f"{level}"
    if summary.get("level_range") and summary["level_range"][0] != summary["level_range"][1]:
        text += f"  [{summary['level_range'][0]}–{summary['level_range'][1]}]"

    pad = int(14 * scale)
    tb = draw.textbbox((0, 0), text, font=f_big)
    bw, bh = tb[2] - tb[0] + pad * 2, tb[3] - tb[1] + pad * 2
    banner = (w - bw - int(16 * scale), int(16 * scale), w - int(16 * scale), int(16 * scale) + bh)
    draw.rounded_rectangle(list(banner), radius=int(10 * scale), fill=(*color, 235))
    draw.text((w - bw - int(16 * scale) + pad, int(16 * scale) + pad - tb[1]), text, font=f_big, fill=(255, 255, 255, 255))

    def _overlaps(bx, by, r, rect) -> bool:
        return (bx + r > rect[0] and bx - r < rect[2] and by + r > rect[1] and by - r < rect[3])

    # 编号只标面积最大的若干个，否则几十个圆点会把图糊满；完整清单在 JSON 与详情页里
    max_badges = 15
    badged = sorted(picked, key=lambda i: -i["area_px"])[:max_badges]
    for inst in badged:
        x1, y1, x2, y2 = inst["bbox"]
        r = max(11, int(15 * scale))
        cx, cy = x1 + r, max(y1 - r, r)
        if _overlaps(cx, cy, r, banner):        # 会压在等级角标上 → 挪到目标下沿
            cy = min(y2 + r, h - r)
        cx = min(max(cx, r), w - r)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20, 225), outline=(255, 255, 255, 240), width=2)
        label = str(inst["id"])
        tbb = draw.textbbox((0, 0), label, font=f_small)
        draw.text((cx - (tbb[2] - tbb[0]) / 2, cy - (tbb[3] - tbb[1]) / 2 - 2),
                  label, font=f_small, fill=(255, 255, 255, 255))

    tag = "UV 395nm" if modality == "uv" else "White light"
    draw.rounded_rectangle([int(16 * scale), int(16 * scale),
                            int(16 * scale) + draw.textbbox((0, 0), tag, font=f_small)[2] + pad * 2,
                            int(16 * scale) + int(34 * scale)],
                           radius=int(8 * scale), fill=(0, 0, 0, 170))
    draw.text((int(16 * scale) + pad, int(20 * scale)), tag, font=f_small, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def thumbnail(bgr: np.ndarray, max_side: int = 480) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def crop_instance(bgr: np.ndarray, inst: Dict[str, Any], pad_ratio: float = 0.25) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = inst["bbox"]
    pw, ph = int((x2 - x1) * pad_ratio) + 8, int((y2 - y1) * pad_ratio) + 8
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    return bgr[y1:y2, x1:x2].copy()
