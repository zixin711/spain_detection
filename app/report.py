"""一页式 PDF 检测报告。reportlab 不可用时抛 RuntimeError，由接口层转成 503。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .config import LEVEL_COLORS

_CJK_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    for path in _CJK_FONT_FILES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", path))
                return "CJK"
            except Exception:
                continue
    try:  # reportlab 自带的中日韩 CID 字体，无需系统字体文件
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def build(result: Dict[str, Any], record_path: Path, out_path: Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as pdfcanvas
        from reportlab.lib.utils import ImageReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安装 reportlab，无法生成 PDF 报告") from exc

    font = _register_font()
    s = result["summary"]
    W, H = A4
    c = pdfcanvas.Canvas(str(out_path), pagesize=A4)

    lvl_color = colors.Color(*[v / 255.0 for v in LEVEL_COLORS.get(s["level"], (128, 128, 128))])
    c.setFillColor(lvl_color)
    c.rect(0, H - 26 * mm, W, 26 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont(font, 18)
    c.drawString(16 * mm, H - 14 * mm, f"墙面霉菌检测报告　{s['level']}　{s['level_name']}")
    c.setFont(font, 9)
    c.drawString(16 * mm, H - 21 * mm,
                 f"记录 {result['record_id']}　检测时间 {result['created_at']}　"
                 f"综合指数 I_mold={s['i_mold']}　置信度 {s['confidence']}")

    y = H - 34 * mm
    imgs = sorted(record_path.glob("src*_annotated.png"))[:2]
    if imgs:
        box_w = (W - 32 * mm - (6 * mm if len(imgs) > 1 else 0)) / len(imgs)
        x = 16 * mm
        for p in imgs:
            img = ImageReader(str(p))
            iw, ih = img.getSize()
            scale = min(box_w / iw, 78 * mm / ih)
            c.drawImage(img, x, y - ih * scale, width=iw * scale, height=ih * scale)
            x += box_w + 6 * mm
        y -= 84 * mm

    c.setFillColor(colors.black)
    c.setFont(font, 11)
    c.drawString(16 * mm, y, "检测结论")
    y -= 6 * mm
    c.setFont(font, 9)

    rng = s["level_range"]
    lines = [
        f"等级：{s['level']} {s['level_name']}" + (f"（可确定区间 {rng[0]}–{rng[1]}）" if rng[0] != rng[1] else ""),
        f"同级严重度：{s['severity_in_level']:.2f}　　检出实例：{s['instance_count']} 个",
        f"证据通道：" + "　".join(
            f"{k}={'—' if v is None else f'{v:.3f}'}(权重 {s['channel_weights'].get(k, 0):.2f})"
            for k, v in s["channels"].items()),
    ]
    if s.get("visible_area_ratio") is not None:
        lines.append(f"白光可见霉斑面积占比：{s['visible_area_ratio']:.2%}")
    if s.get("fluorescent_area_ratio") is not None:
        lines.append(f"紫外荧光区面积占比：{s['fluorescent_area_ratio']:.2%}")
    if s.get("confirmation_hint"):
        lines.append(f"提示：{s['confirmation_hint']}")
    for ln in lines:
        c.drawString(16 * mm, y, ln)
        y -= 5 * mm

    rec = result["recommendations"]
    y -= 3 * mm
    c.setFont(font, 11)
    c.drawString(16 * mm, y, f"处理建议 · {rec['action']}")
    y -= 6 * mm
    c.setFont(font, 9)
    for step in rec["steps"]:
        c.drawString(18 * mm, y, f"· {step}")
        y -= 5 * mm
    if rec.get("ppe"):
        c.drawString(18 * mm, y, f"· 防护：{rec['ppe']}")
        y -= 5 * mm
    if rec.get("recheck"):
        c.drawString(18 * mm, y, f"· 复检：{rec['recheck']}")
        y -= 5 * mm

    for note in rec.get("class_notes", [])[:4]:
        c.drawString(18 * mm, y, f"· {note[:60]}")
        y -= 5 * mm

    c.setFont(font, 7.5)
    c.setFillColor(colors.grey)
    c.drawString(16 * mm, 12 * mm, result.get("disclaimer", ""))
    c.showPage()
    c.save()
    return out_path
