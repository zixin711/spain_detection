#!/usr/bin/env python3
"""用配对样本标定紫外/白光两套面积阈值的映射关系，并回写 configs/grading.yaml。

数据放置：
    data/paired_calib/white/<name>.jpg
    data/paired_calib/uv/<name>.jpg      同名即为一对

    python scripts/calibrate_grading.py            # 拟合并打印
    python scripts/calibrate_grading.py --write    # 拟合并回写配置
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import inference, preprocess  # noqa: E402
from app.channels import FLUOR_POSITIVE  # noqa: E402
from app.config import CONFIG_DIR, DATA_DIR  # noqa: E402
from app.grading import MOLD_CLASSES  # noqa: E402
from app.postprocess import build_instances  # noqa: E402

PAIRED = DATA_DIR / "paired_calib"


def area_ratio(path: Path, modality: str) -> float:
    bgr = preprocess.resize_long_side(preprocess.load_image(path.read_bytes()))
    wall, _ = preprocess.segment_wall(bgr, modality)
    dets = inference.run(bgr, wall, modality)
    insts = build_instances(bgr, dets, int(wall.sum()), modality, 0)
    keep = FLUOR_POSITIVE if modality == "uv" else MOLD_CLASSES
    return float(sum(i["area_ratio"] for i in insts if i["cls"] in keep))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="把标定结果写回 configs/grading.yaml")
    args = ap.parse_args()

    wdir, udir = PAIRED / "white", PAIRED / "uv"
    if not wdir.exists() or not udir.exists():
        print(f"没有配对数据。请把同名的白光/紫外照片分别放到：\n  {wdir}\n  {udir}")
        print("目标规模 150~300 组，见 docs/DESIGN.md 3.4 节。")
        return 1

    xs, ys = [], []
    for wp in sorted(wdir.iterdir()):
        up = udir / wp.name
        if not up.exists():
            continue
        aw, au = area_ratio(wp, "white"), area_ratio(up, "uv")
        if aw <= 0 and au <= 0:
            continue
        xs.append(aw); ys.append(au)
        print(f"  {wp.name:<40} white={aw:.4f}  uv={au:.4f}")

    if len(xs) < 10:
        print(f"\n只有 {len(xs)} 组有效配对，样本太少，标定结果不可信（建议 ≥150 组）。")
        return 1

    k, b = np.polyfit(np.array(xs), np.array(ys), 1)
    resid = np.array(ys) - (k * np.array(xs) + b)
    r2 = 1 - float((resid ** 2).sum() / max(((np.array(ys) - np.mean(ys)) ** 2).sum(), 1e-9))
    print(f"\n拟合结果：A_uv ≈ {k:.4f} · A_white + {b:.4f}　　R² = {r2:.3f}　样本 {len(xs)} 组")

    if args.write:
        path = CONFIG_DIR / "grading.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        wt = cfg["area_thresholds"]["white"]
        cfg["area_thresholds"]["uv"] = {
            "l2_max": round(float(k * wt["l2_max"] + b), 4),
            "l3_max": round(float(k * wt["l3_max"] + b), 4),
        }
        cfg["uv_area_mapping"] = {"k": round(float(k), 4), "b": round(float(b), 4),
                                  "calibrated": True, "n_pairs": len(xs), "r2": round(r2, 4)}
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"已回写 {path}")
        print(f"  紫外阈值：l2_max={cfg['area_thresholds']['uv']['l2_max']}  l3_max={cfg['area_thresholds']['uv']['l3_max']}")
    else:
        print("加 --write 可把结果回写到 configs/grading.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
