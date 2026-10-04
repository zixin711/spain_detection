#!/usr/bin/env python3
"""评估：分割指标 + 负样本误报清单 + 等级一致性（加权 Kappa）。

    python scripts/evaluate.py --domain white
    python scripts/evaluate.py --grading data/eval/level_labels.csv   # 等级一致性
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, LEVEL_ORDER, MODEL_DIR, ROOT  # noqa: E402

NEGATIVE = {"white": {"damp_stain", "efflorescence", "dirt_stain"},
            "uv": {"fluor_artifact"}}


def eval_segmentation(domain: str) -> dict:
    from ultralytics import YOLO

    weights = MODEL_DIR / f"seg_{domain}.pt"
    data = DATA_DIR / "processed" / domain / "data.yaml"
    if not weights.exists() or not data.exists():
        print(f"缺少权重或数据集：{weights} / {data}")
        return {}

    model = YOLO(str(weights))
    m = model.val(data=str(data), split="test", imgsz=1024, plots=True)
    out = {
        "mask_mAP50": float(getattr(m.seg, "map50", 0.0)),
        "mask_mAP50_95": float(getattr(m.seg, "map", 0.0)),
        "box_mAP50": float(getattr(m.box, "map50", 0.0)),
    }
    print(json.dumps(out, indent=2))
    return out


def false_positive_report(domain: str, conf: float = 0.25) -> dict:
    """在纯负样本图（标签里只有负类或没有目标）上统计假阳性，按置信度排序输出，便于数据回流。"""
    from ultralytics import YOLO

    weights = MODEL_DIR / f"seg_{domain}.pt"
    root = DATA_DIR / "processed" / domain
    if not weights.exists():
        print(f"缺少权重 {weights}")
        return {}

    import yaml

    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    neg_idx = {i for i, n in names.items() if n in NEGATIVE[domain]}

    model = YOLO(str(weights))
    rows, n_neg_images, n_fp_images = [], 0, 0

    for img in sorted((root / "images" / "test").glob("*.jpg")):
        lab = root / "labels" / "test" / f"{img.stem}.txt"
        classes = {int(l.split()[0]) for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()} if lab.exists() else set()
        if classes - neg_idx:
            continue                      # 图里有正类，不算纯负样本
        n_neg_images += 1

        res = model.predict(str(img), conf=conf, verbose=False)
        fps = []
        for r in res:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                ci = int(r.boxes.cls[i])
                if ci in neg_idx:
                    continue
                fps.append({"cls": names[ci], "conf": round(float(r.boxes.conf[i]), 4)})
        if fps:
            n_fp_images += 1
            rows.append({"image": img.name, "false_positives": sorted(fps, key=lambda d: -d["conf"])})

    rate = n_fp_images / max(n_neg_images, 1)
    rows.sort(key=lambda r: -max(f["conf"] for f in r["false_positives"]))
    out_path = ROOT / "runs" / f"false_positives_{domain}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"negative_images": n_neg_images, "images_with_fp": n_fp_images,
         "false_positive_rate": round(rate, 4), "target": 0.08, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"纯负样本 {n_neg_images} 张，其中 {n_fp_images} 张出现假阳性 → 误报率 {rate:.2%}（目标 ≤ 8%）")
    print(f"明细：{out_path}")
    return {"false_positive_rate": rate}


def quadratic_kappa(a, b, k: int) -> float:
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    ha, hb = np.bincount(a, minlength=k), np.bincount(b, minlength=k)
    E = np.outer(ha, hb).astype(float)
    E *= O.sum() / max(E.sum(), 1e-9)
    denom = (w * E).sum()
    return float(1 - (w * O).sum() / denom) if denom > 0 else 0.0


def grading_consistency(csv_path: Path) -> dict:
    """CSV 需要两列：human_level, system_level（值为 L0~L5）。"""
    if not csv_path.exists():
        print(f"找不到 {csv_path}")
        return {}
    a, b, under = [], [], 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h, s = row["human_level"].strip(), row["system_level"].strip()
            if h not in LEVEL_ORDER or s not in LEVEL_ORDER:
                continue
            hi, si = LEVEL_ORDER.index(h), LEVEL_ORDER.index(s)
            a.append(hi); b.append(si)
            if hi - si >= 2:
                under += 1
    if not a:
        return {}
    kappa = quadratic_kappa(a, b, len(LEVEL_ORDER))
    rate = under / len(a)
    print(f"样本 {len(a)} 条　加权 Kappa = {kappa:.3f}（目标 ≥ 0.7）")
    print(f"低估 ≥2 级的漏判率 = {rate:.2%}（目标 ≤ 2%）")
    return {"kappa": kappa, "under_rate": rate, "n": len(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["white", "uv"])
    ap.add_argument("--fp", action="store_true", help="只跑负样本误报统计")
    ap.add_argument("--grading", type=Path, help="等级一致性 CSV")
    args = ap.parse_args()

    if args.grading:
        grading_consistency(args.grading)
        return 0
    if not args.domain:
        ap.error("需要 --domain 或 --grading")
    if args.fp:
        false_positive_report(args.domain)
    else:
        eval_segmentation(args.domain)
        false_positive_report(args.domain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
