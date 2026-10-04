#!/usr/bin/env python3
"""标定 / 训练光源判别器（白光 vs 395nm 紫外）。

阶段一（默认）：用已标注的样本目录标定规则版阈值，验证准确率，必要时回写 configs/grading.yaml。
阶段二（--sklearn）：样本够多时训练一个逻辑回归，权重存 models/modality_clf.json，
                     app/modality.py 会自动优先加载它。

数据放置：
    data/modality_clf/white/*.jpg
    data/modality_clf/uv/*.jpg

    python scripts/train_modality_clf.py
    python scripts/train_modality_clf.py --sklearn --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import modality as modality_mod  # noqa: E402
from app.config import DATA_DIR, MODEL_DIR  # noqa: E402

SRC = DATA_DIR / "modality_clf"
FEATURE_ORDER = ["b_minus_r_mean", "value_mean", "violet_ratio", "fluor_ratio",
                 "gray_like_ratio", "hue_entropy", "saturation_mean"]


def load(split: str):
    X, y, paths = [], [], []
    for label, name in ((0, "white"), (1, "uv")):
        d = SRC / name
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            f = modality_mod.extract_features(img)
            X.append([f[k] for k in FEATURE_ORDER])
            y.append(label)
            paths.append(p)
    return np.array(X, dtype=np.float32), np.array(y), paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sklearn", action="store_true", help="训练逻辑回归代替规则版")
    ap.add_argument("--write", action="store_true", help="保存模型 / 回写配置")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"没有标注数据。请建立目录并各放几十张：\n  {SRC / 'white'}\n  {SRC / 'uv'}")
        return 1

    X, y, paths = load("all")
    if len(y) < 10:
        print(f"样本太少（{len(y)} 张），至少各放 20 张")
        return 1
    print(f"样本：白光 {int((y == 0).sum())} 张，紫外 {int((y == 1).sum())} 张")

    # 规则版准确率
    correct, wrong = 0, []
    for p, label in zip(paths, y):
        img = cv2.imread(str(p))
        res = modality_mod.classify(img)
        pred = 1 if res.modality == "uv" else (0 if res.modality == "white" else -1)
        if pred == label:
            correct += 1
        else:
            wrong.append((p.name, res.modality, res.score))
    print(f"规则版准确率：{correct}/{len(y)} = {correct / len(y):.3f}（目标 ≥ 0.99）")
    for name, pred, score in wrong[:15]:
        print(f"  错判 {name}: 判为 {pred} (score={score})")

    if args.sklearn:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 2), scoring="accuracy")
        print(f"逻辑回归 5 折交叉验证准确率：{scores.mean():.4f} ± {scores.std():.4f}")
        clf.fit(X, y)
        if args.write:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            lr = clf.named_steps["logisticregression"]
            sc = clf.named_steps["standardscaler"]
            out = MODEL_DIR / "modality_clf.json"
            out.write_text(json.dumps({
                "features": FEATURE_ORDER,
                "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                "n_samples": int(len(y)), "cv_accuracy": float(scores.mean()),
            }, indent=2), encoding="utf-8")
            print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
