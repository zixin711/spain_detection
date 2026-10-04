#!/usr/bin/env python3
"""数据清洗管线：data/raw/ → data/processed/{white,uv}/ 标准 YOLO-seg 数据集。

流程（对应 docs/DESIGN.md 6.3）：
  1 格式统一转 JPEG，剔除损坏文件
  2 pHash 跨数据集去重
  3 质量过滤（模糊 / 曝光；紫外图单独设阈值）
  4 自动打模态标签（白光 / 紫外），低置信度进人工复核
  5 原始类名 → 本项目类别（白光 9 类 / 紫外 6 类）
  6 只有检测框的数据集用 SAM2 以框为 prompt 生成掩膜（装了 SAM2 才启用）
  7 负样本补充：分类目录型数据集里的 normal/stain 作为背景负样本（空标签）
  8 按场景分组 8:1:1 划分，防止同场景连拍跨集泄漏
  9 输出 data.yaml + 统计报告

用法：
    python scripts/prepare_dataset.py                 # 处理全部
    python scripts/prepare_dataset.py --modality uv   # 只处理紫外
    python scripts/prepare_dataset.py --no-sam        # 跳过 SAM2 掩膜生成
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, grading_config, labels_uv, labels_white  # noqa: E402
from app import modality as modality_mod  # noqa: E402

RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
REVIEW = DATA_DIR / "review"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 清洗阶段只剔除"真的没法用"的图，不做严格挑选。
# 阈值取 configs/grading.yaml 里的 hard 档：公开数据集里的图常被缩到 512/640，
# 拉普拉斯方差天然偏低，用运行时的 warn 阈值（60）会误杀掉四成训练数据。
_G = grading_config()["unknown"]
BLUR_MIN = float(_G.get("blur_var_hard_min", 10.0))
BLUR_MIN_UV = float(_G.get("blur_var_hard_min_uv", 5.0))
BRIGHT_MIN, BRIGHT_MAX = 25, 235
UV_BRIGHT_MIN = 6            # 紫外图本来就暗，单独放宽
MAX_SIDE = 2048
PHASH_DISTANCE = 5
SPLIT = (0.8, 0.1, 0.1)

# 分类目录型数据集：目录名 → 处理方式
FOLDER_ROLE = {
    "normal": "background",      # 干净墙面 → 空标签背景图，直接压误报
    "stain": "background",
    "major crack": "skip",
    "minor crack": "skip",
    "algae": "review",           # 是霉/藻，但没有框，进人工复核队列
    "peeling": "review",
    "spalling": "review",
}


# --------------------------------------------------------------------------- #
def phash(img: np.ndarray) -> int:
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)


def _hash_bytes(ph: int) -> np.ndarray:
    return np.frombuffer(int(ph).to_bytes(8, "big"), dtype=np.uint8)


class HashIndex:
    """pHash 去重索引。

    朴素写法是每来一张就和之前所有的逐个比，5000 张图要做 1200 万次 Python 循环，
    慢到没法用。这里把哈希存成 (n, 8) 的字节矩阵，用 numpy 一次性异或 + 查表popcount，
    整批比较是一次向量运算。
    """

    def __init__(self, max_distance: int = PHASH_DISTANCE):
        self.max_distance = max_distance
        self._rows: List[np.ndarray] = []
        self._mat: Optional[np.ndarray] = None

    def is_duplicate(self, ph: int) -> bool:
        if self._mat is None or len(self._rows) == 0:
            return False
        x = _hash_bytes(ph)
        dist = _POPCOUNT[np.bitwise_xor(self._mat, x)].sum(axis=1)
        return bool((dist <= self.max_distance).any())

    def add(self, ph: int) -> None:
        self._rows.append(_hash_bytes(ph))
        self._mat = np.vstack(self._rows)


def load_source_names(ds_dir: Path) -> List[str]:
    """读 Roboflow 导出的 data.yaml，拿原始类名顺序。"""
    for name in ("data.yaml", "data.yml"):
        p = ds_dir / name
        if p.exists():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            names = cfg.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return names
    return []


def build_class_index(modality: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    cfg = labels_uv() if modality == "uv" else labels_white()
    name_to_idx = {m["name"]: int(i) for i, m in cfg["classes"].items()}
    source_map = {k.lower(): v for k, v in (cfg.get("source_map") or {}).items()}
    return name_to_idx, source_map


# --------------------------------------------------------------------------- #
def parse_yolo_label(path: Path) -> List[Tuple[int, List[float]]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), [float(x) for x in parts[1:]]))
    return out


def poly_from_box(vals: List[float]) -> List[float]:
    """把 xywh 归一化框转成矩形多边形（YOLO-seg 格式）。"""
    cx, cy, w, h = vals[:4]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def sam_masks(image: np.ndarray, boxes_xyxy: List[List[float]], predictor) -> List[Optional[np.ndarray]]:
    """SAM2 以框为 prompt 生成掩膜。"""
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out: List[Optional[np.ndarray]] = []
    for box in boxes_xyxy:
        try:
            masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=False)
            m = masks[0].astype(np.uint8)
            # 合理性检查：掩膜面积不能超过框面积，也不能小到只剩边角
            bw, bh = box[2] - box[0], box[3] - box[1]
            frac = m.sum() / max(bw * bh, 1.0)
            out.append(m if 0.05 <= frac <= 1.05 else None)
        except Exception:
            out.append(None)
    return out


def load_sam(enable: bool):
    if not enable:
        return None
    try:
        from ultralytics import SAM

        model = SAM("sam2_b.pt")
        print("  SAM2 已加载，将把检测框转成掩膜")
        return model
    except Exception as exc:
        print(f"  未启用 SAM2（{type(exc).__name__}），检测框将退化成矩形掩膜")
        return None


# --------------------------------------------------------------------------- #
def collect(raw_dir: Path) -> List[Dict]:
    """遍历 data/raw，收集所有候选样本。"""
    items: List[Dict] = []
    for ds_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        src_names = load_source_names(ds_dir)
        # 数据源在 configs/datasets.yaml 里已经声明了模态，直接用它，
        # 比让判别器去猜可靠得多——公开数据集全是白光，判别器偶尔看走眼一两张，
        # 就会把白光图混进紫外训练集里，污染另一个域。
        declared = None
        mf = ds_dir / "manifest.json"
        if mf.exists():
            try:
                declared = (json.loads(mf.read_text(encoding="utf-8")) or {}).get("modality")
            except Exception:
                declared = None
        if declared not in ("white", "uv"):
            declared = None
        for img_path in sorted(ds_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_EXT or not img_path.is_file():
                continue
            label_path = None
            if "images" in img_path.parts:
                parts = list(img_path.parts)
                parts[parts.index("images")] = "labels"
                cand = Path(*parts).with_suffix(".txt")
                if cand.exists():
                    label_path = cand
            folder = img_path.parent.name.lower()
            items.append({
                "image": img_path, "label": label_path, "dataset": ds_dir.name,
                "source_names": src_names, "folder": folder, "declared_modality": declared,
                "scene": f"{ds_dir.name}/{img_path.stem.rsplit('_', 1)[0]}",
            })
    return items


def process(items: List[Dict], modality_filter: Optional[str], sam, dry: bool,
            args_min_blur: Optional[float] = None) -> Dict:
    name_to_idx = {"white": build_class_index("white"), "uv": build_class_index("uv")}
    buckets: Dict[str, List[Dict]] = {"white": [], "uv": []}
    hash_index = HashIndex()
    stats = Counter()
    review_queue: List[Dict] = []

    for it in items:
        img = cv2.imread(str(it["image"]))
        if img is None:
            stats["corrupt"] += 1
            continue

        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # 2 去重
        ph = phash(img)
        if hash_index.is_duplicate(ph):
            stats["duplicate"] += 1
            continue
        hash_index.add(ph)

        # 4 模态：数据源声明了就用声明的，没声明才让判别器猜
        if it.get("declared_modality"):
            mod = it["declared_modality"]
            stats[f"modality_declared_{mod}"] += 1
        else:
            mres = modality_mod.classify(img)
            mod = mres.modality
            if mod == "unknown":
                stats["modality_unknown"] += 1
                review_queue.append({"image": str(it["image"]), "reason": "modality_unknown",
                                     "score": mres.score, "features": mres.features})
                mod = "white"
        if modality_filter and mod != modality_filter:
            continue

        # 3 质量过滤
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())
        lo = UV_BRIGHT_MIN if mod == "uv" else BRIGHT_MIN
        blur_min = BLUR_MIN_UV if mod == "uv" else BLUR_MIN
        if args_min_blur is not None:
            blur_min = args_min_blur
        if blur < blur_min:
            stats["blurry"] += 1
            continue
        if mean < lo or mean > BRIGHT_MAX:
            stats["exposure"] += 1
            continue

        idx_map, source_map = name_to_idx[mod]
        rows: List[str] = []

        anns = parse_yolo_label(it["label"]) if it["label"] else []
        if anns:
            boxes_xyxy, keep = [], []
            for cls_i, vals in anns:
                src_name = (it["source_names"][cls_i] if cls_i < len(it["source_names"]) else "").lower()
                mapped = source_map.get(src_name)
                if mapped is None or mapped not in idx_map:
                    stats["label_unmapped"] += 1
                    continue
                keep.append((mapped, vals))
                if len(vals) == 4:
                    cx, cy, bw, bh = vals
                    boxes_xyxy.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                       (cx + bw / 2) * w, (cy + bh / 2) * h])
                else:
                    boxes_xyxy.append(None)

            masks = [None] * len(keep)
            if sam is not None and any(b is not None for b in boxes_xyxy):
                try:
                    res = sam.predict(img, bboxes=[b for b in boxes_xyxy if b is not None], verbose=False)
                    if res and res[0].masks is not None:
                        got = list(res[0].masks.data.cpu().numpy())
                        gi = 0
                        for i, b in enumerate(boxes_xyxy):
                            if b is not None and gi < len(got):
                                masks[i] = got[gi].astype(np.uint8)
                                gi += 1
                except Exception:
                    pass

            for i, (mapped, vals) in enumerate(keep):
                ci = idx_map[mapped]
                if masks[i] is not None:
                    m = masks[i]
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        eps = 0.004 * cv2.arcLength(cnt, True)
                        pts = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
                        pts[:, 0] /= w
                        pts[:, 1] /= h
                        if len(pts) >= 3:
                            rows.append(f"{ci} " + " ".join(f"{v:.6f}" for v in pts.flatten()))
                            stats["ann_sam_mask"] += 1
                            continue
                if len(vals) > 4:       # 原本就是多边形
                    rows.append(f"{ci} " + " ".join(f"{v:.6f}" for v in vals))
                    stats["ann_polygon"] += 1
                else:
                    rows.append(f"{ci} " + " ".join(f"{v:.6f}" for v in poly_from_box(vals)))
                    stats["ann_box_rect"] += 1
        else:
            role = FOLDER_ROLE.get(it["folder"], "review")
            if role == "skip":
                stats["folder_skipped"] += 1
                continue
            if role == "review":
                review_queue.append({"image": str(it["image"]), "reason": f"unlabeled:{it['folder']}"})
                stats["to_review"] += 1
                continue
            stats["background"] += 1     # 空标签背景图，YOLO 支持，用来压误报

        buckets[mod].append({"img": img, "rows": rows, "scene": it["scene"],
                             "stem": f"{it['dataset']}__{it['image'].stem}"})
        stats[f"kept_{mod}"] += 1

    if not dry:
        REVIEW.mkdir(parents=True, exist_ok=True)
        (REVIEW / "queue.json").write_text(json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"buckets": buckets, "stats": stats, "review": review_queue}


def _safe_clear(root: Path) -> None:
    """清空目录内容。用普通 unlink 逐个删，比 shutil.rmtree 更能容忍受限文件系统。

    删不掉的会明确报出来——静默失败会导致上一轮的产物残留在新数据集里，
    统计数字对不上还找不到原因。
    """
    if not root.exists():
        return
    failed = 0
    for p in sorted(root.rglob("*"), key=lambda x: -len(x.parts)):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except OSError:
            failed += 1
    if failed:
        print(f"  ! {root} 下有 {failed} 个文件删不掉，新数据集可能混入上一轮的残留。"
              f"\n    手动清掉再重跑：rm -rf {root}")


def write_split(bucket: List[Dict], modality: str) -> Dict:
    """按场景分组划分，防止同场景连拍图跨集泄漏。"""
    out_root = PROCESSED / modality
    _safe_clear(out_root)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    scenes = defaultdict(list)
    for item in bucket:
        scenes[item["scene"]].append(item)
    keys = sorted(scenes)
    random.Random(20260828).shuffle(keys)

    n = len(keys)
    n_tr = int(n * SPLIT[0])
    n_va = int(n * SPLIT[1])
    assign = {k: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, k in enumerate(keys)}

    counts = Counter()
    per_class = Counter()
    for scene, items in scenes.items():
        split = assign[scene]
        for item in items:
            stem = item["stem"]
            cv2.imwrite(str(out_root / "images" / split / f"{stem}.jpg"), item["img"],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_root / "labels" / split / f"{stem}.txt").write_text("\n".join(item["rows"]), encoding="utf-8")
            counts[split] += 1
            for r in item["rows"]:
                per_class[int(r.split()[0])] += 1

    cfg = labels_uv() if modality == "uv" else labels_white()
    names = {int(i): m["name"] for i, m in cfg["classes"].items()}
    (out_root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out_root.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": names, "nc": len(names),
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")

    return {"splits": dict(counts), "per_class": {names[k]: v for k, v in sorted(per_class.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["white", "uv"], help="只处理指定模态")
    ap.add_argument("--no-sam", action="store_true", help="跳过 SAM2 掩膜生成")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 个候选文件，用来快速验证流程（不要用于正式产出）")
    ap.add_argument("--min-blur", type=float, default=None,
                    help=f"清晰度下限，默认白光 {BLUR_MIN} / 紫外 {BLUR_MIN_UV}")
    args = ap.parse_args()

    if not RAW.exists() or not any(RAW.iterdir()):
        print(f"没有找到原始数据：{RAW}\n先跑 python scripts/download_datasets.py")
        return 1

    print(f"扫描 {RAW} …")
    items = collect(RAW)
    print(f"  候选文件 {len(items)} 个")
    if args.limit:
        import random as _r
        _r.Random(0).shuffle(items)
        items = items[:args.limit]
        print(f"  --limit 生效，只处理其中 {len(items)} 个")

    sam = load_sam(not args.no_sam)
    res = process(items, args.modality, sam, args.dry_run, args.min_blur)

    print("\n清洗统计：")
    for k, v in sorted(res["stats"].items()):
        print(f"  {k:<20} {v}")

    report = {"stats": dict(res["stats"]), "outputs": {}}
    if not args.dry_run:
        for mod, bucket in res["buckets"].items():
            if not bucket:
                continue
            info = write_split(bucket, mod)
            report["outputs"][mod] = info
            print(f"\n{mod} 数据集 → {PROCESSED / mod}")
            print(f"  划分：{info['splits']}")
            print(f"  各类实例数：{info['per_class'] or '（全是背景负样本）'}")

        PROCESSED.mkdir(parents=True, exist_ok=True)
        (PROCESSED / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n统计报告：{PROCESSED / 'report.json'}")
        if res["review"]:
            print(f"待人工复核 {len(res['review'])} 张：python scripts/review_server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
