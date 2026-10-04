#!/usr/bin/env python3
"""训练白光 / 紫外专家分割模型（Ubuntu + NVIDIA GPU）。

    python scripts/train.py --domain white --model yolo11l-seg.pt --device 0
    python scripts/train.py --domain uv    --model yolo11m-seg.pt --device 1

两张卡架构不同（3090 sm_86 / RTX PRO 5000 Blackwell sm_120）时不要混合 DDP，
并行跑两个实验的吞吐更高：PRO 5000 主训练，3090 跑另一个域或消融实验。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, ROOT  # noqa: E402

PRESETS = {
    "white": dict(model="yolo11l-seg.pt", imgsz=1024, batch=16, epochs=300, name="seg_white"),
    "uv":    dict(model="yolo11m-seg.pt", imgsz=1024, batch=12, epochs=300, name="seg_uv"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["white", "uv"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="0", help="GPU 序号，如 0 或 1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default=None, help="实验名，默认 seg_white / seg_uv")
    ap.add_argument("--cache", choices=["none", "disk", "ram"], default="disk",
                    help="图像缓存。ram 最快但很吃内存（约 图数×3MB），disk 折中，none 最省")
    ap.add_argument("--dry-run", action="store_true", help="只做训练前检查，不真的开训")
    args = ap.parse_args()

    preset = PRESETS[args.domain]
    root = DATA_DIR / "processed" / args.domain
    data = root / "data.yaml"
    if not data.exists():
        print(f"找不到数据集：{data}")
        print("先跑：python scripts/download_datasets.py && python scripts/prepare_dataset.py")
        return 1

    # ---- 训练前检查：宁可现在报错，也别跑了两小时才发现数据是空的 ----
    import yaml

    counts, labeled, instances = {}, 0, 0
    for split in ("train", "val", "test"):
        imgs = list((root / "images" / split).glob("*.jpg"))
        counts[split] = len(imgs)
        for img in imgs:
            lab = root / "labels" / split / f"{img.stem}.txt"
            if lab.exists():
                rows = [r for r in lab.read_text(encoding="utf-8").splitlines() if r.strip()]
                if rows:
                    labeled += 1
                    instances += len(rows)

    names = (yaml.safe_load(data.read_text(encoding="utf-8")) or {}).get("names", {})
    print(f"数据集 {root}")
    print(f"  图片   train {counts['train']} / val {counts['val']} / test {counts['test']}"
          f"  共 {sum(counts.values())} 张")
    print(f"  标注   {labeled} 张有目标，共 {instances} 个实例，{len(names)} 个类别")

    if sum(counts.values()) == 0:
        print("\n✗ 数据集是空的。重跑 prepare_dataset.py，并检查 data/raw 下有没有东西。")
        return 1
    if instances == 0:
        print("\n✗ 一个标注实例都没有，训练出来的模型什么都检测不到。")
        print("  常见原因：原始数据集只有图没有 YOLO 标签，或者类名没被 configs/label_map_*.yaml 的 source_map 覆盖。")
        return 1
    if counts["val"] == 0:
        print("\n✗ 验证集为空，无法计算 mAP 也无法早停。")
        print("  多半是样本太少，按场景分组后没分到验证集。补充数据后重跑 prepare_dataset.py。")
        return 1
    if counts["train"] < 50:
        print(f"\n! 训练集只有 {counts['train']} 张，训出来的模型基本没有实用价值，仅够验证流程能跑通。")

    if args.dry_run:
        print("\n✓ 检查通过（--dry-run，未开始训练）")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics。GPU 机器上：")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("  pip install ultralytics")
        return 1

    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(int(str(args.device).split(",")[0]))
        print(f"GPU: {torch.cuda.get_device_name(0)}  compute capability sm_{cap[0]}{cap[1]}")
        if cap[0] >= 12 and not torch.version.cuda.startswith(("12.8", "12.9", "13")):
            print("⚠️  Blackwell（sm_120）需要 cu128 及以上的 PyTorch，当前可能不匹配，会报 no kernel image")
    else:
        print("⚠️  没有检测到 CUDA，将在 CPU 上训练（非常慢）")

    model = YOLO(args.model or preset["model"])

    # 增强策略：白光的色相是关键特征，抖动幅度要小；
    # 紫外的荧光色相本身就是判据，完全不做色相抖动，改做亮度/曝光与低光噪点模拟。
    aug = dict(hsv_h=0.010, hsv_s=0.5, hsv_v=0.4) if args.domain == "white" \
        else dict(hsv_h=0.0, hsv_s=0.3, hsv_v=0.6)

    run_name = args.name or preset["name"]
    print(f"\n开始训练：{run_name}  模型 {args.model or preset['model']}  "
          f"imgsz {args.imgsz or preset['imgsz']}  batch {args.batch or preset['batch']}  "
          f"epochs {args.epochs or preset['epochs']}  device {args.device}\n")

    model.train(
        data=str(data),
        project=str(ROOT / "runs" / "seg"),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        epochs=args.epochs or preset["epochs"],
        imgsz=args.imgsz or preset["imgsz"],
        batch=args.batch or preset["batch"],
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=50,
        amp=True,
        cache=(False if args.cache == "none" else args.cache),
        close_mosaic=15,
        copy_paste=0.3,
        flipud=0.0,          # 霉菌有重力/结露方向先验，不做上下翻转
        fliplr=0.5,
        degrees=5.0,
        perspective=0.0005,  # 斜拍墙面
        **aug,
    )

    best = ROOT / "runs" / "seg" / run_name / "weights" / "best.pt"
    if best.exists():
        target = ROOT / "models" / f"{preset['name']}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(best.read_bytes())
        print(f"\n最优权重已复制到 {target}，服务重启后自动生效")
        print(f"导出推理引擎：python scripts/export_model.py --domain {args.domain}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
