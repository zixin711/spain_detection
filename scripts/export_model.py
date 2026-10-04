#!/usr/bin/env python3
"""导出 ONNX 与 TensorRT 引擎。

    python scripts/export_model.py --domain white --formats onnx engine

⚠️ TensorRT 引擎不跨 GPU 架构通用：3090(sm_86) 与 RTX PRO 5000(sm_120) 必须各自 build 一份。
本脚本会自动把架构写进文件名（seg_white_sm86.engine），服务启动时按实测卡自动挑选。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import MODEL_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["white", "uv"], required=True)
    ap.add_argument("--formats", nargs="+", default=["onnx"], choices=["onnx", "engine"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--half", action="store_true", default=True)
    args = ap.parse_args()

    src = MODEL_DIR / f"seg_{args.domain}.pt"
    if not src.exists():
        print(f"找不到权重 {src}，先训练：python scripts/train.py --domain {args.domain}")
        return 1

    from ultralytics import YOLO
    import torch

    arch = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm{major}{minor}"
        print(f"当前 GPU：{torch.cuda.get_device_name(0)}  {arch}")

    model = YOLO(str(src))
    for fmt in args.formats:
        if fmt == "engine" and arch is None:
            print("没有 CUDA，跳过 TensorRT 导出")
            continue
        print(f"导出 {fmt} …")
        out = model.export(format=fmt, imgsz=args.imgsz, half=args.half,
                           dynamic=(fmt == "onnx"), opset=17 if fmt == "onnx" else None)
        out = Path(out)
        target = MODEL_DIR / (f"seg_{args.domain}_{arch}.engine" if fmt == "engine"
                              else f"seg_{args.domain}.onnx")
        shutil.copy2(out, target)
        print(f"  → {target}")

    print("\n提示：ONNX 作为换卡与 CI 的兜底保留；两张卡请各自跑一次 --formats engine。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
