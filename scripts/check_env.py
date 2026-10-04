#!/usr/bin/env python3
"""环境自检：一条命令告诉你缺什么、能不能开始训练。

    python scripts/check_env.py
    python scripts/check_env.py --train      # 额外检查训练相关的依赖与数据集
"""
from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD = "✓", "!", "✗"
_problems: list[str] = []
_warnings: list[str] = []


def line(status: str, name: str, detail: str = "") -> None:
    print(f"  {status} {name:<26} {detail}")


def bad(name: str, detail: str, fix: str) -> None:
    line(BAD, name, detail)
    _problems.append(f"{name}: {fix}")


def warn(name: str, detail: str, fix: str) -> None:
    line(WARN, name, detail)
    _warnings.append(f"{name}: {fix}")


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}  ({platform.machine()}, {platform.system()})"
    if v < (3, 10):
        bad("Python", detail, "需要 3.10+，建议 3.11")
    else:
        line(OK, "Python", detail)


def check_runtime_deps() -> None:
    for mod, pkg in [("cv2", "opencv-python-headless"), ("numpy", "numpy"), ("PIL", "Pillow"),
                     ("yaml", "PyYAML"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                     ("multipart", "python-multipart")]:
        try:
            m = importlib.import_module(mod)
            line(OK, pkg, getattr(m, "__version__", ""))
        except ImportError:
            bad(pkg, "未安装", f"pip install -r requirements.txt")

    for mod, pkg, why in [("reportlab", "reportlab", "PDF 报告会返回 503"),
                          ("qrcode", "qrcode", "启动时不打印二维码"),
                          ("pillow_heif", "pillow-heif", "iPhone 直传的 HEIC 无法解码")]:
        try:
            importlib.import_module(mod)
            line(OK, pkg)
        except ImportError:
            warn(pkg, f"未安装 —— {why}", f"pip install {pkg}")


def check_font() -> None:
    """标注图与 PDF 里的中文全靠系统 CJK 字体。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    hit = next((p for p in candidates if Path(p).exists()), None)
    if hit:
        line(OK, "中文字体", Path(hit).name)
    else:
        warn("中文字体", "没找到 CJK 字体，标注图和 PDF 里的中文会变方块",
             "sudo apt install fonts-noto-cjk")


def check_storage() -> None:
    import os
    import sqlite3

    from app.config import STORAGE_DIR

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    probe = STORAGE_DIR / ".probe.db"
    try:
        conn = sqlite3.connect(probe)
        conn.execute("create table if not exists t(x)")
        conn.close()
        probe.unlink(missing_ok=True)
        line(OK, "存储目录", str(STORAGE_DIR))
    except sqlite3.OperationalError as exc:
        bad("存储目录", f"{STORAGE_DIR} 不支持 SQLite（{exc}）",
            "换到本地盘：export MOLD_STORAGE_DIR=~/mold-storage")


def check_gpu() -> None:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            for i, row in enumerate(out.splitlines()):
                line(OK, f"GPU {i}", row.strip())
        except Exception as exc:
            warn("nvidia-smi", str(exc), "检查驱动安装")
    else:
        line(WARN, "nvidia-smi", "未找到（没有 NVIDIA 卡就正常）")

    try:
        import torch
    except ImportError:
        warn("PyTorch", "未安装 —— 只能用传统 CV 基线，无法训练",
             "GPU 机器：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        return

    line(OK, "PyTorch", f"{torch.__version__}  (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        warn("CUDA", "torch 看不到 GPU", "确认驱动已装、torch 是 CUDA 版本而不是 CPU 版")
        return

    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch = f"sm_{major}{minor}"
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        line(OK, f"CUDA 设备 {i}", f"{name}  {arch}  {mem:.0f}GB")

        # Blackwell(sm_120) 必须 cu128 及以上，否则一跑就 no kernel image
        if major >= 12:
            cuda_ver = (torch.version.cuda or "0").split(".")
            try:
                ok = (int(cuda_ver[0]), int(cuda_ver[1])) >= (12, 8)
            except (ValueError, IndexError):
                ok = False
            if not ok:
                bad(f"设备 {i} 兼容性",
                    f"{arch} 需要 cu128 及以上，当前 torch 是 CUDA {torch.version.cuda}",
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall")

    # 真跑一次矩阵乘，光看 is_available() 不够
    try:
        import torch as t

        x = t.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()
        t.cuda.synchronize()
        line(OK, "GPU 实测", "矩阵乘通过")
    except Exception as exc:
        bad("GPU 实测", f"{type(exc).__name__}: {str(exc)[:80]}",
            "多半是 torch 的 CUDA 版本与显卡架构不匹配，换 cu128 重装")


def check_train_deps() -> None:
    for mod, pkg in [("ultralytics", "ultralytics"), ("roboflow", "roboflow"),
                     ("sklearn", "scikit-learn"), ("tqdm", "tqdm")]:
        try:
            m = importlib.import_module(mod)
            line(OK, pkg, getattr(m, "__version__", ""))
        except ImportError:
            bad(pkg, "未安装", "pip install -r requirements-train.txt")


def check_datasets() -> None:
    import yaml

    from app.config import DATA_DIR

    for domain in ("white", "uv"):
        root = DATA_DIR / "processed" / domain
        cfg = root / "data.yaml"
        if not cfg.exists():
            warn(f"{domain} 数据集", "还没生成",
                 "python scripts/download_datasets.py && python scripts/prepare_dataset.py")
            continue

        counts, labeled = {}, 0
        for split in ("train", "val", "test"):
            imgs = list((root / "images" / split).glob("*.jpg"))
            counts[split] = len(imgs)
            for p in imgs:
                lab = root / "labels" / split / f"{p.stem}.txt"
                if lab.exists() and lab.read_text(encoding="utf-8").strip():
                    labeled += 1

        total = sum(counts.values())
        names = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("names", {})
        detail = f"共 {total} 张（train {counts['train']} / val {counts['val']} / test {counts['test']}），有标注 {labeled} 张，{len(names)} 类"
        if total == 0:
            warn(f"{domain} 数据集", "目录在但一张图都没有", "重跑 prepare_dataset.py")
        elif counts["val"] == 0:
            warn(f"{domain} 数据集", detail + " —— 验证集为空，训练时无法早停",
                 "样本太少导致划分不出验证集，补充数据后重跑 prepare_dataset.py")
        elif labeled == 0:
            warn(f"{domain} 数据集", detail + " —— 全是背景负样本，学不到东西",
                 "确认原始数据带 YOLO 标签，且类名映射覆盖了它们")
        else:
            line(OK, f"{domain} 数据集", detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="额外检查训练依赖与数据集")
    args = ap.parse_args()

    print("\n── 基础环境 ──")
    check_python()
    check_runtime_deps()
    check_font()
    check_storage()

    print("\n── GPU ──")
    check_gpu()

    if args.train:
        print("\n── 训练依赖 ──")
        check_train_deps()
        print("\n── 数据集 ──")
        check_datasets()

    print()
    if _problems:
        print(f"✗ {len(_problems)} 个问题必须先解决：")
        for p in _problems:
            print(f"    · {p}")
    if _warnings:
        print(f"! {len(_warnings)} 个提醒（不影响跑起来）：")
        for w in _warnings:
            print(f"    · {w}")
    if not _problems and not _warnings:
        print("✓ 一切正常")
    elif not _problems:
        print("✓ 没有阻塞性问题，可以开始")
    return 1 if _problems else 0


if __name__ == "__main__":
    sys.exit(main())
