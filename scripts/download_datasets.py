#!/usr/bin/env python3
"""按 configs/datasets.yaml 下载数据集到 data/raw/。

用法：
    python scripts/download_datasets.py                # 下载所有能下的，其余打印指引
    python scripts/download_datasets.py --only bd3_samples mould_detection_v3
    python scripts/download_datasets.py --list         # 只列清单不下载

凭据（放环境变量或 ~/.kaggle/kaggle.json）：
    export ROBOFLOW_API_KEY=xxxx      # https://app.roboflow.com/settings/api 免费获取
    ~/.kaggle/kaggle.json             # https://www.kaggle.com/settings 下载
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, datasets_config  # noqa: E402

RAW = DATA_DIR / "raw"
UA = {"User-Agent": "mold-detect-dataset-fetcher/0.3"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _write_manifest(out: Path, entry: Dict[str, Any], files: int, status: str, note: str = "") -> None:
    manifest = {
        "name": entry["name"],
        "modality": entry.get("modality"),
        "role": entry.get("role"),
        "license": entry.get("license"),
        "attribution": entry.get("attribution"),
        "url": entry.get("url"),
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "status": status,
        "note": note,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
def fetch_github_contents(entry: Dict[str, Any]) -> str:
    """从 GitHub 仓库按目录抓文件（公开仓库，无需 token）。"""
    repo = entry["repo"]
    out = RAW / entry["name"]
    total = 0
    for rel in entry["paths"]:
        api = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(rel)}"
        try:
            listing = json.loads(_get(api))
        except Exception as exc:
            print(f"    ! 取目录失败 {rel}: {exc}")
            continue
        sub = out / Path(rel).name
        sub.mkdir(parents=True, exist_ok=True)
        for item in listing:
            if item.get("type") != "file":
                continue
            dst = sub / item["name"]
            if dst.exists() and dst.stat().st_size == item.get("size", -1):
                total += 1
                continue
            try:
                dst.write_bytes(_get(item["download_url"]))
                total += 1
                time.sleep(0.05)
            except Exception as exc:
                print(f"    ! 下载失败 {item['name']}: {exc}")
    _write_manifest(out, entry, total, "ok" if total else "empty")
    print(f"    → {total} 个文件 → {out}")
    return "ok" if total else "failed"


def _safe_rmtree(root: Path) -> None:
    """删目录。用普通 unlink 逐个删，比 shutil.rmtree 更能容忍受限文件系统。"""
    if not root.exists():
        return
    try:
        shutil.rmtree(root)
        return
    except OSError:
        pass
    for p in sorted(root.rglob("*"), key=lambda x: -len(x.parts)):
        try:
            p.unlink() if (p.is_file() or p.is_symlink()) else p.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def fetch_roboflow(entry: Dict[str, Any], force: bool = False) -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("    ! 缺少 ROBOFLOW_API_KEY。")
        print("      到 https://app.roboflow.com/settings/api 复制 **Private API Key**（不是 Publishable Key），然后：")
        print("        export ROBOFLOW_API_KEY=你的private key")
        return "skipped"
    try:
        from roboflow import Roboflow
    except ImportError:
        print("    ! 未安装 roboflow：pip install roboflow")
        return "skipped"

    if key.startswith("rf_"):
        print("    ! 你填的是 Publishable Key（rf_ 开头）。它只能做推理和下载模型权重，")
        print("      下载数据集需要 Private API Key —— 同一个页面 app.roboflow.com/settings/api 里的另一个。")
        return "skipped"

    out = RAW / entry["name"]
    fmt = entry.get("format", "yolov8")

    # 已经下过就跳过。注意判断依据是"有没有图片"，不是"目录在不在"——
    # 只有一个 manifest.json 的空目录不算下载成功。
    if out.exists() and _count_images(out) > 0 and not force:
        n = _count_images(out)
        print(f"    已存在 {n} 张图，跳过（加 --force 可重下）")
        return "ok"

    # ⚠️ 关键：roboflow SDK 在 location 已存在且 overwrite=False 时会静默跳过下载。
    # 所以这里绝不能先把目录建出来，而且要显式 overwrite=True。
    _safe_rmtree(out)

    try:
        rf = Roboflow(api_key=key)
        project = rf.workspace(entry["workspace"]).project(entry["project"])

        want = entry.get("version", "latest")
        if want in (None, "latest", "Latest"):
            nums = []
            for v in project.versions():
                vid = str(getattr(v, "version", "") or getattr(v, "id", ""))
                tail = vid.rstrip("/").split("/")[-1]
                if tail.isdigit():
                    nums.append(int(tail))
            want = max(nums) if nums else 1
            print(f"    取最新版本 v{want}")

        version = project.version(int(want))
        version.download(fmt, location=str(out), overwrite=True)

        # 兜底：个别 SDK 版本会忽略 location，把数据落到当前目录下的
        # "<Project Name>-<version>"，这里把它搬回来
        if _count_images(out) == 0:
            print("    location 参数似乎没生效，改用默认路径重试…")
            _safe_rmtree(out)
            ds = version.download(fmt, overwrite=True)
            src = Path(getattr(ds, "location", "") or "")
            if src.exists() and src.is_dir() and _count_images(src) > 0:
                shutil.move(str(src), str(out))

    except Exception as exc:
        print(f"    ! Roboflow 下载失败：{type(exc).__name__}: {exc}")
        print(f"      手动方式：打开 {entry.get('url')} → Download this Dataset → YOLOv8 → 解压到 {out}")
        _write_manifest(out, entry, 0, "failed", str(exc))
        return "failed"

    n_img = _count_images(out)
    n_all = sum(1 for _ in out.rglob("*") if _.is_file())
    if n_img == 0:
        print("    ! 下载完成但一张图都没有。常见原因：")
        print("      · 这个项目在 Universe 上没有开放数据集下载（只公开了模型）")
        print("      · 你的 workspace 没有接受该数据集的许可条款——先在网页上点一次 Download 试试")
        print(f"      手动方式：打开 {entry.get('url')} → Download this Dataset → YOLOv8 → 解压到 {out}")
        _write_manifest(out, entry, 0, "empty")
        return "failed"

    _write_manifest(out, entry, n_all, "ok")
    print(f"    → {n_img} 张图（共 {n_all} 个文件）→ {out}")
    return "ok"


def fetch_kaggle(entry: Dict[str, Any]) -> str:
    out = RAW / entry["name"]
    try:
        import kagglehub
    except ImportError:
        print("    ! 未安装 kagglehub：pip install kagglehub")
        return "skipped"
    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not os.environ.get("KAGGLE_KEY"):
        print("    ! 缺少 Kaggle 凭据。到 https://www.kaggle.com/settings → Create New Token")
        print("      把 kaggle.json 放到 ~/.kaggle/kaggle.json 并 chmod 600")
        return "skipped"
    try:
        path = kagglehub.dataset_download(entry["slug"])
    except Exception as exc:
        print(f"    ! Kaggle 下载失败：{exc}")
        return "failed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "PATH.txt").write_text(str(path), encoding="utf-8")
    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    _write_manifest(out, entry, n, "ok", f"kagglehub 缓存路径：{path}")
    print(f"    → {n} 个文件（kagglehub 缓存于 {path}）")
    return "ok"


def print_manual(entry: Dict[str, Any]) -> str:
    print("    ! 该数据集需要手动获取：")
    for line in (entry.get("note") or "").strip().splitlines():
        print(f"      {line}")
    for k in ("url", "search_url", "paper"):
        if entry.get(k):
            print(f"      {k}: {entry[k]}")
    out = RAW / entry["name"]
    _write_manifest(out, entry, 0, "manual")
    return "manual"


HANDLERS = {
    "github_contents": fetch_github_contents,
    "roboflow": fetch_roboflow,
    "kaggle": fetch_kaggle,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载墙面霉菌检测所需的公开数据集")
    ap.add_argument("--only", nargs="*", help="只处理指定名称的数据集")
    ap.add_argument("--list", action="store_true", help="只列出清单")
    ap.add_argument("--force", action="store_true", help="已下载过的也重新下载")
    args = ap.parse_args()

    entries: List[Dict[str, Any]] = datasets_config()["datasets"]
    if args.only:
        entries = [e for e in entries if e["name"] in set(args.only)]

    if args.list:
        for e in entries:
            print(f"{e['name']:<28} {e.get('modality','-'):<7} {e.get('access','-'):<9} {e.get('role','-'):<12} {e.get('url','')}")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}
    for e in entries:
        print(f"\n[{e['name']}]  模态={e.get('modality')}  获取方式={e.get('access')}")
        access, provider = e.get("access"), e.get("provider")
        if access == "manual" or provider not in HANDLERS:
            status = print_manual(e)
        elif provider == "roboflow":
            status = fetch_roboflow(e, force=args.force)
        else:
            status = HANDLERS[provider](e)
        stats[status] = stats.get(status, 0) + 1

    print("\n" + "=" * 60)
    print("汇总：" + "　".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"原始数据目录：{RAW}")
    print("下一步：python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
