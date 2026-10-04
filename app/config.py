"""集中管理路径与配置加载。所有阈值都来自 configs/*.yaml，不写死在代码里。"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
STORAGE_DIR = Path(os.environ.get("MOLD_STORAGE_DIR", ROOT / "storage"))
KNOWLEDGE_FILE = DATA_DIR / "knowledge" / "advice.yaml"

LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]
LEVEL_NAMES = {
    "L0": "未见异常",
    "L0_UNVERIFIED": "未见可见霉斑（未排除隐形期）",
    "L1": "微轻度（隐形/早期萌发）",
    "L2": "轻度（点状/表面着色）",
    "L3": "中度（块状/缝隙侵入）",
    "L4": "重度（片状/厚质结块）",
    "L5": "严重（结构破坏/墙皮剥落）",
    "UNKNOWN": "无法判定",
}
LEVEL_COLORS = {
    "L0": (110, 170, 110), "L1": (46, 139, 87), "L2": (47, 114, 184),
    "L3": (184, 134, 11), "L4": (192, 57, 43), "L5": (123, 63, 168),
    "UNKNOWN": (128, 128, 128), "L0_UNVERIFIED": (128, 140, 128),
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=None)
def grading_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "grading.yaml")


@functools.lru_cache(maxsize=None)
def labels_white() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "label_map_white.yaml")


@functools.lru_cache(maxsize=None)
def labels_uv() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "label_map_uv.yaml")


@functools.lru_cache(maxsize=None)
def knowledge() -> Dict[str, Any]:
    return _load_yaml(KNOWLEDGE_FILE)


@functools.lru_cache(maxsize=None)
def datasets_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "datasets.yaml")


def class_table(modality: str) -> Dict[str, Dict[str, Any]]:
    """按类名索引的类别表。"""
    raw = (labels_uv() if modality == "uv" else labels_white())["classes"]
    out = {}
    for idx, meta in raw.items():
        meta = dict(meta)
        meta["index"] = int(idx)
        out[meta["name"]] = meta
    return out


def class_by_index(modality: str, index: int) -> Dict[str, Any]:
    raw = (labels_uv() if modality == "uv" else labels_white())["classes"]
    meta = dict(raw[index])
    meta["index"] = index
    return meta


def reload_all() -> None:
    """标定脚本回写配置后调用，清掉缓存。"""
    for fn in (grading_config, labels_white, labels_uv, knowledge, datasets_config):
        fn.cache_clear()
