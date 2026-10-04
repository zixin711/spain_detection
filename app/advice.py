"""处理建议知识库查询 + 机器人执行指令生成（设计第 10、11 章）。"""
from __future__ import annotations

from typing import Any, Dict, List

from .config import knowledge


def _class_notes(instances: List[Dict[str, Any]]) -> List[str]:
    kb = knowledge().get("by_class", {})
    seen, notes = set(), []
    # 按面积从大到小，最相关的说明排前面
    for inst in sorted(instances, key=lambda i: -i.get("area_ratio", 0.0)):
        cls = inst["cls"]
        if cls in seen or cls not in kb:
            continue
        seen.add(cls)
        entry = kb[cls]
        prefix = "已排除项 · " if entry.get("negative") else ""
        notes.append(f"{prefix}{inst.get('cls_cn', cls)}：{entry['note']}")
    return notes[:6]


def recommend(level: str, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    kb = knowledge()
    entry = kb["levels"].get(level) or kb["levels"]["UNKNOWN"]
    return {
        "title": entry["title"],
        "action": entry["action"],
        "steps": list(entry.get("steps", [])),
        "ppe": entry.get("ppe"),
        "recheck": entry.get("recheck"),
        "class_notes": _class_notes(instances),
        "safety": list(kb.get("general_safety", [])),
    }


def robot_action(level: str, instances: List[Dict[str, Any]], level_range: List[str]) -> Dict[str, Any]:
    """按区域下发指令：同一面墙上不同等级的区域用不同参数。"""
    kb = knowledge()["robot_actions"]
    entry = dict(kb.get(level) or kb["UNKNOWN"])
    params = dict(entry.get("params", {}))
    note = ""

    # 等级区间跨度 >= 2 级时先切光源二次扫描，不要按上限直接执行（设计 11 章）
    if len(level_range) == 2 and level_range[0] != level_range[1]:
        from .grading import _level_index

        if abs(_level_index(level_range[1]) - _level_index(level_range[0])) >= 1 and level != "L5":
            note = "等级区间未收敛，建议先切换光源二次扫描再决定动作"

    regions = []
    for inst in instances:
        if inst["group"] in ("negative", "fluor_negative"):
            continue
        regions.append({
            "instance_id": inst["id"],
            "source_index": inst["source_index"],
            "polygon": inst["polygon"],
            "bbox": inst["bbox"],
            "cls": inst["cls"],
            "density": inst.get("density"),
            "depth_band": inst.get("depth_band"),
        })

    return {"action": entry.get("action", "rescan"), "params": params,
            "target_regions": regions, "note": note}
