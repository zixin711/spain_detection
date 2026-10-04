"""检测主流程编排（设计第 5 章算法链路）。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import advice, channels, grading, inference, modality as modality_mod, postprocess, preprocess, render, storage
from .config import grading_config, knowledge


def _serializable(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉不能进 JSON 的掩膜数组。"""
    out = []
    for inst in instances:
        d = {k: v for k, v in inst.items() if not k.startswith("_")}
        out.append(d)
    return out


def _try_fuse(images: List[np.ndarray], infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """同时有白光和紫外图时尝试配准（设计 3.8）。"""
    white_idx = [i for i, m in enumerate(infos) if m["modality"] == "white"]
    uv_idx = [i for i, m in enumerate(infos) if m["modality"] == "uv"]
    if not white_idx or not uv_idx:
        return {"paired": False, "registration_ok": None, "inlier_ratio": None}

    H, inliers = preprocess.register(images[uv_idx[0]], images[white_idx[0]])
    ok = H is not None and inliers >= 0.25
    return {"paired": True, "registration_ok": bool(ok), "inlier_ratio": round(float(inliers), 4),
            "white_index": white_idx[0], "uv_index": uv_idx[0]}


def detect(files: List[Tuple[str, bytes]], hints: Optional[List[Optional[str]]] = None,
           px_per_cm: Optional[float] = None) -> Tuple[Dict[str, Any], List[np.ndarray], List[np.ndarray]]:
    """主入口。files 为 (filename, bytes) 列表，返回 (result_dict, 原图列表, 标注图列表)。"""
    hints = hints or [None] * len(files)
    cfg = grading_config()

    images: List[np.ndarray] = []
    infos: List[Dict[str, Any]] = []
    wall_masks: List[np.ndarray] = []

    for idx, (name, blob) in enumerate(files):
        bgr = preprocess.resize_long_side(preprocess.load_image(blob))
        hint = hints[idx] if idx < len(hints) else None
        mres = modality_mod.classify(bgr, hint)
        eff_modality = mres.modality if mres.modality != "unknown" else "white"

        quality = preprocess.check_quality(bgr, eff_modality)
        wall_mask, fallback = preprocess.segment_wall(bgr, eff_modality)

        images.append(bgr)
        wall_masks.append(wall_mask)
        h, w = bgr.shape[:2]
        infos.append({
            "index": idx,
            "filename": name,
            "modality": mres.modality,
            "modality_conf": mres.confidence,
            "modality_source": mres.source,
            "width": w, "height": h,
            "quality": {
                "blur_var": quality.blur_var, "exposure": quality.exposure,
                "overexposed_ratio": quality.overexposed_ratio,
                "underexposed_ratio": quality.underexposed_ratio,
                "passed": quality.passed, "reasons": quality.reasons,
                "warnings": quality.warnings,
            },
            "wall_area_px": int(wall_mask.sum()),
            "wall_area_ratio": round(float(wall_mask.sum()) / float(h * w), 4),
            "wall_area_fallback": fallback,
            "_effective_modality": eff_modality,
        })

    # ---- 检测 ----
    instances: List[Dict[str, Any]] = []
    next_id = 1
    for idx, bgr in enumerate(images):
        eff = infos[idx]["_effective_modality"]
        dets = inference.run(bgr, wall_masks[idx], eff)
        built = postprocess.build_instances(
            bgr, dets, int(wall_masks[idx].sum()), eff, idx, start_id=next_id, px_per_cm=px_per_cm)
        next_id += len(built)
        instances.extend(built)

    has_white = any(i["_effective_modality"] == "white" for i in infos)
    has_uv = any(i["_effective_modality"] == "uv" for i in infos)

    # compute_H 内部按 source_index 过滤实例，所以这里按原始下标对齐传全量
    aligned = list(zip(images, wall_masks))
    chans, weights, key = channels.compute_channels(instances, aligned, has_white, has_uv)
    imold = channels.i_mold(chans, weights)

    fusion = _try_fuse(images, infos)
    reg_failed = bool(fusion.get("paired") and fusion.get("registration_ok") is False)

    summary = grading.grade(infos, instances, chans, weights, key, imold, reg_failed)
    summary.update({
        "channels": chans,
        "channel_weights": weights,
        "available_modalities": [m for m, ok in (("white", has_white), ("uv", has_uv)) if ok],
        "instance_count": len([i for i in instances if i["group"] not in ("negative", "fluor_negative")]),
        "by_class": _count(instances, positive=True),
        "excluded": _count(instances, positive=False),
        "depth_band": _dominant_depth(instances),
        "dominant_fluor_pattern": _dominant_fluor(instances),
    })
    summary.setdefault("visible_area_ratio", None)
    summary.setdefault("fluorescent_area_ratio", None)

    rec = advice.recommend(summary["level"], instances)
    robot = advice.robot_action(summary["level"], instances, summary["level_range"])

    annotated = [render.annotate(images[i], instances, summary, i, infos[i]["_effective_modality"])
                 for i in range(len(images))]

    for info in infos:
        info.pop("_effective_modality", None)

    result = {
        "record_id": storage.new_record_id(),
        "created_at": datetime.now(storage.CST).isoformat(timespec="seconds"),
        "inputs": infos,
        "fusion": fusion,
        "calibration": {"mode": "manual" if px_per_cm else "none", "px_per_cm": px_per_cm},
        "summary": summary,
        "instances": _serializable(instances),
        "recommendations": rec,
        "robot_action": robot,
        "model": {
            "white": inference.backend_name("white"),
            "uv": inference.backend_name("uv"),
            "grading_config": cfg.get("version", "0"),
        },
        "disclaimer": knowledge().get("disclaimer", ""),
    }
    return result, images, annotated


def _count(instances, positive: bool) -> Dict[str, int]:
    neg_groups = ("negative", "fluor_negative")
    out: Dict[str, int] = {}
    for inst in instances:
        is_neg = inst["group"] in neg_groups
        if is_neg == positive:
            continue
        out[inst["cls"]] = out.get(inst["cls"], 0) + 1
    return out


def _dominant_depth(instances) -> Optional[str]:
    cands = [(i.get("depth_band"), i["area_px"]) for i in instances
             if i.get("depth_band") and i["group"] in ("mold", "damage")]
    if not cands:
        return None
    agg: Dict[str, int] = {}
    for band, area in cands:
        agg[band] = agg.get(band, 0) + area
    return max(agg.items(), key=lambda kv: kv[1])[0]


def _dominant_fluor(instances) -> Optional[str]:
    cands = [(i["cls"], i["area_px"]) for i in instances if i["modality"] == "uv" and i["group"] == "fluor"]
    if not cands:
        return None
    agg: Dict[str, int] = {}
    for cls, area in cands:
        agg[cls] = agg.get(cls, 0) + area
    return max(agg.items(), key=lambda kv: kv[1])[0]
