"""分级：硬规则优先，I_mold 指数校验，输出等级区间与 UNKNOWN 回退（设计第 4 章）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import LEVEL_NAMES, LEVEL_ORDER, grading_config
from .channels import FLUOR_POSITIVE, MOLD_GROUPS

MOLD_CLASSES = {"mold_black", "mold_green", "mold_white", "mold_pink", "grout_mold"}


def _level_index(level: str) -> int:
    base = level.replace("+", "").replace("?", "").split("_")[0]
    return LEVEL_ORDER.index(base) if base in LEVEL_ORDER else 0


def _clamp(level: str, lo: str, hi: str) -> str:
    i, a, b = _level_index(level), _level_index(lo), _level_index(hi)
    return LEVEL_ORDER[int(np.clip(i, a, b))]


def check_unknown(inputs: List[Dict[str, Any]], instances: List[Dict[str, Any]]) -> List[str]:
    """返回 UNKNOWN 的触发原因列表，空列表表示不触发。"""
    cfg = grading_config()["unknown"]
    reasons: List[str] = []

    for info in inputs:
        q = info["quality"]
        if not q["passed"]:
            reasons.extend(f"第 {info['index'] + 1} 张：{r}" for r in q["reasons"])
        if info["wall_area_ratio"] < cfg["wall_area_ratio_min"]:
            reasons.append(f"第 {info['index'] + 1} 张：有效墙面区域占比仅 {info['wall_area_ratio']:.0%}，可能被遮挡或拍的不是墙面")
        if info["modality"] == "unknown" and info["modality_conf"] < cfg["modality_conf_min"]:
            reasons.append(f"第 {info['index'] + 1} 张：无法判断是白光照还是紫光照，请手动指定")

    if instances:
        confs = [i["conf"] for i in instances]
        if float(np.mean(confs)) < cfg["mean_conf_min"]:
            reasons.append(f"检出目标的平均置信度仅 {np.mean(confs):.2f}，低于阈值")

        pos = [i["conf"] for i in instances if i["group"] in MOLD_GROUPS or i["cls"] in FLUOR_POSITIVE]
        neg = [i["conf"] for i in instances if i["group"] in ("negative", "fluor_negative")]
        if pos and neg and abs(max(pos) - max(neg)) < cfg["mold_vs_negative_margin"]:
            reasons.append("霉菌类与干扰类（水渍/返碱/荧光增白剂）的置信度接近，无法可靠区分")

    return reasons


def _white_rules(instances, area_ratio: float, L_score: Optional[float]) -> Tuple[Optional[str], List[str]]:
    cfg = grading_config()
    rules = cfg["rules"]
    th = cfg["area_thresholds"]["white"]
    hit: List[str] = []

    molds_all = [i for i in instances if i["cls"] in MOLD_CLASSES]
    damage = [i for i in instances if i["cls"] == "substrate_damage"]
    damage_area = sum(i["area_ratio"] for i in damage)
    # L5 的定义是"墙体严重发黑脱落，伴随空鼓/开裂/剥落"——是霉菌 + 结构破坏，不是单纯有裂缝。
    # 一面干净墙上的裂缝是建筑缺陷，不是霉菌等级，只作为附带发现报出来。
    if damage_area >= rules["substrate_damage_min_area"] and area_ratio > 0:
        hit.append("substrate_damage_with_mold")
        return "L5", hit
    if damage_area >= rules["substrate_damage_min_area"]:
        hit.append("substrate_damage_without_mold")   # 记录但不据此定级
    if L_score is not None and L_score >= cfg["depth"]["l5_score"]:
        hit.append("depth_score_l5")
        return "L5", hit

    molds = [i for i in instances if i["cls"] in MOLD_CLASSES]
    # 面积加权的厚度分，比"任意一个实例超阈值"稳得多——单个小噪点不该定级
    thick_score = 0.0
    if molds:
        areas = np.array([i["area_px"] for i in molds], dtype=np.float64)
        ts = np.array([i.get("thickness_score") or 0.0 for i in molds], dtype=np.float64)
        thick_score = float((ts * areas).sum() / max(areas.sum(), 1.0))
    thick = (thick_score >= cfg["depth"]["thick_layer_score"]
             and area_ratio >= cfg["depth"].get("thick_layer_min_area", 0.20))

    if area_ratio > th["l3_max"]:
        # 覆盖极广且衰减极深，但没拍到明确的空鼓/剥落 → 报 L4 并提示可能已到 L5
        deep = 0.0
        if molds_all:
            areas = np.array([i["area_px"] for i in molds_all], dtype=np.float64)
            att = np.array([1.0 - (i.get("features", {}).get("val_mean") or 1.0) for i in molds_all])
            deep = float((att * areas).sum() / max(areas.sum(), 1.0))
        if (area_ratio >= th.get("l5_hint_area", 0.45)
                and deep >= th.get("l5_hint_attenuation", 0.38)):
            hit.append("area_and_attenuation_near_l5")
        else:
            hit.append("area_over_30_percent")
        return "L4", hit
    # 厚层单独不足以判 L4：L4 的定义是"大面积 + 立体厚层"，
    # 一个很小的厚霉点属于 L2/L3。厚度代理量本身也还没标定，不能让它单独决定等级。
    if thick:
        hit.append("thick_layer_with_area")
        return "L4", hit

    grout = [i for i in molds if i["cls"] == "grout_mold"]
    grout_area = sum(i["area_ratio"] for i in grout)
    grout_share = (grout_area / area_ratio) if area_ratio > 0 else 0.0
    if area_ratio >= th["l2_max"]:
        hit.append("area_5_to_30_percent")
        return "L3", hit
    if (grout_share > rules["grout_continuous_ratio"]
            and grout_area >= rules.get("grout_min_area", 0.02)):
        hit.append("grout_continuous")
        return "L3", hit
    if area_ratio > 0:
        # L1 的定义是白光下肉眼不可见（或仅有极微弱灰斑）。面积极小且颜色极浅时，
        # 白光证据不足以判 L2，交给紫外去区分 L1/L2；只有白光时会落到 L0_UNVERIFIED。
        atten = 0.0
        if molds:
            areas = np.array([i["area_px"] for i in molds], dtype=np.float64)
            av = np.array([i.get("attenuation") or 0.0 for i in molds], dtype=np.float64)
            atten = float((av * areas).sum() / max(areas.sum(), 1.0))
        if (area_ratio >= th.get("l2_min_area", 0.03)
                or atten >= th.get("l2_min_attenuation", 0.18)):
            hit.append("isolated_spots_under_5_percent")
            return "L2", hit
        hit.append("faint_marks_below_l2")
        return None, hit
    return None, hit


def _uv_rules(instances, area_ratio: float) -> Tuple[Optional[str], List[str], bool]:
    """紫外规则。返回 (等级, 命中规则, 是否只是"疑似")。

    面积阈值用紫外自己的一套（紫外异常区通常大于白光可见区），
    数值由 data/paired_calib 的 L1~L5 配对样张标定。
    """
    cfg = grading_config()
    rules = cfg["rules"]
    th = cfg["area_thresholds"]["uv"]
    hit: List[str] = []

    by_cls: Dict[str, List[Dict[str, Any]]] = {}
    for i in instances:
        if i["cls"] in FLUOR_POSITIVE:
            by_cls.setdefault(i["cls"], []).append(i)
    positives = [i for group in by_cls.values() for i in group]

    # 面积加权的平均衰减深度
    depth = 0.0
    if positives:
        areas = np.array([i["area_px"] for i in positives], dtype=np.float64)
        att = np.array([i.get("uv_attenuation") or 0.0 for i in positives], dtype=np.float64)
        depth = float((att * areas).sum() / max(areas.sum(), 1.0))

    chaotic_area = sum(i["area_ratio"] for i in by_cls.get("fluor_chaotic", []))
    if chaotic_area >= rules["uv_chaotic_min_area"]:
        hit.append("uv_chaotic_fluorescence")
        return "L5", hit, True          # 紫外只能给"疑似 L5"，需白光复核

    if area_ratio > th["l3_max"]:
        if depth >= th.get("l5_attenuation", 0.30):
            hit.append("uv_wide_and_deep")
            return "L4", hit, True      # 覆盖广且衰减深 → 疑似已到 L5，需白光确认
        hit.append("uv_coverage_over_l3")
        return "L4", hit, False
    if area_ratio > th["l2_max"]:
        hit.append("uv_coverage_l3_band")
        return "L3", hit, False
    if area_ratio > th["l1_max"]:
        hit.append("uv_coverage_l2_band")
        return "L2", hit, False
    if area_ratio > 0:
        hit.append("uv_faint_anomaly_only")
        return "L1", hit, False
    return None, hit, False


def _index_level(i_mold_value: float) -> str:
    bands = grading_config()["index_bands"]
    for level in ["L5", "L4", "L3", "L2", "L1"]:
        lo, hi = bands[level]
        if lo <= i_mold_value < hi:
            return level
    return "L1"


def grade(inputs: List[Dict[str, Any]], instances: List[Dict[str, Any]],
          channels: Dict[str, Optional[float]], weights: Dict[str, float],
          modality_key: str, i_mold_value: float,
          registration_failed: bool = False) -> Dict[str, Any]:
    """主分级入口。返回可直接填进 Summary 的字典。"""
    cfg = grading_config()
    matrix = cfg["modality_matrix"][modality_key]

    unknown_reasons = check_unknown(inputs, instances)
    if unknown_reasons:
        return {
            "level": "UNKNOWN", "level_name": LEVEL_NAMES["UNKNOWN"],
            "level_range": ["UNKNOWN", "UNKNOWN"], "severity_in_level": 0.0,
            "needs_confirmation": True,
            "confirmation_hint": "请按提示重新拍摄后再检测",
            "suspect": False, "i_mold": i_mold_value,
            "rules_hit": [], "conflict": False, "confidence": 0.0,
            "unknown_reasons": unknown_reasons,
        }

    has_white = modality_key in ("white", "both")
    has_uv = modality_key in ("uv", "both")

    white_inst = [i for i in instances if i["modality"] == "white"]
    uv_inst = [i for i in instances if i["modality"] == "uv"]
    visible_ratio = sum(i["area_ratio"] for i in white_inst if i["cls"] in MOLD_CLASSES)
    fluor_ratio = sum(i["area_ratio"] for i in uv_inst if i["cls"] in FLUOR_POSITIVE)

    rules_hit: List[str] = []
    suspect = False
    level: Optional[str] = None
    white_level: Optional[str] = None
    uv_level: Optional[str] = None

    if has_white:
        white_level, hit = _white_rules(white_inst, visible_ratio, channels.get("L"))
        level = white_level
        rules_hit += hit
    if has_uv:
        uv_level, uv_hit, uv_suspect = _uv_rules(uv_inst, fluor_ratio)
        rules_hit += uv_hit
        if uv_level is not None:
            if level is None or _level_index(uv_level) > _level_index(level):
                # 紫外给出更严重的结论时采用它；但白光已确证 L5 时不降级
                if not (uv_suspect and level is not None and _level_index(level) >= 4):
                    level = uv_level
                    suspect = uv_suspect and not has_white

    # 配准失败本身不判 UNKNOWN：按设计 3.8，不融合、各自出结论、取更严重的一级；
    # 只有当两个模态的独立结论相差 >= 2 级时才认为证据互相矛盾，退回 UNKNOWN。
    registration_note = ""
    if registration_failed and white_level and uv_level:
        gap = abs(_level_index(white_level) - _level_index(uv_level))
        if gap >= cfg["conflict_level_gap"]:
            return {
                "level": "UNKNOWN", "level_name": LEVEL_NAMES["UNKNOWN"],
                "level_range": ["UNKNOWN", "UNKNOWN"], "severity_in_level": 0.0,
                "needs_confirmation": True,
                "confirmation_hint": "两张照片未能对齐，且白光与紫外给出的结论相差过大，请对同一部位重新拍摄",
                "suspect": False, "i_mold": i_mold_value,
                "rules_hit": rules_hit, "conflict": True, "confidence": 0.0,
                "unknown_reasons": [
                    f"两张照片未能对齐，白光判为 {white_level}、紫外判为 {uv_level}，相差 {gap} 级"],
            }
        registration_note = "两张照片未能对齐，已按两次独立检测取更严重的一级"
        rules_hit.append("registration_failed_no_fusion")

    # 安全兜底：单块过大 → 至少 L4；大面积水痕 → 至少 L4 并提示先治水
    esc = cfg["rules"]["escalation"]
    if white_inst:
        biggest = max((i["area_ratio"] for i in white_inst if i["cls"] in MOLD_CLASSES), default=0.0)
        if biggest >= esc["single_blob_area_ratio"]:
            rules_hit.append("single_blob_too_large")
            if level is None or _level_index(level) < 4:
                level = "L4"
        damp = sum(i["area_ratio"] for i in white_inst if i["cls"] == "damp_stain")
        if damp >= esc["large_damp_stain_ratio"]:
            rules_hit.append("large_damp_area")
            if level is None or _level_index(level) < 4:
                level = "L4"

    if level is None:
        if has_uv:
            level = "L0"
        else:
            level = "L0_UNVERIFIED"
            rules_hit.append("no_visible_mold_no_uv")

    # ---- 指数校验 ----
    conflict = False
    index_level = _index_level(i_mold_value) if i_mold_value > 0 else "L0"
    check = cfg.get("index_check", {"enabled": True, "calibrated": False, "max_escalation_uncalibrated": 1})
    if check.get("enabled", True) and level in LEVEL_ORDER and level != "L0":
        gap = _level_index(index_level) - _level_index(level)
        if abs(gap) >= cfg["conflict_level_gap"]:
            conflict = True
            if gap > 0:
                # 宁可高估不可漏判：往严重的方向抬。
                # 但 index_bands 未标定时最多抬 1 级，免得未经校准的指数产生离谱结论。
                step = gap if check.get("calibrated") else min(gap, int(check.get("max_escalation_uncalibrated", 1)))
                level = LEVEL_ORDER[_level_index(level) + step]
                rules_hit.append(f"index_escalated_{step}")

    # ---- 等级区间与置信度 ----
    lo = matrix.get("level_min", "L1")
    hi = matrix.get("level_max", "L5")
    if level in LEVEL_ORDER and level != "L0":
        level = _clamp(level, lo, hi)
        level_range = [level, level]
        if modality_key == "uv" and level == "L4":
            level_range = ["L4", "L5"]          # 紫外分不开 L4/L5
        if "area_and_attenuation_near_l5" in rules_hit and level == "L4":
            level_range = ["L4", "L5"]          # 覆盖极广且衰减极深，可能已到 L5
        if suspect:
            level_range = ["L4", "L5"]
    elif level == "L0_UNVERIFIED":
        level_range = ["L0", "L1"]
    else:
        level_range = ["L0", "L0"]

    severity = 0.0
    if level in LEVEL_ORDER and level != "L0":
        band = cfg["index_bands"][level]
        span = max(band[1] - band[0], 1e-6)
        severity = float(np.clip((i_mold_value - band[0]) / span, 0.0, 1.0))

    confs = [i["conf"] for i in instances] or [0.6]
    confidence = float(np.clip(np.mean(confs) * matrix.get("confidence_factor", 1.0), 0.0, 1.0))
    if any(info["wall_area_fallback"] for info in inputs):
        confidence *= 0.85
    if any(info["quality"].get("warnings") for info in inputs):
        confidence *= 0.85
    if conflict:
        confidence *= 0.9

    needs_confirm, hint = _confirmation(modality_key, level, suspect, rules_hit)
    if registration_note:
        hint = (hint + "　" if hint else "") + registration_note
        needs_confirm = True

    return {
        "level": level,
        "level_name": LEVEL_NAMES.get(level, level),
        "level_range": level_range,
        "severity_in_level": round(severity, 4),
        "needs_confirmation": needs_confirm,
        "confirmation_hint": hint,
        "suspect": suspect,
        "i_mold": i_mold_value,
        "rules_hit": rules_hit,
        "conflict": conflict,
        "confidence": round(confidence, 4),
        "unknown_reasons": [],
        "index_level": index_level,
        "visible_area_ratio": round(visible_ratio, 6) if has_white else None,
        "fluorescent_area_ratio": round(fluor_ratio, 6) if has_uv else None,
    }


def _confirmation(modality_key: str, level: str, suspect: bool, rules_hit=()) -> Tuple[bool, str]:
    if "area_and_attenuation_near_l5" in rules_hit:
        return True, ("霉变覆盖极广且颜色极深，可能已经达到 L5。"
                      "请补拍一张能看清墙面是否空鼓、开裂、起皮剥落的照片来确认。")
    if modality_key == "white":
        if level == "L0_UNVERIFIED":
            return True, "白光下未见可见霉斑。L1 隐形萌发期肉眼不可见，建议补一张 395nm 紫光照以排除。"
        return True, "本次仅有白光照，无法判定 L1 隐形萌发期。补一张 395nm 紫光照可覆盖全部等级。"
    if modality_key == "uv":
        if suspect:
            return True, "紫外下检出杂乱深度荧光，疑似基材结构破坏。请补一张白光照确认是否存在空鼓、开裂或剥落。"
        return True, "本次仅有紫外照，面积口径按荧光区换算，且无法评估厚度与基材损伤。补一张白光照可缩小等级区间。"
    return False, ""
