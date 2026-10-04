"""端到端冒烟测试：合成图 → 检测 → 分级 → 建议，不需要模型权重。

    pytest tests/ -v
    或直接 python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import channels, grading, modality, pipeline, preprocess  # noqa: E402
from app.config import grading_config  # noqa: E402


def _add_noise(img: np.ndarray, rng, amp: int) -> np.ndarray:
    noise = rng.integers(-amp, amp + 1, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def synth_white(with_mold: bool = True, coverage: float = 0.08) -> bytes:
    """合成一张米色墙面照，可选在上面画黑色霉斑。"""
    rng = np.random.default_rng(7)
    img = np.full((720, 960, 3), (196, 202, 208), np.uint8)
    img = _add_noise(img, rng, 14)   # 高频噪声，让合成图能过清晰度质检
    if with_mold:
        target = int(720 * 960 * coverage)
        painted = 0
        while painted < target:
            cx, cy = int(rng.integers(80, 880)), int(rng.integers(80, 640))
            r = int(rng.integers(12, 34))
            cv2.circle(img, (cx, cy), r, (38, 42, 45), -1)
            painted += int(np.pi * r * r)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def synth_uv(n_points: int = 6) -> bytes:
    """合成一张紫外照：深蓝紫底 + 黄绿荧光点。"""
    rng = np.random.default_rng(11)
    img = np.full((720, 960, 3), (78, 26, 30), np.uint8)      # BGR：偏蓝紫，整体偏暗
    img = _add_noise(img, rng, 12)
    for _ in range(n_points):
        cx, cy = int(rng.integers(100, 860)), int(rng.integers(100, 620))
        cv2.circle(img, (cx, cy), int(rng.integers(8, 16)), (60, 235, 190), -1)   # 黄绿荧光
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def test_modality_classifier():
    w = preprocess.load_image(synth_white())
    u = preprocess.load_image(synth_uv())
    assert modality.classify(w).modality == "white", "白光图被判成了紫外"
    assert modality.classify(u).modality == "uv", "紫外图被判成了白光"
    assert modality.classify(w, hint="uv").source == "user_hint"


def test_white_only_cannot_reach_l1():
    """仅白光时等级下限是 L2，绝不能报 L1。"""
    result, _, _ = pipeline.detect([("w.jpg", synth_white(coverage=0.02))])
    s = result["summary"]
    assert s["level"] != "L1", "仅白光却报出了 L1，违反设计约束"
    assert s["channels"]["H"] is None, "没有紫外图时 H 通道应为 None"
    assert s["needs_confirmation"] is True
    assert "紫光" in s["confirmation_hint"]


def test_clean_wall_white_only_is_unverified():
    result, _, _ = pipeline.detect([("w.jpg", synth_white(with_mold=False))])
    s = result["summary"]
    assert s["level"] in ("L0_UNVERIFIED", "UNKNOWN"), f"干净墙面不该判成 {s['level']}"


def test_uv_only_channels():
    result, _, _ = pipeline.detect([("u.jpg", synth_uv())])
    s = result["summary"]
    assert s["channels"]["L"] is None, "紫外单模态不应给出深度分"
    assert s["channels"]["H"] is not None
    assert s["available_modalities"] == ["uv"]


def test_both_modalities_full_weights():
    result, _, _ = pipeline.detect([("w.jpg", synth_white()), ("u.jpg", synth_uv())])
    s = result["summary"]
    if s["level"] != "UNKNOWN":
        w = s["channel_weights"]
        assert abs(w["H"] - 0.40) < 1e-6 and abs(w["M"] - 0.35) < 1e-6 and abs(w["L"] - 0.25) < 1e-6


def test_weights_renormalized():
    cfg = grading_config()["modality_matrix"]
    for key in ("white", "uv", "both"):
        total = sum(cfg[key]["weights"].values())
        assert abs(total - 1.0) < 0.01, f"{key} 的权重之和是 {total}，应当归一化到 1"


def test_grading_never_downgrades_on_conflict():
    """指数远高于规则判级时必须往严重方向抬级，且标记 conflict。"""
    inputs = [{"index": 0, "quality": {"passed": True, "reasons": [], "warnings": []},
               "wall_area_ratio": 0.9, "wall_area_fallback": False,
               "modality": "white", "modality_conf": 0.95}]
    instances = [{"id": 1, "source_index": 0, "modality": "white", "cls": "mold_black",
                  "group": "mold", "conf": 0.9, "area_ratio": 0.02, "area_px": 1800,
                  "attenuation": 0.30, "density": "H", "thickness_score": 0.2,
                  "depth_band": "0.1~0.5mm", "features": {"val_mean": 0.4}}]
    out = grading.grade(inputs, instances, {"H": None, "M": 0.5, "L": 0.4},
                        {"H": 0.0, "M": 0.583, "L": 0.417}, "white", 0.62)
    assert out["index_level"] == "L4", f"I_mold=0.62 应落在 L4 区间，实际 {out['index_level']}"
    assert out["conflict"] is True, "规则判 L2、指数判 L4，相差 2 级却没标记 conflict"
    assert grading._level_index(out["level"]) > grading._level_index("L2"), \
        f"应当往严重方向抬级，实际 {out['level']}"


def test_calibration_samples_reproduce_levels():
    """用 data/paired_calib 的 L1~L5 真实配对样张做回归。

    期望（与设计一致，不是"每个模态都要判对"）：
      白光单模态：L1 判不出来（下限 L2 / 或 L0_UNVERIFIED），L2~L4 应准确，L5 至少落在 [L4,L5]
      紫外单模态：L1~L4 应准确，L5 至少落在 [L4,L5]
      双模态：L1~L4 应准确
    """
    root = Path(__file__).resolve().parent.parent / "data" / "paired_calib"
    if not (root / "white").exists() or not (root / "uv").exists():
        print("      （跳过：没有 data/paired_calib 样张）")
        return

    def read(sub, lv):
        hits = sorted((root / sub).glob(f"{lv}.*"))
        return hits[0].read_bytes() if hits else None

    for lv in ["L1", "L2", "L3", "L4", "L5"]:
        w, u = read("white", lv), read("uv", lv)
        if w is None or u is None:
            continue
        for label, files in (("紫外", [("u.jpg", u)]), ("双模态", [("w.jpg", w), ("u.jpg", u)])):
            res, _, _ = pipeline.detect(files)
            s = res["summary"]
            lo, hi = s["level_range"]
            if lv in ("L1", "L2", "L3"):
                assert s["level"] == lv, f"{lv} 的{label}结果是 {s['level']}，应为 {lv}"
            else:
                assert grading._level_index(lo) <= grading._level_index(lv) <= grading._level_index(hi), \
                    f"{lv} 的{label}等级区间 [{lo},{hi}] 没有覆盖真值"


def test_advice_and_robot_action_present():
    result, _, _ = pipeline.detect([("w.jpg", synth_white())])
    assert result["recommendations"]["steps"], "处理建议不能为空"
    assert result["robot_action"]["action"], "机器人指令不能为空"
    assert result["disclaimer"]


def test_serializable():
    import json

    result, _, _ = pipeline.detect([("w.jpg", synth_white())])
    json.dumps(result, ensure_ascii=False)      # 掩膜数组必须已被剔除
    for inst in result["instances"]:
        assert not any(k.startswith("_") for k in inst)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)
