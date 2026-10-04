"""API 响应结构（与 docs/DESIGN.md 8.2 节一致）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityInfo(BaseModel):
    blur_var: float
    exposure: str
    overexposed_ratio: float
    underexposed_ratio: float
    passed: bool
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class InputInfo(BaseModel):
    index: int
    filename: str
    modality: str                     # white | uv | unknown
    modality_conf: float
    modality_source: str              # auto | user_hint
    width: int
    height: int
    quality: QualityInfo
    wall_area_px: int
    wall_area_ratio: float
    wall_area_fallback: bool


class Instance(BaseModel):
    id: int
    source_index: int
    modality: str
    cls: str
    cls_cn: str
    group: str                        # mold | damage | negative | fluor | fluor_negative
    conf: float
    bbox: List[int]
    polygon: List[List[int]]
    area_px: int
    area_ratio: float
    area_cm2: Optional[float] = None
    position_hint: str = ""
    density: Optional[str] = None
    density_score: Optional[float] = None
    depth_band: Optional[str] = None
    thickness_score: Optional[float] = None
    uv_response: Optional[float] = None
    uv_response_normalized: Optional[float] = None
    uv_attenuation: Optional[float] = None
    attenuation: Optional[float] = None
    features: Dict[str, float] = Field(default_factory=dict)
    advice_key: str = ""


class Summary(BaseModel):
    level: str
    level_name: str
    level_range: List[str]
    severity_in_level: float
    needs_confirmation: bool
    confirmation_hint: str = ""
    suspect: bool = False
    i_mold: float
    channels: Dict[str, Optional[float]]
    channel_weights: Dict[str, float]
    available_modalities: List[str]
    visible_area_ratio: Optional[float] = None
    fluorescent_area_ratio: Optional[float] = None
    depth_band: Optional[str] = None
    dominant_fluor_pattern: Optional[str] = None
    instance_count: int
    by_class: Dict[str, int]
    excluded: Dict[str, int]
    rules_hit: List[str]
    conflict: bool
    confidence: float
    unknown_reasons: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    title: str
    action: str
    steps: List[str]
    ppe: Optional[str] = None
    recheck: Optional[str] = None
    class_notes: List[str] = Field(default_factory=list)
    safety: List[str] = Field(default_factory=list)


class RobotAction(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    target_regions: List[Dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class DetectResult(BaseModel):
    record_id: str
    created_at: str
    inputs: List[InputInfo]
    fusion: Dict[str, Any]
    calibration: Dict[str, Any]
    summary: Summary
    instances: List[Instance]
    recommendations: Recommendation
    robot_action: RobotAction
    model: Dict[str, str]
    disclaimer: str
