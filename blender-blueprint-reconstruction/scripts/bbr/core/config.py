"""Pipeline configuration (thresholds, iterations, resolutions)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from .. import SpecError


CONSISTENCY_GATES = ("warn", "strict")
DIMENSION_TOLERANCE = 0.03     # relative error of each overall model dimension against the target (validation 'dimensions')


@dataclass
class PipelineConfig:
    """Tunable parameters of the reconstruction pipeline."""

    max_iterations: int = 3
    error_threshold: float = 0.05
    blockout_threshold: float = 0.12
    auto_correct: bool = True
    consistency_gate: str = "warn"   # "warn" = report only; "strict" = stop when the references disagree
    consistency_ignore_flags: list[str] = field(default_factory=list)   # "type" / "type:view" flags judged false alarms
    consistency_roles_override: dict[str, list[str]] = field(default_factory=dict)   # {view: roles} judged false alarms
    dimension_tolerance: Optional[float] = None   # None = by reference quality (dimcheck.resolve_tolerance): 3 % synthetic / blueprint, 5 % AI-generated
    hull_resolution: int = 320
    hull_margin: float = 0.03
    top_consistency_tol: float = 0.15
    smooth_iterations: int = 14
    hull_min_iou: float = 0.95
    render_scale: float = 1.0
    bg_tolerance: float = 0.12
    max_hole_frac: float = 0.002
    min_component_frac: float = 0.0005
    edge_distance_cap_px: int = 32
    ref_plane_offset_factor: float = 0.75
    ref_plane_offset_base: float = 0.1
    view_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PipelineConfig":
        """Build from a dict; unknown keys raise SpecError, missing keys use defaults."""
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SpecError(f"unknown PipelineConfig keys: {unknown}")
        if data.get("consistency_gate", "warn") not in CONSISTENCY_GATES:
            raise SpecError(f"consistency_gate must be one of {CONSISTENCY_GATES}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (JSON-able)."""
        return asdict(self)
