"""Validation functions; each returns a dict with a ``passed`` bool. ``validate_reconstruction`` aggregates them.

Hard checks decide ``geometry.passed``: dimensions, scale, modifiers, materials, scene structure.
Soft checks are reported and listed under ``warnings`` but never fail the run: symmetry, normals, non-manifold.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import bmesh
import bpy

from .. import get_logger
from ..core import correction, dimcheck
from ..core.config import DIMENSION_TOLERANCE, PipelineConfig
from ..core.metrics import ViewComparison, aggregate
from ..core.summary import failed_checks
from . import mesh, modifiers, scene

log = get_logger(__name__)

SCALE_TOLERANCE = 1e-4
SYMMETRY_TOLERANCE = 0.02      # |bbox centre| / extent
DIM_NAMES = ("width", "depth", "height")


def validate_dimensions(target: Mapping[str, float], tol: float = DIMENSION_TOLERANCE,
                        context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Overall bbox of the rendered model vs target width/depth/height (relative tolerance ``tol``). With ``context``
    (``boxes`` = per-component world boxes, ``reference`` = :func:`dimcheck.reference_extents`, ``hull_ids``) a failure also
    gets ``limiting``: the component(s) that set the model's edge on each failing axis (:func:`dimcheck.attribute`)."""
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        return {"passed": False, "reason": "AI_MODEL has no mesh geometry"}
    extent = box[1] - box[0]
    actual = dict(zip(DIM_NAMES, (float(e) for e in extent)))
    rel = {k: (actual[k] / target[k] - 1.0) if target.get(k) else 0.0 for k in DIM_NAMES}
    out = {"passed": all(abs(v) <= tol for v in rel.values()), "actual": actual,
           "target": {k: float(target[k]) for k in DIM_NAMES if target.get(k)}, "relative_error": rel,
           "tolerance": tol, "floor_z": float(box[0][2])}
    if not out["passed"] and context:
        out["limiting"] = dimcheck.attribute(rel, tol, context.get("boxes") or {}, context.get("reference") or {},
                                             set(context.get("hull_ids") or ()))
    return out


def validate_scale(tol: float = SCALE_TOLERANCE) -> dict[str, Any]:
    """Every AI_MODEL mesh object has unit scale (geometry is built at true size)."""
    bad = {o.name: [round(float(s), 6) for s in o.scale] for o in scene.model_meshes(include_cutters=True)
           if max(abs(float(s) - 1.0) for s in o.scale) > tol}
    return {"passed": not bad, "non_unit_scale": bad}


def validate_symmetry(symmetry: Mapping[str, bool], tol: float = SYMMETRY_TOLERANCE) -> dict[str, Any]:
    """For each symmetric axis the bbox centre sits at 0 (soft check)."""
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        return {"passed": False, "reason": "AI_MODEL has no mesh geometry"}
    centre, extent = (box[0] + box[1]) / 2.0, box[1] - box[0]
    offsets: dict[str, float] = {}
    for i, axis in enumerate("xyz"):
        if symmetry.get(axis):
            offsets[axis] = float(centre[i] / max(extent[i], 1e-9))
    return {"passed": all(abs(v) <= tol for v in offsets.values()), "centre_offset_ratio": offsets}


def validate_normals() -> dict[str, Any]:
    """Meshes with negative signed volume have inward-facing normals (soft check)."""
    inverted = [o.name for o in scene.model_meshes() if mesh.signed_volume(o.data) < 0]
    return {"passed": not inverted, "inverted": inverted}


def validate_non_manifold() -> dict[str, Any]:
    """Count non-manifold edges per mesh (soft check; voxel hulls and touching frame bars may have some)."""
    counts: dict[str, int] = {}
    for obj in scene.model_meshes():
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        counts[obj.name] = sum(1 for e in bm.edges if not e.is_manifold)
        bm.free()
    return {"passed": not any(counts.values()), "non_manifold_edges": {k: v for k, v in counts.items() if v}}


def validate_modifiers() -> dict[str, Any]:
    """Modifier stacks follow Mirror -> Boolean -> Bevel -> Weighted Normal."""
    bad: dict[str, list[str]] = {}
    for obj in scene.model_meshes(include_cutters=True):
        types = modifiers.stack_summary(obj)
        ranks = [modifiers.STACK_ORDER.index(t) if t in modifiers.STACK_ORDER else len(modifiers.STACK_ORDER)
                 for t in types]
        if ranks != sorted(ranks):
            bad[obj.name] = types
    return {"passed": not bad, "wrong_order": bad}


def validate_materials() -> dict[str, Any]:
    """Every AI_MODEL mesh has at least one assigned material."""
    missing = [o.name for o in scene.model_meshes(include_cutters=True)
               if not any(m is not None for m in o.data.materials)]
    return {"passed": not missing, "without_material": missing}


def validate_scene_structure() -> dict[str, Any]:
    """Collections exist, generated objects carry an agent_id, no duplicate agent_ids, origin empty present."""
    missing = [n for n in scene.ALL_COLLECTIONS if n not in bpy.data.collections]
    no_id = [o.name for n in (scene.COLL_MODEL, scene.COLL_DEBUG, scene.COLL_REF) for o in scene.objects_in(n)
             if scene.PROP_AGENT_ID not in o.keys()]
    dup = scene.duplicate_agent_ids()
    origin = scene.find_by_agent_id(scene.ORIGIN_AGENT_ID) is not None
    return {"passed": not (missing or no_id or dup) and origin, "missing_collections": missing,
            "without_agent_id": no_id, "duplicate_agent_ids": dup, "origin_empty": origin}


def validate_reference_projection(results: Mapping[str, ViewComparison], config: PipelineConfig) -> dict[str, Any]:
    """Per-view silhouette error against the references, aggregated against ``error_threshold``."""
    if not results:
        return {"passed": False, "reason": "no validation views were compared"}
    agg = aggregate(results, config.view_weights, config.error_threshold)
    out: dict[str, Any] = {"passed": agg["passed"], "total_error": agg["total_error"], "threshold": config.error_threshold}
    for view, info in agg["per_view"].items():
        out[f"{view}_error"] = info["error"]
    out["per_view"] = agg["per_view"]
    return out


def _warning_detail(name: str, check: Mapping[str, Any]) -> str:
    """Short human-readable detail of a failed soft check."""
    if name == "non_manifold":
        return f"{name} ({sum(check.get('non_manifold_edges', {}).values())} edges)"
    if name == "normals":
        return f"{name} ({len(check.get('inverted', []))} inverted objects)"
    if name == "symmetry" and check.get("centre_offset_ratio"):
        worst = max(check["centre_offset_ratio"].values(), key=abs)
        return f"{name} (centre offset {abs(worst) * 100:.1f}% of extent)"
    return name


def validate_reconstruction(results: Mapping[str, ViewComparison], calibs: Mapping[str, Any], dims: Mapping[str, float],
                            symmetry: Mapping[str, bool], config: PipelineConfig,
                            scene_spec: Optional[Mapping[str, Any]] = None, dimension_tolerance: Optional[float] = None,
                            dimension_context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Full report: ``{passed, geometry{passed, checks}, soft_checks, warnings, warning_details, reference_match
    {passed, <view>_error..., per_view{<view>: error..., bbox_delta_m, largest_mismatch_region, components_in_region}},
    diagnosis, recommendation}``. ``scene_spec`` (dict) is used to name the components in each mismatch region.
    ``dimension_tolerance`` (default 3 %) and ``dimension_context`` are those of :func:`validate_dimensions`."""
    hard = {"dimensions": validate_dimensions(dims, DIMENSION_TOLERANCE if dimension_tolerance is None else dimension_tolerance,
                                              dimension_context), "scale": validate_scale(), "modifiers": validate_modifiers(),
            "materials": validate_materials(), "scene_structure": validate_scene_structure()}
    soft = {"symmetry": validate_symmetry(symmetry), "normals": validate_normals(),
            "non_manifold": validate_non_manifold()}
    match = validate_reference_projection(results, config)
    for view, extra in correction.describe_view_results(results, scene_spec).items():
        match["per_view"][view].update(extra)
    geometry_ok = all(c["passed"] for c in hard.values())
    warnings = [name for name, c in soft.items() if not c["passed"]]
    details = [_warning_detail(name, soft[name]) for name in warnings]
    diagnosis = correction.diagnose(results, calibs) if results else []
    failed = failed_checks(hard, match)
    if not match["passed"] and results:
        hint = diagnosis[0]["recommendation"] if diagnosis else "Silhouettes differ in shape; inspect compare_<view>.png."
    else:
        hint = "Fix the failed geometry checks (the model's size / scale / modifiers / materials / structure)."
    recommendation = (f"FAILED: {'; '.join(failed)}. {hint}" if failed
                      else "Reconstruction matches the references within tolerance.")
    report = {"passed": bool(geometry_ok and match["passed"]),
              "geometry": {"passed": geometry_ok, "checks": hard}, "soft_checks": soft, "warnings": warnings,
              "warning_details": details, "reference_match": match, "diagnosis": diagnosis,
              "recommendation": correction.with_warnings(recommendation, details)}
    log.info("validation: passed=%s geometry=%s reference_match=%s warnings=%s", report["passed"], geometry_ok,
             match["passed"], warnings)
    return report
