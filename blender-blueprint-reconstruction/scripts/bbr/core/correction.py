"""Error -> axis / component diagnosis and axis-scale correction of a SceneSpec dict."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import get_logger
from .calibration import ViewCalibration
from .conventions import view_frame
from .metrics import ViewComparison

log = get_logger(__name__)

RELATIVE_TOL = 0.01  # extents / shifts below 1 % of the reference extent are not reported
M_TO_CM = 100.0
AXIS_DIM_NAME = {"X": "width", "Y": "depth", "Z": "height"}
AXIS_FIELDS = ("width", "depth", "height")  # X, Y, Z order of SceneSpec.dimensions


def _rotation_matrix_xyz(rot_deg: Sequence[float]) -> np.ndarray:
    """Blender XYZ-Euler rotation matrix (R = Rz @ Ry @ Rx)."""
    rx, ry, rz = (math.radians(a) for a in rot_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def diagnose(view_results: Mapping[str, ViewComparison],
             calibs: Optional[Mapping[str, ViewCalibration]] = None) -> list[dict[str, Any]]:
    """Map per-view bbox/size errors to world axes with human-readable recommendations.

    Each entry: ``view, axis ('X'|'Y'|'Z'), dimension, kind ('size'|'shift'), delta_m`` (signed, output minus
    reference; for shifts positive = output displaced toward ``direction``), ``direction`` (shifts only),
    ``recommendation``. Sorted by descending |delta_m|. Front/back -> X,Z; left/right/side -> Y,Z; top/bottom -> X,Y.
    """
    items: list[dict[str, Any]] = []
    for view, res in view_results.items():
        cam = calibs[view].camera_axis if calibs and view in calibs else res.camera_axis
        frame = view_frame(cam)
        label = f"{view.capitalize()} view"
        d = res.bbox_delta_m
        rx0, ry0, rx1, ry1 = res.ref_bbox
        for axis_name, positive_axis, size_delta, shift_delta, ref_px in (
                (frame.right_axis, frame.right_axis, d["left"] + d["right"], (d["right"] - d["left"]) / 2.0, rx1 - rx0),
                (frame.up_axis, frame.up_axis, d["top"] + d["bottom"], (d["top"] - d["bottom"]) / 2.0, ry1 - ry0)):
            letter = axis_name[1]
            dim = AXIS_DIM_NAME[letter]
            tol = RELATIVE_TOL * ref_px * res.m_per_px
            if abs(size_delta) > tol:
                word = "large" if size_delta > 0 else "small"
                items.append({
                    "view": view, "axis": letter, "dimension": dim, "kind": "size", "delta_m": size_delta,
                    "recommendation": f"{label}: {dim} too {word} by {abs(size_delta) * M_TO_CM:.1f} cm → {letter} axis"})
            if abs(shift_delta) > tol:
                direction = positive_axis if shift_delta > 0 else ("-" if positive_axis[0] == "+" else "+") + letter
                items.append({
                    "view": view, "axis": letter, "dimension": dim, "kind": "shift", "delta_m": shift_delta,
                    "direction": direction,
                    "recommendation": f"{label}: model shifted {abs(shift_delta) * M_TO_CM:.1f} cm toward {direction}"
                                      f" → move along {letter} axis"})
    items.sort(key=lambda it: -abs(it["delta_m"]))
    return items


def axis_scale_factors(view_results: Mapping[str, ViewComparison]) -> dict[str, float]:
    """Per world axis ``ref extent / model extent`` averaged over the views that see the axis."""
    acc: dict[str, list[float]] = {}
    for res in view_results.values():
        frame = view_frame(res.camera_axis)
        for axis_name, ratio in ((frame.right_axis, res.width_ratio), (frame.up_axis, res.height_ratio)):
            if ratio > 0:
                acc.setdefault(axis_name[1], []).append(1.0 / ratio)
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def apply_axis_scale(scene_spec_dict: Mapping[str, Any], factors: Mapping[str, float]) -> dict[str, Any]:
    """Return a copy of the spec with locations, sizes and dimensions scaled per world axis.

    Scaling is about the world origin (X/Y centre, Z floor at 0, so bottoms stay on the floor). A component's
    local size axis is scaled by the factor of the world axis it is mostly aligned with (rotation-aware).
    ``params`` (radius, thickness, ...) are left untouched.
    """
    out = copy.deepcopy(dict(scene_spec_dict))
    f = np.array([float(factors.get(a, 1.0)) for a in "XYZ"])
    dims = out.get("dimensions")
    if dims:
        for axis_i, key in enumerate(AXIS_FIELDS):
            if dims.get(key) is not None:
                dims[key] = float(dims[key] * f[axis_i])
    for comp in out.get("components", []):
        comp["location"] = [float(x) for x in np.asarray(comp.get("location", [0, 0, 0]), float) * f]
        if comp.get("size"):
            rot = _rotation_matrix_xyz(comp.get("rotation_deg", [0, 0, 0]))
            dominant = np.argmax(np.abs(rot), axis=0)  # world axis each local axis points along
            comp["size"] = [float(s * f[dominant[i]]) for i, s in enumerate(comp["size"])]
    return out


def components_in_region(scene_spec_dict: Mapping[str, Any], world_bbox: Mapping[str, Sequence[Optional[float]]]
                         ) -> list[str]:
    """Ids of components whose world AABB overlaps ``world_bbox`` (``{"min": [x,y,z], "max": [x,y,z]}``).

    ``None`` entries are unbounded (the line-of-sight axis of a 2-D view region). Mirrored components also
    match through their mirror image. Result keeps spec order.
    """
    lo = np.array([-np.inf if v is None else v for v in world_bbox["min"]], dtype=float)
    hi = np.array([np.inf if v is None else v for v in world_bbox["max"]], dtype=float)
    hits: list[str] = []
    for comp in scene_spec_dict.get("components", []):
        if not comp.get("size"):
            continue
        rot = _rotation_matrix_xyz(comp.get("rotation_deg", [0, 0, 0]))
        half = np.abs(rot) @ (np.asarray(comp["size"], float) / 2.0)
        loc = np.asarray(comp.get("location", [0, 0, 0]), float)
        mirror = comp.get("mirror") or {}
        centers = [loc]
        for i, key in enumerate("xyz"):
            if mirror.get(key):
                centers += [c * np.where(np.arange(3) == i, -1.0, 1.0) for c in centers]
        if any(np.all(c - half <= hi) and np.all(c + half >= lo) for c in centers):
            hits.append(comp["id"])
    return hits


def describe_view_results(view_results: Mapping[str, ViewComparison], scene_spec_dict: Optional[Mapping[str, Any]]
                          ) -> dict[str, dict[str, Any]]:
    """Per view: ``bbox_delta_m``, ``largest_mismatch_region`` (with ``world_bbox``) and ``components_in_region``
    (ids of the scene's components overlapping that region; empty without a scene spec or a region)."""
    out: dict[str, dict[str, Any]] = {}
    for view, res in view_results.items():
        region = res.largest_mismatch_region
        hits = components_in_region(scene_spec_dict, region["world_bbox"]) if region and scene_spec_dict else []
        out[view] = {"bbox_delta_m": dict(res.bbox_delta_m), "largest_mismatch_region": region,
                     "components_in_region": hits}
    return out


def with_warnings(recommendation: str, warning_details: Sequence[str]) -> str:
    """Append soft-check warnings to a recommendation, e.g. ``... within tolerance; warnings: non_manifold (14 edges)``."""
    if not warning_details:
        return recommendation
    return f"{recommendation.rstrip('.')}; warnings: {', '.join(warning_details)}"
