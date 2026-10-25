"""Overall-dimension check helpers (numpy only): the tolerance by reference quality and the attribution of a failure.

The pipeline compares the model's bounding box with the target width / depth / height. When one fails, the number alone
says nothing about what to change; :func:`attribute` names the component whose extreme sets the model's edge on the
failing axis, and by how much that edge is off the reference edge (from the reference masks, :func:`reference_extents`).
The usual culprit is an outer part (a fin, a wing tip, a tail) that the label fit shrank a little: pin it with
``fit_locked``, or fix its anchors.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .calibration import ViewCalibration
from .conventions import normalize_axis
from .imageops import mask_bbox

DIMENSION_TOLERANCE = 0.03            # synthetic / blueprint references
DIMENSION_TOLERANCE_AI = 0.05         # AI-generated references (the views disagree by a few percent by nature)
AXIS_OF_DIMENSION = {"width": 0, "depth": 1, "height": 2}
SIDE_NAMES = {0: ("-X", "+X"), 1: ("-Y", "+Y"), 2: ("-Z", "+Z")}
SAME_EXTREME_M = 0.003                # parts within this of the extreme are named together


def is_ai_generated(reference_quality: Mapping[str, Any], calibration_modes: Mapping[str, str]) -> bool:
    """True for AI-generated references: ``reference_quality.ai_generated``, ``orthographic: false``, or views cut from one
    turnaround sheet (calibration mode ``sheet``)."""
    if reference_quality.get("synthetic"):
        return False
    if "ai_generated" in reference_quality:
        return bool(reference_quality["ai_generated"])
    return reference_quality.get("orthographic", True) is False or any(m == "sheet" for m in calibration_modes.values())


def resolve_tolerance(configured: Optional[float], reference_quality: Mapping[str, Any],
                      calibration_modes: Mapping[str, str]) -> tuple[float, str]:
    """``(tolerance, why)``: ``workflow.dimension_tolerance`` when set, else 3 % (synthetic / blueprint) or 5 % (AI-generated)."""
    if configured is not None:
        return float(configured), "workflow.dimension_tolerance"
    if is_ai_generated(reference_quality, calibration_modes):
        return DIMENSION_TOLERANCE_AI, "AI-generated references"
    return DIMENSION_TOLERANCE, "synthetic / blueprint references"


def reference_extents(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration],
                      views: Optional[list[str]] = None) -> dict[int, tuple[float, float]]:
    """World ``(lo, hi)`` of the reference silhouette per axis index (0 X, 1 Y, 2 Z): the median over the views that
    measure the axis (front / back give X and Z, side views Y and Z). Top / bottom views are skipped."""
    est: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
    for view in views or list(masks):
        if view not in calibs or view not in masks or not np.asarray(masks[view]).any():
            continue
        calib = calibs[view]
        if normalize_axis(calib.camera_axis) in ("+Z", "-Z"):
            continue
        x0, y0, x1, y1 = mask_bbox(masks[view])
        corners = calib.px_to_plane(np.array([[x0, y0], [x1, y1]], dtype=float))
        depth = "XYZ".index(normalize_axis(calib.camera_axis)[1])
        for axis in range(3):
            if axis != depth:
                est[axis].append((float(corners[:, axis].min()), float(corners[:, axis].max())))
    return {a: (float(np.median([e[0] for e in v])), float(np.median([e[1] for e in v]))) for a, v in est.items() if v}


def attribute(relative_error: Mapping[str, float], tolerance: float, boxes: Mapping[str, tuple[np.ndarray, np.ndarray]],
              reference: Mapping[int, tuple[float, float]], hull_ids: Optional[set[str]] = None) -> list[dict[str, Any]]:
    """For every dimension off by more than ``tolerance``: which component sets the model's edge that is farthest off the
    reference edge. ``boxes`` = ``{component id: (min xyz, max xyz)}`` of the built meshes, ``reference`` =
    :func:`reference_extents`. Entries: ``{dimension, error, side, part, parts, off_m, direction, text}`` with ``direction``
    ``short`` (the model edge is inside the reference edge) or ``over``; ``text`` is the one-line message."""
    out: list[dict[str, Any]] = []
    if not boxes:
        return out
    for dim, err in relative_error.items():
        axis = AXIS_OF_DIMENSION.get(dim)
        if axis is None or abs(err) <= tolerance or axis not in reference:
            continue
        ref_lo, ref_hi = reference[axis]
        lo_part = min(boxes, key=lambda k: boxes[k][0][axis])
        hi_part = max(boxes, key=lambda k: boxes[k][1][axis])
        model_lo, model_hi = float(boxes[lo_part][0][axis]), float(boxes[hi_part][1][axis])
        short = {0: model_lo - ref_lo, 1: ref_hi - model_hi}               # > 0: the model edge is inside the reference edge
        side = max(short, key=lambda s: short[s] if err < 0 else -short[s])
        off = short[side] if err < 0 else -short[side]
        if off <= 0:                                                       # the reference masks do not explain the error: no attribution
            continue
        edge = model_lo if side == 0 else model_hi
        pick = (lambda k: float(boxes[k][0][axis])) if side == 0 else (lambda k: float(boxes[k][1][axis]))
        parts = sorted((k for k in boxes if abs(pick(k) - edge) <= SAME_EXTREME_M),
                       key=lambda k: abs(pick(k) - edge))[:3]
        main = parts[0]
        direction = "short of" if err < 0 else "beyond"
        entry = {"dimension": dim, "error": round(float(err), 4), "side": SIDE_NAMES[axis][side], "part": main, "parts": parts,
                 "off_m": round(float(abs(off)), 5), "direction": "short" if err < 0 else "over"}
        names = ", ".join(parts)
        advice = (f"the hull sets it: check the silhouette, calibration and the carve views" if main in (hull_ids or set())
                  else f"consider fit_locked on {main} position" if err < 0 else f"check {main}: its anchors or fit stick out")
        entry["text"] = (f"{dim} {err * 100:+.1f}% (limit {tolerance * 100:g}%); model extreme set by {names} "
                         f"({entry['side']} side), {abs(off) * 1000:.0f} mm {direction} the reference edge; {advice}")
        out.append(entry)
    return out


def format_check(check: Mapping[str, Any]) -> list[str]:
    """Lines for a dimension check (``validate_dimensions`` result): one summary line, then one per attributed failure."""
    if "relative_error" not in check:
        return [f"Dimensions : {check.get('reason', 'not measurable')}"]
    tol = float(check.get("tolerance", DIMENSION_TOLERANCE))
    parts = ", ".join(f"{k} {v * 100:+.1f}%" for k, v in check["relative_error"].items() if check["target"].get(k))
    lines = [f"Dimensions : {parts} (limit {tol * 100:g}%) {'PASSED' if check.get('passed') else 'FAILED'}"]
    lines += [f"  {a['text']}" for a in check.get("limiting") or []]
    return lines
