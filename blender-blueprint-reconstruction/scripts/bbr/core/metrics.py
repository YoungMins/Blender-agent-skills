"""Silhouette comparison metrics (reference mask vs rendered mask)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .. import ImageError, get_logger
from .calibration import ViewCalibration
from .conventions import axis_index
from .imageops import (DISTANCE_CAP_PX, distance_to, label_components, mask_bbox, mask_edges)

log = get_logger(__name__)

DEFAULT_PASS_THRESHOLD = 0.05
EDGE_PAD_PX = 2


@dataclass
class ViewComparison:
    """Result of comparing one reference mask with one output mask.

    ``bbox_delta_px`` / ``_m``: signed, positive = output extends beyond the reference on that side.
    ``largest_mismatch_region``: ``{kind, area_px, bbox_px, center_world, world_bbox}`` where ``kind`` is
    ``missing`` (reference only) or ``excess`` (output only) and ``world_bbox`` = ``{min, max}`` with None on
    the axis along the line of sight; None when the masks are identical.
    """

    view: str
    camera_axis: str
    iou: float
    error: float
    bbox_delta_px: dict[str, float]
    bbox_delta_m: dict[str, float]
    ref_bbox: tuple[int, int, int, int]
    out_bbox: Optional[tuple[int, int, int, int]]
    width_ratio: float
    height_ratio: float
    edge_px: float
    m_per_px: float
    largest_mismatch_region: Optional[dict[str, Any]] = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return asdict(self)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two bool masks (1.0 when both are empty)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b)) / union


def bbox_delta_px(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    """Signed bbox side differences in px; positive = output extends beyond the reference."""
    rx0, ry0, rx1, ry1 = mask_bbox(ref)
    if not np.any(out):
        return {"left": -(rx1 - rx0) / 2.0, "right": -(rx1 - rx0) / 2.0,
                "top": -(ry1 - ry0) / 2.0, "bottom": -(ry1 - ry0) / 2.0}
    ox0, oy0, ox1, oy1 = mask_bbox(out)
    return {"left": float(rx0 - ox0), "right": float(ox1 - rx1), "top": float(ry0 - oy0), "bottom": float(oy1 - ry1)}


def edge_distance(ref: np.ndarray, out: np.ndarray, cap: int = DISTANCE_CAP_PX) -> float:
    """Mean symmetric boundary distance in px (chamfer approximation, capped at ``cap``)."""
    ea, eb = mask_edges(ref), mask_edges(out)
    if not ea.any() or not eb.any():
        return float(cap)
    both = ea | eb
    rows, cols = np.flatnonzero(both.any(axis=1)), np.flatnonzero(both.any(axis=0))
    pad = cap + EDGE_PAD_PX
    h, w = ref.shape
    sl = (slice(max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)),
          slice(max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)))
    ea, eb = ea[sl], eb[sl]
    d_ab = distance_to(eb, cap)[ea].mean()
    d_ba = distance_to(ea, cap)[eb].mean()
    return float((d_ab + d_ba) / 2.0)


def symmetry_score(mask: np.ndarray, axis_u_px: float) -> float:
    """IoU of the mask with its mirror image about the vertical line ``u = axis_u_px`` (pixel units)."""
    m = np.asarray(mask, dtype=bool)
    w = m.shape[1]
    src = np.floor(2.0 * axis_u_px - (np.arange(w) + 0.5)).astype(int)
    valid = (src >= 0) & (src < w)
    mirrored = np.zeros_like(m)
    mirrored[:, valid] = m[:, src[valid]]
    return iou(m, mirrored)


def _mismatch_region(ref: np.ndarray, out: np.ndarray, calib: ViewCalibration) -> Optional[dict[str, Any]]:
    diff = ref ^ out
    if not diff.any():
        return None
    labels, n = label_components(diff, full=True)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    areas[0] = 0
    best = int(np.argmax(areas))
    region = labels == best
    x0, y0, x1, y1 = mask_bbox(region)
    missing = int(np.count_nonzero(region & ref))
    excess = int(np.count_nonzero(region & out))
    corners = calib.px_to_plane(np.array([[x0, y0], [x1, y1]], dtype=float))
    lo, hi = corners.min(axis=0).tolist(), corners.max(axis=0).tolist()
    depth_axis = axis_index(calib.camera_axis)[0]
    lo[depth_axis] = hi[depth_axis] = None
    center = calib.px_to_plane(np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]))[0].tolist()
    center[depth_axis] = None
    return {"kind": "missing" if missing >= excess else "excess", "area_px": int(areas[best]),
            "bbox_px": [x0, y0, x1, y1], "center_world": center, "world_bbox": {"min": lo, "max": hi}}


def compare_view(ref_mask: np.ndarray, out_mask: np.ndarray, calib: ViewCalibration,
                 edge_cap_px: int = DISTANCE_CAP_PX) -> ViewComparison:
    """Compare a reference mask with an output mask rendered on the same pixel grid."""
    ref = np.asarray(ref_mask, dtype=bool)
    out = np.asarray(out_mask, dtype=bool)
    if ref.shape != out.shape:
        raise ImageError(f"mask shapes differ for view {calib.view}: {ref.shape} vs {out.shape}")
    value = iou(ref, out)
    delta = bbox_delta_px(ref, out)
    rbox = mask_bbox(ref)
    obox = mask_bbox(out) if out.any() else None
    rw, rh = rbox[2] - rbox[0], rbox[3] - rbox[1]
    wr = (obox[2] - obox[0]) / rw if obox else 0.0
    hr = (obox[3] - obox[1]) / rh if obox else 0.0
    return ViewComparison(
        view=calib.view, camera_axis=calib.camera_axis, iou=value, error=1.0 - value,
        bbox_delta_px=delta, bbox_delta_m={k: v * calib.m_per_px for k, v in delta.items()},
        ref_bbox=rbox, out_bbox=obox, width_ratio=wr, height_ratio=hr,
        edge_px=edge_distance(ref, out, edge_cap_px), m_per_px=calib.m_per_px, largest_mismatch_region=_mismatch_region(ref, out, calib))


def format_error_line(view: str, error: float) -> str:
    """``Front Error : 4.3%``."""
    return f"{view.capitalize()} Error : {error * 100:.1f}%"


def extent_gate(view_results: Mapping[str, ViewComparison], threshold: float) -> dict[str, Any]:
    """Bounding-box gate for a bbox blockout: pass when every view's width / height ratio (output extent /
    reference extent) is within ``1 +- threshold``. ``error`` = the largest ratio deviation."""
    ratios = {v: {"width_ratio": r.width_ratio, "height_ratio": r.height_ratio} for v, r in view_results.items()}
    deviations = [abs(x - 1.0) for pair in ratios.values() for x in pair.values()]
    error = max(deviations) if deviations else 1.0
    return {"mode": "bbox_fallback", "ratios": ratios, "error": error, "threshold": threshold,
            "passed": bool(deviations) and error <= threshold}


def aggregate(view_results: Mapping[str, ViewComparison], weights: Optional[Mapping[str, float]] = None,
              threshold: float = DEFAULT_PASS_THRESHOLD) -> dict[str, Any]:
    """Weighted mean of per-view errors; ``passed`` when ``total_error <= threshold``."""
    if not view_results:
        raise ImageError("no view results to aggregate")
    w = {v: float((weights or {}).get(v, 1.0)) for v in view_results}
    total_w = sum(w.values())
    total = sum(w[v] * r.error for v, r in view_results.items()) / total_w
    per_view = {v: {"iou": r.iou, "error": r.error, "error_pct": round(r.error * 100, 2),
                    "edge_px": r.edge_px, "width_ratio": r.width_ratio, "height_ratio": r.height_ratio,
                    "line": format_error_line(v, r.error)} for v, r in view_results.items()}
    return {"per_view": per_view, "total_error": total, "total_error_pct": round(total * 100, 2),
            "threshold": threshold, "passed": bool(total <= threshold)}
