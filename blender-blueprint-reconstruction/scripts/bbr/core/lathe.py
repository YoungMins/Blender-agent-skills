"""Measure a body of revolution about Z from the reference masks (numpy only): the profile of a mushroom cap, vase, lamp,
bottle or column is read from the front silhouette, its elliptic depth from the side silhouette; nothing is typed.

Per pixel row of the front mask (restricted to a pixel box or a part's label mask) the larger half-width about the axis
(:func:`half_widths`) is the radius at that height; the same rows of the side mask give the axis position (y) and the depth
ratio (side half-width over front half-width, median over the rows). The polyline is simplified to a few points (see
:func:`simplify`) and handed to :func:`bbr.core.partgeom.lathe_mesh`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .. import BBRError
from .calibration import ViewCalibration
from .conventions import view_frame

DEFAULT_TOL_PX = 1.0         # simplification tolerance (reference pixels)
DEFAULT_MAX_POINTS = 40      # profile points kept at most (the tolerance grows until it fits)
MIN_ROWS = 3
SMOOTH_ROWS = 7              # rows of the median filter over the half-widths (3 for a region under 21 rows): spikes up to 3 rows vanish
RATIO_ROWS_MIN_FRAC = 0.3    # depth ratio: rows whose front half-width is below this share of the widest are ignored
RATIO_LIMITS = (0.2, 5.0)


@dataclass
class LatheMeasure:
    """Result of :func:`measure_lathe`: ``profile`` ``[[z, r], ...]`` bottom to top in meters (world z), ``axis_xy`` world
    position of the axis, ``depth_ratio`` (Y extent over X extent), ``info`` (rows, tolerance, sources) for the report."""

    profile: list[list[float]]
    axis_xy: tuple[float, float]
    depth_ratio: float
    info: dict[str, Any] = field(default_factory=dict)


def _window(mask: np.ndarray, box: Optional[Sequence[float]]) -> np.ndarray:
    """``mask`` cut to the pixel box ``[x0, y0, x1, y1]`` (None = unchanged)."""
    region = np.asarray(mask, dtype=bool)
    if box is None:
        return region
    if len(box) != 4:
        raise BBRError(f"bbox_px needs [x0, y0, x1, y1], got {list(box)}")
    x0, y0, x1, y1 = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
    out = np.zeros_like(region)
    out[max(y0, 0):y1, max(x0, 0):x1] = region[max(y0, 0):y1, max(x0, 0):x1]
    return out


def row_extents(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, left, right)`` of every non-empty row: the outermost set pixels as pixel edges (``right`` exclusive)."""
    rows = np.flatnonzero(region.any(axis=1))
    sub = region[rows]
    left = np.argmax(sub, axis=1).astype(float)
    right = region.shape[1] - np.argmax(sub[:, ::-1], axis=1).astype(float)
    return rows, left, right


def half_widths(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    """``(radius per row, axis column)``: the axis is the median centre of the widest rows (at least 90 % of the 95th-percentile half-width, so a few stray rows do not vote:
    rim of a cap, belly of a vase, free of anything stuck on one side) and the radius of a row is its LARGER side about that
    axis, so a label cut, an occluder or a missing flank on ONE side does not shrink the profile."""
    hw = (right - left) / 2.0
    axis = float(np.median(((left + right) / 2.0)[hw >= 0.9 * np.percentile(hw, 95)]))
    return np.maximum(axis - left, right - axis), axis


def simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker: the subset of the polyline ``points (N,2)`` that keeps every dropped point within ``tol``."""
    pts = np.asarray(points, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = pts[b] - pts[a]
        n = float(np.hypot(*d))
        mid = pts[a + 1:b] - pts[a]
        dist = np.hypot(mid[:, 0], mid[:, 1]) if n < 1e-12 else np.abs(d[0] * mid[:, 1] - d[1] * mid[:, 0]) / n
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return pts[keep]


def _check_views(front: ViewCalibration, side: Optional[ViewCalibration]) -> None:
    """Raise unless ``front`` looks along Y and ``side`` along X, both with Z up."""
    for calib, wanted in ((front, ("+X", "-X")), (side, ("+Y", "-Y"))):
        if calib is not None and view_frame(calib.camera_axis).right_axis not in wanted:
            raise BBRError(f"a lathe about Z reads its profile from a front/back view (camera on Y) and its depth from a "
                           f"side view (camera on X); view {calib.view!r} looks along {calib.camera_axis}")
        if calib is not None and abs(float(view_frame(calib.camera_axis).up[2])) < 0.99:
            raise BBRError(f"view {calib.view!r} does not have world Z up")


def measure_lathe(front_mask: np.ndarray, front_calib: ViewCalibration, side_mask: Optional[np.ndarray] = None,
                  side_calib: Optional[ViewCalibration] = None, bbox_px: Optional[Sequence[float]] = None,
                  side_bbox_px: Optional[Sequence[float]] = None, tol_px: float = DEFAULT_TOL_PX,
                  max_points: int = DEFAULT_MAX_POINTS, depth_ratio: Optional[float] = None,
                  bottom_z: Optional[float] = None) -> LatheMeasure:
    """Profile, axis and depth ratio of the body of revolution drawn in ``front_mask`` (a silhouette cut to ``bbox_px``, or a
    part's label mask) and ``side_mask`` (same: ``side_bbox_px``; only the rows of the front part count).

    ``depth_ratio`` overrides the measured one (also without a side view; then 1.0 = circular). The axis y comes from the
    side view, else 0. ``bottom_z``: extend the lowest profile point straight down to this height (0 = the ground), for a
    base whose silhouette curls in at the bottom because the reference camera looked slightly down at its round rim. Raises BBRError for a region with fewer than :data:`MIN_ROWS` rows."""
    _check_views(front_calib, side_calib)
    region = _window(front_mask, bbox_px)
    rows, left, right = row_extents(region)
    if len(rows) < MIN_ROWS:
        raise BBRError(f"lathe profile: the front region has {len(rows)} pixel rows (need >= {MIN_ROWS}); check bbox_px / the label")
    m = front_calib.m_per_px
    hw, axis_col = half_widths(left, right)
    k = SMOOTH_ROWS if len(hw) >= 3 * SMOOTH_ROWS else 3                           # median over k rows (mirrored at the ends): a label's
    padded = np.pad(hw, k // 2, mode="reflect")                                    # ragged edge or a stray stripe is no profile feature
    hw = np.median(np.stack([padded[i:i + len(hw)] for i in range(k)]), axis=0)
    axis_x = float(front_calib.px_to_plane(np.array([[axis_col, 0.0]]))[0, 0])
    # polyline in (row-edge v, half-width) pixels: exact top and bottom edge of the silhouette, row centres between
    v = np.concatenate([[rows[0]], rows + 0.5, [rows[-1] + 1.0]])
    r_px = np.concatenate([[hw[0]], hw, [hw[-1]]])
    poly = np.stack([v, r_px], axis=1)
    tol = float(tol_px)
    kept = simplify(poly, tol)
    while len(kept) > max_points:
        tol *= 1.5
        kept = simplify(poly, tol)
    z = front_calib.px_to_plane(np.stack([np.zeros(len(kept)), kept[:, 0]], axis=1))[:, 2]
    profile = [[round(float(zz), 5) + 0.0, round(float(rr) * m, 5)] for zz, rr in zip(z[::-1], kept[::-1, 1])]
    if bottom_z is not None and float(bottom_z) < profile[0][0]:
        profile.insert(0, [round(float(bottom_z), 5) + 0.0, profile[0][1]])
    info: dict[str, Any] = {"rows": int(len(rows)), "points": len(profile), "tol_px": round(tol, 3),
                            "front_px": [int(left.min()), int(rows[0]), int(right.max()), int(rows[-1] + 1)]}
    axis_y, ratio, ratio_src = 0.0, 1.0, "default"
    if side_mask is not None and side_calib is not None:
        sreg = _window(side_mask, side_bbox_px)
        s_rows, s_left, s_right = row_extents(sreg) if sreg.any() else (np.array([], int), np.array([]), np.array([]))
        ms = side_calib.m_per_px
        ratios = []
        if len(s_rows):
            s_hw, s_axis = half_widths(s_left, s_right)
            lookup = {int(r): w for r, w in zip(s_rows, s_hw)}
            for r, h in zip(rows, hw):
                zr = float(front_calib.px_to_plane(np.array([[0.0, r + 0.5]]))[0, 2])
                sv = int(np.floor(side_calib.origin_px[1] - zr / ms))
                if sv in lookup and h >= RATIO_ROWS_MIN_FRAC * hw.max():
                    ratios.append((lookup[sv] * ms) / (h * m))
        if ratios:
            ratio, ratio_src = float(np.clip(np.median(ratios), *RATIO_LIMITS)), "side"
            axis_y = float(side_calib.px_to_plane(np.array([[s_axis, 0.0]]))[0, 1])
            info["side_rows"] = len(ratios)
        else:
            info["warning"] = "the side region shares no rows with the front part: depth_ratio = 1, axis y = 0"
    if depth_ratio is not None:
        ratio, ratio_src = float(depth_ratio), "explicit"
    info["depth_ratio_source"] = ratio_src
    return LatheMeasure(profile, (round(axis_x, 5), round(axis_y, 5)), round(ratio, 4), info)
