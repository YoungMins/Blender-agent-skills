"""Continuous hull field (silhouette distance + superellipse cross-sections) on a voxel grid (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .. import CalibrationError, get_logger
from .calibration import ViewCalibration
from .conventions import axis_index, view_frame
from .distance import signed_distance
from .imageops import mask_bbox

log = get_logger(__name__)

DEFAULT_MARGIN = 0.03
DEFAULT_TOP_CONSISTENCY_TOL = 0.15
MIN_RESOLUTION = 8
_TOP_BOTTOM = ("+Z", "-Z")
CROSS_SECTIONS = ("superellipse", "rectangle")
DEFAULT_SLICE_COHERENCE = 0.03   # hull param default: Z thickness (share of the height) below which a spike cannot inflate a ring
SPIKE_MAX_FRAC = 0.3             # a bulge sticking out by more than this share of the run half-width is kept (hat brim, arm)
FIN_BETA = 0.25                  # sharpness of the raw-extent fin around the carve planes (see _cross_section_term)


@dataclass
class VoxelGrid:
    """Occupancy ``[nx, ny, nz]``; ``origin`` = world min corner; cubic cells of size ``cell``."""

    occ: np.ndarray
    origin: np.ndarray
    cell: float

    def centers(self, axis: int) -> np.ndarray:
        """World coordinates of voxel centres along world ``axis`` (0=X, 1=Y, 2=Z)."""
        return self.origin[axis] + (np.arange(self.occ.shape[axis]) + 0.5) * self.cell


def _broadcast(table: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """Reshape a ``[n_right, n_up]`` table so it broadcasts against ``[nx, ny, nz]``."""
    f = view_frame(calib.camera_axis)
    ri = axis_index(f.right_axis)[0]
    ui = axis_index(f.up_axis)[0]
    if ri > ui:
        table = table.T
    shape = [1, 1, 1]
    shape[min(ri, ui)], shape[max(ri, ui)] = table.shape
    return table.reshape(shape)


def _grid_geometry(dims: Mapping[str, float], resolution: int, margin: float) -> tuple[np.ndarray, float, tuple]:
    if resolution < MIN_RESOLUTION:
        raise CalibrationError(f"resolution must be >= {MIN_RESOLUTION}")
    w, d, h = float(dims["width"]), float(dims["depth"]), float(dims["height"])
    pad = margin * max(w, d, h)
    lo = np.array([-w / 2 - pad, -d / 2 - pad, -pad])
    hi = np.array([w / 2 + pad, d / 2 + pad, h + pad])
    cell = float((hi - lo).max() / resolution)
    shape = tuple(int(np.ceil((hi - lo)[i] / cell - 1e-9)) for i in range(3))
    return lo, cell, shape


def _bounds_geometry(bounds: tuple[Sequence[float], Sequence[float]], resolution: int) -> tuple[np.ndarray, float, tuple]:
    """Grid over explicit world bounds ``(lo, hi)``: cubic cells, ``resolution`` along the longest side."""
    if resolution < MIN_RESOLUTION:
        raise CalibrationError(f"resolution must be >= {MIN_RESOLUTION}")
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or (hi <= lo).any():
        raise CalibrationError(f"bounds must be (lo, hi) 3-vectors with hi > lo, got {bounds!r}")
    cell = float((hi - lo).max() / resolution)
    shape = tuple(int(np.ceil((hi - lo)[i] / cell - 1e-9)) for i in range(3))
    return lo, cell, shape


def _usable_views(masks, calibs, dims, carve_views, top_consistency_tol) -> list[str]:
    """Carve views with mask + calibration; top/bottom views only when their aspect matches width/depth."""
    out = []
    for name in carve_views:
        if name not in masks or name not in calibs:
            log.warning("carve view %s skipped: missing mask or calibration", name)
            continue
        if calibs[name].camera_axis in _TOP_BOTTOM:
            x0, y0, x1, y1 = mask_bbox(masks[name])
            aspect, expect = (x1 - x0) / (y1 - y0), float(dims["width"]) / float(dims["depth"])
            if abs(aspect / expect - 1.0) > top_consistency_tol:
                log.warning("view %s aspect %.3f vs width/depth %.3f exceeds tol %.2f; validation only",
                            name, aspect, expect, top_consistency_tol)
                continue
        out.append(name)
    return out


def _view_uv(calib: ViewCalibration, coords: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Continuous pixel coordinates (u per right-axis sample, v per up-axis sample) of a view's two world axes."""
    f = view_frame(calib.camera_axis)
    ri, rs = axis_index(f.right_axis)
    ui, us = axis_index(f.up_axis)
    return (calib.origin_px[0] + rs * coords[ri] / calib.m_per_px,
            calib.origin_px[1] - us * coords[ui] / calib.m_per_px)


def _bilinear_table(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` (pixel centres at +0.5) on the grid ``u`` x ``v`` -> ``[len(u), len(v)]``.

    Coordinates outside the image are clamped; the excess distance (px) is added so outside stays positive.
    """
    h, w = img.shape
    x = np.clip(u - 0.5, 0, w - 1)
    y = np.clip(v - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[None, :]
    top = img[y0[None, :], x0[:, None]] * (1 - fx) + img[y0[None, :], x1[:, None]] * fx
    bot = img[y1[None, :], x0[:, None]] * (1 - fx) + img[y1[None, :], x1[:, None]] * fx
    excess_u = np.maximum(0, np.maximum(-(u - 0.5), (u - 0.5) - (w - 1)))
    excess_v = np.maximum(0, np.maximum(-(v - 0.5), (v - 0.5) - (h - 1)))
    return top * (1 - fy) + bot * fy + excess_u[:, None] + excess_v[None, :]


def _row_runs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel of each bool row: start and exclusive end column of the run containing it (valid inside runs)."""
    w = rows.shape[1]
    cols = np.arange(w)[None, :]
    prev = np.pad(rows, ((0, 0), (1, 0)))[:, :-1]
    nxt = np.pad(rows, ((0, 0), (0, 1)))[:, 1:]
    start = np.maximum.accumulate(np.where(rows & ~prev, cols, -1), axis=1)
    end = np.minimum.accumulate(np.where(rows & ~nxt, cols + 1, w + 1)[:, ::-1], axis=1)[:, ::-1]
    return start, end


def _run_extents(mask: np.ndarray, calib: ViewCalibration, coords: Sequence[np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per (horizontal voxel column, Z slice): centre and half-extent (m, along the view's horizontal world axis)
    of the mask run containing it; half-extent 0 where the voxel centre is outside every run."""
    f = view_frame(calib.camera_axis)
    rs = axis_index(f.right_axis)[1]
    u, v = _view_uv(calib, coords)
    h, w = mask.shape
    ci = np.floor(u).astype(int)
    col_ok = (ci >= 0) & (ci < w)
    ci = np.clip(ci, 0, w - 1)
    # average the run extents over the pixel rows covered by each Z cell (removes 1-px row jitter / banding)
    half_rows = int(max(0, round(1.0 * float(np.abs(np.diff(v)).mean() if len(v) > 1 else 1.0) - 0.5)))
    wsum = np.zeros((len(v), len(ci)))
    sa = np.zeros_like(wsum)
    sb = np.zeros_like(wsum)
    for off in range(-half_rows, half_rows + 1):
        rows_i = np.floor(v).astype(int) + off
        row_ok = (rows_i >= 0) & (rows_i < h)
        rows = mask[np.clip(rows_i, 0, h - 1)] & row_ok[:, None]        # [nz, W]
        start, end = _row_runs(rows)
        ins = rows[:, ci] & col_ok[None, :]
        wsum += ins
        sa += np.where(ins, start[:, ci], 0)
        sb += np.where(ins, end[:, ci], 0)
    inside = wsum > 0
    s = sa / np.maximum(wsum, 1)
    e = sb / np.maximum(wsum, 1)
    xa = rs * (s - calib.origin_px[0]) * calib.m_per_px
    xb = rs * (e - calib.origin_px[0]) * calib.m_per_px
    center = (xa + xb) / 2.0
    half = np.abs(xb - xa) / 2.0
    return np.where(inside, center, 0.0).T, np.where(inside, half, 0.0).T   # [n_c, nz]


def _open_along_z(vals: np.ndarray, inside: np.ndarray, n: int) -> np.ndarray:
    """Grey-scale opening (erosion then dilation, flat element of ``n`` slices) of ``vals`` ``[n_c, nz]`` along Z.

    Only ``inside`` samples take part (a column's outside slices are neither eroded into nor grown from), so a bump
    thinner than ``n`` slices is flattened to its neighbours while steps, plateaus and features that only exist in a
    few slices keep their value. Never exceeds ``vals``.
    """
    nz = vals.shape[1]
    pad = n - 1
    big = np.pad(np.where(inside, vals, np.inf), ((0, 0), (0, pad)), constant_values=np.inf)
    eroded = np.min([big[:, i:i + nz] for i in range(n)], axis=0)
    eroded = np.pad(np.where(inside, eroded, -np.inf), ((0, 0), (pad, 0)), constant_values=-np.inf)
    return np.max([eroded[:, i:i + nz] for i in range(n)], axis=0)


def _regularize_extents(center: np.ndarray, half: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Coherent-along-Z version of the run extents ``[n_c, nz]``: no ring per spike in the cross-section term.

    Both run edges (``center -+ half``) are grey-opened along Z: a bulge thinner than ``n`` slices is flattened to its
    neighbours, unless it sticks out by more than ``SPIKE_MAX_FRAC`` of the run half-width (a part: hat brim, arm).
    Columns inside only a few slices (the tip of a spike) keep their own extents; edges that would cross keep their
    original values.
    """
    inside = half > 0
    if n < 2 or not inside.any():
        return center, half
    lo0, hi0 = center - half, center + half
    lo = np.where(inside, -_open_along_z(-lo0, inside, n), 0.0)
    hi = np.where(inside, _open_along_z(hi0, inside, n), 0.0)
    big_bulge = SPIKE_MAX_FRAC * (hi - lo) / 2.0
    lo, hi = np.where(inside & (lo - lo0 > big_bulge), lo0, lo), np.where(inside & (hi0 - hi > big_bulge), hi0, hi)
    ok = inside & (hi > lo)
    return np.where(ok, (lo + hi) / 2.0, center), np.where(ok, (hi - lo) / 2.0, half)


def _slice_runs(center: np.ndarray, half: np.ndarray, tol: float) -> list[tuple[np.ndarray, float, float]]:
    """Split one Z slice's voxel columns into runs: ``(column indices, centre, half-extent)`` (inside columns only).

    A new run starts at a gap in the inside columns or where centre / half-extent jump by more than ``tol``.
    """
    idx = np.flatnonzero(half > 0)
    if idx.size == 0:
        return []
    brk = np.flatnonzero((np.diff(idx) > 1) | (np.abs(np.diff(center[idx])) > tol)
                         | (np.abs(np.diff(half[idx])) > tol)) + 1
    return [(g, float(np.median(center[g])), float(np.median(half[g]))) for g in np.split(idx, brk)]


def select_run_pairs(xa: Sequence[float], yb: Sequence[float], max_aspect: float
                     ) -> list[tuple[int, int, float, float]]:
    """Deterministic x-run / y-run pairing for one Z slice -> ``(i, j, a_eff, b_eff)`` half-extents to use.

    Pairs with ``max/min <= max_aspect`` are kept as is; every run lacking a good partner gets its best-scoring
    partner (score ``|log(a/b)|``) with the long half-extent clamped to ``max_aspect`` x the short one (the
    superellipse then still projects onto the whole run on the short axis). A 1 x 1 slice has no ambiguity.
    """
    nx, ny = len(xa), len(yb)
    if nx == 0 or ny == 0:
        return []
    if nx == 1 and ny == 1:
        return [(0, 0, xa[0], yb[0])]
    score = np.abs(np.log(np.asarray(xa)[:, None] / np.asarray(yb)[None, :]))
    keep = score <= np.log(max_aspect) + 1e-9
    clamped = np.zeros_like(keep)
    for i in np.flatnonzero(~keep.any(axis=1)):
        j = int(np.argmin(score[i]))
        keep[i, j] = clamped[i, j] = True
    for j in np.flatnonzero(~keep.any(axis=0)):
        i = int(np.argmin(score[:, j]))
        keep[i, j] = clamped[i, j] = True
    out = []
    for i, j in zip(*np.nonzero(keep)):
        a, b = float(xa[i]), float(yb[j])
        if clamped[i, j]:
            a, b = min(a, max_aspect * b), min(b, max_aspect * a)
        out.append((int(i), int(j), a, b))
    return out


def _superellipse_radius(dx: np.ndarray, dy: np.ndarray, exponent: float) -> np.ndarray:
    if exponent == 2.0:
        return np.sqrt(dx * dx + dy * dy)
    return (dx ** exponent + dy ** exponent) ** (1.0 / exponent)


def _cross_section_term(masks, calibs, front: str, side: str, coords, exponent: float, big: float,
                        max_aspect: float = 4.0, open_slices: int = 0) -> np.ndarray:
    """Superellipse cross-section field (m) from front-type x-runs and side-type y-runs, per Z slice.

    Only the run pairs chosen by :func:`select_run_pairs` get a superellipse (union of pairs); this avoids phantom
    slabs from pairing e.g. a wide torso x-run with a thin sword y-run.

    ``open_slices`` > 1 sizes the superellipses from Z-regularised run extents (:func:`_regularize_extents`) instead
    of the raw ones, so a spike row cannot inflate a whole ring. The raw superellipse is kept only as a thin fin
    around the two carve planes (penalty ``FIN_BETA * min(a, b) * |sin 2 phi|``, ``phi`` = angle in the normalised
    cross-section), so the hull still reaches the outline of both reference views.
    """
    cx0, ax0 = _run_extents(masks[front], calibs[front], coords)         # [nx, nz]
    cy0, ay0 = _run_extents(masks[side], calibs[side], coords)           # [ny, nz]
    cx, ax = _regularize_extents(cx0, ax0, open_slices)
    cy, ay = _regularize_extents(cy0, ay0, open_slices)
    regularised = open_slices >= 2
    cell = float(coords[0][1] - coords[0][0])
    out = np.full((len(coords[0]), len(coords[1]), len(coords[2])), big, dtype=np.float32)
    for k in range(out.shape[2]):
        xr = _slice_runs(cx[:, k], ax[:, k], cell)
        yr = _slice_runs(cy[:, k], ay[:, k], cell)
        for i, j, a, b in select_run_pairs([r[2] for r in xr], [r[2] for r in yr], max_aspect):
            colx, ccx, hx = xr[i]
            coly, ccy, hy = yr[j]
            ux = np.abs(coords[0][colx] - ccx)[:, None]
            uy = np.abs(coords[1][coly] - ccy)[None, :]
            term = (_superellipse_radius(ux / a, uy / b, exponent) - 1.0) * min(a, b)
            if regularised:
                a0 = float(np.median(ax0[colx, k])) * a / hx                 # raw extents, clamped like a, b
                b0 = float(np.median(ay0[coly, k])) * b / hy
                raw = (_superellipse_radius(np.abs(coords[0][colx] - float(np.median(cx0[colx, k])))[:, None] / a0,
                                            np.abs(coords[1][coly] - float(np.median(cy0[coly, k])))[None, :] / b0,
                                            exponent) - 1.0) * min(a0, b0)
                fin = FIN_BETA * min(a0, b0) * np.abs(np.sin(2.0 * np.arctan2(uy / b, ux / a)))
                term = np.minimum(term, raw + fin)
            sub = out[colx[:, None], coly[None, :], k]
            out[colx[:, None], coly[None, :], k] = np.minimum(sub, term.astype(np.float32))
    return out


def build_hull_field(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration],
                     dims: Mapping[str, float], resolution: int, carve_views: Sequence[str] = (),
                     cross_section: str = "superellipse", exponent: float = 2.0, margin: float = DEFAULT_MARGIN,
                     top_consistency_tol: float = DEFAULT_TOP_CONSISTENCY_TOL,
                     pair_max_aspect: float = 4.0, bounds: tuple[Sequence[float], Sequence[float]] | None = None,
                     slice_coherence: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Continuous hull field ``[nx, ny, nz]`` float32 in meters (negative inside) sampled at voxel centres.

    Silhouette term: per carve view the signed distance to the mask boundary (bilinear at the voxel's projected
    pixel, times ``m_per_px``); the field is the max over views (intersection). With ``cross_section="superellipse"``
    and both a front-type (+-Y) and a side-type (+-X) carve view, each Z slice is additionally rounded: voxel
    (x, y) in front run i x side run j gets ``(r - 1) * min(a, b)`` with ``r`` the superellipse radius of exponent
    ``exponent`` inscribed in the run rectangle (centre ``cx, cy``; half extents ``a, b``); field = max of both.
    Runs are paired by :func:`select_run_pairs` (size-similar pairs within ``pair_max_aspect``, plus coverage pairs
    with the long axis clamped) instead of all-pairs.
    ``bounds=(lo, hi)`` (world, m) replaces the grid derived from ``dims`` + ``margin`` (``resolution`` cells along the
    longest side of the box): a part's own, finer grid (see :mod:`bbr.core.layers`).
    ``slice_coherence`` (share of ``dims["height"]``, 0 = off, hull default :data:`DEFAULT_SLICE_COHERENCE`): the run
    extents that size the cross-sections are made coherent along Z (:func:`_regularize_extents`), so a spiky
    silhouette (hair, fur) cannot inflate a whole ring per spike (a stack of rings, "lampshade"); the spikes stay as
    thin fins in the two carve planes and in the silhouette term (see :func:`_cross_section_term`).
    Returns ``(field, origin (min corner), cell)``.
    """
    if cross_section not in CROSS_SECTIONS:
        raise CalibrationError(f"cross_section must be one of {CROSS_SECTIONS}, got {cross_section!r}")
    if exponent <= 0:
        raise CalibrationError("exponent must be > 0")
    if pair_max_aspect < 1:
        raise CalibrationError("pair_max_aspect must be >= 1")
    if slice_coherence < 0:
        raise CalibrationError("slice_coherence must be >= 0")
    lo, cell, shape = _grid_geometry(dims, resolution, margin) if bounds is None else _bounds_geometry(bounds, resolution)
    coords = [lo[i] + (np.arange(shape[i]) + 0.5) * cell for i in range(3)]
    used = _usable_views(masks, calibs, dims, carve_views, top_consistency_tol)
    if not used:
        raise CalibrationError("no carve view could be used")
    big = float((np.array(shape) * cell).max())
    field = np.full(shape, -big, dtype=np.float32)
    for name in used:
        calib = calibs[name]
        u, v = _view_uv(calib, coords)
        table = _bilinear_table(signed_distance(masks[name]), u, v) * calib.m_per_px
        np.maximum(field, _broadcast(table.astype(np.float32), calib), out=field)
    if cross_section == "superellipse":
        front = next((n for n in used if calibs[n].camera_axis in ("-Y", "+Y")), None)
        side = next((n for n in used if calibs[n].camera_axis in ("-X", "+X")), None)
        if front and side:
            height = float(dims["height"])
            np.maximum(field, _cross_section_term(masks, calibs, front, side, coords, float(exponent), big,
                                                     float(pair_max_aspect), int(float(slice_coherence) * height / cell)), out=field)
        else:
            log.info("superellipse cross-section needs front- and side-type carve views; silhouette term only")
    log.info("hull field: grid %s, cell %.5f m, %d inside cells from views %s", shape, cell,
             int((field < 0).sum()), used)
    return field, lo, cell
