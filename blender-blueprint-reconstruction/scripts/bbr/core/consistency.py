"""Reference consistency gate: do the reference views describe one and the same object?

Pure numpy. Opposite orthographic views of one object have mirror-image silhouettes, so the mirrored IoU of an
opposite pair (front/back, side/right) is a calibration-aware measure of how far the images disagree. Whatever
the reconstruction does, a validate view cannot match better than that agreement: ``floor = 1 - pair IoU``.

Everything runs on world-aligned grids (cells <= ``GRID_MAX_PX`` per side) built from the view calibrations, so
the numbers reflect what the pipeline will compare. Thresholds live in :class:`ConsistencyThresholds`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .calibration import ViewCalibration
from .conventions import view_frame
from .imageops import dilate, erode, label_components
from .symmetry import front_symmetry

GRID_MAX_PX = 512                       # masks are block-averaged down to at most this many px per side
BAND_COUNT = 8                          # row bands for the per-band core width comparison
ALIGN_SCALE_RANGE = 0.06                # rigid alignment search: +-6 % scale ...
ALIGN_SHIFT_H = 0.06                    # ... +-6 % of the height sideways ...
ALIGN_SHIFT_Z = 0.03                    # ... +-3 % of the height vertically
ALIGN_COARSE_PX = 128                   # coarse search grid
_ROLE_TIERS = (("carve", "color", "validate"), ("color", "validate"), ())
_VIEW_PRIORITY = ("front", "side", "left", "right", "back", "top", "bottom")  # who keeps its roles in a failed pair


@dataclass
class ConsistencyThresholds:
    """Calibratable thresholds of the gate."""

    carve_min_iou: float = 0.95            # pair IoU needed to keep the carve role
    color_validate_min_iou: float = 0.92   # pair IoU needed to keep color / validate; below: drop the view
    scale_drift_tol: float = 0.02          # relative core-height deviation between views
    shift_tol: float = 0.01                # baseline deviation, fraction of the core height
    accessory_blob_frac: float = 0.01      # one-sided XOR blob area / union area
    mirrored_margin: float = 0.03          # unmirrored IoU must beat the mirrored one by this much
    core_open_frac: float = 0.03           # morphological opening radius (fraction of height) removing thin protrusions
    bbox_agree_tol: float = 0.015          # views on one shared scale whose full bbox heights agree this well: no scale_drift / vertical_shift
    accessory_open_frac: float = 0.015     # opening radius that removes boundary slivers from the XOR

    def to_dict(self) -> dict[str, float]:
        """Plain-dict form."""
        return asdict(self)


# --------------------------------------------------------------------------- grids
SINGLE_PART_HINT = "a single part may differ (e.g. tail pose); see recipe 'View disagrees on one part' in references/reconstruction-workflow.md"


def _downscale(mask: np.ndarray, calib: ViewCalibration) -> tuple[np.ndarray, ViewCalibration]:
    """Block-average a mask down to at most GRID_MAX_PX per side; the calibration follows."""
    h, w = mask.shape
    s = int(np.ceil(max(h, w) / GRID_MAX_PX))
    if s <= 1:
        return np.asarray(mask, dtype=bool), calib
    hh, ww = h // s * s, w // s * s
    small = mask[:hh, :ww].reshape(hh // s, s, ww // s, s).mean(axis=(1, 3)) >= 0.5
    ox, oy = calib.origin_px
    return small, ViewCalibration(calib.view, calib.camera_axis, (ww // s, hh // s), calib.m_per_px * s,
                                  (ox / s, oy / s))


@dataclass
class _Grid:
    """World raster: column = horizontal coordinate (A's image right), row = height z (row 0 = lowest)."""

    h0: float
    z0: float
    res: float
    nx: int
    nz: int

    def h_centers(self) -> np.ndarray:
        return self.h0 + (np.arange(self.nx) + 0.5) * self.res

    def z_centers(self) -> np.ndarray:
        return self.z0 + (np.arange(self.nz) + 0.5) * self.res


def _make_grid(calibs: Sequence[ViewCalibration]) -> _Grid:
    half = max(max(c.origin_px[0], c.image_size[0] - c.origin_px[0]) * c.m_per_px for c in calibs)
    z_hi = max(c.origin_px[1] * c.m_per_px for c in calibs)
    z_lo = min((c.origin_px[1] - c.image_size[1]) * c.m_per_px for c in calibs)
    res = max(max(2 * half, z_hi - z_lo) / GRID_MAX_PX, min(c.m_per_px for c in calibs))
    return _Grid(-half, z_lo, res, int(np.ceil(2 * half / res)), int(np.ceil((z_hi - z_lo) / res)))


def _to_grid(mask: np.ndarray, calib: ViewCalibration, grid: _Grid, mirror: bool = False) -> np.ndarray:
    """Resample a view mask (pixels) onto the world grid; ``mirror`` negates the horizontal coordinate."""
    h = grid.h_centers()
    u = calib.origin_px[0] + (-h if mirror else h) / calib.m_per_px
    v = calib.origin_px[1] - grid.z_centers() / calib.m_per_px
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    hh, ww = mask.shape
    ok = ((vi >= 0) & (vi < hh))[:, None] & ((ui >= 0) & (ui < ww))[None, :]
    return mask[np.clip(vi, 0, hh - 1)[:, None], np.clip(ui, 0, ww - 1)[None, :]] & ok


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def _warp(m: np.ndarray, grid: _Grid, scale: float, dh: float, dz: float, stride: int = 1) -> np.ndarray:
    """``m`` scaled by ``scale`` about the world origin and shifted by (dh, dz) meters, on a strided grid."""
    h = grid.h_centers()[::stride]
    z = grid.z_centers()[::stride]
    ui = np.floor(((h - dh) / scale - grid.h0) / grid.res).astype(int)
    vi = np.floor(((z - dz) / scale - grid.z0) / grid.res).astype(int)
    ok = ((vi >= 0) & (vi < grid.nz))[:, None] & ((ui >= 0) & (ui < grid.nx))[None, :]
    return m[np.clip(vi, 0, grid.nz - 1)[:, None], np.clip(ui, 0, grid.nx - 1)[None, :]] & ok


def _align(a: np.ndarray, b: np.ndarray, grid: _Grid, height: float) -> tuple[float, tuple[float, float, float]]:
    """Best scale + shift of ``b`` onto ``a`` (coarse scan, then hill climbing). Returns (iou, (scale, dh, dz))."""
    stride = max(1, int(np.ceil(max(grid.nx, grid.nz) / ALIGN_COARSE_PX)))
    a_c = a[::stride, ::stride]
    best = (_iou(a_c, _warp(b, grid, 1.0, 0.0, 0.0, stride)), (1.0, 0.0, 0.0))
    for s in np.linspace(1 - ALIGN_SCALE_RANGE, 1 + ALIGN_SCALE_RANGE, 9):
        for dh in np.linspace(-ALIGN_SHIFT_H, ALIGN_SHIFT_H, 13) * height:
            for dz in np.linspace(-ALIGN_SHIFT_Z, ALIGN_SHIFT_Z, 7) * height:
                v = _iou(a_c, _warp(b, grid, float(s), float(dh), float(dz), stride))
                if v > best[0]:
                    best = (v, (float(s), float(dh), float(dz)))
    fine = max(1, stride // 2)
    a_f = a[::fine, ::fine]
    cur_iou, cur = _iou(a_f, _warp(b, grid, *best[1], stride=fine)), best[1]
    steps = np.array([0.01, 0.01 * height, 0.005 * height])
    for _ in range(3):
        improved = True
        while improved:
            improved = False
            for d in np.ndindex(3, 3, 3):
                delta = (np.array(d) - 1) * steps
                if not delta.any():
                    continue
                cand = (cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2])
                v = _iou(a_f, _warp(b, grid, *cand, stride=fine))
                if v > cur_iou + 1e-6:
                    cur_iou, cur, improved = v, cand, True
        steps = steps / 2.0
    return float(cur_iou), (float(cur[0]), float(cur[1]), float(cur[2]))


def _opened(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Morphological opening (erode, dilate): removes protrusions thinner than 2 * radius_px."""
    return dilate(erode(mask, radius_px), radius_px) if radius_px > 0 else mask


# --------------------------------------------------------------------------- per view / per pair
def _is_upright(axis: str) -> bool:
    return view_frame(axis).up_axis == "+Z"


def _view_core(mask: np.ndarray, calib: ViewCalibration, thr: ConsistencyThresholds) -> dict[str, float]:
    """Core extents of an upright view in meters: protrusions thinner than ``core_open_frac * height`` are ignored.

    The opening also eats legs / feet thinner than that, so the full bbox extents (``bbox_*``) are returned too:
    they are the cross-check of :func:`analyze` (a hat, a staff or thin boots must not fake a scale drift).
    """
    rows = np.flatnonzero(mask.any(axis=1))
    height_px = rows[-1] - rows[0] + 1
    oy, m = calib.origin_px[1], calib.m_per_px
    bbox_top, bbox_bottom = (oy - rows[0]) * m, (oy - (rows[-1] + 1)) * m
    core = _opened(mask, int(round(thr.core_open_frac * height_px)))
    if not core.any():
        core = mask
    rows = np.flatnonzero(core.any(axis=1))
    top, bottom = (oy - rows[0]) * m, (oy - (rows[-1] + 1)) * m
    return {"core_top_z": float(top), "core_bottom_z": float(bottom), "core_height": float(top - bottom),
            "bbox_bottom_z": float(bbox_bottom), "bbox_height": float(bbox_top - bbox_bottom)}


def _row_widths(core: np.ndarray, grid: _Grid, z_lo: float, z_hi: float) -> list[Optional[float]]:
    """Median row width (m) of ``core`` in BAND_COUNT equal bands between z_lo and z_hi."""
    rows_z = grid.z_centers()
    out: list[Optional[float]] = []
    for k in range(BAND_COUNT):
        lo, hi = z_lo + (z_hi - z_lo) * k / BAND_COUNT, z_lo + (z_hi - z_lo) * (k + 1) / BAND_COUNT
        idx = np.flatnonzero((rows_z >= lo) & (rows_z < hi))
        widths = []
        for i in idx:
            cols = np.flatnonzero(core[i])
            if cols.size:
                widths.append((cols[-1] - cols[0] + 1) * grid.res)
        out.append(float(np.median(widths)) if widths else None)
    return out


def _blobs(only: np.ndarray, union_area: int, grid: _Grid, open_px: int) -> Optional[dict[str, float]]:
    """Largest connected blob of a one-sided XOR (after removing boundary slivers); None when nothing is left."""
    cleaned = _opened(only, open_px)
    labels, n = label_components(cleaned)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    rr, cc = np.nonzero(labels == k)
    return {"area_frac": float(sizes[k - 1]) / max(union_area, 1),
            "z": float(grid.z0 + (rr.mean() + 0.5) * grid.res), "h": float(grid.h0 + (cc.mean() + 0.5) * grid.res)}


def _pairs(axes: Mapping[str, str]) -> list[tuple[str, str]]:
    """Opposite upright view pairs (their image-right axes are negatives of each other)."""
    keys = [k for k, a in axes.items() if _is_upright(a)]
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = view_frame(axes[a]).right_axis, view_frame(axes[b]).right_axis
            if ra[1] == rb[1] and ra[0] != rb[0]:
                out.append((a, b))
    return out


def _priority(key: str) -> int:
    return _VIEW_PRIORITY.index(key) if key in _VIEW_PRIORITY else len(_VIEW_PRIORITY)


def recommend_roles(iou: Optional[float], current: Sequence[str], thr: Optional[ConsistencyThresholds] = None,
                    blamed: bool = True) -> list[str]:
    """Roles a view may keep given its pair IoU (never adds roles; ``preview`` is always kept).

    ``blamed=False`` (the pair failed but the other view is the one at fault) keeps the roles unchanged.
    """
    thr = thr or ConsistencyThresholds()
    if iou is None or not blamed:
        return list(current)
    tier = _ROLE_TIERS[0] if iou >= thr.carve_min_iou else _ROLE_TIERS[1] if iou >= thr.color_validate_min_iou else _ROLE_TIERS[2]
    return [r for r in current if r in tier or r == "preview"]


def _pair_result(a: str, b: str, masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration],
                 thr: ConsistencyThresholds) -> dict[str, Any]:
    grid = _make_grid([calibs[a], calibs[b]])
    ga = _to_grid(masks[a], calibs[a], grid)
    gb = _to_grid(masks[b], calibs[b], grid, mirror=True)
    gb_plain = _to_grid(masks[b], calibs[b], grid, mirror=False)
    raw = _iou(ga, gb)
    rows = np.flatnonzero(ga.any(axis=1))
    height = (rows[-1] - rows[0] + 1) * grid.res if rows.size else grid.nz * grid.res
    aligned, (s, dh, dz) = _align(ga, gb, grid, height)
    gb_al = _warp(gb, grid, s, dh, dz)
    union = int((ga | gb_al).sum())
    open_px = int(round(thr.accessory_open_frac * height / grid.res))
    blobs = {a: _blobs(ga & ~gb_al, union, grid, open_px), b: _blobs(gb_al & ~ga, union, grid, open_px)}
    xor = _opened((ga & ~gb_al) | (gb_al & ~ga), open_px)
    single = bool(xor.any()) and label_components(xor)[1] == 1       # the views differ in ONE place: one part (a tail pose), not the whole figure
    core_px = int(round(thr.core_open_frac * height / grid.res))
    ca, cb = _opened(ga, core_px), _opened(gb, core_px)
    z_lo = max(float(grid.z_centers()[ca.any(axis=1)].min()) if ca.any() else 0.0,
               float(grid.z_centers()[cb.any(axis=1)].min()) if cb.any() else 0.0)
    z_hi = min(float(grid.z_centers()[ca.any(axis=1)].max()) if ca.any() else 1.0,
               float(grid.z_centers()[cb.any(axis=1)].max()) if cb.any() else 1.0)
    bands = []
    if z_hi > z_lo:
        wa, wb = _row_widths(ca, grid, z_lo, z_hi), _row_widths(cb, grid, z_lo, z_hi)
        for k, (x, y) in enumerate(zip(wa, wb)):
            if x and y:
                bands.append({"z0": round(k / BAND_COUNT, 3), "z1": round((k + 1) / BAND_COUNT, 3),
                              "width_a": round(x, 5), "width_b": round(y, 5), "rel_diff": round((x - y) / max(x, y), 4)})
    plain = _iou(ga, gb_plain)
    pair = {"a": a, "b": b, "iou_raw": round(raw, 4), "iou_aligned": round(aligned, 4),
            "align": {"scale": round(s, 4), "dh_m": round(dh, 5), "dz_m": round(dz, 5)},
            "iou_unmirrored": round(plain, 4), "floor": round(1.0 - raw, 4), "bands": bands,
            "blobs": {k: v for k, v in blobs.items() if v}, "single_part_differs": single}
    return pair


def pair_overlay(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration], a: str, b: str) -> np.ndarray:
    """XOR overlay of ``a`` and mirrored ``b`` on the common world grid: float RGB, top-down (green both, red only
    ``a``, blue only mirrored ``b``)."""
    grid = _make_grid([calibs[a], calibs[b]])
    ga = _to_grid(masks[a], calibs[a], grid)
    gb = _to_grid(masks[b], calibs[b], grid, mirror=True)
    img = np.zeros(ga.shape + (3,), np.float32)
    img[ga & gb] = (0.16, 0.78, 0.24)
    img[ga & ~gb] = (0.9, 0.16, 0.16)
    img[~ga & gb] = (0.2, 0.35, 0.95)
    return img[::-1]


# --------------------------------------------------------------------------- the analysis
def _flag_ignored(flag: Mapping[str, Any], ignore: Sequence[str]) -> bool:
    """``ignore`` entries are ``"type"`` (any view) or ``"type:view"``."""
    return any(i == flag["type"] or i in (f"{flag['type']}:{v}" for v in flag["views"]) for i in ignore)


def merge_overrides(*sources: Optional[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    """Union of ``ignore_flags`` and per-view ``roles_override`` of several ``{ignore_flags, roles_override}`` dicts, given
    lowest priority first (view spec, scene ``workflow.consistency``); a later source wins a view's roles. Never drops an
    earlier source's flag: a command-line ``--ignore-flag`` adds to the recorded ones instead of replacing them."""
    flags: list[str] = []
    roles: dict[str, list[str]] = {}
    for src in sources:
        for f in (src or {}).get("ignore_flags") or []:
            if f not in flags:
                flags.append(str(f))
        for view, rr in ((src or {}).get("roles_override") or {}).items():
            roles[str(view)] = [str(r) for r in rr]
    return flags, roles


def _loo_median(values: Mapping[str, float], key: str) -> Optional[float]:
    others = [v for k, v in values.items() if k != key]
    return float(np.median(others)) if others else None


def analyze(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration], axes: Mapping[str, str],
            roles: Optional[Mapping[str, Sequence[str]]] = None,
            thresholds: Optional[ConsistencyThresholds] = None,
            labels: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
            asymmetric_parts: Sequence[str] = (),
            ignore_flags: Sequence[str] = (),
            roles_override: Optional[Mapping[str, Sequence[str]]] = None) -> dict[str, Any]:
    """Consistency of the views that have a mask and a calibration.

    Returns a JSON-able dict: ``pairs`` (raw / aligned mirror IoU, floor, per-band widths, one-sided blobs),
    ``views`` (core extents, pair IoU, floor, current / recommended roles), ``flags`` (scale_drift,
    vertical_shift, missing_accessory, mirrored_view) and ``passed`` (False when a view must lose a role or an
    error flag is raised). ``symmetry``: per front / back view the left / right mirror IoU and the ``symmetry.x``
    suggestion (:func:`~bbr.core.symmetry.front_symmetry`; ``labels`` = ``{view: {part: mask}}`` and
    ``asymmetric_parts`` blank the labelled asymmetric parts), a hint only: it never changes ``passed``.

    scale_drift / vertical_shift compare the opened "core" extents, which a hat, a staff or thin legs distort.
    When the views share one scale (sheet calibration) and their full bbox heights agree within
    ``bbox_agree_tol`` (and the bbox bottoms within ``shift_tol``), those flags are not raised.
    ``ignore_flags`` (``"type"`` or ``"type:view"``) and ``roles_override`` (``{view: roles}``) record a judged
    false alarm: ignored flags move to ``ignored_flags`` (no effect on ``passed``), overridden views get
    verdict ``override`` and are listed in ``overrides``.
    """
    thr = thresholds or ConsistencyThresholds()
    keys = [k for k in axes if k in masks and k in calibs]
    ds = {k: _downscale(masks[k], calibs[k]) for k in keys}
    m = {k: ds[k][0] for k in keys}
    c = {k: ds[k][1] for k in keys}
    ax = {k: axes[k] for k in keys}
    upright = [k for k in keys if _is_upright(ax[k]) and m[k].any()]
    views: dict[str, dict[str, Any]] = {}
    for k in upright:
        views[k] = {"axis": ax[k], "m_per_px": round(calibs[k].m_per_px, 6), **_view_core(m[k], c[k], thr)}
    flags: list[dict[str, Any]] = []

    heights = {k: v["core_height"] for k, v in views.items()}
    bottoms = {k: v["core_bottom_z"] for k, v in views.items()}
    bbox_h = [v["bbox_height"] for v in views.values()]
    scales = [v["m_per_px"] for v in views.values()]
    shared_scale = len(views) > 1 and max(scales) / min(scales) - 1.0 < 0.005   # else bbox heights agree by construction
    bbox_agree = shared_scale and max(bbox_h) / min(bbox_h) - 1.0 <= thr.bbox_agree_tol
    bbox_bottoms = [v["bbox_bottom_z"] for v in views.values()]
    bottoms_agree = bbox_agree and max(bbox_bottoms) - min(bbox_bottoms) <= thr.shift_tol * float(np.median(bbox_h))
    for k in views:
        ref = _loo_median(heights, k)
        if ref:
            dev = heights[k] / ref - 1.0
            if abs(dev) > thr.scale_drift_tol and not bbox_agree:
                flags.append({"type": "scale_drift", "severity": "error", "views": [k], "value": round(dev, 4),
                              "detail": f"{k}: core height {heights[k]:.4f} m is {dev * 100:+.1f}% vs the other views "
                                        f"(tolerance {thr.scale_drift_tol * 100:.1f}%)"})
        ref_b = _loo_median(bottoms, k)
        if ref_b is not None and ref:
            shift = bottoms[k] - ref_b
            if abs(shift) > thr.shift_tol * ref and not bottoms_agree:
                flags.append({"type": "vertical_shift", "severity": "error", "views": [k], "value": round(shift, 5),
                              "detail": f"{k}: baseline is {shift * 1000:+.1f} mm off the other views "
                                        f"(tolerance {thr.shift_tol * 100:.1f}% of the height)"})

    pairs: list[dict[str, Any]] = []
    for a, b in _pairs({k: ax[k] for k in upright}):
        pair = _pair_result(a, b, m, c, thr)
        pairs.append(pair)
        if pair["iou_unmirrored"] > pair["iou_raw"] + thr.mirrored_margin:
            odd = a if _priority(a) > _priority(b) else b
            flags.append({"type": "mirrored_view", "severity": "error", "views": [odd], "value": pair["iou_unmirrored"],
                          "detail": f"{a}/{b}: matches better unmirrored ({pair['iou_unmirrored']:.3f}) than mirrored "
                                    f"({pair['iou_raw']:.3f}); {odd} faces the wrong way (flip_horizontal or camera_axis sign)"})
        for owner, blob in pair["blobs"].items():
            if blob["area_frac"] >= thr.accessory_blob_frac:
                other = b if owner == a else a
                flags.append({"type": "missing_accessory", "severity": "warn", "views": [other], "value": round(blob["area_frac"], 4),
                              "detail": f"{a}/{b}: a blob ({blob['area_frac'] * 100:.1f}% of the silhouette) exists in {owner} only, "
                                        f"at z={blob['z']:.3f} m (missing in {other})"
                                        + (f"; {SINGLE_PART_HINT}" if pair.get("single_part_differs") else "")})

    cur_roles = {k: list(roles[k]) if roles and k in roles else None for k in keys}
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        tier_ok = pair["iou_raw"] >= thr.color_validate_min_iou
        loser = None if tier_ok else (a if _priority(a) > _priority(b) else b)
        for k in (a, b):
            if k not in views:
                continue
            cur = cur_roles[k] or []
            rec = recommend_roles(pair["iou_raw"], cur, thr, blamed=(k == loser or tier_ok))
            views[k].update({"pair": b if k == a else a, "pair_iou": pair["iou_raw"], "floor": pair["floor"],
                             "current_roles": cur, "recommended_roles": rec,
                             "verdict": "drop" if (k == loser) else ("unverified" if not tier_ok else
                                        ("ok" if set(rec) == set(cur) else "demote"))})
    for k, v in views.items():
        v.setdefault("current_roles", cur_roles.get(k) or [])
        v.setdefault("recommended_roles", v["current_roles"])
        v.setdefault("verdict", "unpaired")
        v.setdefault("floor", None)
        v.setdefault("pair", None)
        v.setdefault("pair_iou", None)
    overrides = {}
    for k, rr in (roles_override or {}).items():
        if k in views:
            views[k]["recommended_roles"] = list(rr)
            views[k]["verdict"] = "override"
            overrides[k] = list(rr)
    ignored = [f for f in flags if _flag_ignored(f, ignore_flags)]
    flags = [f for f in flags if f not in ignored]
    changed = [k for k, v in views.items() if set(v["current_roles"]) != set(v["recommended_roles"])]
    sym = [front_symmetry(masks[k], calibs[k], (labels or {}).get(k), asymmetric_parts) for k in upright
           if ax[k] in ("-Y", "+Y")]
    return {"thresholds": thr.to_dict(), "pairs": pairs, "views": views, "flags": flags, "symmetry": sym,
            "ignored_flags": ignored, "overrides": overrides, "bbox_heights_agree": bool(bbox_agree),
            "roles_to_change": changed,
            "passed": not changed and not any(f["severity"] == "error" for f in flags)}


# --------------------------------------------------------------------------- text
def add_excess(result: dict[str, Any], view_errors: Mapping[str, float]) -> None:
    """Attach ``error`` and ``excess_error`` (= error - floor) to every validated view that has a floor."""
    for k, err in view_errors.items():
        v = result["views"].get(k)
        if v is not None and v.get("floor") is not None:
            v["error"] = round(float(err), 5)
            v["excess_error"] = round(float(err) - v["floor"], 5)


def format_consistency(result: Mapping[str, Any]) -> str:
    """Human-readable table of an :func:`analyze` result."""
    t = result["thresholds"]
    lines = [f"Consistency (carve >= {t['carve_min_iou']:.2f}, color/validate >= {t['color_validate_min_iou']:.2f}):"]
    for p in result["pairs"]:
        al = p["align"]
        lines.append(f"  {p['a']}<->{p['b']}: mirror IoU {p['iou_raw']:.3f}  aligned {p['iou_aligned']:.3f} "
                     f"(scale {al['scale']:.3f}, dh {al['dh_m'] * 1000:+.1f} mm, dz {al['dz_m'] * 1000:+.1f} mm)  "
                     f"floor {p['floor'] * 100:.1f}%" + (f"  [{SINGLE_PART_HINT}]" if p.get("single_part_differs") and p["iou_raw"] < 0.97 else ""))
    if not result["pairs"]:
        lines.append("  no opposite view pair (front/back, side/right): nothing to cross-check")
    for k, v in result["views"].items():
        rec = "drop" if not [r for r in v["recommended_roles"] if r != "preview"] and v["current_roles"] else ",".join(v["recommended_roles"])
        cur = ",".join(v["current_roles"])
        extra = ""
        if v.get("excess_error") is not None:
            extra = f"  error {v['error'] * 100:.1f}% - floor {v['floor'] * 100:.1f}% = excess {v['excess_error'] * 100:+.1f}%"
        lines.append(f"  {k:<7} core height {v['core_height']:.4f} m (bbox {v['bbox_height']:.4f})  baseline z {v['core_bottom_z'] * 1000:+.1f} mm  "
                     f"roles [{cur}] -> [{rec}] ({v['verdict']}){extra}")
    for s in result.get("symmetry", []):
        lines.append(f"  symmetry {s['view']}: left/right mirror IoU " + (s.get("explanation") or f"{s['iou']:.3f} raw")
                     + ("  (mirrored colour view / paired fitting)" if s["suggest_x"] else ""))
    for f in result["flags"]:
        lines.append(f"  FLAG {f['type']}: {f['detail']}")
    for f in result.get("ignored_flags", []):
        lines.append(f"  ignored (judged false alarm) {f['type']}: {f['detail']}")
    for k, roles in result.get("overrides", {}).items():
        lines.append(f"  roles override {k}: [{','.join(roles)}] (judged false alarm)")
    lines.append("  verdict: " + ("consistent" if result["passed"] else
                                  "INCONSISTENT references (see flags / role changes); fix the images or apply the recommended roles"))
    return "\n".join(lines)
