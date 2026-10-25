"""Pixel <-> world calibration of orthographic views.

Pixel (i, j) has its centre at u = j + 0.5, v = i + 0.5 (u right, v down). The world origin projects
to ``origin_px``; one pixel is ``m_per_px`` meters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .. import CalibrationError, get_logger
from .conventions import view_frame
from .imageops import dilate, erode, mask_bbox

log = get_logger(__name__)

DEFAULT_HEIGHT_M = 1.0
AXIS_DIMENSION = {"X": "width", "Y": "depth", "Z": "height"}
DIMENSIONS = ("width", "depth", "height")
_VIEW_PRIORITY = ("front", "back", "side", "right", "left", "top", "bottom")
SHEET_CORE_OPEN_FRAC = 0.03       # sheet mode: protrusions thinner than 2x this fraction of the height are not "body"
SHEET_LOWER_BODY = (0.05, 0.40)   # sheet mode: body-core rows, as fractions of the height above the soles
BBOX_LOWER_BODY = (0.0, 0.60)     # silhouette_bbox mode: body-core rows (the lower 60%; heads, hats and raised props sit above)
BODY_AXIS_MIN_SHIFT = 0.03       # silhouette_bbox mode: the body axis replaces the bbox centre when it is off by more than this share of the width
BODY_LABEL = "body"               # label whose pixels alone give the body axis (else the mask minus every labelled part)


@dataclass
class ViewCalibration:
    """Scale and origin of one view. ``image_size`` is (W, H) after crop."""

    view: str
    camera_axis: str
    image_size: tuple[int, int]
    m_per_px: float
    origin_px: tuple[float, float]

    def world_to_px(self, pts: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixel coordinates (N,2) float ``(u, v)``."""
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        f = view_frame(self.camera_axis)
        u = self.origin_px[0] + (p @ f.right) / self.m_per_px
        v = self.origin_px[1] - (p @ f.up) / self.m_per_px
        return np.stack([u, v], axis=1)

    def px_to_plane(self, uv: np.ndarray) -> np.ndarray:
        """Pixel coordinates (N,2) -> world points (N,3) on the plane through the origin facing the camera."""
        q = np.atleast_2d(np.asarray(uv, dtype=float))
        f = view_frame(self.camera_axis)
        right = (q[:, 0] - self.origin_px[0]) * self.m_per_px
        up = -(q[:, 1] - self.origin_px[1]) * self.m_per_px
        return right[:, None] * f.right[None, :] + up[:, None] * f.up[None, :]

    def image_extent_m(self) -> tuple[float, float]:
        """(width, height) of the whole image in meters."""
        return self.image_size[0] * self.m_per_px, self.image_size[1] * self.m_per_px

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {"view": self.view, "camera_axis": self.camera_axis, "image_size": list(self.image_size),
                "m_per_px": self.m_per_px, "origin_px": list(self.origin_px)}


def view_dimension_names(camera_axis: str) -> tuple[str, str]:
    """(horizontal, vertical) dimension names seen by a view, e.g. '-Y' -> ('width', 'height')."""
    f = view_frame(camera_axis)
    return AXIS_DIMENSION[f.right_axis[1]], AXIS_DIMENSION[f.up_axis[1]]


def resolve_dimensions(bboxes_px: Mapping[str, tuple], axes: Mapping[str, str],
                       known: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Overall (width, depth, height) in meters from known values and silhouette bbox ratios.

    ``bboxes_px``: view -> (x0, y0, x1, y1); ``axes``: view -> camera_axis. Provenance values:
    ``"known"``, ``"derived:<view>"`` or ``"default"``.
    """
    dims: dict[str, float | None] = {k: (float(known[k]) if known.get(k) else None) for k in DIMENSIONS}
    prov: dict[str, str] = {k: "known" for k in DIMENSIONS if dims[k] is not None}
    order = sorted((v for v in bboxes_px if v in axes),
                   key=lambda v: _VIEW_PRIORITY.index(v) if v in _VIEW_PRIORITY else len(_VIEW_PRIORITY))
    extents = {v: (bboxes_px[v][2] - bboxes_px[v][0], bboxes_px[v][3] - bboxes_px[v][1]) for v in order}

    def propagate() -> None:
        changed = True
        while changed:
            changed = False
            for v in order:
                hd, vd = view_dimension_names(axes[v])
                pw, ph = extents[v]
                if dims[hd] is not None and dims[vd] is None and pw > 0:
                    dims[vd], prov[vd], changed = dims[hd] / pw * ph, f"derived:{v}", True
                elif dims[vd] is not None and dims[hd] is None and ph > 0:
                    dims[hd], prov[hd], changed = dims[vd] / ph * pw, f"derived:{v}", True

    propagate()
    if all(dims[k] is None for k in DIMENSIONS):
        log.warning("no known dimension; assuming height = %.2f m", DEFAULT_HEIGHT_M)
        dims["height"], prov["height"] = DEFAULT_HEIGHT_M, "default"
        propagate()
    for k in DIMENSIONS:  # dimension no view could determine: fall back to the mean of the others
        if dims[k] is None:
            others = [d for d in dims.values() if d is not None]
            dims[k], prov[k] = float(np.mean(others)), "default"
            log.warning("dimension %s not determinable from the views; defaulting to %.3f m", k, dims[k])
            propagate()
    return {k: float(dims[k]) for k in DIMENSIONS}, prov  # type: ignore[arg-type]


def body_core_centre(mask: np.ndarray, bbox: tuple, labels: Optional[Mapping[str, np.ndarray]] = None,
                     rows: tuple[float, float] = BBOX_LOWER_BODY) -> float:
    """Horizontal body axis (pixel column) of one upright view, not dragged by props.

    With ``labels`` ({part: bool mask} of this view) only the ``body`` label counts (without one: the mask minus every
    labelled part; ``_``-prefixed labels are notes, not parts). Then thin protrusions (a held spatula, horns, a tail)
    are opened away, and the result is the median row centre over the ``rows`` band (fractions of the height above the
    soles). Falls back to the bbox centre when nothing is left."""
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    m = np.asarray(mask, dtype=bool)
    if labels:
        named = {k: np.asarray(v, dtype=bool) for k, v in labels.items() if not k.startswith("_")}
        if BODY_LABEL in named and named[BODY_LABEL].any():
            m = m & named[BODY_LABEL]
        else:
            for k, v in named.items():
                m = m & ~v
    r = int(round(SHEET_CORE_OPEN_FRAC * h))
    core = dilate(erode(m, r), r) if r > 0 else m
    lo, hi = int(y1 - rows[1] * h), int(y1 - rows[0] * h)
    centres = []
    for row in range(max(lo, 0), min(hi, core.shape[0])):
        cols = np.flatnonzero(core[row])
        if cols.size:
            centres.append((cols[0] + cols[-1] + 1) / 2.0)
    return float(np.median(centres)) if centres else (x0 + x1) / 2.0


def body_bbox(mask: np.ndarray, bbox: tuple, min_width_frac: float = 0.5) -> tuple[int, int, int, int]:
    """Bbox ``(x0, y0, x1, y1)`` of the BODY: the typical row extent, not the outline's. Handles, a padlock, side rings or
    a raised arm occupy a minority of the rows, so the median left / right edge over the rows at least ``min_width_frac``
    of the widest row wide (thin tops, feet, antennas excluded) is the body's; the vertical extent is the bbox's. (An
    opening does not work here: a protrusion that sits flat against the body fits inside the opening's structuring element.)"""
    m = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return tuple(bbox)
    left = np.array([np.flatnonzero(m[r])[0] for r in rows])
    right = np.array([np.flatnonzero(m[r])[-1] for r in rows])
    keep = (right - left + 1) >= min_width_frac * (right - left + 1).max()
    return (int(np.median(left[keep])), int(bbox[1]), int(np.median(right[keep])) + 1, int(bbox[3]))


def derive_height(masks: Mapping[str, np.ndarray], axes: Mapping[str, str], width: Optional[float] = None,
                  depth: Optional[float] = None, size_of: str = "silhouette") -> tuple[float, str, dict[str, float]]:
    """Object height (m) from a known ``width`` (X) or ``depth`` (Y) measured in an upright view.

    The scale is the given extent over its pixel extent in the first upright view (priority front, back, side, ...) that
    shows that dimension horizontally; the height is that scale times the SILHOUETTE height of the same view (what the
    pipeline calibrates against). ``size_of`` says what the given extent measures: ``"silhouette"`` = the whole outline
    (handles, padlock and all), ``"body"`` = the body (typical row extent, handles and props excluded, :func:`body_bbox`).
    Returns ``(height_m, provenance text, {alternative size_of: height})``; with both ``width`` and ``depth`` the width
    gives the scale and the depth is only cross-checked (a >3% disagreement is logged)."""
    if size_of not in ("silhouette", "body"):
        raise CalibrationError(f"size_of must be 'silhouette' or 'body', got {size_of!r}")
    given = [(name, val) for name, val in (("width", width), ("depth", depth)) if val]
    if not given:
        raise CalibrationError("derive_height needs a width or a depth")
    order = sorted((v for v in masks if v in axes and view_frame(axes[v]).up_axis[1] == "Z"),
                   key=lambda v: _VIEW_PRIORITY.index(v) if v in _VIEW_PRIORITY else len(_VIEW_PRIORITY))

    def height_for(name: str, value: float, kind: str) -> tuple[float, str, float]:
        view = next((v for v in order if view_dimension_names(axes[v])[0] == name), None)
        if view is None:
            raise CalibrationError(f"no upright view shows the {name} horizontally; pass --height instead")
        box = mask_bbox(masks[view])
        used = body_bbox(masks[view], box) if kind == "body" else box
        m_per_px = value / float(used[2] - used[0])
        return m_per_px * (box[3] - box[1]), view, m_per_px

    name, value = given[0]
    height, view, m_per_px = height_for(name, value, size_of)
    other = "body" if size_of == "silhouette" else "silhouette"
    note = (f"derived from {name} {value:g} m of the {size_of} in view {view} ({m_per_px * 1000:.4f} mm/px), "
            f"silhouette height of that view")
    if len(given) > 1:
        h2, _, _ = height_for(given[1][0], given[1][1], size_of)
        if abs(h2 / height - 1.0) > 0.03:
            log.warning("%s %g m and %s %g m imply heights %.3f m and %.3f m (> 3%% apart)", name, value, *given[1], height, h2)
    return height, note, {other: height_for(name, value, other)[0]}


def sheet_calibration(masks: Mapping[str, np.ndarray], bboxes: Mapping[str, tuple], height_m: float,
                      labels: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None) -> tuple[float, dict[str, float], dict[str, float]]:
    """Shared calibration of upright views cut from ONE turnaround sheet.

    One m/px for all (``height_m`` over the median bbox height), one baseline row (median sole row, when the
    images share a height; else each view's own), and a horizontal origin from the body core: the median row
    centre of the lower body with thin protrusions (horns, weapons, tails) opened away - not the bbox centre,
    which accessories drag around. Returns ``(m_per_px, {view: baseline_row}, {view: origin_u})``.
    """
    heights = {k: b[3] - b[1] for k, b in bboxes.items()}
    m_per_px = height_m / float(np.median(list(heights.values())))
    same_rows = len({masks[k].shape[0] for k in bboxes}) == 1
    shared_row = float(np.median([b[3] for b in bboxes.values()]))
    baselines = {k: shared_row if same_rows else float(b[3]) for k, b in bboxes.items()}
    origins = {k: body_core_centre(masks[k], b, (labels or {}).get(k), SHEET_LOWER_BODY) for k, b in bboxes.items()}
    return m_per_px, baselines, origins


def _body_axes(masks: Mapping[str, np.ndarray], bboxes: Mapping[str, tuple],
               labels: Optional[Mapping[str, Mapping[str, np.ndarray]]]) -> dict[str, float]:
    """Body-axis column of the upright front / back views (world X is their horizontal), or ``{}``.

    The body axis replaces the bbox centre only when a view's is more than ``BODY_AXIS_MIN_SHIFT`` of its width off: a few px
    are noise, and the bbox centre registers front and back extents against each other. The decision is joint: when one
    of the views needs it, all of them take the body axis, so front and back stay registered."""
    core = {k: body_core_centre(m, bboxes[k], (labels or {}).get(k)) for k, m in masks.items()}
    if any(abs(core[k] - (bboxes[k][0] + bboxes[k][2]) / 2.0) > BODY_AXIS_MIN_SHIFT * (bboxes[k][2] - bboxes[k][0]) for k in core):
        return core
    return {}


def frame_offset_m(calibs: Mapping[str, ViewCalibration], bboxes: Mapping[str, tuple]) -> dict[str, float]:
    """World offset ``{"x": m, "y": m}`` of the silhouette bbox centre from the model origin, measured in the upright views.

    The body axis (X) or a sheet origin moves the model origin off the bbox centre; the model is then the same shape
    translated by this offset relative to a frame centred on the bbox. Scoring against ground truth subtracts it
    (``bench/score.py``). Axes no upright view shows horizontally are absent; the first view in priority order wins."""
    out: dict[str, float] = {}
    for key in sorted(calibs, key=lambda v: _VIEW_PRIORITY.index(v) if v in _VIEW_PRIORITY else len(_VIEW_PRIORITY)):
        f = view_frame(calibs[key].camera_axis)
        axis = f.right_axis[1].lower()
        if f.up_axis[1] != "Z" or axis in out or axis not in "xy":
            continue
        x0, _, x1, _ = bboxes[key]
        sign = 1.0 if f.right_axis[0] == "+" else -1.0
        out[axis] = sign * ((x0 + x1) / 2.0 - calibs[key].origin_px[0]) * calibs[key].m_per_px
    return out


def calibrate_views(view_spec, masks: Mapping[str, np.ndarray],
                    labels: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None) -> tuple[dict[str, ViewCalibration], dict]:
    """Calibrate every enabled orthographic view that has a mask.

    ``silhouette_bbox`` mode: scale from the vertical extent (views whose up axis is Z) or the horizontal
    extent (top/bottom) against the resolved dimensions; origin_px horizontally = body-core centre of upright
    front / back views when one of them is off its bbox centre by more than ``BODY_AXIS_MIN_SHIFT`` of the width (:func:`_body_axes`, :func:`body_core_centre`: only the ``body`` label when ``labels`` = ``{view: {part: mask}}`` is given, else
    the lower 60% of the rows with thin props opened away; a held prop drags the bbox centre off the body axis),
    bbox centre for side and top/bottom views; vertically = bbox bottom (Z-up views) or bbox centre (top/bottom). ``manual`` mode uses px_per_meter / origin_px.
    ``sheet`` mode (upright views of one sheet): shared m/px and baseline, origin from the body core
    (:func:`sheet_calibration`). Returns ``(calibrations, info)`` with info = dimensions, provenance, bboxes_px
    and, when used, ``sheet`` = {m_per_px, views}.
    """
    bboxes: dict[str, tuple] = {}
    axes: dict[str, str] = {}
    for key, axis in view_spec.view_axes().items():
        if key not in masks:
            log.warning("view %s has no mask; not calibrated", key)
            continue
        bboxes[key] = mask_bbox(masks[key])
        axes[key] = axis
    if not bboxes:
        raise CalibrationError("no usable view with a mask to calibrate")
    dims, prov = resolve_dimensions(bboxes, axes, view_spec.known_dimensions)
    for k, note in (getattr(view_spec, "dimension_provenance", None) or {}).items():
        if prov.get(k) == "known":
            prov[k] = f"known ({note})"
    calibs: dict[str, ViewCalibration] = {}
    sheet_keys = [k for k in bboxes if view_spec.views[k].calibration.mode == "sheet"]
    sheet_info: dict[str, Any] = {}
    for k in [k for k in sheet_keys if view_frame(axes[k]).up_axis[1] != "Z"]:
        log.warning("view %s: sheet calibration needs an upright view; using silhouette_bbox", k)
        sheet_keys.remove(k)
    if sheet_keys:
        sheet_m, sheet_rows, sheet_us = sheet_calibration(
            {k: masks[k] for k in sheet_keys}, {k: bboxes[k] for k in sheet_keys}, dims["height"], labels)
        sheet_info = {"m_per_px": sheet_m, "views": sheet_keys}
    body_axes = _body_axes({k: masks[k] for k in bboxes if k not in sheet_keys and view_spec.views[k].calibration.mode != "manual"
                            and view_frame(axes[k]).up_axis[1] == "Z" and view_frame(axes[k]).right_axis[1] == "X"},
                           bboxes, labels)
    for key, (x0, y0, x1, y1) in bboxes.items():
        entry = view_spec.views[key]
        frame = view_frame(axes[key])
        h, w = masks[key].shape[:2]
        cfg = entry.calibration
        if cfg.mode == "manual":
            if not cfg.px_per_meter or cfg.origin_px is None:
                raise CalibrationError(f"view {key}: manual calibration needs px_per_meter and origin_px")
            m_per_px, origin = 1.0 / cfg.px_per_meter, (float(cfg.origin_px[0]), float(cfg.origin_px[1]))
        elif key in sheet_keys:
            m_per_px, origin = sheet_m, (sheet_us[key], sheet_rows[key])
        else:
            z_up = frame.up_axis[1] == "Z"
            m_per_px = dims["height"] / (y1 - y0) if z_up else dims["width"] / (x1 - x0)
            u = body_axes[key] if key in body_axes else (x0 + x1) / 2.0
            origin = (u, float(y1) if z_up else (y0 + y1) / 2.0)
        calibs[key] = ViewCalibration(key, axes[key], (w, h), float(m_per_px), origin)
        log.info("calibrated %s: %.6f m/px origin=%s", key, m_per_px, origin)
    info: dict[str, Any] = {"dimensions": dims, "provenance": prov, "bboxes_px": bboxes,
                            "frame_offset_m": frame_offset_m(calibs, bboxes)}
    if sheet_info:
        info["sheet"] = sheet_info
    return calibs, info
