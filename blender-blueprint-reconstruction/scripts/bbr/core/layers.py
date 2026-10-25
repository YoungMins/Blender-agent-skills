"""Layered parts for concavities (numpy only): a face recessed inside a helmet, a visor, a recessed panel.

A visual hull of front + side silhouettes cannot carve a concavity: the face is flush with the helmet front, so the
side reference's profile (eye, ear, cheek) has no surface to land on. A *layer* is a small hull of its own, carved from
the layer's label masks (the front and side pixels of "face" in ``labels.json``) over its own, finer bounding box, and
the body is recessed behind it:

* :func:`build_layer_field`: the layer's continuous hull field (same carve as the body, :mod:`bbr.core.carve`, with
  explicit ``bounds``) plus per-column (x, z) front / back surface depth and the footprint signed distance;
* :func:`recess_field`: removes the body from the volume ``footprint prism x {y < layer back}``. Inside the footprint
  the body's front surface moves back to the layer's back surface; outside it (the rim, the cheek guards) the body keeps
  its full depth, so the rim protrudes in front of the face exactly where the silhouettes say so.

The layer faces the same way as the front view (camera ``-Y``: the layer front is at its smallest y; the body is pushed
toward +y). The layer mesh stays separate from the body (own texture gated by its label), overlapping it by ``embed``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import CalibrationError
from .calibration import ViewCalibration
from .carve import _bilinear_table, build_hull_field
from .conventions import view_frame
from .distance import edt, signed_distance
from .imageops import mask_bbox

LAYER_FRONT_AXIS = "-Y"
DEFAULT_LAYER_RESOLUTION = 160    # cells along the longest side of the layer's bounding box
DEFAULT_LAYER_PAD_CELLS = 3       # empty cells around the layer's bounding box
DEFAULT_EMBED_M = 0.002           # the layer's back sinks this far into the recessed body (no gap, no z-fighting)


@dataclass
class LayerField:
    """Continuous field ``[nx, ny, nz]`` (m, negative inside) of a layer on its own grid, plus per-column surface data.

    ``y_front`` / ``y_back`` / ``footprint_sd`` are ``[nz, nx]`` (rows = z, columns = x; the layout of an image):
    front / back surface y of each (x, z) column (NaN where the layer has no material) and the signed distance (m,
    negative inside) of the footprint, the set of columns that contain material.
    """

    field: np.ndarray
    origin: np.ndarray
    cell: float
    y_front: np.ndarray
    y_back: np.ndarray
    footprint_sd: np.ndarray

    def centers(self, axis: int) -> np.ndarray:
        """World coordinates of the grid cell centres along world ``axis`` (0=X, 1=Y, 2=Z)."""
        return self.origin[axis] + (np.arange(self.field.shape[axis]) + 0.5) * self.cell


def layer_bounds(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration], front: str, side: str,
                 pad: float) -> tuple[np.ndarray, np.ndarray]:
    """World bounding box ``(lo, hi)`` of a layer: x from the ``front`` mask, y from the ``side`` mask, z where both
    masks have pixels, padded by ``pad`` m."""
    lo, hi = np.zeros(3), np.zeros(3)
    z_ranges = []
    for name, axis_i in ((front, 0), (side, 1)):
        calib = calibs[name]
        if abs(float(view_frame(calib.camera_axis).up[2])) < 0.99:
            raise CalibrationError(f"layer view {name} must have world Z as its image up axis")
        x0, y0, x1, y1 = mask_bbox(masks[name])
        plane = calib.px_to_plane(np.array([[x0, y0], [x1, y1]], dtype=float))
        lo[axis_i], hi[axis_i] = float(plane[:, axis_i].min()), float(plane[:, axis_i].max())
        z_ranges.append((float(plane[:, 2].min()), float(plane[:, 2].max())))
    lo[2], hi[2] = max(z[0] for z in z_ranges), min(z[1] for z in z_ranges)
    if hi[2] <= lo[2]:
        raise CalibrationError(f"the layer's {front} and {side} masks do not overlap in z "
                               f"({z_ranges[0]} vs {z_ranges[1]}): check the labels")
    return lo - pad, hi + pad


def column_extent(field: np.ndarray, origin: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Front (smallest y) and back (largest y) surface of every (x, z) column of ``field``, ``[nz, nx]`` each, NaN for
    columns without a negative voxel. The surface is placed on the field's zero crossing (linear), not on a voxel."""
    inside = field < 0
    nx, ny, nz = field.shape
    yc = origin[1] + (np.arange(ny) + 0.5) * cell
    has = inside.any(axis=1)
    i_front = inside.argmax(axis=1)
    i_back = ny - 1 - inside[:, ::-1, :].argmax(axis=1)

    def crossing(i_in: np.ndarray, i_out: np.ndarray) -> np.ndarray:
        f_in = np.take_along_axis(field, i_in[:, None, :], axis=1)[:, 0, :]
        f_out = np.take_along_axis(field, np.clip(i_out, 0, ny - 1)[:, None, :], axis=1)[:, 0, :]
        t = np.clip(f_in / np.where(f_in == f_out, 1.0, f_in - f_out), 0.0, 1.0)   # 0 at the inside centre, 1 outside
        return yc[i_in] + t * (yc[np.clip(i_out, 0, ny - 1)] - yc[i_in])

    front = np.where(has, crossing(i_front, i_front - 1), np.nan)
    back = np.where(has, crossing(i_back, i_back + 1), np.nan)
    return front.T.astype(np.float32), back.T.astype(np.float32)


def build_layer_field(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration], front: str, side: str,
                      resolution: int = DEFAULT_LAYER_RESOLUTION, exponent: float = 2.0, cross_section: str = "superellipse",
                      pad_cells: int = DEFAULT_LAYER_PAD_CELLS, pair_max_aspect: float = 4.0,
                      dims: Optional[Mapping[str, float]] = None) -> LayerField:
    """The layer's hull from its label masks ``masks[front]`` (a ``-Y`` view) and ``masks[side]`` (a ``+-X`` view),
    carved like the body (:func:`bbr.core.carve.build_hull_field`) over the layer's own bounding box at ``resolution``
    cells along its longest side."""
    if calibs[front].camera_axis != LAYER_FRONT_AXIS:
        raise CalibrationError(f"layer front view {front} must look from {LAYER_FRONT_AXIS} (camera_axis), "
                               f"got {calibs[front].camera_axis}")
    if calibs[side].camera_axis not in ("-X", "+X"):
        raise CalibrationError(f"layer side view {side} must look from -X or +X, got {calibs[side].camera_axis}")
    for name in (front, side):
        if name not in masks or not np.asarray(masks[name]).any():
            raise CalibrationError(f"the layer has no label pixels in view {name}")
    use = {name: np.asarray(masks[name], dtype=bool) for name in (front, side)}
    coarse = layer_bounds(use, calibs, front, side, 0.0)
    pad = pad_cells * float((coarse[1] - coarse[0]).max()) / max(resolution - 2 * pad_cells, 1)
    bounds = layer_bounds(use, calibs, front, side, pad)
    extent = bounds[1] - bounds[0]
    field, origin, cell = build_hull_field(
        use, {front: calibs[front], side: calibs[side]},
        dims or {"width": float(extent[0]), "depth": float(extent[1]), "height": float(extent[2])},
        int(resolution), carve_views=[front, side], cross_section=cross_section, exponent=exponent,
        pair_max_aspect=pair_max_aspect, bounds=bounds)
    if not (field < 0).any():
        raise CalibrationError("the layer is empty: its front and side label masks do not intersect "
                               "(check the labels and camera_axis)")
    y_front, y_back = column_extent(field, origin, cell)
    footprint = np.isfinite(y_front)
    sd = signed_distance(footprint, cap=int(DEFAULT_LAYER_RESOLUTION)) * cell
    return LayerField(field, origin, cell, y_front, y_back, sd.astype(np.float32))


def _fill_nearest(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``img`` with every invalid pixel replaced by its nearest valid pixel's value."""
    _, iy, ix = edt(valid, cap=max(img.shape), return_index=True)
    out = img.copy()
    bad = ~valid & (iy >= 0)
    out[bad] = img[iy[bad], ix[bad]]
    return out


def recess_field(field: np.ndarray, origin: Sequence[float], cell: float, layer: LayerField,
                 embed: float = DEFAULT_EMBED_M) -> tuple[np.ndarray, dict[str, Any]]:
    """Body field with the volume ``{(x, z) inside the layer footprint} x {y < layer back - embed}`` removed.

    ``embed`` (m) is how far the body front stands in front of the layer's back surface (the layer sinks into the
    body), clamped to half the layer thickness of the column so the body never pokes through the layer front.
    Returns ``(new field, stats)``; the input is not modified. Outside the footprint the field is unchanged in sign:
    the rim around the layer keeps its full depth.
    """
    origin = np.asarray(origin, dtype=float)
    nx, ny, nz = field.shape
    xc = origin[0] + (np.arange(nx) + 0.5) * cell
    yc = origin[1] + (np.arange(ny) + 0.5) * cell
    zc = origin[2] + (np.arange(nz) + 0.5) * cell
    lo = layer.origin
    u = (xc - lo[0]) / layer.cell
    v = (zc - lo[2]) / layer.cell
    valid = np.isfinite(layer.y_front)
    thickness = np.where(valid, layer.y_back - layer.y_front, 0.0)
    y_cut = np.where(valid, layer.y_back - np.minimum(embed, 0.5 * thickness), 0.0)
    y_cut = _fill_nearest(y_cut.astype(np.float32), valid)
    sd_tab = _bilinear_table(layer.footprint_sd / layer.cell, u, v) * layer.cell      # [nx, nz] m, > 0 outside
    cut_tab = _bilinear_table(y_cut, u, v)                                           # [nx, nz] m; only used inside
    sdv = np.maximum(sd_tab[:, None, :], yc[None, :, None] - cut_tab[:, None, :])    # sdf of the carved volume
    new = np.maximum(field, -sdv.astype(np.float32)).astype(np.float32)
    carved = (field < 0) & (new >= 0)
    per_col = carved.sum(axis=1) * cell
    stats = {"layer_cell_m": round(float(layer.cell), 6), "region_columns": int((sd_tab < 0).sum()),
             "carved_cells": int(carved.sum()), "max_recess_m": round(float(per_col.max()), 4)}
    return new, stats


DEFAULT_SHELL_THICKNESS_M = 0.0015   # glass of a visor / helmet bubble


def shell_mesh(layer: LayerField, thickness: float = DEFAULT_SHELL_THICKNESS_M) -> tuple[np.ndarray, np.ndarray]:
    """Closed thin shell over the layer's front surface: a glass visor in front of a face layer.

    Every (x, z) column with material gets a front vertex at ``y_front`` and a back vertex ``thickness`` behind it
    (+y); each 2x2 block of such columns is a front quad (facing -Y) and a back quad (facing +Y), and every boundary
    edge of the front sheet is closed by a wall quad. Returns ``(verts [n, 3], quads [m, 4])`` (outward winding)."""
    if thickness <= 0:
        raise ValueError(f"shell thickness must be > 0, got {thickness}")
    valid = np.isfinite(layer.y_front)                     # [nz, nx]
    if not valid.any():
        raise CalibrationError("the shell has no front surface (empty layer)")
    nz, nx = valid.shape
    xc, zc = layer.centers(0), layer.centers(2)
    index = -np.ones((nz, nx), dtype=np.int64)
    rows, cols = np.nonzero(valid)
    index[rows, cols] = np.arange(len(rows))
    n = len(rows)
    yf = layer.y_front[rows, cols].astype(float)
    front = np.stack([xc[cols], yf, zc[rows]], axis=1)
    back = front + np.array([0.0, thickness, 0.0])
    verts = np.concatenate([front, back])
    a, b = index[:-1, :-1], index[:-1, 1:]                 # (z, x), (z, x+1)
    c, d = index[1:, 1:], index[1:, :-1]                   # (z+1, x+1), (z+1, x)
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    # seen from -Y (x right, z up) a -> b -> c -> d is counter-clockwise: the front quad faces -Y
    fq = np.stack([a[ok], b[ok], c[ok], d[ok]], axis=1)
    if len(fq) == 0:
        raise CalibrationError("the shell footprint is too thin for one grid cell: raise the layer resolution")
    bq = fq[:, ::-1] + n
    edges = np.stack([fq, np.roll(fq, -1, axis=1)], axis=2).reshape(-1, 2)    # directed edges of the front quads
    key = np.sort(edges, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inv.ravel()] == 1]
    # a front boundary edge u -> v runs counter-clockwise around the sheet seen from -Y; the wall u, u', v', v faces out
    walls = np.stack([boundary[:, 1], boundary[:, 0], boundary[:, 0] + n, boundary[:, 1] + n], axis=1)
    return verts, np.concatenate([fq, bq, walls]).astype(np.int64)


# --------------------------------------------------------------------------- face detail: localised profile relief
DEFAULT_RELIEF_SMOOTH = 0.08   # profile smoothing sigma, share of the label's height: nose, lips, chin are finer than this
RELIEF_BASE_ITERATIONS = 12    # asymmetric baseline: smoothing with the protrusions clipped, repeated
DEFAULT_RELIEF_WIDTH = 0.03    # falloff sigma across the face, share of the layer's width (a skin-coloured nose: this alone)
DEFAULT_FEATURE_COLOR_DIST = 0.18   # RGB distance from the skin colour that marks a feature (hair fringe, mouth) in the front view
FEATURE_WIDTH_SMOOTH_ROWS = 5       # median over this many rows of the front-view feature width
FEATURE_WIDTH_BLUR = 0.03           # taper of the width over the rows, share of the label height: no ledge where a fringe ends
RELIEF_DY_BLUR_ROWS = 1.5           # Gaussian over the rows of the feature depth: pixel steps of the profile are not features


@dataclass
class ProfileRelief:
    """The side profile of a layer split into a smooth base and its features.

    ``mask``: the side label mask cut back / filled out to the smoothed profile (no features: carving it gives no ridge);
    ``z`` (ascending) and ``dy`` (m, world y): per height, the profile's front minus the base front (< 0: a feature such
    as the nose sticks out toward -Y; > 0: a hollow)."""
    mask: np.ndarray
    z: np.ndarray
    dy: np.ndarray
    half_width: Optional[np.ndarray] = None   # per z (m): flat core of the feature across the face (front view)

    @property
    def amplitude(self) -> float:
        """Largest feature depth (m)."""
        return float(np.abs(self.dy).max()) if len(self.dy) else 0.0


def _gaussian_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(float)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(values.astype(float), r, mode="reflect" if len(values) > r else "edge")
    return np.convolve(padded, k, mode="valid")


def profile_relief(mask: np.ndarray, calib: ViewCalibration, smooth: float = DEFAULT_RELIEF_SMOOTH) -> ProfileRelief:
    """Split the front contour of a side-view layer mask (camera ``+-X``) into a smooth base and its features.

    Per image row the contour is the mask pixel nearest the front (-Y); the base is a Gaussian smoothing over the rows
    (sigma ``smooth`` times the mask's height) repeated with the protrusions clipped to it, so it runs behind the nose,
    lips and chin instead of through them. The returned mask ends at the smoothed contour, so the layer carved from it has
    a featureless front; :func:`apply_relief` puts the features back on the face's midline only (a visual hull would
    extrude a nose seen in profile across the whole face width: a horizontal ridge)."""
    mask = np.asarray(mask, dtype=bool)
    f = view_frame(calib.camera_axis)
    if abs(float(f.right[1])) < 0.99:
        raise CalibrationError(f"relief needs a side view (camera +-X), got {calib.camera_axis}")
    front_is_left = float(f.right[1]) > 0          # image right = +y: the front (-y) is the leftmost pixel
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) < 3:
        raise CalibrationError("the layer's side label is too small for a profile relief")
    rows = np.arange(rows[0], rows[-1] + 1)
    present = mask[rows].any(axis=1)
    cols = np.arange(mask.shape[1])
    edge = np.where(mask[rows], cols[None, :], np.inf if front_is_left else -np.inf)
    front = edge.min(axis=1) if front_is_left else edge.max(axis=1)
    front = np.interp(rows, rows[present], front[present])            # rows without pixels (a gap) take their neighbours
    forward = -front if front_is_left else front                     # larger = further toward -Y
    clipped = forward.copy()
    for _ in range(RELIEF_BASE_ITERATIONS):    # baseline under the features: a protrusion must not pull the base forward
        base_f = _gaussian_1d(clipped, smooth * len(rows))
        clipped = np.minimum(forward, base_f)
    base = -base_f if front_is_left else base_f
    out = mask.copy()
    for r, fu, bu in zip(rows, front, base):
        lo, hi = (int(np.ceil(bu)), mask.shape[1]) if front_is_left else (0, int(np.floor(bu)) + 1)
        line = np.zeros(mask.shape[1], dtype=bool)
        line[max(lo, 0):min(hi, mask.shape[1])] = True
        if mask[r].any():   # cut a feature back to the base; fill a hollow from the old front out to the base
            out[r] = line & (mask[r] | _between(cols, fu, bu))
    z = calib.px_to_plane(np.stack([np.zeros(len(rows)), rows + 0.5], axis=1).astype(float))[:, 2]
    y_front = calib.px_to_plane(np.stack([front + 0.5, rows + 0.5], axis=1))[:, 1]
    y_base = calib.px_to_plane(np.stack([base + 0.5, rows + 0.5], axis=1))[:, 1]
    order = np.argsort(z)
    return ProfileRelief(out, z[order], _gaussian_1d(y_front - y_base, RELIEF_DY_BLUR_ROWS)[order])


def _between(cols: np.ndarray, a: float, b: float) -> np.ndarray:
    lo, hi = min(a, b), max(a, b)
    return (cols >= np.floor(lo)) & (cols <= np.ceil(hi))


def skin_color(rgb: np.ndarray, mask: np.ndarray, levels: int = 12) -> np.ndarray:
    """Dominant colour of ``mask`` in an RGB image (0..1): the median of the most frequent quantised colour bin."""
    px = np.asarray(rgb, dtype=float)[..., :3][np.asarray(mask, dtype=bool)]
    q = np.clip((px * levels).astype(int), 0, levels - 1)
    key = (q[:, 0] * levels + q[:, 1]) * levels + q[:, 2]
    return np.median(px[key == np.bincount(key).argmax()], axis=0)


def feature_half_widths(rgb: np.ndarray, mask: np.ndarray, calib: ViewCalibration, x0: float,
                        threshold: float = DEFAULT_FEATURE_COLOR_DIST) -> tuple[np.ndarray, np.ndarray]:
    """How wide each height's profile feature is across the face, read off the front view (camera ``-Y``).

    Per image row of ``mask`` (the face label): the run of pixels through the midline ``x0`` whose colour differs from
    the skin (:func:`skin_color`) by more than ``threshold``; its half-width in metres (0 where the midline pixel is
    skin, e.g. a nose drawn by shading only). A hair fringe spans the face, a mouth its own width. Returns ``(z ascending,
    half_width)``; widths are a running median over FEATURE_WIDTH_SMOOTH_ROWS rows, then their ends tapered (the max with a FEATURE_WIDTH_BLUR
    blur)."""
    rgb = np.asarray(rgb, dtype=float)[..., :3]
    rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
    mask = np.asarray(mask, dtype=bool)
    off = (np.linalg.norm(rgb - skin_color(rgb, mask), axis=2) > threshold) & mask
    rows = np.nonzero(mask.any(axis=1))[0]
    xc = int(np.clip(np.floor(calib.world_to_px(np.array([[x0, 0.0, 0.0]]))[0, 0]), 0, mask.shape[1] - 1))
    widths = np.zeros(len(rows))
    for i, r in enumerate(rows):
        line = off[r]
        if not line[xc]:
            continue
        left = xc - (np.argmin(line[xc::-1]) if not line[xc::-1].all() else xc + 1)
        right = xc + (np.argmin(line[xc:]) if not line[xc:].all() else line.size - xc)
        widths[i] = 0.5 * (right - left - 1) * calib.m_per_px
    k = FEATURE_WIDTH_SMOOTH_ROWS // 2
    padded = np.pad(widths, k, mode="edge")
    widths = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1), axis=1)
    widths = np.maximum(widths, _gaussian_1d(widths, FEATURE_WIDTH_BLUR * len(rows)))   # taper the ends, keep the widths
    z = calib.px_to_plane(np.stack([np.zeros(len(rows)), rows + 0.5], axis=1).astype(float))[:, 2]
    order = np.argsort(z)
    return z[order], widths[order]


def apply_relief(verts: np.ndarray, layer: LayerField, relief: ProfileRelief, x0: float,
                 width: float) -> np.ndarray:
    """Put the profile features of ``relief`` back onto a layer carved from ``relief.mask``: every vertex in the front
    half of its (x, z) column moves along y by ``dy(z) * exp(-(max(|x - x0| - half_width(z), 0) / width)^2 / 2)``: flat
    over the feature's front-view half-width (``relief.half_width``, 0 without it), a Gaussian falloff of sigma ``width``
    beyond; fading to 0 at the column's middle (the back of the layer, where the body is recessed, does not move).
    Returns new vertices."""
    v = np.asarray(verts, dtype=float).copy()
    dy = np.interp(v[:, 2], relief.z, relief.dy, left=0.0, right=0.0)
    core = (np.interp(v[:, 2], relief.z, relief.half_width, left=0.0, right=0.0) if relief.half_width is not None
            else np.zeros(len(v)))
    across = np.exp(-0.5 * (np.maximum(np.abs(v[:, 0] - x0) - core, 0.0) / max(width, 1e-9)) ** 2)
    nz, nx = layer.y_front.shape
    ix = np.clip(np.round((v[:, 0] - layer.origin[0]) / layer.cell - 0.5).astype(int), 0, nx - 1)
    iz = np.clip(np.round((v[:, 2] - layer.origin[2]) / layer.cell - 0.5).astype(int), 0, nz - 1)
    yf, yb = layer.y_front[iz, ix].astype(float), layer.y_back[iz, ix].astype(float)
    half = np.where(np.isfinite(yf) & np.isfinite(yb), 0.5 * (yb - yf), np.nan)
    front_ness = np.where(np.isfinite(half) & (half > 0), np.clip(1.0 - (v[:, 1] - yf) / np.where(half > 0, half, 1.0), 0.0, 1.0), 0.0)
    v[:, 1] += dy * across * front_ness
    return v
