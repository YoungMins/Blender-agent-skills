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
