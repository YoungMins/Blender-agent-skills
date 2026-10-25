"""Orthographic depth buffer of a triangle mesh on a view's pixel grid (numpy only).

Depth is the height along the view's ``camera_axis`` (larger = closer to the camera), so a surface point is
visible from the view when its height is >= the buffer value minus a tolerance.
"""
from __future__ import annotations

import numpy as np

from .calibration import ViewCalibration
from .conventions import AXES
from .uvraster import EDGE_EPS, barycentric, pair_chunks

NO_SURFACE = -np.inf


def view_height(points: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """Height of world points along the camera axis (distance from the plane through the origin)."""
    return np.asarray(points, dtype=float) @ AXES[calib.camera_axis]


def depth_buffer(verts: np.ndarray, tris: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """``(H, W)`` buffer of the nearest surface height per pixel (``-inf`` where no triangle covers the pixel centre)."""
    w, h = calib.image_size
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    zbuf = np.full((h, w), NO_SURFACE)
    if len(tris) == 0:
        return zbuf
    txy = calib.world_to_px(verts)[tris]
    tz = view_height(verts, calib)[tris]
    for tri, px, py in pair_chunks(txy, (w, h)):
        t = txy[tri]
        bary = barycentric(np.stack([px + 0.5, py + 0.5], axis=1), t[:, 0], t[:, 1], t[:, 2])
        inside = (bary >= -EDGE_EPS).all(axis=1)
        z = np.einsum("nk,nk->n", bary[inside], tz[tri[inside]])
        np.maximum.at(zbuf, (py[inside], px[inside]), z)
    return zbuf


def visible(points: np.ndarray, calib: ViewCalibration, zbuf: np.ndarray, tol: float) -> np.ndarray:
    """Bool per world point: not behind the buffer surface by more than ``tol`` (meters).

    Pixels without any surface sample count as visible (silhouette-edge texels land there).
    """
    h, w = zbuf.shape
    uv = calib.world_to_px(points)
    px = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 1)
    return view_height(points, calib) >= zbuf[py, px] - tol
