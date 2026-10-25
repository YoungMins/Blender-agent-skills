"""Camera pose of held-out views that are not on an axis (numpy only): azimuth / elevation frames and a small
pose search (azimuth, scale) that fits a rendered silhouette to a reference mask.

Azimuth 0 is the front (camera on -Y), 90 the +X side, 180 the back, 270 the -X side; elevation is degrees above
the horizon. The frame of an azimuth that falls on an axis equals the :mod:`conventions` frame of that axis.
A fit is a similarity transform (uniform scale + translation) from the model's plane coordinates (meters,
right / up of the view, origin = the centre of the canonical render) to the reference pixel grid: the model's
bbox bottom-centre is put on the reference bbox bottom-centre and ``scale`` multiplies the height-fit size.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from .conventions import AXES, view_frame
from .imageops import mask_bbox
from .metrics import iou

MAX_ELEVATION_DEG = 89.0
DEFAULT_AZIMUTH_RANGE = 15.0
DEFAULT_AZIMUTH_STEP = 5.0
DEFAULT_SCALE_RANGE = 0.05
DEFAULT_SCALE_STEPS = 5
_AXIS_ANGLES = {"-Y": 0.0, "+X": 90.0, "+Y": 180.0, "-X": 270.0}


def axis_angles(camera_axis: str) -> tuple[float, float]:
    """``(azimuth, elevation)`` of a camera axis (``+Z`` -> elevation 90, ``-Z`` -> -90)."""
    if camera_axis in _AXIS_ANGLES:
        return _AXIS_ANGLES[camera_axis], 0.0
    return 0.0, 90.0 if camera_axis == "+Z" else -90.0


def view_basis(azimuth: float, elevation: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(direction, right, up)`` unit vectors of a camera at ``azimuth`` / ``elevation`` degrees.

    ``direction`` points from the origin TO the camera (like ``camera_axis``); the camera looks along ``-direction``;
    ``right`` / ``up`` are the image axes (up has a positive Z component). At |elevation| = 90 the frame of the
    ``+Z`` / ``-Z`` axis is used.
    """
    el = float(np.clip(elevation, -90.0, 90.0))
    if abs(el) >= 90.0:
        f = view_frame("+Z" if el > 0 else "-Z")
        return -f.forward, f.right, f.up
    az, e = math.radians(azimuth), math.radians(el)
    d = np.array([math.sin(az) * math.cos(e), -math.cos(az) * math.cos(e), math.sin(e)])
    right = np.cross(-d, AXES["+Z"])
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    return d, right, up / np.linalg.norm(up)


def plane_bbox(mask: np.ndarray, m_per_px: float) -> dict[str, float]:
    """Silhouette bbox of a render centred on the plane origin, in meters: left / right (image right) and bottom / top (up)."""
    x0, y0, x1, y1 = mask_bbox(mask)
    h, w = mask.shape
    return {"left": (x0 - w / 2.0) * m_per_px, "right": (x1 - w / 2.0) * m_per_px,
            "top": -(y0 - h / 2.0) * m_per_px, "bottom": -(y1 - h / 2.0) * m_per_px}


def fit_transform(ref_bbox: tuple[int, int, int, int], model_bbox: dict[str, float], ref_shape: tuple[int, int],
                  scale: float = 1.0) -> dict[str, float]:
    """Reference-grid camera that puts the model on the reference silhouette.

    ``k`` = meters per reference pixel (height fit divided by ``scale``: a larger ``scale`` draws the model larger),
    ``p_r`` / ``p_up`` = plane coordinates of the reference image centre.
    """
    x0, y0, x1, y1 = ref_bbox
    h, w = ref_shape
    k = (model_bbox["top"] - model_bbox["bottom"]) / max(y1 - y0, 1) / scale
    return {"k": k,
            "p_r": (model_bbox["left"] + model_bbox["right"]) / 2.0 - ((x0 + x1) / 2.0 - w / 2.0) * k,
            "p_up": model_bbox["bottom"] + (y1 - h / 2.0) * k}


def warp_canonical(canon: np.ndarray, canon_m_per_px: float, ref_shape: tuple[int, int], fit: dict[str, float]
                   ) -> np.ndarray:
    """Resample the canonical render onto the reference grid with a :func:`fit_transform` (nearest neighbour)."""
    h, w = ref_shape
    ch, cw = canon.shape
    xs = (np.arange(w) + 0.5 - w / 2.0) * fit["k"] + fit["p_r"]
    ys = -(np.arange(h) + 0.5 - h / 2.0) * fit["k"] + fit["p_up"]
    u = np.floor(xs / canon_m_per_px + cw / 2.0).astype(int)
    v = np.floor(-ys / canon_m_per_px + ch / 2.0).astype(int)
    inside_u, inside_v = (u >= 0) & (u < cw), (v >= 0) & (v < ch)
    out = np.zeros((h, w), dtype=bool)
    out[np.ix_(inside_v, inside_u)] = canon[np.ix_(v[inside_v], u[inside_u])]
    return out


def search_pose(ref_mask: np.ndarray, render_fn: Callable[[float, float], np.ndarray], canon_m_per_px: float,
                nominal_azimuth: float, elevation: float = 0.0, azimuth_range: float = DEFAULT_AZIMUTH_RANGE,
                azimuth_step: float = DEFAULT_AZIMUTH_STEP, scale_range: float = DEFAULT_SCALE_RANGE,
                scale_steps: int = DEFAULT_SCALE_STEPS) -> dict[str, Any]:
    """Find the azimuth and scale at which the model silhouette best matches ``ref_mask``.

    ``render_fn(azimuth, elevation)`` returns the model silhouette (bool) rendered with an orthographic camera
    centred on the plane origin at ``canon_m_per_px`` (the same canvas every call). Candidates: ``nominal_azimuth``
    +- ``azimuth_range`` in ``azimuth_step`` steps, ``scale`` in ``1 +- scale_range`` (``scale_steps`` values).
    Returns the best ``{azimuth, scale, iou, k, p_r, p_up}`` plus ``nominal_iou`` (best over scales at the nominal
    azimuth), ``by_azimuth`` ({azimuth: best iou}) and the step sizes. Ties prefer the azimuth nearest the nominal one;
    the achievable angular accuracy is the step, and shallow IoU-vs-azimuth curves (blobby shapes) make it coarser.
    """
    ref = np.asarray(ref_mask, dtype=bool)
    ref_bbox = mask_bbox(ref)
    if azimuth_range > 0 and azimuth_step > 0:
        n = int(round(azimuth_range / azimuth_step))
        azimuths = [nominal_azimuth + i * azimuth_step for i in range(-n, n + 1)]
    else:
        azimuths = [float(nominal_azimuth)]
    scales = [1.0] if scale_steps <= 1 or scale_range <= 0 else list(np.linspace(1 - scale_range, 1 + scale_range, scale_steps))
    best: Optional[dict[str, Any]] = None
    by_az: dict[float, float] = {}
    for az in azimuths:
        canon = np.asarray(render_fn(az, elevation), dtype=bool)
        by_az[round(az, 3)] = 0.0
        if not canon.any():
            continue
        box = plane_bbox(canon, canon_m_per_px)
        for s in scales:
            fit = fit_transform(ref_bbox, box, ref.shape, s)
            value = iou(ref, warp_canonical(canon, canon_m_per_px, ref.shape, fit))
            by_az[round(az, 3)] = max(by_az[round(az, 3)], value)
            better = best is None or value > best["iou"] + 1e-9 or (
                abs(value - best["iou"]) <= 1e-9 and abs(az - nominal_azimuth) < abs(best["azimuth"] - nominal_azimuth))
            if better:
                best = {"azimuth": float(az), "scale": float(s), "iou": float(value), **fit}
    if best is None:
        best = {"azimuth": float(nominal_azimuth), "scale": 1.0, "iou": 0.0, "k": 0.0, "p_r": 0.0, "p_up": 0.0}
    best["nominal_iou"] = by_az.get(round(float(nominal_azimuth), 3), 0.0)
    best["by_azimuth"] = {str(k): v for k, v in by_az.items()}
    best["azimuth_step"], best["scale_range"] = float(azimuth_step), float(scale_range)
    return best
