"""Synthetic reference images: render the built model through ortho cameras onto a solid background.

Used to create test references for the table/chair examples (``run_pipeline.py --make-synthetic-references``).
The pixel scale is chosen so the model bbox falls exactly on pixel boundaries: the silhouette calibration the
pipeline later derives from the masks then reproduces these cameras exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .. import BBRError, get_logger
from ..core import imageio
from ..core.calibration import ViewCalibration, view_dimension_names
from ..core.conventions import view_frame
from . import mesh, render, scene

log = get_logger(__name__)

SYNTHETIC_PX_PER_M = 500.0
MARGIN_FRACTION = 0.2
MIN_MARGIN_PX = 40
CENTER_WARN_M = 0.001


def model_dimensions() -> tuple[dict[str, float], np.ndarray]:
    """(width/depth/height, bbox centre) of the built AI_MODEL meshes."""
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        raise BBRError("no model geometry to render synthetic references from")
    ext = box[1] - box[0]
    return {"width": float(ext[0]), "depth": float(ext[1]), "height": float(ext[2])}, (box[0] + box[1]) / 2.0


def synthetic_calibration(view: str, camera_axis: str, dims: Mapping[str, float],
                          px_per_m: float = SYNTHETIC_PX_PER_M) -> ViewCalibration:
    """Calibration of a synthetic view: object centred horizontally, floor (Z views) or centre (top/bottom) vertically."""
    hdim, vdim = view_dimension_names(camera_axis)
    w_px, h_px = dims[hdim] * px_per_m, dims[vdim] * px_per_m
    margin = max(MIN_MARGIN_PX, round(MARGIN_FRACTION * max(w_px, h_px)))
    width, height = int(round(w_px)) + 2 * margin, int(round(h_px)) + 2 * margin
    z_up = view_frame(camera_axis).up_axis[1] == "Z"
    origin = (width / 2.0, float(height - margin) if z_up else height / 2.0)
    return ViewCalibration(view, camera_axis, (width, height), 1.0 / px_per_m, origin)


def make_synthetic_references(view_spec, px_per_m: float = SYNTHETIC_PX_PER_M,
                              dims: Optional[Mapping[str, float]] = None) -> dict[str, str]:
    """Render every enabled non-preview view of the built model to its ``path`` (solid background, or a
    transparent one when the view's ``background.mode`` is ``alpha``)."""
    measured, centre = model_dimensions()
    if abs(centre[0]) > CENTER_WARN_M or abs(centre[1]) > CENTER_WARN_M:
        log.warning("model bbox centre is (%.4f, %.4f) m, not at the origin; calibration from these references "
                    "will place the origin at the bbox centre", centre[0], centre[1])
    use = dict(dims or measured)
    written: dict[str, str] = {}
    for key, axis in view_spec.view_axes().items():
        entry = view_spec.views[key]
        calib = synthetic_calibration(key, axis, use, px_per_m)
        rgba = render.render_color(key, calib, transparent=entry.background.mode == "alpha")
        if entry.flip_horizontal:
            rgba = rgba[:, ::-1]
        imageio.save_rgba(Path(entry.path), rgba)
        written[key] = str(entry.path)
        log.info("synthetic reference %s -> %s (%dx%d)", key, entry.path, *calib.image_size)
    return written


# --------------------------------------------------------------------------- benchmark helpers (ground truth + held-out renders)
HOLDOUT_AZIMUTHS = tuple(float(a) for a in range(0, 360, 45))
HOLDOUT_MARGIN = 1.25


def model_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Merged world-space ``(verts, tris)`` of the modifier-evaluated AI_MODEL meshes."""
    from . import hullpost
    return hullpost.world_triangles(scene.model_meshes())


def export_obj(path: Path | str) -> dict[str, int]:
    """Write the AI_MODEL meshes (evaluated, world space) as one Wavefront OBJ (for the benchmark's mesh scores)."""
    from ..core import geomscore
    verts, tris = model_triangles()
    if len(tris) == 0:
        raise BBRError("no model geometry to export")
    geomscore.write_obj(path, verts, tris)
    return {"vertices": int(len(verts)), "triangles": int(len(tris))}


def holdout_camera(px_per_m: float = SYNTHETIC_PX_PER_M, elevation: float = 0.0) -> dict:
    """Camera of a held-out sheet: one square canvas and bbox-centre target shared by all azimuths (JSON-able)."""
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        raise BBRError("no model geometry to render held-out views from")
    side = int(np.ceil(HOLDOUT_MARGIN * float(np.linalg.norm(box[1] - box[0])) * px_per_m))
    return {"px_per_m": float(px_per_m), "size": [side, side], "centre": [float(x) for x in (box[0] + box[1]) / 2.0],
            "elevation": float(elevation), "azimuths": list(HOLDOUT_AZIMUTHS)}


def render_azimuths(cam: Mapping, out_dir: Path | str) -> dict[float, str]:
    """Flat colour renders (transparent background, no antialiasing) of the model from every azimuth of ``cam``
    (:func:`holdout_camera`) into ``<out_dir>/az_<NNN>.png``."""
    out = Path(out_dir)
    written: dict[float, str] = {}
    for az in cam["azimuths"]:
        path = out / f"az_{int(round(az)):03d}.png"
        rgba = render.render_color_dir(az, cam["elevation"], tuple(cam["size"]), 1.0 / cam["px_per_m"],
                                       np.asarray(cam["centre"], dtype=float), transparent=True)
        imageio.save_rgba(path, rgba)
        written[float(az)] = str(path)
    return written
