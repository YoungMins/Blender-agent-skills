"""Cameras: exact-framing orthographic cameras derived from a ViewCalibration and a 3/4 preview camera."""
from __future__ import annotations

import math
from typing import Sequence

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .. import get_logger
from ..core.calibration import ViewCalibration
from ..core import pose
from ..core.conventions import view_frame
from . import scene

log = get_logger(__name__)

CLIP_START = 0.01
CLIP_MARGIN_FACTOR = 4.0
PREVIEW_DIRECTION = (-0.75, -1.0, 0.6)  # front-right-above (the object's right side is -X)
PREVIEW_LENS_MM = 50.0
PREVIEW_FIT_MARGIN = 1.1
PREVIEW_AGENT_ID = "cam:preview"


def _camera_object(agent_id: str, name: str) -> bpy.types.Object:
    data = bpy.data.cameras.get(agent_id) or bpy.data.cameras.new(agent_id)
    obj = scene.ensure_object(agent_id, scene.ensure_collection(scene.COLL_DEBUG), data)
    obj.hide_render = False
    return obj


def ensure_ortho_camera(view: str, calib: ViewCalibration, distance: float) -> bpy.types.Object:
    """Orthographic camera ``cam:<view>`` whose pixel grid equals the calibrated reference image.

    ``ortho_scale`` spans the LARGER image side (sensor_fit AUTO), so the camera framing covers exactly
    ``image_size * m_per_px`` meters, centred on the world point that projects to the image centre.
    """
    obj = _camera_object(f"cam:{view}", view)
    cam = obj.data
    w, h = calib.image_size
    f = view_frame(calib.camera_axis)
    right_m = (w / 2.0 - calib.origin_px[0]) * calib.m_per_px
    up_m = -(h / 2.0 - calib.origin_px[1]) * calib.m_per_px
    centre = f.right * right_m + f.up * up_m
    cam.type = "ORTHO"
    cam.sensor_fit = "AUTO"
    cam.ortho_scale = max(w, h) * calib.m_per_px
    cam.shift_x = cam.shift_y = 0.0
    cam.clip_start = CLIP_START
    cam.clip_end = distance * CLIP_MARGIN_FACTOR
    basis = Matrix(np.column_stack([f.right, f.up, -f.forward]).tolist())  # camera X, Y, Z (looks along -Z)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = basis.to_quaternion()
    obj.location = Vector(centre - f.forward * distance)
    return obj


def ensure_ortho_camera_dir(name: str, azimuth: float, elevation: float, size: tuple[int, int], m_per_px: float,
                            centre: Sequence[float], distance: float) -> bpy.types.Object:
    """Orthographic camera ``cam:<name>`` looking at world point ``centre`` from ``azimuth`` / ``elevation`` degrees
    (see :mod:`bbr.core.pose`), framing ``size`` pixels of ``m_per_px`` meters each (the larger side spans the sensor)."""
    obj = _camera_object(f"cam:{name}", name)
    cam = obj.data
    direction, right, up = pose.view_basis(azimuth, elevation)
    cam.type = "ORTHO"
    cam.sensor_fit = "AUTO"
    cam.ortho_scale = max(size) * m_per_px
    cam.shift_x = cam.shift_y = 0.0
    cam.clip_start = CLIP_START
    cam.clip_end = distance * CLIP_MARGIN_FACTOR
    basis = Matrix(np.column_stack([right, up, direction]).tolist())
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = basis.to_quaternion()
    obj.location = Vector(np.asarray(centre, dtype=float) + direction * distance)
    return obj


def ensure_preview_camera(bbox_min: Sequence[float], bbox_max: Sequence[float],
                          direction: Sequence[float] = PREVIEW_DIRECTION) -> bpy.types.Object:
    """Perspective 3/4 camera that frames the bounding box (default front-right-above; ``direction`` = from centre to camera)."""
    obj = _camera_object(PREVIEW_AGENT_ID, "preview")
    cam = obj.data
    lo, hi = np.asarray(bbox_min, float), np.asarray(bbox_max, float)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    cam.type = "PERSP"
    cam.lens = PREVIEW_LENS_MM
    cam.sensor_fit = "AUTO"
    half_fov = math.atan(cam.sensor_width / (2.0 * cam.lens))
    distance = PREVIEW_FIT_MARGIN * radius / math.sin(half_fov)
    direction = Vector(direction).normalized()
    cam.clip_start = CLIP_START
    cam.clip_end = distance * CLIP_MARGIN_FACTOR
    obj.rotation_mode = "QUATERNION"
    obj.location = Vector(centre) + direction * distance
    obj.rotation_quaternion = (-direction).to_track_quat("-Z", "Y")
    return obj
