"""Reference image planes (REF_BLUEPRINT): one unlit, semi-transparent plane per enabled view."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Optional

import bpy
import numpy as np

from .. import get_logger
from ..core import imageio
from ..core.calibration import ViewCalibration
from ..core.config import PipelineConfig
from ..core.conventions import AXES
from . import material, mesh, scene

log = get_logger(__name__)

PLANE_OPACITY = 0.5
UV_CORNERS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))  # BL, BR, TR, TL


def _image_for(key: str, entry, rgba: Optional[np.ndarray], cache_dir: Path) -> bpy.types.Image:
    """Blender image of the (cropped / flipped) reference; the original file is used when nothing was changed."""
    if entry.crop_px or entry.flip_horizontal:
        if rgba is None:
            rgba = imageio.load_view_image(entry)
        path = cache_dir / f"{key}.png"
        imageio.save_rgba(path, rgba)
    else:
        path = entry.path
    img = bpy.data.images.load(str(path), check_existing=True)
    img.reload()
    return img


def setup_reference_views(view_spec, calibs: Mapping[str, ViewCalibration], dims: Mapping[str, float],
                          config: Optional[PipelineConfig] = None, images: Optional[Mapping[str, np.ndarray]] = None,
                          cache_dir: Optional[Path | str] = None) -> dict[str, bpy.types.Object]:
    """Create/update one image plane ``ref:<view>`` per calibrated view; pixels land on their world positions.

    The plane lies on the plane through the origin facing the camera, then is pushed back along ``-camera_axis`` by
    ``ref_plane_offset_factor * max(dims) + ref_plane_offset_base`` so it sits behind the model.
    Planes are not selectable and not rendered. Re-running updates the same objects.
    """
    cfg = config or PipelineConfig()
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "bbr_refs"
    offset = cfg.ref_plane_offset_factor * max(dims.values()) + cfg.ref_plane_offset_base
    coll = scene.ensure_collection(scene.COLL_REF)
    planes: dict[str, bpy.types.Object] = {}
    for key, calib in calibs.items():
        entry = view_spec.views[key]
        w, h = calib.image_size
        corners_px = np.array([[0, h], [w, h], [w, 0], [0, 0]], dtype=float)
        back = -AXES[calib.camera_axis]
        verts = calib.px_to_plane(corners_px) + back * offset
        agent_id = f"ref:{key}"
        data = mesh.get_or_create_mesh(agent_id)
        mesh.mesh_from_arrays(data, verts, np.array([[0, 1, 2, 3]]))
        uv = data.uv_layers.get("UVMap") or data.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", np.array(UV_CORNERS, dtype=np.float32).ravel())
        image = _image_for(key, entry, None if images is None else images.get(key), cache)
        data.materials.clear()
        data.materials.append(material.ensure_reference_material(f"ref:{key}", image, PLANE_OPACITY))
        obj = scene.ensure_object(agent_id, coll, data)
        obj.hide_select = True
        obj.hide_render = True
        planes[key] = obj
    for obj in scene.objects_in(scene.COLL_REF):
        if str(obj.get(scene.PROP_AGENT_ID, "")).startswith("ref:") and obj.name not in {p.name for p in planes.values()}:
            scene.remove_object(obj)
    log.info("reference planes: %s (offset %.3f m)", sorted(planes), offset)
    return planes

