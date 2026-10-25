"""Workbench renders of masks, colors and the 3/4 preview. Scene settings and visibility are always restored."""
from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import bpy
import numpy as np
from mathutils import Vector

from .. import BBRError, get_logger
from ..core import imageio, imageops, pose
from ..core.calibration import ViewCalibration
from . import camera, mesh, quiet, scene
from .material import REF_TEXTURE_MATERIAL

log = get_logger(__name__)

ENGINE = "BLENDER_WORKBENCH"
MASK_ALPHA_THRESHOLD = 0.5
MASK_AA = "8"
MASK_FILTER_SIZE = 1.0
CRISP_AA = "OFF"  # pixel-centre sampling: used for synthetic references (exact, no blended edge pixels)
SYNTHETIC_BG = (0.8, 0.8, 0.8)
PREVIEW_BG = (0.45, 0.5, 0.55)
PREVIEW_SIZE = (1024, 768)
PREVIEW_AA = "16"
PREVIEW_SAMPLES = 32
PREVIEW_WORLD = "bbr:preview_world"
PREVIEW_SKY = ((0.0, 0.02), (0.5, 0.18), (1.0, 0.45))   # (position, grey) of the world gradient: ground, horizon, zenith
RIG_PREFIX = "bbr:preview_light_"
# Studio 3-point rig ported from EL4CTEO/blender-skills (MIT, turntable.py): (name, azimuth deg from the camera,
# elevation deg, power factor, size as a fraction of the model radius)
STUDIO_RIG = (("key", -45.0, 35.0, 1.0, 1.0), ("fill", 50.0, 15.0, 0.35, 1.5), ("rim", 160.0, 40.0, 0.8, 0.6))
RIG_POWER_PER_M2 = 18.0

ID_LEVELS = 4                          # object-ID colours: 5 sRGB levels per channel (0, .25, ... 1) -> 124 objects
TURNTABLE_TILE = (320, 320)
TURNTABLE_SAMPLES = 16
TURNTABLE_ELEVATION = 15.0
TURNTABLE_COLUMNS = 4

_RENDER_ATTRS = ("engine", "resolution_x", "resolution_y", "resolution_percentage", "film_transparent",
                 "filepath", "filter_size", "use_file_extension", "dither_intensity")
_IMAGE_ATTRS = ("file_format", "color_mode", "color_depth")
_VIEW_ATTRS = ("view_transform", "look", "exposure", "gamma")
_SHADING_ATTRS = ("light", "color_type", "single_color", "show_object_outline", "show_cavity",
                  "show_shadows", "show_specular_highlight")


def _record(owner: Any, names: Iterable[str]) -> list[tuple[Any, str, Any]]:
    return [(owner, n, getattr(owner, n)) for n in names if hasattr(owner, n)]


@contextlib.contextmanager
def render_session(visible: Iterable[bpy.types.Object]) -> Iterator[bpy.types.Scene]:
    """Make only ``visible`` objects render; restore render settings, world color and visibility on exit."""
    sc = bpy.context.scene
    saved = (_record(sc.render, _RENDER_ATTRS) + _record(sc.render.image_settings, _IMAGE_ATTRS)
             + _record(sc.view_settings, _VIEW_ATTRS) + _record(sc.display, ("render_aa",))
             + _record(sc.display.shading, _SHADING_ATTRS) + _record(sc, ("camera",))
             + (_record(sc.cycles, ("samples", "use_denoising", "device")) if hasattr(sc, "cycles") else []))
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    saved += _record(sc.world, ("color",))
    saved += [(o, "hide_render", o.hide_render) for o in bpy.data.objects]
    keep = set(visible)
    try:
        for obj in bpy.data.objects:
            obj.hide_render = obj not in keep
        yield sc
    finally:
        for owner, attr, value in saved:
            try:
                setattr(owner, attr, value)
            except (AttributeError, TypeError, ReferenceError) as exc:
                log.warning("could not restore %s: %s", attr, exc)


def _configure(sc: bpy.types.Scene, cam: bpy.types.Object, size: tuple[int, int], *, transparent: bool,
               aa: str, filter_size: float, light: str, color_type: str, bg: Sequence[float]) -> None:
    sc.render.engine = ENGINE
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    sc.render.filter_size = filter_size
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display.render_aa = aa
    shading = sc.display.shading
    shading.light = light
    shading.color_type = color_type
    shading.show_object_outline = False
    shading.show_cavity = False
    shading.show_shadows = False
    shading.show_specular_highlight = light != "FLAT"
    sc.world.color = tuple(bg)
    sc.camera = cam


def _render_file(sc: bpy.types.Scene, path: Path) -> np.ndarray:
    """Render the current camera to ``path`` and load it as float RGBA (top-down)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sc.render.use_file_extension = False
    sc.render.filepath = str(path)
    with bpy.context.temp_override(scene=sc), quiet.native_stdout():  # the operator renders exactly the scene we configured
        bpy.ops.render.render(write_still=True)
    if not path.is_file():
        raise BBRError(f"render produced no file at {path}")
    return imageio.load_rgba(path)


@contextlib.contextmanager
def _scratch_path(out_path: Optional[Path | str]) -> Iterator[Path]:
    if out_path:
        yield Path(out_path)
        return
    fd, name = tempfile.mkstemp(prefix="bbr_render_", suffix=".png")
    os.close(fd)
    try:
        yield Path(name)
    finally:
        Path(name).unlink(missing_ok=True)


def _view_distance() -> float:
    box = mesh.world_bbox(scene.model_meshes())
    extent = 1.0 if box is None else float(max(np.abs(box[0]).max(), np.abs(box[1]).max()))
    return 2.0 * extent + 1.0


def _color_type() -> str:
    """TEXTURE display when a model mesh carries the projected texture material, else MATERIAL."""
    for obj in scene.model_meshes():
        if any(m is not None and m.name.startswith(REF_TEXTURE_MATERIAL) for m in obj.data.materials):
            return "TEXTURE"
    return "MATERIAL"


def render_mask(view: str, calib: ViewCalibration, out_path: Optional[Path | str] = None,
                render_scale: float = 1.0, aa: str = MASK_AA,
                objects: Optional[Sequence[bpy.types.Object]] = None) -> np.ndarray:
    """Silhouette of the AI_MODEL meshes (or only ``objects``) from ``view`` on the calibrated pixel grid (bool ``H x W``)."""
    w, h = calib.image_size
    size = (max(1, round(w * render_scale)), max(1, round(h * render_scale)))
    cam = camera.ensure_ortho_camera(view, calib, _view_distance())
    with render_session(scene.model_meshes() if objects is None else objects) as sc, _scratch_path(out_path) as path:
        _configure(sc, cam, size, transparent=True, aa=aa, filter_size=MASK_FILTER_SIZE, light="FLAT",
                   color_type="SINGLE", bg=(0.0, 0.0, 0.0))
        rgba = _render_file(sc, path)
    mask = rgba[..., 3] > MASK_ALPHA_THRESHOLD
    return imageops.resize_nearest(mask, (w, h)) if size != (w, h) else mask


def render_color(view: str, calib: ViewCalibration, out_path: Optional[Path | str] = None,
                 render_scale: float = 1.0, bg: Sequence[float] = SYNTHETIC_BG, aa: str = CRISP_AA,
                 flat: bool = True, transparent: bool = False) -> np.ndarray:
    """Color render from ``view`` on a solid ``bg`` (or transparent); the material / projected color attribute is the color."""
    w, h = calib.image_size
    size = (max(1, round(w * render_scale)), max(1, round(h * render_scale)))
    cam = camera.ensure_ortho_camera(view, calib, _view_distance())
    with render_session(scene.model_meshes()) as sc, _scratch_path(out_path) as path:
        _configure(sc, cam, size, transparent=transparent, aa=aa, filter_size=MASK_FILTER_SIZE,
                   light="FLAT" if flat else "STUDIO", color_type=_color_type(), bg=bg)
        return _render_file(sc, path)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def id_color(index: int) -> tuple[float, float, float]:
    """sRGB colour (0..1 per channel) that marks object ``index`` (0-based) in an ID render."""
    code, base = index + 1, ID_LEVELS + 1
    if code >= base ** 3:
        raise BBRError(f"too many objects for an ID render ({index + 1} > {base ** 3 - 1})")
    return (code % base) / ID_LEVELS, (code // base % base) / ID_LEVELS, (code // base ** 2) / ID_LEVELS


def decode_ids(rgba: np.ndarray) -> np.ndarray:
    """ID map ``H x W`` (0 = background, ``index + 1``) from a flat-colour ID render; transparent pixels are 0."""
    base = ID_LEVELS + 1
    lv = np.clip(np.rint(rgba[..., :3] * ID_LEVELS), 0, ID_LEVELS).astype(np.int32)
    ids = lv[..., 0] + lv[..., 1] * base + lv[..., 2] * base ** 2
    return np.where(rgba[..., 3] > MASK_ALPHA_THRESHOLD, ids, 0).astype(np.int32)


def render_object_ids(view: str, calib: ViewCalibration, objects: Sequence[bpy.types.Object]) -> np.ndarray:
    """Object-ID render from ``view`` on the calibrated grid: every object of ``objects`` gets a flat unique colour
    (:func:`id_color`); the map holds ``index in objects + 1`` per pixel (0 = background). No antialiasing."""
    w, h = calib.image_size
    cam = camera.ensure_ortho_camera(view, calib, _view_distance())
    saved = [(o, tuple(o.color)) for o in objects]
    try:
        for i, obj in enumerate(objects):
            obj.color = (*(_srgb_to_linear(c) for c in id_color(i)), 1.0)
        with render_session(objects) as sc, _scratch_path(None) as path:
            _configure(sc, cam, (w, h), transparent=True, aa=CRISP_AA, filter_size=MASK_FILTER_SIZE, light="FLAT",
                       color_type="OBJECT", bg=(0.0, 0.0, 0.0))
            sc.render.dither_intensity = 0.0
            rgba = _render_file(sc, path)
    finally:
        for obj, color in saved:
            obj.color = color
    return decode_ids(rgba)


def render_mask_dir(azimuth: float, elevation: float, size: tuple[int, int], m_per_px: float, centre: np.ndarray,
                    objects: Optional[Sequence[bpy.types.Object]] = None, aa: str = MASK_AA) -> np.ndarray:
    """Silhouette from an arbitrary orthographic direction (see :mod:`pose`); ``centre`` = world point at the image
    centre, ``m_per_px`` the pixel size. Bool ``H x W`` with ``size = (W, H)``."""
    cam = camera.ensure_ortho_camera_dir("dir", azimuth, elevation, size, m_per_px, centre, _view_distance())
    with render_session(scene.model_meshes() if objects is None else objects) as sc, _scratch_path(None) as path:
        _configure(sc, cam, size, transparent=True, aa=aa, filter_size=MASK_FILTER_SIZE, light="FLAT",
                   color_type="SINGLE", bg=(0.0, 0.0, 0.0))
        rgba = _render_file(sc, path)
    return rgba[..., 3] > MASK_ALPHA_THRESHOLD


def render_color_dir(azimuth: float, elevation: float, size: tuple[int, int], m_per_px: float, centre: np.ndarray,
                     out_path: Optional[Path | str] = None, bg: Sequence[float] = SYNTHETIC_BG, aa: str = CRISP_AA,
                     transparent: bool = False) -> np.ndarray:
    """Flat colour render from an arbitrary orthographic direction (framing as :func:`render_mask_dir`)."""
    cam = camera.ensure_ortho_camera_dir("dir", azimuth, elevation, size, m_per_px, centre, _view_distance())
    with render_session(scene.model_meshes()) as sc, _scratch_path(out_path) as path:
        _configure(sc, cam, size, transparent=transparent, aa=aa, filter_size=MASK_FILTER_SIZE, light="FLAT",
                   color_type=_color_type(), bg=bg)
        return _render_file(sc, path)


@contextlib.contextmanager
def preview_world(sc: bpy.types.Scene) -> Iterator[bpy.types.World]:
    """Temporary gradient world for the lit preview: ground 0.02 / horizon 0.18 / zenith 0.45, so metal reflects something
    other than a flat dark grey (the default world's colour never reaches Cycles). The scene's own world is restored and
    the temporary one deleted on exit. TexCoord.Generated -> Z -> Map Range (-1..1 -> 0..1) -> Color Ramp -> Background."""
    old = sc.world
    world = bpy.data.worlds.new(PREVIEW_WORLD)
    try:
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        coord = nt.nodes.new("ShaderNodeTexCoord")
        split = nt.nodes.new("ShaderNodeSeparateXYZ")
        rng = nt.nodes.new("ShaderNodeMapRange")
        rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = -1.0, 1.0
        rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 0.0, 1.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        elements = ramp.color_ramp.elements
        while len(elements) < len(PREVIEW_SKY):
            elements.new(0.5)
        for el, (pos, grey) in zip(elements, PREVIEW_SKY):
            el.position, el.color = pos, (grey, grey, grey, 1.0)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 1.0
        out = nt.nodes.new("ShaderNodeOutputWorld")
        chain = (coord, split, rng, ramp, bg, out)
        for i, node in enumerate(chain):
            node.location = (i * 220, 0)
        nt.links.new(coord.outputs["Generated"], split.inputs["Vector"])
        nt.links.new(split.outputs["Z"], rng.inputs["Value"])
        nt.links.new(rng.outputs["Result"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        sc.world = world
        yield world
    finally:
        sc.world = old
        bpy.data.worlds.remove(world)


@contextlib.contextmanager
def studio_rig(center: np.ndarray, radius: float, direction: Sequence[float]) -> Iterator[list[bpy.types.Object]]:
    """Temporary key / fill / rim area lights around ``center`` (azimuth relative to the preview camera); removed on exit.

    The lights live in AI_DEBUG with an agent_id while they exist and never reach the validation (silhouette) renders.
    """
    coll = scene.ensure_collection(scene.COLL_DEBUG)
    d = np.asarray(direction, dtype=float)
    cam_az = math.atan2(d[1], d[0])
    dist = 2.4 * radius
    lights: list[bpy.types.Object] = []
    try:
        for name, az, elev, factor, size in STUDIO_RIG:
            a, e = cam_az + math.radians(az), math.radians(elev)
            loc = Vector(center) + Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))) * dist
            data = bpy.data.lights.new(RIG_PREFIX + name, "AREA")
            data.energy = RIG_POWER_PER_M2 * dist * dist * factor
            data.size = radius * size
            obj = bpy.data.objects.new(RIG_PREFIX + name, data)
            obj[scene.PROP_AGENT_ID] = RIG_PREFIX + name
            coll.objects.link(obj)
            obj.location = loc
            obj.rotation_euler = (Vector(center) - loc).to_track_quat("-Z", "Y").to_euler()
            lights.append(obj)
        yield lights
    finally:
        for obj in lights:
            data = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.lights.remove(data)


def _render_lit_preview(cam: bpy.types.Object, size: tuple[int, int], box: tuple[np.ndarray, np.ndarray],
                        direction: Sequence[float], path_out: Optional[Path | str],
                        samples: int = PREVIEW_SAMPLES) -> np.ndarray:
    """Cycles (CPU) render under the studio rig, composited over the preview background."""
    centre = (box[0] + box[1]) / 2.0
    radius = float(np.linalg.norm(box[1] - box[0])) / 2.0
    with studio_rig(centre, radius, direction) as lights:
        with render_session(list(scene.model_meshes()) + lights) as sc, preview_world(sc), _scratch_path(path_out) as path:
            sc.render.engine = "CYCLES"
            sc.cycles.samples = samples
            sc.cycles.device = "CPU"
            sc.cycles.use_denoising = True
            _configure(sc, cam, size, transparent=True, aa=PREVIEW_AA, filter_size=MASK_FILTER_SIZE, light="STUDIO",
                       color_type="MATERIAL", bg=PREVIEW_BG)
            sc.render.engine = "CYCLES"  # _configure selects the Workbench engine
            rgba = _render_file(sc, path)
            alpha = rgba[..., 3:4]
            rgba = np.concatenate([rgba[..., :3] * alpha + np.asarray(PREVIEW_BG, dtype=np.float32) * (1.0 - alpha),
                                   np.ones_like(alpha)], axis=-1)
            imageio.save_rgba(path, rgba)
            return rgba


def render_preview(out_path: Optional[Path | str] = None, size: tuple[int, int] = PREVIEW_SIZE,
                   direction: Sequence[float] = camera.PREVIEW_DIRECTION, lit: bool = True) -> np.ndarray:
    """Perspective color render of the model from ``direction`` (default 3/4 front-right-above).

    ``lit``: studio 3-point light rig rendered with Cycles (falls back to the Workbench studio light on failure);
    projected colors / textures come through the material. Silhouette and color-metric renders never use this.
    """
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        raise BBRError("nothing to preview: AI_MODEL has no mesh geometry")
    cam = camera.ensure_preview_camera(*box, direction=direction)
    if lit:
        try:
            return _render_lit_preview(cam, size, box, direction, out_path)
        except Exception as exc:  # noqa: BLE001 - any Cycles/device problem must not break the workflow
            log.warning("lit preview failed (%s); falling back to the Workbench studio light", exc)
    with render_session(scene.model_meshes()) as sc, _scratch_path(out_path) as path:
        _configure(sc, cam, size, transparent=False, aa=PREVIEW_AA, filter_size=MASK_FILTER_SIZE,
                   light="STUDIO", color_type=_color_type(), bg=PREVIEW_BG)
        return _render_file(sc, path)


def render_turntable(out_path: Path | str, n: int = 8, tile: tuple[int, int] = TURNTABLE_TILE,
                     elevation: float = TURNTABLE_ELEVATION) -> np.ndarray:
    """Contact sheet of the lit model from ``n`` evenly spaced azimuths (0 = front, then counter-clockwise seen from
    above; row-major, ``TURNTABLE_COLUMNS`` per row). Small tiles and few samples keep it cheap."""
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        raise BBRError("nothing to render: AI_MODEL has no mesh geometry")
    tiles = []
    for i in range(n):
        direction, _, _ = pose.view_basis(360.0 * i / n, elevation)
        cam = camera.ensure_preview_camera(*box, direction=direction)
        tiles.append(_render_lit_preview(cam, tile, box, direction, None, TURNTABLE_SAMPLES)[..., :3])
    cols = min(TURNTABLE_COLUMNS, n)
    rows = []
    for r in range(0, n, cols):
        row = tiles[r:r + cols]
        rows.append(np.concatenate(row + [np.zeros_like(tiles[0])] * (cols - len(row)), axis=1))
    sheet = np.concatenate(rows, axis=0)
    imageio.save_rgba(out_path, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,), dtype=sheet.dtype)], axis=2))
    return sheet
