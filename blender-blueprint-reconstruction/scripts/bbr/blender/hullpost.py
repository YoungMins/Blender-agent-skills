"""Silhouette-hull post-processing: voxel cleanup, retopo, UV unwrap and texel colour baking.

Adapted from EL4CTEO/blender-skills (MIT; see THIRD_PARTY_NOTICES.md): the headless-safe voxel remesh via a collapsed
Remesh modifier (sculpt_ops.py), the quadriflow / decimate + shrinkwrap retopo (auto_retopo.py) and the smart-project +
pack UV unwrap (uv_pack.py). The island removal, UV-space rasterizer, depth-buffer visibility and texel projection
are new numpy code in ``bbr.core``.

Data API first; ``bpy.ops`` only for quadriflow, smart_project and pack_islands (no data-API equivalent), always under
``temp_override`` and always returning to OBJECT mode. Sculpt Mode is never used (it crashes with ``-b``).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import bmesh
import bpy
import numpy as np

from .. import BBRError, get_logger
from ..core.islands import small_island_vertices
from ..core.meshsmooth import taubin_smooth_edges, triangle_vertex_normals
from ..core.despill import background_leak_ratio
from ..core.texelproj import (DONOR_MIN_TRUST, dilate_colors, inpaint_colors, project_texel_colors,
                              source_map_colors, texture_shares)
from ..core.uvraster import rasterize_uv_mesh
from ..core.zbuffer import depth_buffer
from . import material

log = get_logger(__name__)

REMESH_DIVISIONS = 250          # default voxel size = height / this
MAX_REMESH_DIVISIONS = 1000     # finest allowed voxel size = height / this
ISLAND_RATIO = 0.002            # islands below this share of the total area are dropped
POST_SMOOTH_ITERATIONS = 4
TARGET_FACES = 20000
TEXTURE_SIZE = 2048
UV_ANGLE_DEG = 66.0
RETOPO_MODES = ("none", "decimate", "quadriflow")
TEXTURE_SUBDIR = "textures"
SOURCE_MAP_BG = 0.1             # source-map colour of texels outside the UV islands (inpainted texels are gray)
QUADRIFLOW_PRE_DECIMATE = 3     # quadriflow input = this many x target faces (triangles)


# --------------------------------------------------------------------------- mesh plumbing
def _evaluated_copy(obj: bpy.types.Object) -> bpy.types.Mesh:
    """New mesh datablock holding the modifier-evaluated geometry of ``obj`` (headless-safe collapse)."""
    return bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))


def _swap_geometry(dst: bpy.types.Mesh, src: bpy.types.Mesh) -> None:
    """Replace the geometry of ``dst`` with that of ``src`` (kept as the same datablock) and delete ``src``."""
    bm = bmesh.new()
    bm.from_mesh(src)
    bm.to_mesh(dst)
    bm.free()
    bpy.data.meshes.remove(src)
    dst.update()


def _evaluate_with(obj: bpy.types.Object, mods: Sequence[Any]) -> bpy.types.Mesh:
    """Evaluate ``obj`` with the (already added) modifiers ``mods``, then remove them again."""
    names = [m.name for m in mods]
    try:
        return _evaluated_copy(obj)
    finally:
        for name in names:
            mod = obj.modifiers.get(name)
            if mod is not None:
                obj.modifiers.remove(mod)


def _array(coll: Any, attr: str, n: int, dtype: Any, width: int = 1) -> np.ndarray:
    buf = np.empty(n * width, dtype=dtype)
    coll.foreach_get(attr, buf)
    return buf.reshape(-1, width) if width > 1 else buf


def vertices(me: bpy.types.Mesh) -> np.ndarray:
    return _array(me.vertices, "co", len(me.vertices), np.float32, 3).astype(np.float64)


def triangles(me: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(tris (T,3) vertex indices, tri_loops (T,3) loop indices)`` of the triangulated mesh."""
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    return (_array(me.loop_triangles, "vertices", n, np.int32, 3).astype(np.int64),
            _array(me.loop_triangles, "loops", n, np.int32, 3).astype(np.int64))


def blend_dir() -> Optional[Path]:
    """Directory of the saved .blend (texture destination when no output directory was given)."""
    return Path(bpy.data.filepath).parent if bpy.data.filepath else None


def world_vertices(obj: bpy.types.Object) -> np.ndarray:
    """Base-mesh vertices of ``obj`` in world space ``(N,3)`` (no modifiers)."""
    m = np.array(obj.matrix_world)
    return vertices(obj.data) @ m[:3, :3].T + m[:3, 3]


def world_triangles(objs: Sequence[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    """Merged world-space ``(verts, tris)`` of the modifier-evaluated meshes of ``objs`` (occluder geometry)."""
    all_v, all_t, offset = [], [], 0
    dg = bpy.context.evaluated_depsgraph_get()
    for obj in objs:
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            if len(me.vertices) == 0:
                continue
            m = np.array(obj.matrix_world)
            v = vertices(me) @ m[:3, :3].T + m[:3, 3]
            t, _ = triangles(me)
        finally:
            ev.to_mesh_clear()
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(all_v), np.concatenate(all_t)


def scene_depth_buffers(objs: Sequence[bpy.types.Object], calibs: Sequence[Any]) -> list[np.ndarray]:
    """Depth buffer per calibration of all ``objs`` together (the visibility test of the texel projection)."""
    verts, tris = world_triangles(objs)
    return [depth_buffer(verts, tris, calib) for calib in calibs]


def face_count(me: bpy.types.Mesh) -> int:
    return len(me.polygons)


def triangle_count(me: bpy.types.Mesh) -> int:
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def non_manifold_edges(me: bpy.types.Mesh) -> int:
    bm = bmesh.new()
    bm.from_mesh(me)
    count = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return count


def _quadriflow_ready(me: bpy.types.Mesh) -> bool:
    """Quadriflow input check: consistent outward face normals, no non-manifold edges or vertices (normals are fixed)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-7)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    ready = not any(not e.is_manifold for e in bm.edges) and not any(not v.is_manifold for v in bm.verts)
    bm.free()
    me.update()
    return ready


def vertex_normals_of(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    return triangle_vertex_normals(verts, tris)


# --------------------------------------------------------------------------- step 1: cleanup
def _drop_small_islands(me: bpy.types.Mesh, ratio: float) -> int:
    """Delete connected components below ``ratio`` of the total surface area; returns the number of removed vertices."""
    n = len(me.vertices)
    if ratio <= 0 or n == 0:
        return 0
    edges = _array(me.edges, "vertices", len(me.edges), np.int32, 2)
    area = _array(me.polygons, "area", len(me.polygons), np.float32).astype(np.float64)
    total = _array(me.polygons, "loop_total", len(me.polygons), np.int32)
    loop_verts = _array(me.loops, "vertex_index", len(me.loops), np.int32)
    share = np.repeat(area / np.maximum(total, 1), total)
    weight = np.bincount(loop_verts, weights=share, minlength=n)
    drop = small_island_vertices(n, edges, weight, ratio) | (weight <= 0)   # loose / zero-area vertices go too
    count = int(drop.sum())
    if count:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.flatnonzero(drop)], context="VERTS")
        bm.to_mesh(me)
        bm.free()
        me.update()
    return count


def _smooth_in_place(me: bpy.types.Mesh, iterations: int) -> None:
    if iterations <= 0 or len(me.vertices) == 0:
        return
    edges = _array(me.edges, "vertices", len(me.edges), np.int32, 2)
    co = taubin_smooth_edges(vertices(me), edges, iterations)
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()


def voxel_cleanup(obj: bpy.types.Object, voxel: float, island_ratio: float, smooth_iterations: int) -> dict[str, Any]:
    """Voxel remesh (watertight, spike/flap-free), drop small islands, light Taubin smoothing."""
    me = obj.data
    before = face_count(me)
    mod = obj.modifiers.new("bbr_voxel_remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = float(voxel)
    mod.adaptivity = 0.0
    mod.use_smooth_shade = True
    _swap_geometry(me, _evaluate_with(obj, [mod]))
    dropped = _drop_small_islands(me, island_ratio)
    _smooth_in_place(me, smooth_iterations)
    return {"voxel_size": float(voxel), "faces_before": before, "faces_after": face_count(me),
            "island_vertices_removed": dropped, "non_manifold_edges": non_manifold_edges(me)}


# --------------------------------------------------------------------------- step 2: retopo
def _object_override(obj: bpy.types.Object) -> Any:
    return bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                     selected_editable_objects=[obj])


def _quadriflow(obj: bpy.types.Object, target_faces: int) -> bool:
    """Quadriflow remesh in place; False when the operator refuses or fails."""
    try:
        with _object_override(obj):
            res = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=int(target_faces), seed=0,
                                                   use_mesh_symmetry=False, use_preserve_sharp=False,
                                                   use_preserve_boundary=False, smooth_normals=False)
        return res == {"FINISHED"}
    except RuntimeError as exc:
        log.warning("quadriflow failed: %s", exc)
        return False


def _add_decimate(obj: bpy.types.Object, triangles_wanted: int) -> Any:
    dec = obj.modifiers.new("bbr_decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = min(1.0, triangles_wanted / max(triangle_count(obj.data), 1))
    dec.use_collapse_triangulate = True
    return dec


def retopo(obj: bpy.types.Object, collection: bpy.types.Collection, mode: str, target_faces: int) -> dict[str, Any]:
    """Reduce to ~``target_faces`` faces (decimate or quadriflow), then shrinkwrap back onto the cleaned mesh."""
    me = obj.data
    before = face_count(me)
    info: dict[str, Any] = {"requested": mode, "target_faces": int(target_faces), "faces_before": before}
    if mode == "none" or triangle_count(me) <= target_faces:
        info["method"] = "none"
        info["faces_after"] = before
        return info
    ref_mesh = me.copy()
    ref_obj = bpy.data.objects.new(f"{obj.name}__bbr_retopo_ref", ref_mesh)
    collection.objects.link(ref_obj)
    try:
        method = "decimate"
        if mode == "quadriflow":
            # dense voxel-remesh output is rejected by quadriflow; a pre-decimated triangle mesh is accepted
            if triangle_count(me) > QUADRIFLOW_PRE_DECIMATE * target_faces:
                _swap_geometry(me, _evaluate_with(obj, [_add_decimate(obj, QUADRIFLOW_PRE_DECIMATE * target_faces)]))
            if not _quadriflow_ready(me):
                info["note"] = "quadriflow needs a manifold mesh: fell back to decimate"
            elif _quadriflow(obj, target_faces):
                method = "quadriflow"
            else:
                info["note"] = "quadriflow failed: fell back to decimate"
        mods = []
        if method == "decimate":
            mods.append(_add_decimate(obj, target_faces))
        wrap = obj.modifiers.new("bbr_shrinkwrap", "SHRINKWRAP")
        wrap.target = ref_obj
        wrap.wrap_method = "NEAREST_SURFACEPOINT"
        mods.append(wrap)
        _swap_geometry(me, _evaluate_with(obj, mods))
    finally:
        bpy.data.objects.remove(ref_obj)
        bpy.data.meshes.remove(ref_mesh)
    info.update(method=method, faces_after=face_count(me), non_manifold_edges=non_manifold_edges(me))
    return info


# --------------------------------------------------------------------------- step 3: UV unwrap
def unwrap(obj: bpy.types.Object, angle_deg: float, margin_px: float, size: int, shape_method: str = "CONCAVE") -> dict[str, Any]:
    """Smart UV project + island packing (bpy.ops under ``temp_override``; back to OBJECT mode afterwards). ``shape_method``
    is the packer's island shape: ``CONCAVE`` (organic hulls) or ``AABB`` (boxy hard-surface parts: the concave packer takes
    10 s on a 12-triangle box)."""
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    margin = margin_px / float(size)
    view_layer = bpy.context.view_layer
    previous = view_layer.objects.active
    for other in view_layer.objects:
        other.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    try:
        with _object_override(obj):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle_deg), margin_method="FRACTION",
                                     island_margin=margin, area_weight=0.0, correct_aspect=True,
                                     scale_to_bounds=False)
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(rotate=True, rotate_method="ANY", margin_method="FRACTION", margin=margin,
                                    shape_method=shape_method)
    finally:
        with _object_override(obj):
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        view_layer.objects.active = previous
    uv = _array(me.uv_layers.active.data, "uv", len(me.loops), np.float32, 2)
    return {"uv_layer": me.uv_layers.active.name, "uv_min": [round(float(x), 4) for x in uv.min(axis=0)],
            "uv_max": [round(float(x), 4) for x in uv.max(axis=0)]}


# --------------------------------------------------------------------------- step 4: texel colours
def _save_png(path: Path, rgb: np.ndarray) -> None:
    """Write an (H,W,3) float sRGB image (row 0 = bottom, like Blender pixels) as PNG via a throw-away Blender image."""
    h, w = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.ones((h, w, 1))], axis=2).astype(np.float32)
    tmp = bpy.data.images.new(path.stem, w, h, alpha=False)
    try:
        tmp.colorspace_settings.name = "sRGB"
        tmp.pixels.foreach_set(rgba.ravel())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.filepath_raw = str(path)
        tmp.file_format = "PNG"
        tmp.save()
    finally:
        bpy.data.images.remove(tmp)


def bake_texture(obj: bpy.types.Object, name: str, views: Sequence[tuple], size: int, padding_px: int,
                 out_dir: Optional[Path], projection: Optional[dict[str, Any]] = None,
                 bg_colors: Sequence[Sequence[float]] = (), zbuffers: Optional[Sequence[np.ndarray]] = None,
                 flat_normals: bool = False) -> tuple[bpy.types.Image, dict[str, Any]]:
    """Project the colour views onto the UV layout of ``obj`` and return the packed base-colour image.

    ``projection`` holds the :func:`~bbr.core.texelproj.project_texel_colors` settings ``seam_blend_width``,
    ``conflict_threshold``, ``edge_distrust_px``. Texels no view sees well are inpainted (:func:`~bbr.core.texelproj.inpaint_colors`). The PNG goes to ``<out_dir>/textures/<name>_basecolor.png``
    (when ``out_dir`` is known) with a ``<name>_source_map.png`` showing which view each texel came from, and the image
    is packed into the .blend, so every saved checkpoint stays self-contained. ``bg_colors`` (one backdrop colour per view)
    feeds the ``background_leak_ratio`` stat (:func:`~bbr.core.despill.background_leak_ratio`). Texel positions are
    world coordinates (``obj.matrix_world`` applied); ``zbuffers`` (one per view, see :func:`scene_depth_buffers`) are
    the occluders: pass the buffers of ALL model meshes so parts and body hide each other, else the object's own.
    ``flat_normals``: per-face instead of smooth vertex normals for the facing test (hard-surface parts).
    """
    t0 = time.perf_counter()
    me = obj.data
    if not me.uv_layers:
        raise BBRError(f"{obj.name} has no UV layer to bake into")
    verts = world_vertices(obj)
    tris, tri_loops = triangles(me)
    uv = _array(me.uv_layers.active.data, "uv", len(me.loops), np.float32, 2).astype(np.float64)
    if flat_normals:   # hard-surface: a face keeps its own normal, so a flat flank is not tilted toward the views that see its neighbour
        fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
        nrm = np.repeat((fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12))[:, None, :], 3, axis=1)
    else:
        nrm = vertex_normals_of(verts, tris)
    raster = rasterize_uv_mesh(verts, tris, uv[tri_loops], nrm, size)
    t_raster = time.perf_counter()
    zbufs = list(zbuffers) if zbuffers is not None else [depth_buffer(verts, tris, calib) for _, _, calib in views]
    mask = raster["mask"]
    ys, xs = np.nonzero(mask)
    projection = dict(projection or {})
    rgb, seen, info = project_texel_colors(raster["position"][ys, xs], raster["normal"][ys, xs], views, zbufs,
                                           **projection)
    img = np.empty((size, size, 3))
    img[...] = rgb[seen].mean(axis=0) if seen.any() else 0.5
    img[ys, xs] = rgb
    valid = np.zeros(mask.shape, dtype=bool)
    valid[ys, xs] = seen
    donor = np.zeros(mask.shape, dtype=bool)
    donor[ys, xs] = seen & (info["trust"] >= DONOR_MIN_TRUST)      # silhouette-edge samples (backdrop-mixed, dark) never fill an unseen face
    img, method = inpaint_colors(img, valid, mask, raster["island"], raster["position"], donor=donor)
    leak = background_leak_ratio(img, mask, bg_colors)   # before padding: only texels the UV islands cover
    shares = texture_shares(info["weights"], seen, info["conflict"], info["view_keys"])
    source = np.full((size, size, 3), SOURCE_MAP_BG)
    source[ys, xs] = source_map_colors(info["weights"], seen, info["view_keys"])
    img, _ = dilate_colors(img, mask, padding_px)
    source, _ = dilate_colors(source, mask, padding_px)
    rgba = np.concatenate([img, np.ones((size, size, 1))], axis=2).astype(np.float32)

    image_name = f"{name}_basecolor"
    old = bpy.data.images.get(image_name)
    if old is not None:
        bpy.data.images.remove(old)
    image = bpy.data.images.new(image_name, size, size, alpha=False)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(rgba.ravel())
    path: Optional[Path] = None
    if out_dir is not None:
        path = Path(out_dir) / TEXTURE_SUBDIR / f"{image_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        _save_png(path.with_name(f"{name}_source_map.png"), source)
    image.pack()
    stats = {"size": size, "path": str(path) if path else None,
             "source_map_path": str(path.with_name(f"{name}_source_map.png")) if path else None,
             "covered_texels": int(mask.sum()), "coverage_pct": round(100.0 * float(mask.mean()), 1),
             "texels_from_visible_samples_pct": round(100.0 * float(seen.mean()), 1) if len(seen) else 0.0,
             **shares,
             "background_leak_ratio": round(leak, 5), "inpainted_uv_share": round(float((method == 1).sum()) / max(len(seen), 1), 4),
             "inpainted_3d_share": round(float((method == 2).sum()) / max(len(seen), 1), 4),
             **({"label_rejected_ratio": round(float(info["label_rejected"].mean()), 4)} if projection.get("label_masks") else {}),
             "raster_s": round(t_raster - t0, 2), "total_s": round(time.perf_counter() - t0, 2)}
    return image, stats


def assign_texture_material(obj: bpy.types.Object, image: bpy.types.Image, name: str = material.REF_TEXTURE_MATERIAL,
                            pbr: Optional[dict[str, float]] = None, maps: Optional[dict[str, bpy.types.Image]] = None) -> None:
    """Give ``obj`` the projected-texture material; ``pbr`` (:func:`bbr.core.pbr.resolve`) sets its Principled values."""
    material.assign_material(obj, material.ensure_texture_material(
        image, name, (pbr or {}).get("roughness", material.REF_COLOR_ROUGHNESS), pbr, maps))
