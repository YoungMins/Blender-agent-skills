"""Silhouette hull component: continuous hull field (silhouette distance + superellipse cross-sections)
-> Surface Nets quads -> light Taubin smoothing -> post-processing (see :mod:`hullpost`: voxel cleanup, retopo,
UV unwrap, texel colour projection; adapted from EL4CTEO/blender-skills, MIT).

The surface is extracted from a continuous field with Surface Nets (no banding) and lightly smoothed with a numpy
Taubin filter (shrink-free, deterministic); the Remesh modifier is only used afterwards, as the watertight cleanup.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import bpy

from .. import BBRError, get_logger
from ..core.carve import DEFAULT_SLICE_COHERENCE, build_hull_field
from ..core import labels as labels_core
from ..core import layers as layers_core
from ..core import debris as debris_core
from ..core import partmask
from ..core import imageio
from ..core.ringmetric import ring_score
from ..core.surface import surface_nets
from ..core.meshsmooth import taubin_smooth
from ..core.texelproj import DEFAULT_PADDING_PX
from ..core.spec import Component
from . import hullpost, material, mesh, partstex, scene
from .components import (HULL_TYPE, LAYER_TYPE, BuildContext, apply_symmetry, build_component, defers_to_target, material_for,
                         object_for, register_component)

log = get_logger(__name__)

ISLAND_MIN_FRAC = 0.01          # surface-nets groups below this share of the largest one are dropped (disconnected parts stay)
LAYER_SMOOTH_ITERATIONS = 8
LAYER_TARGET_FACES = 5000


def _prebuild_parts(comp: Component, ctx: BuildContext) -> dict[str, bpy.types.Object]:
    """Build every other (non-hull) component first, with its mirror: they are the occluders of the texture projection
    and the ``exclude_components`` silhouettes the hull is carved without. Idempotent (the normal build repeats it).
    Parts placed on a surface (``attach_to``, or mirroring one) are left out: they need the finished hull, so they are
    built after it in the normal stage order (and cannot be in ``exclude_components``)."""
    objs: dict[str, bpy.types.Object] = {}
    for other in ctx.scene.components_by_stage():
        if other.type != HULL_TYPE and not defers_to_target(other, ctx.scene):
            obj = build_component(other, ctx)
            apply_symmetry(obj, other)
            objs[other.id] = obj
    return objs


def _excluded_masks(comp: Component, ctx: BuildContext, carve_views: list[str]) -> tuple[Any, dict[str, Any]]:
    """Carve masks without the ``params.exclude_components``.

    A view with part labels (``parts_labels``) loses exactly the label pixels of those parts
    (:func:`partmask.exclude_label_pixels`; no fitted mesh involved). A view without labels falls back to the
    silhouettes of the built part meshes (:mod:`bbr.core.partmask`)."""
    hi, p = ctx.hull_inputs, comp.params
    ids = [str(i) for i in p.get("exclude_components") or []]
    if not ids:
        return hi.masks, {}
    known = {c.id for c in ctx.scene.components if c.type != HULL_TYPE}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise BBRError(f"{comp.id}: exclude_components {unknown} are not (other) components of the scene")
    labels = partstex.label_masks(ctx)
    labelled = [k for k in carve_views if k in labels]
    mesh_views = [k for k in carve_views if k not in labels]
    dilate = int(p.get("exclude_dilate_px", partmask.DEFAULT_EXCLUDE_DILATE_PX if mesh_views else partmask.DEFAULT_LABEL_DILATE_PX))
    masks, per_view = dict(hi.masks), {}
    for key in labelled:
        cut = labels_core.union(labels[key], ids)
        if cut is None:
            log.warning("hull %s: none of the excluded parts is labelled in view %s", comp.id, key)
            continue
        masks[key] = partmask.exclude_label_pixels(hi.masks[key], cut, int(p.get("exclude_dilate_px", partmask.DEFAULT_LABEL_DILATE_PX)))
        per_part = {}
        for pid in ids:
            own = labels_core.union(labels[key], [pid])
            if own is not None:
                own = own & hi.masks[key]
                per_part[pid] = {"removed_px": int((own & ~masks[key]).sum()), "restored_px": int((own & masks[key]).sum())}
        per_view[key] = {"mask_px_before": int(hi.masks[key].sum()), "mask_px_after": int(masks[key].sum()),
                         "part_px": int(cut.sum()), "source": "labels", "parts": per_part}
    if mesh_views:
        objs = _prebuild_parts(comp, ctx)
        missing = [i for i in ids if i not in objs]
        if missing:
            raise BBRError(f"{comp.id}: exclude_components {missing} are built after the hull (attach_to) and cannot be "
                           f"excluded from it without labels")
        parts = [hullpost.world_triangles([objs[i]]) for i in ids]
        mesh_masks, mesh_stats = partmask.exclude_parts_from_masks(hi.masks, hi.calibs, parts, mesh_views, dilate)
        masks.update({k: mesh_masks[k] for k in mesh_stats})
        per_view.update({k: {**v, "source": "part meshes"} for k, v in mesh_stats.items()})
    out_dir = ctx.extra.get("out_dir")
    if out_dir is not None:
        for key in per_view:
            imageio.save_mask(Path(out_dir) / "hull_carve_masks" / f"{comp.id}_{key}.png", masks[key])
    return masks, {"components": ids, "dilate_px": dilate, "views": per_view}


def _layer_field(comp: Component, ctx: BuildContext) -> layers_core.LayerField:
    """The layer's hull field from its label masks (``params.label``, default the id) in the two ``params.views``
    (default front, side); cached per build context. Needs ``parts_labels``."""
    cache = ctx.extra.setdefault("_layer_fields", {})
    if comp.id not in cache:
        hi, p = ctx.hull_inputs, comp.params
        if hi is None:
            raise BBRError(f"{comp.id}: a layer needs reference views: pass a ViewSpec (HullInputs) to the build")
        label = str(p.get("label", comp.id))
        masks = labels_core.part_masks(partstex.label_masks(ctx), label)
        for other in p.get("cover") or []:        # a shell's footprint: its own label plus those of the layers it covers
            other_comp = ctx.scene.get_component(str(other))
            other_label = str(other_comp.params.get("label", other_comp.id))
            for view, mask in labels_core.part_masks(partstex.label_masks(ctx), other_label).items():
                masks[view] = mask | masks[view] if view in masks else mask
        views = [str(v) for v in p.get("views") or ("front", "side")]
        if len(views) != 2 or any(v not in masks or v not in hi.calibs for v in views):
            raise BBRError(f"{comp.id}: a layer is carved from the label {label!r} in two views {views} "
                           f"(front-type, side-type); labelled views: {sorted(masks)} (needs parts_labels)")
        relief = _relief_params(comp)
        if relief is not None:     # face detail: carve a featureless profile, the features go back on the midline later
            rel = layers_core.profile_relief(masks[views[1]], hi.calibs[views[1]],
                                             float(relief.get("smooth", layers_core.DEFAULT_RELIEF_SMOOTH)))
            masks = {**masks, views[1]: rel.mask}
            front = masks[views[0]]
            cols = np.nonzero(front.any(axis=0))[0]
            xs = hi.calibs[views[0]].px_to_plane(np.stack([cols + 0.5, np.zeros(len(cols))], axis=1))[:, 0]
            weights = front[:, cols].sum(axis=0)
            ctx.extra.setdefault("_layer_reliefs", {})[comp.id] = (
                rel, float(relief.get("center_x", np.average(xs, weights=weights))),
                float(relief.get("width", layers_core.DEFAULT_RELIEF_WIDTH)) * float(xs.max() - xs.min()))
        cache[comp.id] = layers_core.build_layer_field(
            masks, hi.calibs, views[0], views[1], int(p.get("resolution", layers_core.DEFAULT_LAYER_RESOLUTION)),
            float(p.get("exponent", 2.0)), str(p.get("cross_section", "superellipse")),
            pair_max_aspect=float(p.get("pair_max_aspect", 4.0)))
    return cache[comp.id]


def _relief_params(comp: Component) -> Optional[dict[str, Any]]:
    """``params.relief`` of a layer as a dict (``true`` = defaults), None when off."""
    relief = comp.params.get("relief")
    if not relief:
        return None
    return dict(relief) if isinstance(relief, dict) else {}


def _recess_layers(comp: Component, ctx: BuildContext, field: Any, origin: Any, cell: float) -> dict[str, Any]:
    """``params.layers`` (ids of ``layer_hull`` components): recess the body field behind each layer. Returns ``{}`` when
    there are none, else ``{"field": new field, <layer id>: stats}``."""
    ids = [str(i) for i in comp.params.get("layers") or []]
    out: dict[str, Any] = {}
    for lid in ids:
        layer = ctx.scene.get_component(lid)
        if layer.type != LAYER_TYPE:
            raise BBRError(f"{comp.id}: layers entry {lid!r} is a {layer.type}, not a {LAYER_TYPE}")
        field, out[lid] = layers_core.recess_field(field, origin, cell, _layer_field(layer, ctx),
                                                   float(layer.params.get("embed", layers_core.DEFAULT_EMBED_M)))
    return {"field": field, **out} if ids else {}


@register_component(HULL_TYPE)
def build_hull(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Carve the hull from the ``carve`` views, smooth it, then clean / retopo / unwrap / colour it (see ``_post_process``).

    ``params.exclude_components`` (ids of part primitives) are removed from the carve masks first, so thin parts the
    hull cannot represent (horns, a sword) are not baked into it as fans / blobs. ``params.slice_coherence`` (share of the
    height, default :data:`DEFAULT_SLICE_COHERENCE`, 0 = off) keeps a spiky silhouette (hair, fur) from becoming a stack of
    rings (:func:`bbr.core.carve.build_hull_field`); ``stats["rings"]`` reports the ring metric of the carved field
    (:func:`bbr.core.ringmetric.ring_score`)."""
    hi = ctx.hull_inputs
    if hi is None:
        raise BBRError("silhouette_hull needs reference views: pass a ViewSpec (HullInputs) to the build")
    p, cfg = comp.params, ctx.config
    t0 = time.perf_counter()
    carve_views = list(p.get("carve_views") or [v.key for v in hi.view_spec.enabled_views("carve")])
    masks, exclusion = _excluded_masks(comp, ctx, carve_views)
    slice_coherence = float(p.get("slice_coherence", DEFAULT_SLICE_COHERENCE))
    field, origin, cell = build_hull_field(
        masks, hi.calibs, hi.dims, int(p.get("resolution", cfg.hull_resolution)), carve_views=carve_views,
        cross_section=str(p.get("cross_section", "superellipse")), exponent=float(p.get("exponent", 2.0)),
        pair_max_aspect=float(p.get("pair_max_aspect", 4.0)), slice_coherence=slice_coherence,
        margin=float(p.get("margin", cfg.hull_margin)), top_consistency_tol=cfg.top_consistency_tol)
    height = float(hi.dims.get("height") or max(hi.dims.values()))
    stats_rings = {"slice_coherence": slice_coherence, "ring_pct": round(100.0 * ring_score(field, cell, height), 3)}
    recess = _recess_layers(comp, ctx, field, origin, cell)
    if recess:
        field = recess.pop("field")
    islands: list[dict[str, Any]] = []
    if p.get("drop_debris", True) and float(p.get("island_min_frac", ISLAND_MIN_FRAC)) < 1.0:
        drop, islands = debris_core.find_debris(field, origin, cell, masks, hi.calibs, carve_views, partstex.label_masks(ctx) or None)
        if drop.any():
            field = debris_core.drop_debris(field, drop)
        for line in debris_core.format_debris_report(islands, comp.id):
            log.warning("hull_debris: %s", line)
    verts, quads = surface_nets(field, origin + 0.5 * cell, cell, min_island_frac=float(p.get("island_min_frac", ISLAND_MIN_FRAC)))
    if len(quads) == 0:
        raise BBRError("silhouette hull is empty: the carve views do not intersect (check camera_axis / calibration)")
    verts = taubin_smooth(verts, quads, int(p.get("smooth_iterations", cfg.smooth_iterations)))
    obj, data = object_for(comp, ctx)
    mesh.mesh_from_arrays(data, verts, quads)
    mesh.set_smooth(data, True)
    stats = _post_process(comp, ctx, obj, carve_views)
    stats["rings"] = stats_rings
    if islands:
        stats["debris"] = islands
    if exclusion:
        stats["exclusion"] = exclusion
    if recess:
        stats["layers"] = recess
    log.info("hull %s: %d faces from views %s in %.1fs", comp.id, len(data.polygons), carve_views,
             time.perf_counter() - t0)
    ctx.extra.setdefault("hull_stats", {})[comp.id] = stats
    return obj


def _post_process(comp: Component, ctx: BuildContext, obj: bpy.types.Object, carve_views: list[str]) -> dict[str, Any]:
    """Cleanup -> retopo -> UV unwrap -> texture; returns the stage stats.

    With ``ctx.extra["defer_texture"]`` (the scene has textured parts) the texture bake is left to
    :func:`partstex.texture_pass`, which sees all meshes as occluders; the hull is only unwrapped here."""
    hi, p = ctx.hull_inputs, comp.params
    data = obj.data
    retopo = str(p.get("retopo", "decimate"))
    if retopo not in hullpost.RETOPO_MODES:
        raise BBRError(f"{comp.id}: retopo must be one of {hullpost.RETOPO_MODES}, got {retopo!r}")
    stats: dict[str, Any] = {"carve_views": carve_views}
    height = float(hi.dims.get("height") or max(hi.dims.values()))
    voxel = float(p.get("remesh_voxel", height / hullpost.REMESH_DIVISIONS))
    if voxel > 0:
        voxel = max(voxel, height / hullpost.MAX_REMESH_DIVISIONS)   # a tiny voxel size would exhaust memory
        stats["cleanup"] = hullpost.voxel_cleanup(obj, voxel, float(p.get("remove_islands_ratio", hullpost.ISLAND_RATIO)),
                                                  int(p.get("post_smooth_iterations", hullpost.POST_SMOOTH_ITERATIONS)))
    stats["retopo"] = hullpost.retopo(obj, ctx.model, retopo, int(p.get("target_faces", hullpost.TARGET_FACES)))
    mesh.set_smooth(data, True)
    if not p.get("color_projection", True):
        material.assign_material(obj, material_for(comp, ctx))
        return stats
    usable, inputs, _ = partstex.color_inputs(hi, p)
    if not inputs:
        log.warning("hull %s: color_projection requested but no color view is available", comp.id)
        material.assign_material(obj, material_for(comp, ctx))
        return stats
    size = int(p.get("texture_size", hullpost.TEXTURE_SIZE))
    padding = int(p.get("texture_padding_px", DEFAULT_PADDING_PX))
    stats["uv"] = hullpost.unwrap(obj, float(p.get("uv_angle_deg", hullpost.UV_ANGLE_DEG)), padding, size)
    if ctx.extra.get("defer_texture"):
        stats["texture_deferred"] = True
        material.assign_material(obj, material_for(comp, ctx))
        return stats
    zbufs = dict(zip(usable, hullpost.scene_depth_buffers(scene.model_meshes(opaque=True), [hi.calibs[k] for k in usable])))
    stats.update(partstex.bake_object(comp, ctx, obj, unwrap=False, zbuffers=zbufs))
    return stats


def _build_shell(comp: Component, ctx: BuildContext, lf: layers_core.LayerField, t0: float) -> bpy.types.Object:
    """``params.shell``: a thin closed sheet over the layer's front surface (glass visor, helmet bubble; see
    :func:`bbr.core.layers.shell_mesh`), lightly smoothed; no voxel cleanup or retopo (they would close up the sheet)."""
    p = comp.params
    verts, quads = layers_core.shell_mesh(lf, float(p.get("thickness", layers_core.DEFAULT_SHELL_THICKNESS_M)))
    verts = taubin_smooth(verts, quads, int(p.get("smooth_iterations", LAYER_SMOOTH_ITERATIONS)))
    obj, data = object_for(comp, ctx)
    mesh.mesh_from_arrays(data, verts, quads)
    mesh.set_smooth(data, True)
    ctx.extra.setdefault("hull_stats", {})[comp.id] = {"shell": True, "cell_m": round(float(lf.cell), 6),
                                                       "faces": len(data.polygons)}
    log.info("shell %s: %d faces in %.1fs", comp.id, len(data.polygons), time.perf_counter() - t0)
    return obj


@register_component(LAYER_TYPE)
def build_layer(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """A layer (face in a helmet, visor, recessed panel): its own small hull from its label masks, Surface Nets ->
    Taubin -> cleanup / retopo like the body, finer grid over its own bounding box (see :mod:`bbr.core.layers`).
    The mesh stays separate from the body (which lists it in ``params.layers``); the texture pass projects the colour
    views onto it through its own label (``color_projection``, default true)."""
    p, cfg = comp.params, ctx.config
    t0 = time.perf_counter()
    lf = _layer_field(comp, ctx)
    if p.get("shell"):
        return _build_shell(comp, ctx, lf, t0)
    verts, quads = surface_nets(lf.field, lf.origin + 0.5 * lf.cell, lf.cell)
    if len(quads) == 0:
        raise BBRError(f"{comp.id}: the layer hull is empty")
    verts = taubin_smooth(verts, quads, int(p.get("smooth_iterations", LAYER_SMOOTH_ITERATIONS)))
    stats: dict[str, Any] = {"cell_m": round(float(lf.cell), 6), "grid": list(lf.field.shape)}
    relief = ctx.extra.get("_layer_reliefs", {}).get(comp.id)
    if relief is not None:
        rel, x0, width = relief
        verts = layers_core.apply_relief(verts, lf, rel, x0, width)
        stats["relief"] = {"amplitude_m": round(rel.amplitude, 4), "center_x": round(x0, 4), "width_m": round(width, 4)}
    obj, data = object_for(comp, ctx)
    mesh.mesh_from_arrays(data, verts, quads)
    mesh.set_smooth(data, True)
    voxel = float(p.get("remesh_voxel", lf.cell))
    if voxel > 0:
        stats["cleanup"] = hullpost.voxel_cleanup(obj, voxel, float(p.get("remove_islands_ratio", hullpost.ISLAND_RATIO)),
                                                  int(p.get("post_smooth_iterations", hullpost.POST_SMOOTH_ITERATIONS)))
    retopo = str(p.get("retopo", "decimate"))
    if retopo not in hullpost.RETOPO_MODES:
        raise BBRError(f"{comp.id}: retopo must be one of {hullpost.RETOPO_MODES}, got {retopo!r}")
    stats["retopo"] = hullpost.retopo(obj, ctx.model, retopo, int(p.get("target_faces", LAYER_TARGET_FACES)))
    mesh.set_smooth(data, True)
    ctx.extra.setdefault("hull_stats", {})[comp.id] = stats
    log.info("layer %s: %d faces in %.1fs", comp.id, len(data.polygons), time.perf_counter() - t0)
    return obj
