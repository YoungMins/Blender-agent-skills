"""Texture misattribution and camera-facing occlusion of every AI_MODEL object (Blender side of
:mod:`bbr.core.attribution`).

Each object is sampled on its modifier-evaluated surface; the colour view that would paint a sample is chosen with the
texture projection itself (:func:`~bbr.core.texelproj.project_texel_colors`, same views, depth buffers and settings as
the bake), its source pixel is looked up in an object-ID render of that view, and the samples that face a colour camera
but fail its depth test are counted (an embedded or hidden part such as a shield sunk into the body).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import bpy
import numpy as np

from .. import get_logger
from ..core import attribution as core
from ..core.geomscore import sample_surface, triangle_areas
from ..core.texelproj import project_texel_colors
from . import hullpost, partstex, render, scene

log = get_logger(__name__)

TOTAL_SAMPLES = 150_000
MIN_SAMPLES = 600
SEED = 7


def evaluated_world_mesh(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    """World-space ``(verts, tris)`` of ``obj`` after its modifiers."""
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    try:
        if len(me.vertices) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        m = np.array(obj.matrix_world)
        verts = hullpost.vertices(me) @ m[:3, :3].T + m[:3, 3]
        tris, _ = hullpost.triangles(me)
    finally:
        ev.to_mesh_clear()
    return verts, tris


def sample_objects(objs: list[bpy.types.Object], total: int = TOTAL_SAMPLES, min_each: int = MIN_SAMPLES,
                   seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-proportional surface samples of ``objs``: ``(points, outward normals, owner index)``."""
    meshes = [evaluated_world_mesh(o) for o in objs]
    areas = np.array([triangle_areas(v, t).sum() if len(t) else 0.0 for v, t in meshes])
    rng = np.random.default_rng(seed)
    pts, nrm, owner = [], [], []
    for k, (verts, tris) in enumerate(meshes):
        n = max(min_each, int(round(total * areas[k] / max(areas.sum(), 1e-12)))) if areas[k] > 0 else 0
        p, nr = sample_surface(verts, tris, n, rng)
        pts.append(p), nrm.append(nr), owner.append(np.full(len(p), k, dtype=np.int64))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(owner)


def component_name(spec: Any, obj: bpy.types.Object) -> str:
    """Component id of a model object (its agent_id minus the ``<scene>:`` prefix), else the object name."""
    aid = str(obj.get(scene.PROP_AGENT_ID, ""))
    prefix = f"{spec.name}:"
    return aid[len(prefix):] if aid.startswith(prefix) else (aid or obj.name)


def analyze(inp: Any, config: Any, total: int = TOTAL_SAMPLES) -> dict[str, Any]:
    """Misattribution and occlusion of every model object against the colour views of ``inp`` (see module doc).

    Returns ``{views, samples, facing_threshold, occlusion_warn, objects{id: {...}}, misattribution_max,
    occlusion_max, occluded_objects, misattributed_objects}``; ``{"skipped": reason}`` without colour views.
    """
    hi = inp.hull
    objs = scene.model_meshes()
    if hi is None or not objs:
        return {"skipped": "no ViewSpec or no model meshes"}
    hull = partstex.hull_component(inp.scene)
    usable, inputs, _ = partstex.color_inputs(hi, partstex.effective_params(inp.scene, hull) if hull else {})
    if not usable:
        return {"skipped": "no colour view"}
    calibs = [hi.calibs[k] for k in usable]
    points, normals, owner = sample_objects(objs, total)
    zbufs = hullpost.scene_depth_buffers(objs, calibs)
    _, has, info = project_texel_colors(points, normals, inputs, zbufs,
                                        **partstex.projection_params(partstex.effective_params(inp.scene, hull) if hull else {}))
    source = np.where(has, info["weights"].argmax(axis=1), -1)
    id_maps = [render.render_object_ids(k, hi.calibs[k], objs) for k in usable]
    mis = core.misattribution_by_object(points, owner, len(objs), source, calibs, id_maps)
    occ = core.occlusion_by_object(points, normals, owner, len(objs), calibs, zbufs)
    names = [component_name(inp.scene, o) for o in objs]
    objects: dict[str, Any] = {}
    for k, name in enumerate(names):
        top = mis[k]["top_confusion"]
        objects[name] = {
            "samples": occ[k]["samples"], "misattribution": mis[k]["share"],
            "top_confusion": None if top is None else {"object": names[top["object"]], "share": top["share"]},
            "occlusion_share": occ[k]["occlusion_share"], "best_view": occ[k]["best_view"],
            "pooled_occlusion": occ[k]["pooled_share"], "hidden_share": occ[k]["hidden_share"],
            "occlusion_per_view": {v: i["share"] for v, i in occ[k]["per_view"].items()}}
    return {"views": usable, "samples": int(len(points)), "facing_threshold": core.FACING,
            "occlusion_warn": core.OCCLUSION_WARN, "misattribution_warn": core.MISATTRIBUTION_WARN,
            "objects": objects, **summarize(objects)}


def summarize(objects: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Worst objects and the flagged lists of an ``objects`` block (also used by the report and the bench)."""
    def worst(key: str) -> Optional[dict[str, Any]]:
        vals = {n: o[key] for n, o in objects.items() if o.get(key) is not None}
        if not vals:
            return None
        name = max(vals, key=vals.get)
        return {"object": name, "share": vals[name]}
    return {"misattribution_max": worst("misattribution"), "occlusion_max": worst("occlusion_share"),
            "occluded_objects": sorted((n for n, o in objects.items()
                                        if (o.get("occlusion_share") or 0.0) > core.OCCLUSION_WARN),
                                       key=lambda n: -objects[n]["occlusion_share"]),
            "misattributed_objects": sorted((n for n, o in objects.items()
                                             if (o.get("misattribution") or 0.0) > core.MISATTRIBUTION_WARN),
                                            key=lambda n: -objects[n]["misattribution"])}
