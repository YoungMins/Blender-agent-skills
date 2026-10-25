"""Texture projection for every character component: the silhouette hull AND the thin-part primitives.

Each textured object gets its own UV layout and its own base-colour image (``<scene>_<id>_basecolor.png``) with the
same per-texel view selection, inpainting, despill and padding as the hull (:func:`hullpost.bake_texture`). Visibility
uses the depth buffers of ALL model meshes together, so a horn hides the helmet behind it and the body hides the
back of a shield: a texel only takes colour from a view that really sees it.

Which components are textured: in a character scene (it has a ``silhouette_hull`` or ``parts_labels``) EVERY non-hull
component (``capsule``, ``sphere``, ``cylinder``, ``gem``, boxes ... as well as ``cone`` / ``blade`` / ``disc``); in a
scene without either (hard-surface) only ``cone`` / ``blade`` / ``disc`` and components with ``params.color_projection: true``
- or EVERY component when the scene projects (``style.texture`` ``project``, or ``auto`` with real, non-synthetic colour
references: :func:`bbr.core.spec.project_textures`). ``color_projection: false`` opts a part out; hidden boolean cutters are never
textured and never occlude (:func:`scene.model_meshes` leaves them out of every depth buffer). Parts inherit the colour / projection
settings of the scene's silhouette hull (``color_views``, ``spill_*``...) and may override them;
``texture_size`` defaults to :data:`PART_TEXTURE_SIZE` (hard-surface parts: one texel per reference pixel, 128-1024,
:func:`bbr.core.texelproj.suggest_texture_size`). A symmetric hard-surface scene (``symmetry.x``) uses its side views mirrored through X = 0 for the flank no real view sees
(``mirror_color_views``; ``params.mirror_color_views: []`` opts out). Character parts with a ``mirror`` modifier are not textured; in a projecting hard-surface scene they are: the base instance is baked and the Mirror modifier reuses its UVs on the copy.

Mirrored virtual colour views (hull params, inherited by the parts): ``mirror_color_views: ["side"]`` adds, for each
listed X-axis view, the view reflected through X = 0 (:mod:`bbr.core.symmetry`) as a low-priority colour view
(``mirror_penalty``, default :data:`MIRROR_PENALTY`), so the flank no real view sees takes the mirrored colours.
``mirror_pairs`` ``{"horn_l": "horn_r"}`` adds to the left / right naming convention for the label swap, and
``asymmetric_parts`` (ids or globs such as ``sword_*``) are never sampled from a virtual view, nor are those parts'
pixels in it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import bpy
import numpy as np

from .. import get_logger
from ..core import labels as labels_core
from ..core import pbr
from ..core import symmetry
from ..core.despill import DEFAULT_SPILL_EDGE_PX, decontaminate_view
from ..core.imageops import border_background_color
from ..core.spec import Component, SceneSpec, is_textured_component, project_textures
from ..core.texelproj import (DEFAULT_CONFLICT_THRESHOLD, DEFAULT_EDGE_DISTRUST_PX, DEFAULT_PADDING_PX,
                              DEFAULT_SEAM_BLEND_WIDTH, suggest_texture_size)
from . import hullpost, material, scene
from .components import HULL_TYPE, BuildContext

log = get_logger(__name__)

PART_TEXTURE_SIZE = 512
HARD_SURFACE_MIN_FACING = 0.3  # hard-surface parts ignore a view that sees a face this obliquely (cos of the angle): grazing samples only smear edge pixels
MIRROR_PENALTY = 0.5          # score penalty of a mirrored virtual view: it only wins where no real view faces the texel nearly as well


# --------------------------------------------------------------------------- which components
def is_character_scene(spec: SceneSpec) -> bool:
    """True when the scene has a silhouette hull, or part labels in a scene that is not ``style.hard_surface`` (the default):
    its parts are textured by default. Part labels alone do not turn a hard-surface scene into a character one: they only gate
    which pixels each component samples (:func:`label_gates`), the hard-surface texture rules (flat face normals, boxy UV
    unwrap, ``min_facing``, mirrored side view) stay."""
    return any(c.type == HULL_TYPE for c in spec.components) or (bool(spec.parts_labels) and not spec.style.get("hard_surface", True))


def is_textured_part(comp: Component, character: bool = False, project: bool = False) -> bool:
    """True for a non-hull, non-cutter component that gets a projected texture of its own (see
    :func:`bbr.core.spec.is_textured_component`): ``params.color_projection`` when set, else every type in a ``character``
    scene (:func:`is_character_scene`) or when ``project`` (hard-surface ``style.texture``), and only ``cone`` / ``blade`` /
    ``disc`` elsewhere. Hidden boolean cutters never are."""
    return is_textured_component(comp, character, project)


def textured_parts(spec: SceneSpec, project: bool = False) -> list[Component]:
    """Parts (see :func:`is_textured_part`) of the scene in build order; ``project``: see :func:`project_textures`."""
    character = is_character_scene(spec)
    return [c for c in spec.components_by_stage() if is_textured_part(c, character, project)]


def hull_component(spec: SceneSpec) -> Optional[Component]:
    """The scene's (first) silhouette hull, else None."""
    return next((c for c in spec.components if c.type == HULL_TYPE), None)


def effective_params(spec: SceneSpec, comp: Component) -> dict[str, Any]:
    """Projection settings of ``comp``: a part inherits the hull's params, its own take precedence."""
    if comp.type == HULL_TYPE:
        return dict(comp.params)
    hull = hull_component(spec)
    return {**(hull.params if hull else {}), **comp.params}


# --------------------------------------------------------------------------- shared settings
def background_color(hi: Any, key: str) -> np.ndarray:
    """Backdrop colour of a view: the ViewSpec ``background.color``, else the median of the image border."""
    color = hi.view_spec.views[key].background.color
    return np.asarray(color, dtype=np.float32)[:3] if color else border_background_color(hi.rgba[key][..., :3])


def color_inputs(hi: Any, p: dict[str, Any]) -> tuple[list[str], list[tuple], list[np.ndarray]]:
    """``(view keys, (rgba, mask, calib) per view, backdrop colour per view)`` of the colour views, despilled unless
    ``spill_suppression`` is false."""
    color_views = list(p.get("color_views") or [v.key for v in hi.view_spec.enabled_views("color")])
    usable = [k for k in color_views if k in hi.calibs and k in hi.rgba]
    inputs = [(hi.rgba[k], hi.masks[k], hi.calibs[k]) for k in usable]
    bg_colors = [background_color(hi, k) for k in usable]
    if p.get("spill_suppression", True):
        edge_px = int(p.get("spill_edge_px", DEFAULT_SPILL_EDGE_PX))
        inputs = [(*decontaminate_view(rgba, mask, bg, edge_px, hi.view_spec.views[k].background.tolerance), calib)
                  for (rgba, mask, calib), bg, k in zip(inputs, bg_colors, usable)]
    return usable, inputs, bg_colors


def label_masks(ctx: BuildContext) -> dict[str, dict[str, np.ndarray]]:
    """``{view: {part: mask}}`` of the scene's ``parts_labels`` (see :mod:`bbr.core.labels`), ``{}`` without labels or
    reference views. Loaded once per build context."""
    spec, hi = ctx.scene, ctx.hull_inputs
    if hi is None or not spec.parts_labels:
        return {}
    if "label_masks" not in ctx.extra:
        path = Path(spec.parts_labels)
        if not path.is_absolute() and spec.source_path is not None:
            path = Path(spec.source_path).parent / path
        ctx.extra["label_masks"] = labels_core.load_label_masks(path, hi.masks)
        log.info("part labels %s: views %s", path, sorted(ctx.extra["label_masks"]))
    return ctx.extra["label_masks"]


def label_name(comp: Component) -> str:
    """Label an object samples with: ``body`` for the silhouette hull, else its component id."""
    return labels_core.BODY if comp.type == HULL_TYPE else comp.id


def label_gates(ctx: BuildContext, comp: Component, views: list[str],
                extra: Optional[dict[str, dict[str, np.ndarray]]] = None) -> Optional[list[Optional[np.ndarray]]]:
    """Per colour view, the pixels ``comp`` may sample (its label), None per view without labels; None overall when the
    scene has no labels or ``comp`` is a part that is not labelled in any view (it is then not gated). ``extra``: labels
    of virtual views (``{view key: {part: mask}}``)."""
    masks = {**label_masks(ctx), **(extra or {})}
    name = label_name(comp)
    if not masks:
        return None
    if name not in labels_core.label_names(masks):
        (log.warning if is_character_scene(ctx.scene) else log.info)(
            "%s has no label in any view: its texture is not gated by the part labels", comp.id)
        return None
    return labels_core.allowed_masks(masks, name, views)


def projection_params(p: dict[str, Any]) -> dict[str, Any]:
    """Texel projection settings (see :func:`bbr.core.texelproj.project_texel_colors`)."""
    return {"edge_distrust_px": int(p.get("edge_distrust_px", DEFAULT_EDGE_DISTRUST_PX)),
            "seam_blend_width": float(p.get("seam_blend_width", DEFAULT_SEAM_BLEND_WIDTH)),
            "conflict_threshold": float(p.get("conflict_threshold", DEFAULT_CONFLICT_THRESHOLD)),
            **({"min_facing": float(p["min_facing"])} if "min_facing" in p else {})}


def texture_material_name(comp: Component) -> str:
    """Material name of a textured object: the hull keeps ``M_ref_texture``, parts get ``M_ref_texture:<id>``."""
    return material.REF_TEXTURE_MATERIAL if comp.type == HULL_TYPE else f"{material.REF_TEXTURE_MATERIAL}:{comp.id}"


def mirror_extras(ctx: BuildContext, p: dict[str, Any], usable: list[str], inputs: list[tuple], bg_colors: list[np.ndarray]
                  ) -> tuple[list[str], list[tuple], list[np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """``(keys, (rgba, mask, calib) per virtual view, backdrop colours, {key: labels})`` of ``mirror_color_views``: each
    listed colour view along X reflected through X = 0 (see :func:`bbr.core.symmetry.mirror_view`), built from the
    already despilled ``inputs``. Views that are not usable colour views or do not look along X are skipped."""
    keys: list[str] = []
    out: list[tuple] = []
    bgs: list[np.ndarray] = []
    labels: dict[str, dict[str, np.ndarray]] = {}
    wanted = [str(k) for k in p.get("mirror_color_views") or []]
    if not wanted:
        return keys, out, bgs, labels
    masks = label_masks(ctx)
    names = {n for parts in masks.values() for n in parts}
    mirror_of = {c.id: str(c.params["mirror_of"]) for c in ctx.scene.components if c.params.get("mirror_of")}
    pairs = symmetry.mirror_pairs(sorted(names), p.get("mirror_pairs"), mirror_of)
    asym = [str(a) for a in p.get("asymmetric_parts") or []]
    for src in wanted:
        if src not in usable or inputs[usable.index(src)][2].camera_axis not in ("+X", "-X"):
            log.warning("mirror_color_views: %r is not a colour view looking along X; skipped", src)
            continue
        i = usable.index(src)
        rgba, mask, calib = inputs[i]
        mv = symmetry.mirror_view(rgba, mask, calib, masks.get(src), pairs, asym)
        keys.append(mv.key)
        out.append((mv.rgba, mv.mask, mv.calib))
        bgs.append(bg_colors[i])
        labels[mv.key] = mv.labels
    return keys, out, bgs, labels


# --------------------------------------------------------------------------- baking
def bake_object(comp: Component, ctx: BuildContext, obj: bpy.types.Object, *, unwrap: bool,
                zbuffers: Optional[dict[str, np.ndarray]] = None, cache: Optional[dict] = None) -> dict[str, Any]:
    """Unwrap (when ``unwrap``) and bake the projected texture of one object; returns ``{"uv": ..., "texture": ...}``.

    ``zbuffers``: view key -> depth buffer of all model meshes (see :func:`hullpost.scene_depth_buffers`).
    """
    hi = ctx.hull_inputs
    p = effective_params(ctx.scene, comp)
    if "mirror_color_views" not in p and comp.type != HULL_TYPE and not is_character_scene(ctx.scene) and ctx.scene.symmetry.get("x"):
        along_x = [v.key for v in hi.view_spec.enabled_views("color") if v.key in hi.calibs and hi.calibs[v.key].camera_axis in ("+X", "-X")]
        p = {**p, "mirror_color_views": along_x}    # hard-surface, symmetric in X: the unseen flank takes the mirrored side view
    if "min_facing" not in p and comp.type != HULL_TYPE and not is_character_scene(ctx.scene):
        p = {**p, "min_facing": HARD_SURFACE_MIN_FACING}
    key = (tuple(p.get("color_views") or ()), bool(p.get("spill_suppression", True)), p.get("spill_edge_px"))
    cache = cache if cache is not None else {}
    if key not in cache:
        cache[key] = color_inputs(hi, p)
    usable, inputs, bg_colors = cache[key]
    stats: dict[str, Any] = {}
    if not inputs:
        log.warning("%s: color projection requested but no color view is available", comp.id)
        return stats
    default_size = hullpost.TEXTURE_SIZE if comp.type == HULL_TYPE else PART_TEXTURE_SIZE
    if comp.type != HULL_TYPE and not is_character_scene(ctx.scene) and "texture_size" not in comp.params:
        area = float(sum(poly.area for poly in obj.data.polygons))
        default_size = suggest_texture_size(area, min(hi.calibs[k].m_per_px for k in usable))   # hard-surface: by surface vs reference resolution
    size = int(comp.params.get("texture_size", default_size))
    padding = int(p.get("texture_padding_px", DEFAULT_PADDING_PX))
    if unwrap:
        stats["uv"] = hullpost.unwrap(obj, float(p.get("uv_angle_deg", hullpost.UV_ANGLE_DEG)), padding, size,
                                       "CONCAVE" if is_character_scene(ctx.scene) else "AABB")
    mkeys: list[str] = []
    mlabels: dict[str, dict[str, np.ndarray]] = {}
    if p.get("mirror_color_views") and not symmetry.is_asymmetric(comp.id, p.get("asymmetric_parts") or []):
        mkey = ("mirror", *key, tuple(p["mirror_color_views"]), repr(p.get("mirror_pairs")), repr(p.get("asymmetric_parts")))
        if mkey not in cache:
            cache[mkey] = mirror_extras(ctx, p, usable, inputs, bg_colors)
        mkeys, minputs, mbg, mlabels = cache[mkey]
        inputs, bg_colors = inputs + minputs, bg_colors + mbg
        stats["mirror_views"] = mkeys
    zb = None
    if zbuffers is not None:
        zb = [zbuffers[k] for k in usable]
        for k, (_, _, calib) in zip(mkeys, inputs[len(usable):]):
            if ("zbuf", k) not in cache:
                cache[("zbuf", k)] = hullpost.scene_depth_buffers(scene.model_meshes(), [calib])[0]
            zb.append(cache[("zbuf", k)])
    projection = projection_params(p)
    gates = label_gates(ctx, comp, usable + mkeys, mlabels)
    if gates is not None:
        projection["label_masks"] = gates
    if mkeys:
        projection["view_penalty"] = [0.0] * len(usable) + [float(p.get("mirror_penalty", MIRROR_PENALTY))] * len(mkeys)
    out_dir = ctx.extra.get("out_dir") or hullpost.blend_dir()
    image, stats["texture"] = hullpost.bake_texture(
        obj, ctx.scene.agent_id(comp.id).replace(":", "_"), inputs, size, padding, out_dir, projection,
        bg_colors, zb, flat_normals=comp.type != HULL_TYPE and not is_character_scene(ctx.scene))
    pbr_values = pbr.resolve(pbr.entry_for(comp.material, ctx.scene.materials), pbr.column_for(True, ctx.scene.style))
    if pbr_values:
        stats["material"] = {"key": comp.material, **pbr_values}
    hullpost.assign_texture_material(obj, image, texture_material_name(comp), pbr_values or None)
    return stats


def texture_pass(ctx: BuildContext, only: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Bake every textured object of the scene (deferred hull texture + parts) against the final scene geometry.

    Call after all components are built (and bevelled / normal-cleaned). ``only`` restricts it to one component id.
    Returns the per-component stats, also merged into ``ctx.extra["hull_stats"]`` (shown in the report).
    """
    hi = ctx.hull_inputs
    spec = ctx.scene
    if hi is None:
        return {}
    project = project_textures(spec.style, hi.view_spec)
    character = is_character_scene(spec)
    targets: list[tuple[Component, bool]] = []
    hull = hull_component(spec)
    if hull is not None and ctx.extra.get("defer_texture") and hull.params.get("color_projection", True):
        targets.append((hull, False))
    for part in textured_parts(spec, project):
        if part.mirror and any(part.mirror.values()) and (character or not project):
            log.warning("%s: mirrored parts are not textured (the mirror copy would share one projection)", part.id)
            continue
        targets.append((part, True))
    targets = [(c, u) for c, u in targets if only is None or c.id == only]
    found = [(c, u, scene.find_by_agent_id(spec.agent_id(c.id))) for c, u in targets]
    found = [(c, u, o) for c, u, o in found if o is not None]
    if not found:
        return {}
    keys = [k for k in hi.calibs if k in hi.rgba]
    zbufs = dict(zip(keys, hullpost.scene_depth_buffers(scene.model_meshes(), [hi.calibs[k] for k in keys])))
    cache: dict = {}
    results: dict[str, dict[str, Any]] = {}
    for comp, unwrap, obj in found:
        res = bake_object(comp, ctx, obj, unwrap=unwrap, zbuffers=zbufs, cache=cache)
        results[comp.id] = res
        stats = ctx.extra.setdefault("hull_stats", {}).setdefault(comp.id, {})
        stats.pop("texture_deferred", None)
        stats.update(res)
        log.info("textured %s", comp.id)
    return results
