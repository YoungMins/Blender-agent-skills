"""Component registry: SceneSpec component type -> builder, plus the per-component modifier passes.

Add a new component type with ``@register_component("my_type")``; the builder receives
``(component, BuildContext)``, fills the object's mesh IN PLACE and returns the object. Common work
(transform, custom properties, material) is done by :func:`build_component`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import bpy
import numpy as np

from .. import BBRError, get_logger
from ..core import lathe, partfit, partgeom, pbr, surfaceattach, symmetry, triangulate, zbuffer
from ..core.config import PipelineConfig
from ..core.spec import Component, SceneSpec
from . import material, mesh, modifiers, primitive, scene

log = get_logger(__name__)

HULL_TYPE = "silhouette_hull"
LAYER_TYPE = "layer_hull"   # a hull of its own that sits in a recess of the body hull (see core/layers.py)
DEFAULT_MATERIAL_KEY = "default"
SHELL_MATERIAL = "glass_clear"   # default class of a layer shell (a visor / helmet glass over a face layer)
DEFAULT_BEVEL_WIDTH = 0.003
DEFAULT_BEVEL_SEGMENTS = 2
DEFAULT_FRAME_THICKNESS = 0.02
PART_TYPES = ("cone", "blade", "disc", "strand")  # thin-part primitives: textured by default, see partstex
NO_AUTO_BEVEL_TYPES = ("rounded_box", "torus", "lathe", HULL_TYPE, LAYER_TYPE, "gem") + PART_TYPES  # already rounded / organic / faceted


@dataclass
class HullInputs:
    """Reference data a silhouette hull needs (set by the reconstruction driver)."""

    view_spec: Any
    rgba: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    calibs: Mapping[str, Any]
    dims: Mapping[str, float]


@dataclass
class BuildContext:
    """Everything a component builder may need."""

    scene: SceneSpec
    config: PipelineConfig
    model: bpy.types.Collection
    hull_inputs: Optional[HullInputs] = None
    extra: dict[str, Any] = field(default_factory=dict)


Builder = Callable[[Component, BuildContext], bpy.types.Object]
COMPONENT_REGISTRY: dict[str, Builder] = {}


def register_component(*types: str) -> Callable[[Builder], Builder]:
    """Decorator registering a builder for one or more component type names."""
    def deco(fn: Builder) -> Builder:
        for t in types:
            COMPONENT_REGISTRY[t] = fn
        return fn
    return deco


def object_for(comp: Component, ctx: BuildContext) -> tuple[bpy.types.Object, bpy.types.Mesh]:
    """Find-or-create the component's mesh and object (identity kept across re-runs)."""
    agent_id = ctx.scene.agent_id(comp.id)
    data = mesh.get_or_create_mesh(agent_id)
    return scene.ensure_object(agent_id, ctx.model, data), data


def _require_size(comp: Component) -> list[float]:
    if not comp.size:
        raise BBRError(f"component {comp.id!r} of type {comp.type!r} needs a size")
    return comp.size


# --------------------------------------------------------------------------- builders
@register_component("box", "panel")
def _build_box(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    obj, data = object_for(comp, ctx)
    primitive.create_box(data, _require_size(comp))
    return obj


@register_component("plate")
def _build_plate(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Two things share the name. A plate placed from ``anchors_px`` / ``mirror_of`` (wing, fin, ear fin, leaf) is a
    contour plate: its outline is the part's label mask in ``params.face_view``, extruded ``params.thickness`` along the plane
    normal, optionally curved by ``params.bend`` degrees along its root -> tip axis (:func:`partgeom.plate_mesh`; the solved
    ``outline`` / ``outline_tris`` / ``axis`` / ``normal`` arrive in the params). Without anchors it is the flat box of ``size``
    (hard-surface). Smooth shading with a sharp-edge angle: flat faces stay flat."""
    p = comp.params
    if "outline" not in p:
        return _build_box(comp, ctx)
    obj, data = object_for(comp, ctx)
    verts, faces = partgeom.plate_mesh(p["outline"], p["outline_tris"], float(p["thickness"]), p["axis"], p["normal"],
                                       float(p.get("bend", 0.0)), int(p.get("subdivisions", 0)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("rounded_box")
def _build_rounded_box(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Box with rounded edges: ``params.radius`` a number (all edges) or ``[rx, ry, rz]`` (reach of the rounding along each
    axis, 0 = sharp that way), ``segments`` per rounded zone."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    radius = p.get("radius", 0.005)
    if isinstance(radius, (list, tuple)):
        verts, faces = partgeom.rounded_box_mesh(_require_size(comp), [float(r) for r in radius], int(p.get("segments", 3)))
        mesh.mesh_from_polygons(data, verts, faces)
        return obj
    primitive.create_rounded_box(data, _require_size(comp), radius, p.get("segments", 3))
    return obj


@register_component("arch", "half_cylinder")
def _build_arch(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Half cylinder (lid, vault) of ``size`` [x, y, z]: ``params.axis`` = long axis (X default), ``bulge`` = signed axis the
    dome rises along (default +Z), ``thickness`` > 0 = half-ring band (shell segment), ``segments``. See partgeom.arch_mesh."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    verts, faces = partgeom.arch_mesh(_require_size(comp), str(p.get("axis", "X")), p.get("bulge"), float(p.get("thickness", 0.0)),
                                      int(p.get("segments", partgeom.DEFAULT_ARCH_SEGMENTS)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("torus")
def _build_torus(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Ring: ``params.major_radius`` (number or [r1, r2] elliptic) and ``minor_radius`` (required, true size, no ``size``),
    ``axis`` = ring-plane normal (Z default), ``major_segments`` / ``minor_segments``. Origin = ring centre."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    for key in ("major_radius", "minor_radius"):
        if key not in p:
            raise BBRError(f"component {comp.id!r}: torus needs params.{key}")
    verts, faces = partgeom.torus_mesh(p["major_radius"], float(p["minor_radius"]), str(p.get("axis", "Z")),
                                       int(p.get("major_segments", partgeom.DEFAULT_TORUS_MAJOR_SEGMENTS)),
                                       int(p.get("minor_segments", partgeom.DEFAULT_TORUS_MINOR_SEGMENTS)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("lathe")
def _build_lathe(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Body of revolution about local Z (mushroom cap, vase, lamp, column): ``params.profile`` ``[[z, r], ...]`` in meters (or
    measured from the masks with ``profile_from``, see :func:`measure_lathe_params`), ``depth_ratio`` (Y over X extent),
    ``segments``, ``cap_top`` / ``cap_bottom``. Smooth shaded, sharp at the caps. See partgeom.lathe_mesh."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    if not p.get("profile"):
        raise BBRError(f"component {comp.id!r}: lathe needs params.profile ([[z, r], ...] meters) or params.profile_from")
    verts, faces = partgeom.lathe_mesh(p["profile"], int(p.get("segments", partgeom.DEFAULT_LATHE_SEGMENTS)),
                                       float(p.get("depth_ratio", 1.0)), bool(p.get("cap_top", True)), bool(p.get("cap_bottom", True)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("cylinder")
def _build_cylinder(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    obj, data = object_for(comp, ctx)
    p = comp.params
    primitive.create_cylinder(data, _require_size(comp), p.get("axis", "Z"),
                              p.get("vertices", primitive.DEFAULT_CYLINDER_VERTICES))
    return obj


@register_component("sphere")
def _build_sphere(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    obj, data = object_for(comp, ctx)
    primitive.create_sphere(data, _require_size(comp), comp.params.get("segments", primitive.DEFAULT_SPHERE_SEGMENTS))
    return obj


@register_component("capsule")
def _build_capsule(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    obj, data = object_for(comp, ctx)
    size, p = _require_size(comp), comp.params
    r0, r1 = size[0] / 2.0, float(p.get("tip_radius", size[0] / 2.0))
    if abs(r1 - r0) > 1e-9 and p.get("axis", "Z") == "Z":        # tapered capsule (fitted): end radii differ
        verts, faces = partgeom.capsule_mesh(r0, r1, size[2], int(p.get("segments", primitive.DEFAULT_CAPSULE_SEGMENTS)))
        mesh.mesh_from_polygons(data, verts, faces)
        mesh.set_smooth_by_angle(data)
        return obj
    primitive.create_capsule(data, size, p.get("axis", "Z"), p.get("segments", primitive.DEFAULT_CAPSULE_SEGMENTS))
    return obj


@register_component("gem")
def _build_gem(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Crystal: bipyramid of ``params.facets`` (default 6; 4 = octahedron) sides, ``size`` [2r, 2r, length] with the apexes on
    the local Z axis (anchors base / tip + radius_px place it like a capsule); ``params.waist`` (0..1, default 0.5) moves the
    widest ring toward the tip. Flat shaded."""
    obj, data = object_for(comp, ctx)
    size, p = _require_size(comp), comp.params
    verts, faces = partgeom.gem_mesh(size[0] / 2.0, size[2], int(p.get("facets", partgeom.DEFAULT_GEM_FACETS)),
                                     float(p.get("waist", 0.5)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth(data, False)
    return obj


@register_component("leg")
def _build_leg(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Vertical leg: ``params.profile`` square|round, optional ``params.taper`` (bottom / top footprint ratio)."""
    obj, data = object_for(comp, ctx)
    size = _require_size(comp)
    p = comp.params
    if p.get("profile", "square") == "round":
        primitive.create_cylinder(data, size, "Z", p.get("vertices", primitive.DEFAULT_CYLINDER_VERTICES))
    else:
        primitive.create_box(data, size)
    taper = float(p.get("taper", 1.0))
    if taper != 1.0:
        co = mesh.vertices_array(data)
        t = co[:, 2] / size[2] + 0.5  # 0 at the foot, 1 at the top
        scale = taper + (1.0 - taper) * t
        co[:, 0] *= scale
        co[:, 1] *= scale
        data.vertices.foreach_set("co", co.astype(np.float32).ravel())
        data.update()
    return obj


@register_component("frame")
def _build_frame(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    obj, data = object_for(comp, ctx)
    p = comp.params
    primitive.create_frame(data, _require_size(comp), p.get("thickness", DEFAULT_FRAME_THICKNESS), p.get("axis", ""))
    return obj


def _param_vec(p: Mapping[str, Any], key: str, default: Optional[tuple]) -> Optional[tuple]:
    v = p.get(key, default)
    return None if v is None else tuple(float(x) for x in v)


@register_component("cone")
def _build_cone(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Horn: tapered tube from ``base_radius`` to ``tip_radius`` over ``length`` along ``axis`` (local vector, default Z;
    origin = base centre), optionally bent by ``bend`` degrees toward ``bend_dir`` (``flatten`` < 1 thins the cross-section
    perpendicular to the bend plane); ``segments`` / ``sides`` set the tessellation. Smooth shaded, true size (no ``size`` needed)."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    for key in ("base_radius", "length"):
        if key not in p:
            raise BBRError(f"component {comp.id!r}: cone needs params.{key}")
    verts, faces = partgeom.cone_mesh(
        float(p["base_radius"]), float(p.get("tip_radius", 0.0)), float(p["length"]), _param_vec(p, "axis", (0, 0, 1)),
        float(p.get("bend", 0.0)), _param_vec(p, "bend_dir", None), int(p.get("segments", partgeom.DEFAULT_SEGMENTS)),
        int(p.get("sides", partgeom.DEFAULT_SIDES)),
        float(p.get("flatten", 1.0)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("strand")
def _build_strand(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Tube through ``params.points`` (N >= 3 ordered points in the part's local frame, the first at the origin) along a
    Catmull-Rom spline with ``params.radii`` (one per point: the taper), optional ``flatten`` (< 1: thinner in depth) and
    ``segments`` (rings per span) / ``sides``. Placed from ``anchors_px`` p0 ... pN-1 + ``radius_px`` (twin tails, braids,
    long hair locks, cables). Smooth shaded, true size (no ``size`` needed)."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    for key in ("points", "radii"):
        if key not in p:
            raise BBRError(f"component {comp.id!r}: strand needs params.{key} (or anchors_px p0, p1, p2 ... + radius_px)")
    verts, faces = partgeom.strand_mesh(p["points"], p["radii"], float(p.get("flatten", 1.0)),
                                        int(p.get("segments", partgeom.DEFAULT_STRAND_SPAN_SEGMENTS)), int(p.get("sides", partgeom.DEFAULT_SIDES)))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


@register_component("blade")
def _build_blade(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Sword / flat part: ``params`` length, width, thickness (+ optional tip_width, tip_thickness, tip_length for a
    pointed tip, axis, width_dir); origin = centre of the base end. Flat shaded."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    for key in ("length", "width", "thickness"):
        if key not in p:
            raise BBRError(f"component {comp.id!r}: blade needs params.{key}")
    verts, faces = partgeom.blade_mesh(
        float(p["length"]), float(p["width"]), float(p["thickness"]), p.get("tip_width"), p.get("tip_thickness"),
        float(p.get("tip_length", 0.0)), _param_vec(p, "axis", (0, 0, 1)), _param_vec(p, "width_dir", None))
    mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth(data, False)
    return obj


@register_component("disc")
def _build_disc(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Shield: cylinder ``radius`` x ``thickness`` centred on the origin along ``axis`` (default Z) with an optional
    dome (``dome_height``, ``dome_radius``) on its +axis face; ``radius_ratio`` makes it elliptical (extent along local Y =
    ``radius * radius_ratio``, e.g. an oval hat brim). Without a dome or ratio it is the plain cylinder builder."""
    obj, data = object_for(comp, ctx)
    p = comp.params
    for key in ("radius", "thickness"):
        if key not in p:
            raise BBRError(f"component {comp.id!r}: disc needs params.{key}")
    radius, thickness, segments = float(p["radius"]), float(p["thickness"]), int(p.get("segments", partgeom.DEFAULT_DISC_SEGMENTS))
    axis = _param_vec(p, "axis", (0, 0, 1))
    ratio = float(p.get("radius_ratio", 1.0))
    if float(p.get("dome_height", 0.0)) <= 0 and tuple(axis) == (0.0, 0.0, 1.0) and ratio == 1.0:
        primitive.create_cylinder(data, (2 * radius, 2 * radius, thickness), "Z", segments)
    else:
        verts, faces = partgeom.disc_mesh(radius, thickness, float(p.get("dome_height", 0.0)), p.get("dome_radius"),
                                          segments, int(p.get("dome_rings", partgeom.DEFAULT_DOME_RINGS)), axis, ratio)
        mesh.mesh_from_polygons(data, verts, faces)
    mesh.set_smooth_by_angle(data)
    return obj


# --------------------------------------------------------------------------- common handling
def material_key(comp: Component) -> Optional[str]:
    """The component's ``material`` value; a layer ``shell`` (glass over a face) defaults to ``SHELL_MATERIAL``."""
    return comp.material or (SHELL_MATERIAL if comp.type == LAYER_TYPE and comp.params.get("shell") else None)


def material_values(comp: Component, spec: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """``(material entry, resolved flat-material PBR values)`` of a component (see :func:`material_for`)."""
    key = material_key(comp)
    params = pbr.entry_for(key, spec.materials) if key else dict(spec.materials.get(DEFAULT_MATERIAL_KEY, {}))
    return params, pbr.resolve(params, pbr.column_for(False, spec.style))


def material_for(comp: Component, ctx: BuildContext) -> bpy.types.Material:
    """Principled material of a component (``materials[key]`` of the SceneSpec or a material class name, else a neutral
    default); a class contributes its :mod:`bbr.core.pbr` values, explicit keys win."""
    params, values = material_values(comp, ctx.scene)
    return material.ensure_material(f"{ctx.scene.name}:{material_key(comp) or DEFAULT_MATERIAL_KEY}", params.get("base_color"),
                                    values.get("roughness", material.DEFAULT_ROUGHNESS), values.get("metallic", 0.0), values)


def finalize_object(obj: bpy.types.Object, comp: Component, ctx: BuildContext) -> None:
    """Transform, custom properties, material and cutter flag shared by every component type."""
    obj.location = comp.location
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = tuple(np.radians(comp.rotation_deg))
    obj.scale = (1.0, 1.0, 1.0)
    obj[scene.PROP_CONFIDENCE] = comp.confidence
    obj[scene.PROP_TYPE] = comp.type
    obj[scene.PROP_STAGE] = comp.stage
    if comp.type != HULL_TYPE:
        material.assign_material(obj, material_for(comp, ctx))
    if pbr.is_transparent(material_values(comp, ctx.scene)[1]):
        obj[scene.PROP_TRANSPARENT] = 1
    elif scene.PROP_TRANSPARENT in obj.keys():
        del obj[scene.PROP_TRANSPARENT]
    if comp.boolean:
        obj[scene.PROP_CUTTER] = 1
        obj.display_type = "WIRE"
        obj.hide_render = True
    else:
        obj.pop(scene.PROP_CUTTER, None)
        obj.display_type = "TEXTURED"
        obj.hide_render = False


# --------------------------------------------------------------------------- anchor-based placement
def is_anchored(comp: Component) -> bool:
    """True for a part placed from image anchors (``params.anchors_px`` / ``params.mirror_of``) instead of a typed pose."""
    return bool(comp.params.get("anchors_px") or comp.params.get("mirror_of"))


def defers_to_target(comp: Component, spec: SceneSpec, _seen: tuple[str, ...] = ()) -> bool:
    """True when the part is placed on another component's surface (``attach_to``, or a ``mirror_of`` of such a part):
    it can only be built after that component, so the hull's pre-build pass leaves it for the normal stage order."""
    if comp.params.get("attach_to"):
        return True
    partners = [comp.params.get("mirror_of"), (comp.params.get("joint") or {}).get("part")]
    return any(q and q not in _seen and defers_to_target(spec.get_component(str(q)), spec, _seen + (comp.id,)) for q in partners)


def _depth_of(ctx: BuildContext, target: str, view: str) -> Optional[np.ndarray]:
    """Depth buffer of the built component ``target`` in ``view`` (cached per build context); None when it is not built."""
    from . import hullpost
    key = f"_zbuf:{target}:{view}"
    if key not in ctx.extra:
        obj = scene.find_by_agent_id(ctx.scene.agent_id(target))
        if obj is None:
            return None
        bpy.context.view_layer.update()
        verts, tris = hullpost.world_triangles([obj])
        ctx.extra[key] = zbuffer.depth_buffer(verts, tris, ctx.hull_inputs.calibs[view])
    return ctx.extra[key]


def _target_depth(comp: Component, ctx: BuildContext, view: str) -> np.ndarray:
    """Depth buffer of the ``attach_to`` component in ``view`` (cached per build context)."""
    target = str(comp.params["attach_to"])
    zbuf = _depth_of(ctx, target, view)
    if zbuf is None:
        raise BBRError(f"{comp.id}: attach_to {target!r} is not built yet; it must come earlier in the build order "
                       f"(an earlier stage, or earlier in the component list)")
    return zbuf


def auto_target_id(spec: SceneSpec) -> Optional[str]:
    """Component an unmeasured coordinate falls back on: the silhouette hull, else a component named ``body``."""
    hull = next((c for c in spec.components if c.type == HULL_TYPE), None)
    if hull is not None:
        return hull.id
    return next((c.id for c in spec.components if c.id == "body"), None)


def _auto_partner(comp: Component, ctx: BuildContext) -> Optional[triangulate.PartSolution]:
    """Plain solution of the ``_l`` / ``_r`` partner (:func:`~bbr.core.symmetry.swap_side`) of an anchored part: only a partner
    with its own ``anchors_px`` and no ``mirror_of`` counts (it does not need this part, so there is no cycle)."""
    pid = symmetry.swap_side(comp.id)
    if pid is None or pid == comp.id:
        return None
    key = f"_auto_partner:{pid}"
    if key not in ctx.extra:
        sol = None
        partner = next((c for c in ctx.scene.components if c.id == pid), None)
        if partner is not None and partner.type == comp.type and partner.params.get("anchors_px") and not partner.params.get("mirror_of"):
            try:
                sol = triangulate.solve_part(pid, partner.type, partner.params, ctx.hull_inputs.calibs,
                                             depth_for=(lambda v: _target_depth(partner, ctx, v)) if partner.params.get("attach_to") else None,
                                             label_for=lambda v: _label_view_masks(ctx, pid).get(v))
            except BBRError:
                sol = None
        ctx.extra[key] = sol
    return ctx.extra[key]


def _record_placement(ctx: BuildContext, comp_id: str, sol: triangulate.PartSolution) -> None:
    """Log the placement warnings and merge the solution report into ``<out_dir>/placement.json``."""
    ctx.extra.setdefault("placement", {})[comp_id] = sol.report()
    for w in sol.warnings:
        log.warning("placement: %s", w)
    out_dir = ctx.extra.get("out_dir")
    if out_dir is None:
        return
    path = Path(out_dir) / "placement.json"
    try:
        merged = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        merged = {}
    merged[comp_id] = sol.report()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


_FIT_CACHE: dict[str, tuple[triangulate.PartSolution, dict[str, Any]]] = {}   # same inputs -> same fit (rebuilds)


def _merge_json(ctx: BuildContext, name: str, part_id: str, entry: Mapping[str, Any]) -> None:
    """Merge ``{part_id: entry}`` into ``<out_dir>/<name>`` (no-op without an output directory)."""
    out_dir = ctx.extra.get("out_dir")
    if out_dir is None:
        return
    path = Path(out_dir) / name
    try:
        merged = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        merged = {}
    merged[part_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def _label_view_masks(ctx: BuildContext, part_id: str) -> dict[str, np.ndarray]:
    """``{view: label mask}`` of a part in the calibrated views where it has label pixels (empty without part labels)."""
    from . import partstex
    return {v: parts[part_id] for v, parts in partstex.label_masks(ctx).items()
            if part_id in parts and parts[part_id].any() and v in ctx.hull_inputs.calibs}


def _fit_key(comp: Component, sol: triangulate.PartSolution, ctx: BuildContext, targets: list[partfit.Target],
             extra: bytes) -> str:
    """Hash of everything a fit depends on (params, solution, calibrations, label and silhouette pixels)."""
    h = hashlib.sha1(json.dumps([comp.type, comp.params, sol.report(), {v: c.to_dict() for v, c in ctx.hull_inputs.calibs.items()},
                                 [(t.part_id, t.mirrored) for t in targets]], sort_keys=True, default=str).encode())
    for t in targets:
        for v in sorted(t.labels):
            h.update(v.encode() + np.packbits(np.asarray(t.labels[v], bool)).tobytes() + np.packbits(ctx.hull_inputs.masks[v]).tobytes())
    h.update(extra)
    return h.hexdigest()


def _refine_with_labels(comp: Component, sol: triangulate.PartSolution, ctx: BuildContext,
                        sibling_for: Callable[[str], triangulate.PartSolution]) -> triangulate.PartSolution:
    """Fit the triangulated part to its label masks (``params.fit``, default on when the part is labelled;
    ``fit_locked`` = parameter names left at the triangulated value) and record the per-view IoU. A ``mirror_of`` partner
    without anchors is fitted jointly with its sibling (the exact mirror, extra label terms), so it is only evaluated here."""
    hi, p = ctx.hull_inputs, comp.params
    own = _label_view_masks(ctx, comp.id)
    if not own:
        return sol
    pure_mirror = bool(p.get("mirror_of")) and not p.get("anchors_px")
    report: dict[str, Any] = {"type": comp.type, "fitted": False}
    targets: list[partfit.Target] = []
    fit_on = bool(p.get("fit", True)) and comp.type in partfit.FIT_TYPES and not pure_mirror
    if fit_on:
        partners = [c for c in ctx.scene.components if c.params.get("mirror_of") == comp.id and not c.params.get("anchors_px")
                    and c.type == comp.type and bool(c.params.get("fit", True))]
        targets = [partfit.Target(comp.id, False, own)] + [partfit.Target(c.id, True, _label_view_masks(ctx, c.id)) for c in partners]
    elif pure_mirror:
        report["note"] = f"exact mirror of {p['mirror_of']}"
    else:
        report["note"] = "fit disabled" if not p.get("fit", True) else f"no fitting for type {comp.type}"
    from . import partstex
    others = {v: np.logical_or.reduce([m for n, m in parts.items() if n != comp.id and n != "body" and not n.startswith("_")]
                                      + [np.zeros_like(hi.masks[v])])
              for v, parts in partstex.label_masks(ctx).items() if v in hi.masks}
    init_iou = partfit.part_iou(comp.type, sol, p, hi.calibs, hi.masks, own, others)
    if fit_on:
        depth_for = (lambda view: _target_depth(comp, ctx, view)) if p.get("attach_to") else None
        sibling, tol, weight = None, 0.0, partfit.MIRROR_WEIGHT
        hint = symmetry.pair_hints(partstex.effective_params(ctx.scene, comp).get("symmetric_pairs")).get(comp.id)
        sib_id = str(p["mirror_of"]) if p.get("mirror_of") else hint[0] if hint else None
        sib_comp = ctx.scene.get_component(sib_id) if sib_id else None
        if sib_comp is not None and (sib_comp.type != comp.type or not is_anchored(sib_comp)):
            log.warning("%s: symmetric partner %s is not an anchored %s; no symmetry term", comp.id, sib_id, comp.type)
        elif sib_id:
            sibling = sibling_for(sib_id)
            if hint and hint[0] == sib_id:
                tol, weight = hint[1], partfit.PAIR_WEIGHT
        depth_sig = b"" if depth_for is None else np.nan_to_num(_target_depth(comp, ctx, next(iter(own))), neginf=-1.0).tobytes()
        locked = [str(x) for x in p.get("fit_locked") or []]
        joint_point = triangulate._parse_joint(comp.id, comp.type, p)[2] if p.get("joint") else None
        key = _fit_key(comp, sol, ctx, targets, depth_sig + repr((sibling.report() if sibling else None, sorted(locked), joint_point, tol, weight)).encode())
        if key not in _FIT_CACHE:
            try:
                _FIT_CACHE[key] = partfit.fit_part(
                    comp.id, comp.type, sol, p, hi.calibs, hi.masks, targets, depth_for=depth_for, locked=locked,
                    taper=bool(p.get("fit_taper", True)), seed=int(p.get("fit_seed", 0)), mirror_sibling=sibling,
                    mirror_tol=tol, mirror_weight=weight, joint_point=joint_point)
            except BBRError as exc:
                # a mirror_of partner without its own face-on anchor takes the placement of its partner: not worth a warning
                (log.info if isinstance(exc, partfit.MissingFaceView) and p.get("mirror_of") else log.warning)(
                    "partfit %s: %s; keeping the triangulated placement", comp.id, exc)
                _FIT_CACHE[key] = (sol, {"fitted": False, "note": str(exc)})
        sol, fit_report = _FIT_CACHE[key]
        report.update(fit_report)
    report["iou_init"] = init_iou
    report["iou"] = partfit.part_iou(comp.type, sol, p, hi.calibs, hi.masks, own, others)
    log.info("%s", partfit.format_line(comp.id, report))
    _merge_json(ctx, "partfit_report.json", comp.id, report)
    if report.get("fitted"):
        pose = {"location": sol.location, "rotation_deg": sol.rotation_deg, "size": sol.size, "params": sol.params}
        _merge_json(ctx, "fitted_parts.json", comp.id, {"type": comp.type, "iou": report["iou"], **triangulate._py(pose)})
        for t in targets:
            if t.mirrored:
                _merge_json(ctx, "fitted_parts.json", t.part_id, {"type": comp.type, "mirror_of": comp.id})
    return sol


def _solve_anchored(comp_id: str, ctx: BuildContext, stack: tuple[str, ...]) -> triangulate.PartSolution:
    cache = ctx.extra.setdefault("_placement", {})
    if comp_id in cache:
        return cache[comp_id]
    if comp_id in stack:
        raise BBRError(f"mirror_of cycle: {' -> '.join(stack + (comp_id,))}")
    comp = ctx.scene.get_component(comp_id)
    if not is_anchored(comp):
        raise BBRError(f"mirror_of {comp_id!r}: that component has no anchors_px")
    if ctx.hull_inputs is None:
        raise BBRError(f"{comp_id}: anchors_px need reference views (pass a ViewSpec)")

    def sibling_for(sib: str) -> triangulate.PartSolution:
        return _solve_anchored(sib, ctx, stack + (comp_id,))

    target = auto_target_id(ctx.scene)
    sol = triangulate.solve_part(
        comp_id, comp.type, comp.params, ctx.hull_inputs.calibs,
        depth_for=lambda view: _target_depth(comp, ctx, view), sibling_for=sibling_for, joint_for=sibling_for,
        auto_sibling=lambda: _auto_partner(comp, ctx),
        auto_depth_for=(lambda view: _depth_of(ctx, target, view)) if target and target != comp_id else None,
        auto_target=target or "body", label_for=lambda view: _label_view_masks(ctx, comp_id).get(view))
    sol = _refine_with_labels(comp, sol, ctx, sibling_for)
    cache[comp_id] = sol
    _record_placement(ctx, comp_id, sol)
    return sol


def measure_lathe_params(comp: Component, ctx: BuildContext) -> Component:
    """A ``lathe`` with ``params.profile_from`` as the same lathe with its measured ``profile``, ``depth_ratio`` and axis
    position (``location`` = [x, y, 0]; the profile z are world heights). ``profile_from`` = ``{"view": "front", "bbox_px":
    [x0, y0, x1, y1]}`` (the silhouette inside the box) or ``{"part": label_id}`` (that part's label mask; the side view uses
    its label there too when it has one), optional ``side_view`` / ``side_bbox_px`` / ``bottom_z``. An explicit ``params.profile`` wins."""
    from . import partstex
    hi, p = ctx.hull_inputs, comp.params
    src = p.get("profile_from")
    if p.get("profile") or not src:
        return comp
    if hi is None:
        raise BBRError(f"{comp.id}: lathe profile_from needs reference views (pass a ViewSpec)")
    view = str(src.get("view", "front"))
    side = src.get("side_view") or next((k for k, c in hi.calibs.items() if c.camera_axis in ("+X", "-X")), None)
    if view not in hi.calibs or (side is not None and side not in hi.calibs):
        raise BBRError(f"{comp.id}: profile_from views {view!r} / {side!r} are not calibrated; views: {sorted(hi.calibs)}")
    labelled = partstex.label_masks(ctx)
    if src.get("part"):
        part = str(src["part"])
        front_mask = labelled.get(view, {}).get(part)
        if front_mask is None or not front_mask.any():
            raise BBRError(f"{comp.id}: label {part!r} has no pixels in view {view!r}; labelled views: "
                           f"{sorted(v for v, parts in labelled.items() if part in parts)}")
        side_part = labelled.get(side, {}).get(part) if side else None
        side_mask = side_part if side_part is not None and side_part.any() else hi.masks.get(side)
    elif src.get("bbox_px"):
        front_mask, side_mask = hi.masks[view], hi.masks.get(side) if side else None
    else:
        raise BBRError(f"{comp.id}: profile_from needs bbox_px [x0, y0, x1, y1] or part (a label id)")
    m = lathe.measure_lathe(front_mask, hi.calibs[view], side_mask, hi.calibs[side] if side else None,
                            None if src.get("part") else src["bbox_px"], src.get("side_bbox_px"),
                            float(src.get("tol_px", lathe.DEFAULT_TOL_PX)), int(src.get("max_points", lathe.DEFAULT_MAX_POINTS)),
                            p.get("depth_ratio"), src.get("bottom_z"))
    _merge_json(ctx, "lathe_report.json", comp.id, {"profile": m.profile, "axis_xy": m.axis_xy, "depth_ratio": m.depth_ratio, **m.info})
    log.info("lathe %s: %d profile points, axis (%.3f, %.3f), depth ratio %.3f (%s)", comp.id, len(m.profile), *m.axis_xy,
             m.depth_ratio, m.info["depth_ratio_source"])
    return dataclasses.replace(comp, location=[m.axis_xy[0], m.axis_xy[1], 0.0],
                               params={**p, "profile": m.profile, "depth_ratio": m.depth_ratio})


def resolve_placement(comp: Component, ctx: BuildContext) -> Component:
    """The component with the pose / primitive parameters its image anchors imply (location, rotation, size, params).
    A ``lathe`` with ``profile_from`` gets its measured profile and axis position. Parts without anchors (explicit pose,
    the "advanced" path) come back unchanged."""
    if comp.type == "lathe":
        return measure_lathe_params(comp, ctx)
    if not is_anchored(comp):
        return comp
    sol = _solve_anchored(comp.id, ctx, ())
    confidence = "inferred" if sol.inferred and comp.confidence == "observed" else comp.confidence
    return dataclasses.replace(comp, location=sol.location, rotation_deg=sol.rotation_deg, size=sol.size,
                               params={**comp.params, **sol.params}, confidence=confidence)


def place_on_surface(comp: Component, ctx: BuildContext, obj: bpy.types.Object) -> Component:
    """A hard-surface part with ``params.attach_to`` + ``params.attach_at`` seated on that component's surface: the target's own
    (modifier-evaluated) mesh is ray-cast at ``attach_at`` (``{"view": "front", "px": [x, y]}`` or ``{"azimuth_deg": a,
    "height": z}``), the part's local ``face`` axis (``params.face``, ``-Y`` default or ``+Y``) turns along the outward
    surface normal with local Z kept upright, and its back face (read from the built mesh) rests ``params.standoff`` meters
    (default 0, negative sinks it in) off the surface. See :mod:`bbr.core.surfaceattach`."""
    from . import hullpost
    p = comp.params
    target = scene.find_by_agent_id(ctx.scene.agent_id(str(p["attach_to"])))
    if target is None:
        raise BBRError(f"{comp.id}: attach_to {p['attach_to']!r} is not built yet; it must come earlier in the build order "
                       f"(an earlier stage, or earlier in the component list)")
    bpy.context.view_layer.update()
    verts, tris = hullpost.world_triangles([target])
    calibs = ctx.hull_inputs.calibs if ctx.hull_inputs is not None else {}
    res = surfaceattach.place_part(verts, tris, p["attach_at"], calibs, mesh.vertices_array(obj.data), str(p.get("face", "-Y")),
                                   float(p.get("standoff", 0.0)))
    _merge_json(ctx, "surface_attach.json", comp.id, {"target": p["attach_to"], **res.report()})
    log.info("attached %s to %s at %s, normal %s", comp.id, p["attach_to"], res.point.round(4).tolist(), res.normal.round(3).tolist())
    return dataclasses.replace(comp, location=res.location.tolist(), rotation_deg=res.rotation_deg)


def build_component(comp: Component, ctx: BuildContext) -> bpy.types.Object:
    """Build (or rebuild in place) one component."""
    builder = COMPONENT_REGISTRY.get(comp.type)
    if builder is None:
        raise BBRError(f"no builder registered for component type {comp.type!r}; known: {sorted(COMPONENT_REGISTRY)}")
    comp = resolve_placement(comp, ctx)
    obj = builder(comp, ctx)
    if comp.params.get("attach_at"):
        comp = place_on_surface(comp, ctx, obj)
    finalize_object(obj, comp, ctx)
    log.info("built %s (%s, %s)", obj[scene.PROP_AGENT_ID], comp.type, comp.stage)
    return obj


# --------------------------------------------------------------------------- modifier passes
def resolve_bevel(comp: Component, style: Mapping[str, Any]) -> tuple[Optional[float], int]:
    """(width, segments) for a component, or (None, _) when it gets no bevel."""
    segments = int(style.get("bevel_segments", DEFAULT_BEVEL_SEGMENTS))
    if comp.bevel is None:
        if not style.get("bevel", True) or comp.type in NO_AUTO_BEVEL_TYPES or (comp.type == "plate" and is_anchored(comp)):
            return None, segments
        return float(style.get("bevel_width", DEFAULT_BEVEL_WIDTH)), segments
    if isinstance(comp.bevel, Mapping):
        return (float(comp.bevel.get("width", style.get("bevel_width", DEFAULT_BEVEL_WIDTH))),
                int(comp.bevel.get("segments", segments)))
    if isinstance(comp.bevel, bool):
        return (float(style.get("bevel_width", DEFAULT_BEVEL_WIDTH)), segments) if comp.bevel else (None, segments)
    return (float(comp.bevel) or None), segments


def apply_symmetry(obj: bpy.types.Object, comp: Component) -> None:
    """Step 7: Mirror modifier about the AI_ORIGIN empty, or about the component's own pivot empty (``mirror_center``)."""
    pivot = scene.ensure_mirror_pivot(obj[scene.PROP_AGENT_ID], comp.mirror_center if any(comp.mirror.values()) else None)
    modifiers.apply_mirror(obj, comp.mirror, pivot or scene.ensure_origin_empty())


def apply_booleans(spec: SceneSpec, objects: Mapping[str, bpy.types.Object]) -> int:
    """Step 10: every component with ``boolean`` cuts its target; stale boolean modifiers are pruned."""
    cutters: dict[str, set[str]] = {}
    for comp in spec.components:
        if comp.boolean and comp.id in objects and comp.boolean["target"] in objects:
            cutters.setdefault(comp.boolean["target"], set()).add(objects[comp.id].name)
    count = 0
    for comp in spec.components:
        target = objects.get(comp.id)
        if target is None:
            continue
        modifiers.prune_booleans(target, cutters.get(comp.id, set()))
    for comp in spec.components:
        if comp.boolean and comp.id in objects and comp.boolean["target"] in objects:
            modifiers.apply_boolean(objects[comp.boolean["target"]], objects[comp.id], comp.boolean["operation"])
            count += 1
    return count


def apply_bevel(obj: bpy.types.Object, comp: Component, spec: SceneSpec) -> None:
    """Step 11: bevel modifier from the component override or the scene style."""
    width, segments = resolve_bevel(comp, spec.style)
    modifiers.apply_bevel(obj, width, segments)


def apply_normal_cleanup(obj: bpy.types.Object, comp: Component, spec: SceneSpec) -> None:
    """Step 12: consistent outward normals, smooth shading + Weighted Normal (hard-surface style)."""
    if comp.type in (HULL_TYPE, LAYER_TYPE):
        return
    mesh.recalc_normals(obj.data)
    if comp.type == "gem":      # faceted: stays flat shaded, no Weighted Normal smoothing
        return
    weighted = bool(spec.style.get("weighted_normals", True))
    if weighted and resolve_bevel(comp, spec.style)[0] is None:
        # no Bevel modifier will cut the sharp edges, and smoothing across a 90 degree edge (a cylinder's wall into its flat cap)
        # blends the cap normal over the whole wall: a black band below, a washed-out top. Smooth only below 35 degrees.
        mesh.set_smooth_by_angle(obj.data)
    else:
        mesh.set_smooth(obj.data, weighted)
    modifiers.apply_weighted_normal(obj, weighted)


from . import hull  # noqa: E402,F401  (registers the silhouette_hull builder)
