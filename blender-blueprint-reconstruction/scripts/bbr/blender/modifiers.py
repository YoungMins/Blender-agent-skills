"""Idempotent modifier management. Stack order is always Mirror -> Boolean -> Bevel -> Weighted Normal."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import bpy

from .. import get_logger

log = get_logger(__name__)

MOD_MIRROR = "BBR_Mirror"
MOD_BOOLEAN_PREFIX = "BBR_Boolean"
MOD_BEVEL = "BBR_Bevel"
MOD_WEIGHTED_NORMAL = "BBR_WeightedNormal"
STACK_ORDER = ("MIRROR", "BOOLEAN", "BEVEL", "WEIGHTED_NORMAL")
BEVEL_ANGLE_DEG = 30.0
MIRROR_MERGE_THRESHOLD = 0.0005
WEIGHTED_NORMAL_THRESHOLD = 0.01


def ensure_modifier(obj: bpy.types.Object, name: str, mod_type: str, **props: Any) -> bpy.types.Modifier:
    """Find modifier ``name`` on ``obj`` (create it when missing or of another type) and set ``props``."""
    mod = obj.modifiers.get(name)
    if mod is not None and mod.type != mod_type:
        obj.modifiers.remove(mod)
        mod = None
    if mod is None:
        mod = obj.modifiers.new(name, mod_type)
    for key, value in props.items():
        setattr(mod, key, value)
    return mod


def remove_modifier(obj: bpy.types.Object, name: str) -> bool:
    """Remove modifier ``name`` if present; True when something was removed."""
    mod = obj.modifiers.get(name)
    if mod is None:
        return False
    obj.modifiers.remove(mod)
    return True


def enforce_order(obj: bpy.types.Object) -> None:
    """Reorder modifiers so types follow :data:`STACK_ORDER` (stable within a type; unknown types last)."""
    def rank(mod: bpy.types.Modifier) -> int:
        return STACK_ORDER.index(mod.type) if mod.type in STACK_ORDER else len(STACK_ORDER)

    desired = sorted(obj.modifiers, key=rank)
    for index, mod in enumerate(desired):
        current = list(obj.modifiers).index(mod)
        if current != index:
            obj.modifiers.move(current, index)


def apply_mirror(obj: bpy.types.Object, mirror: Mapping[str, bool], origin: bpy.types.Object) -> None:
    """Mirror about the world origin through the ``AI_ORIGIN`` empty; removed when no axis is set."""
    axes = (bool(mirror.get("x")), bool(mirror.get("y")), bool(mirror.get("z")))
    if not any(axes):
        remove_modifier(obj, MOD_MIRROR)
        return
    ensure_modifier(obj, MOD_MIRROR, "MIRROR", use_axis=axes, mirror_object=origin,
                    use_mirror_merge=False, merge_threshold=MIRROR_MERGE_THRESHOLD)
    enforce_order(obj)


def apply_boolean(target: bpy.types.Object, cutter: bpy.types.Object, operation: str = "DIFFERENCE") -> None:
    """Boolean on ``target`` with ``cutter`` (wire display, not rendered)."""
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    ensure_modifier(target, f"{MOD_BOOLEAN_PREFIX}_{cutter.name}", "BOOLEAN", operation=operation,
                    operand_type="OBJECT", object=cutter, solver="EXACT")
    enforce_order(target)


def prune_booleans(target: bpy.types.Object, keep_cutters: set[str]) -> None:
    """Drop BBR boolean modifiers whose cutter is not in ``keep_cutters`` (object names)."""
    for mod in list(target.modifiers):
        if mod.name.startswith(MOD_BOOLEAN_PREFIX) and (mod.object is None or mod.object.name not in keep_cutters):
            target.modifiers.remove(mod)


def apply_bevel(obj: bpy.types.Object, width: Optional[float], segments: int = 2) -> None:
    """Bevel (angle-limited). ``width`` None or <= 0 removes it."""
    if not width or width <= 0:
        remove_modifier(obj, MOD_BEVEL)
        return
    ensure_modifier(obj, MOD_BEVEL, "BEVEL", width=float(width), segments=int(segments), limit_method="ANGLE",
                    angle_limit=math.radians(BEVEL_ANGLE_DEG))
    enforce_order(obj)


def apply_weighted_normal(obj: bpy.types.Object, enabled: bool) -> None:
    """Weighted Normal with Keep Sharp (the mesh must be smooth-shaded for it to matter)."""
    if not enabled:
        remove_modifier(obj, MOD_WEIGHTED_NORMAL)
        return
    ensure_modifier(obj, MOD_WEIGHTED_NORMAL, "WEIGHTED_NORMAL", keep_sharp=True, thresh=WEIGHTED_NORMAL_THRESHOLD)
    enforce_order(obj)


def stack_summary(obj: bpy.types.Object) -> list[str]:
    """Modifier types in stack order."""
    return [m.type for m in obj.modifiers]
