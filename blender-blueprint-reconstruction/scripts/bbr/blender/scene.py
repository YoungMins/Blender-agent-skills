"""Scene structure: collections, agent_id lookup (find -> update), origin empty, inspection, cleanup."""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import bpy

from .. import get_logger

log = get_logger(__name__)

COLL_REF = "REF_BLUEPRINT"
COLL_MODEL = "AI_MODEL"
COLL_DEBUG = "AI_DEBUG"
ALL_COLLECTIONS = (COLL_REF, COLL_MODEL, COLL_DEBUG)

PROP_AGENT_ID = "agent_id"
PROP_CONFIDENCE = "reconstruction_confidence"
PROP_TYPE = "bbr_component_type"
PROP_STAGE = "bbr_stage"
PROP_CUTTER = "bbr_cutter"
PROP_TRANSPARENT = "bbr_transparent"   # see-through (glass): out of the colour metric, texture occlusion and object-ID renders
PROP_CHECKPOINTS = "bbr_checkpoints"

ORIGIN_NAME = "AI_ORIGIN"
ORIGIN_AGENT_ID = "bbr:origin"
ORIGIN_EMPTY_SIZE = 0.05


def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    """Return collection ``name`` (created if missing), linked under ``parent`` (default: the scene root)."""
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def ensure_structure() -> dict[str, bpy.types.Collection]:
    """Create REF_BLUEPRINT / AI_MODEL / AI_DEBUG and the origin empty; return the collections by name."""
    colls = {name: ensure_collection(name) for name in ALL_COLLECTIONS}
    ensure_origin_empty()
    return colls


def find_by_agent_id(agent_id: str) -> Optional[bpy.types.Object]:
    """Object whose custom property ``agent_id`` equals ``agent_id`` (first match), else None."""
    found = [o for o in bpy.data.objects if o.get(PROP_AGENT_ID) == agent_id]
    if len(found) > 1:
        log.warning("%d objects share agent_id %s; using %s", len(found), agent_id, found[0].name)
    return found[0] if found else None


def ensure_object(agent_id: str, collection: bpy.types.Collection,
                  data: Optional[bpy.types.ID] = None) -> bpy.types.Object:
    """Find the object by ``agent_id`` (update) or create it (``data`` = mesh/camera/None for an empty).

    The object ends up linked to ``collection`` only. Object identity is kept across re-runs.
    """
    obj = find_by_agent_id(agent_id)
    if obj is None:
        obj = bpy.data.objects.new(agent_id, data)
        obj[PROP_AGENT_ID] = agent_id
        log.debug("created object %s", agent_id)
    elif data is not None and obj.data is not data:
        obj.data = data
    for other in list(obj.users_collection):
        if other is not collection:
            other.objects.unlink(obj)
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    return obj


def ensure_origin_empty() -> bpy.types.Object:
    """The ``AI_ORIGIN`` empty at the world origin (mirror pivot), in AI_MODEL."""
    obj = ensure_object(ORIGIN_AGENT_ID, ensure_collection(COLL_MODEL))
    obj.name = ORIGIN_NAME
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = ORIGIN_EMPTY_SIZE
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return obj


def ensure_mirror_pivot(owner_agent_id: str, center: Optional[Sequence[float]]) -> Optional[bpy.types.Object]:
    """Empty at ``center`` (world metres) that a component with ``mirror_center`` mirrors about; None (and the empty removed)
    without a centre. Only the mirrored axes of the centre matter; agent_id ``bbr:pivot:<owner agent_id>``."""
    agent_id = f"bbr:pivot:{owner_agent_id}"
    if center is None:
        stale = find_by_agent_id(agent_id)
        if stale is not None:
            remove_object(stale)
        return None
    obj = ensure_object(agent_id, ensure_collection(COLL_MODEL))
    obj.name = f"AI_PIVOT_{owner_agent_id.split(':', 1)[-1]}"
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = ORIGIN_EMPTY_SIZE
    obj.location = tuple(float(c) for c in center)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return obj


def remove_object(obj: bpy.types.Object) -> None:
    """Delete an object and its mesh/camera data when nothing else uses it."""
    data = obj.data
    data_name = None if data is None else data.name
    bpy.data.objects.remove(obj, do_unlink=True)
    if data_name is None:
        return
    for registry in (bpy.data.meshes, bpy.data.cameras):  # Blender may already have freed the orphaned data
        orphan = registry.get(data_name)
        if orphan is not None and orphan.users == 0:
            registry.remove(orphan)


def objects_in(collection_name: str) -> list[bpy.types.Object]:
    """Objects directly in a collection (empty list when it does not exist)."""
    coll = bpy.data.collections.get(collection_name)
    return list(coll.objects) if coll else []


def model_meshes(include_cutters: bool = False, opaque: bool = False) -> list[bpy.types.Object]:
    """Mesh objects in AI_MODEL (cutters excluded unless asked). ``opaque``: also leave out see-through objects
    (``PROP_TRANSPARENT``, a clear glass shell): the reference shows what lies behind them."""
    return [o for o in objects_in(COLL_MODEL)
            if o.type == "MESH" and (include_cutters or not o.get(PROP_CUTTER)) and not (opaque and o.get(PROP_TRANSPARENT))]


def duplicate_agent_ids() -> dict[str, int]:
    """agent_id -> number of objects carrying it, for ids that occur more than once."""
    counts: dict[str, int] = {}
    for obj in bpy.data.objects:
        aid = obj.get(PROP_AGENT_ID)
        if aid:
            counts[aid] = counts.get(aid, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def inspect_scene() -> dict[str, Any]:
    """JSON-able summary: collections, generated objects (agent_id, type, dimensions, confidence, modifiers)."""
    objects = []
    for obj in bpy.data.objects:
        if PROP_AGENT_ID not in obj.keys():
            continue
        objects.append({
            "name": obj.name, "agent_id": obj[PROP_AGENT_ID], "type": obj.type,
            "component_type": obj.get(PROP_TYPE), "stage": obj.get(PROP_STAGE),
            "confidence": obj.get(PROP_CONFIDENCE),
            "dimensions": [round(float(d), 6) for d in obj.dimensions],
            "location": [round(float(c), 6) for c in obj.location],
            "collections": [c.name for c in obj.users_collection],
            "modifiers": [m.type for m in obj.modifiers],
        })
    try:
        checkpoints = json.loads(bpy.context.scene.get(PROP_CHECKPOINTS, "[]"))
    except ValueError:
        checkpoints = []
    return {
        "collections": {name: [o.name for o in objects_in(name)] for name in ALL_COLLECTIONS
                        if name in bpy.data.collections},
        "objects": objects, "checkpoints": checkpoints, "duplicate_agent_ids": duplicate_agent_ids(),
    }
