"""Mesh data helpers: arrays <-> Mesh, smoothing flags, color attributes, evaluated geometry."""
from __future__ import annotations

from typing import Iterable, Optional

import bmesh
import bpy
import numpy as np

from .. import get_logger

log = get_logger(__name__)


def get_or_create_mesh(name: str) -> bpy.types.Mesh:
    """Mesh datablock ``name`` (created empty when missing)."""
    return bpy.data.meshes.get(name) or bpy.data.meshes.new(name)


def mesh_from_arrays(mesh: bpy.types.Mesh, verts: np.ndarray, faces: np.ndarray) -> bpy.types.Mesh:
    """Replace the geometry of ``mesh`` in place with ``verts (N,3)`` and ``faces (M,k)``."""
    mesh.clear_geometry()
    mesh.from_pydata(np.asarray(verts, dtype=float).tolist(), [], np.asarray(faces, dtype=int).tolist())
    mesh.update()
    return mesh


def mesh_from_polygons(mesh: bpy.types.Mesh, verts: np.ndarray, faces: list) -> bpy.types.Mesh:
    """Like :func:`mesh_from_arrays` for faces of mixed vertex counts (``faces`` = list of index tuples)."""
    mesh.clear_geometry()
    mesh.from_pydata(np.asarray(verts, dtype=float).tolist(), [], [list(map(int, f)) for f in faces])
    mesh.update()
    return mesh


def set_smooth(mesh: bpy.types.Mesh, smooth: bool = True) -> None:
    """Smooth or flat shading for every face."""
    if smooth:
        mesh.shade_smooth()
    else:
        mesh.shade_flat()


def set_smooth_by_angle(mesh: bpy.types.Mesh, angle_deg: float = 35.0) -> None:
    """Smooth shading with edges sharper than ``angle_deg`` marked sharp (crisp rims on an otherwise smooth part)."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for f in bm.faces:
        f.smooth = True
    limit = np.radians(angle_deg)
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > limit:
            e.smooth = False
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def recalc_normals(mesh: bpy.types.Mesh) -> None:
    """Make face normals consistently outward (normal cleanup)."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def vertices_array(mesh: bpy.types.Mesh) -> np.ndarray:
    """Vertex coordinates ``(N,3)`` float64."""
    buf = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", buf)
    return buf.reshape(-1, 3).astype(np.float64)


def evaluated_world_vertices(obj: bpy.types.Object) -> np.ndarray:
    """World-space vertices of the object after modifiers, ``(N,3)``."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    try:
        co = vertices_array(mesh)
    finally:
        ev.to_mesh_clear()
    mat = np.array(obj.matrix_world)
    return co @ mat[:3, :3].T + mat[:3, 3]


def world_bbox(objs: Iterable[bpy.types.Object]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(min, max) corners of the evaluated meshes in world space; None when there is no geometry."""
    pts = [evaluated_world_vertices(o) for o in objs if o.type == "MESH"]
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    allp = np.concatenate(pts)
    return allp.min(axis=0), allp.max(axis=0)


def signed_volume(mesh: bpy.types.Mesh) -> float:
    """Signed volume from loop triangles (positive when normals point outward on a closed mesh)."""
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    if n == 0:
        return 0.0
    tri = np.empty(n * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri)
    v = vertices_array(mesh)[tri.reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
