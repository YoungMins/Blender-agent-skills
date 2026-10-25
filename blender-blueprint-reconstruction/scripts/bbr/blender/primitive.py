"""Primitive meshes at true size, built with bmesh into an existing Mesh (object scale stays 1).

Every ``create_*`` fills ``mesh`` in place (replacing its geometry), centred on the local origin.
``size`` is the full extents ``[x, y, z]`` in local axes.
"""
from __future__ import annotations

from typing import Sequence

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from .. import BBRError, get_logger

log = get_logger(__name__)

DEFAULT_CYLINDER_VERTICES = 24
DEFAULT_SPHERE_SEGMENTS = 24
DEFAULT_CAPSULE_SEGMENTS = 16
CAPSULE_RING_SEGMENTS = 9  # odd: no ring on the equator, so the two middle rings can be pulled apart
MAX_BEVEL_FRACTION = 0.49  # of the smallest extent
AXES = ("X", "Y", "Z")

# cylinder-frame coordinates (a, b, c) -> local (x, y, z); c is the long axis
_AXIS_PERMUTATION = {
    "X": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    "Y": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "Z": np.eye(3),
}


def _check_size(size: Sequence[float]) -> tuple[float, float, float]:
    if len(size) != 3 or any(s <= 0 for s in size):
        raise BBRError(f"size must be three positive extents, got {list(size)}")
    return float(size[0]), float(size[1]), float(size[2])


def _axis_extents(axis: str, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Extents (a, b, c) of the cylinder frame for a long axis."""
    if axis not in AXES:
        raise BBRError(f"axis must be one of {AXES}, got {axis!r}")
    return {"X": (size[1], size[2], size[0]), "Y": (size[2], size[0], size[1]), "Z": size}[axis]


def _matrix(perm: np.ndarray, scale: Sequence[float]) -> Matrix:
    m = np.eye(4)
    m[:3, :3] = perm @ np.diag(scale)
    return Matrix(m.tolist())


def _commit(bm: bmesh.types.BMesh, mesh: bpy.types.Mesh) -> bpy.types.Mesh:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh


def create_box(mesh: bpy.types.Mesh, size: Sequence[float]) -> bpy.types.Mesh:
    """Axis-aligned box."""
    sx, sy, sz = _check_size(size)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, matrix=_matrix(np.eye(3), (sx, sy, sz)), verts=bm.verts)
    return _commit(bm, mesh)


def create_rounded_box(mesh: bpy.types.Mesh, size: Sequence[float], radius: float = 0.005,
                       segments: int = 3) -> bpy.types.Mesh:
    """Box with all edges rounded by ``radius`` (clamped below half the thinnest extent)."""
    sx, sy, sz = _check_size(size)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, matrix=_matrix(np.eye(3), (sx, sy, sz)), verts=bm.verts)
    width = min(float(radius), MAX_BEVEL_FRACTION * min(sx, sy, sz))
    if width > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=width, offset_type="OFFSET",
                        segments=max(1, int(segments)), affect="EDGES")
    return _commit(bm, mesh)


def create_cylinder(mesh: bpy.types.Mesh, size: Sequence[float], axis: str = "Z",
                    vertices: int = DEFAULT_CYLINDER_VERTICES) -> bpy.types.Mesh:
    """Elliptic cylinder whose long axis is ``axis`` (``size`` = bounding extents in local axes)."""
    a, b, c = _axis_extents(axis, _check_size(size))
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=max(3, int(vertices)),
                          radius1=0.5, radius2=0.5, depth=1.0)
    bmesh.ops.transform(bm, matrix=_matrix(_AXIS_PERMUTATION[axis], (a, b, c)), verts=bm.verts)
    return _commit(bm, mesh)


def create_sphere(mesh: bpy.types.Mesh, size: Sequence[float],
                  segments: int = DEFAULT_SPHERE_SEGMENTS) -> bpy.types.Mesh:
    """Ellipsoid with the given extents."""
    sx, sy, sz = _check_size(size)
    segments = max(4, int(segments))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=max(3, segments // 2), radius=0.5)
    bmesh.ops.transform(bm, matrix=_matrix(np.eye(3), (sx, sy, sz)), verts=bm.verts)
    return _commit(bm, mesh)


def create_capsule(mesh: bpy.types.Mesh, size: Sequence[float], axis: str = "Z",
                   segments: int = DEFAULT_CAPSULE_SEGMENTS) -> bpy.types.Mesh:
    """Capsule (cylinder with hemispherical caps) along ``axis``; cap radius = half the smaller cross extent."""
    a, b, c = _axis_extents(axis, _check_size(size))
    cap = min(a, b)
    length = max(c, cap)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=max(4, int(segments)), v_segments=CAPSULE_RING_SEGMENTS, radius=0.5)
    bmesh.ops.transform(bm, matrix=_matrix(np.eye(3), (a, b, cap)), verts=bm.verts)
    shift = (length - cap) / 2.0
    for v in bm.verts:
        v.co.z += shift if v.co.z > 0 else -shift
    bmesh.ops.transform(bm, matrix=_matrix(_AXIS_PERMUTATION[axis], (1.0, 1.0, 1.0)), verts=bm.verts)
    return _commit(bm, mesh)


def create_frame(mesh: bpy.types.Mesh, size: Sequence[float], thickness: float, axis: str = "") -> bpy.types.Mesh:
    """Rectangular tube frame: four bars around an open window; the window axis is the thinnest extent.

    Bars touch at their ends (not boolean-merged), which is fine for rendering and silhouettes.
    """
    dims = list(_check_size(size))
    ax = AXES.index(axis) if axis in AXES else int(np.argmin(dims))
    i, j = [k for k in range(3) if k != ax]  # in-plane axes
    t = min(float(thickness), MAX_BEVEL_FRACTION * min(dims[i], dims[j]))
    bars = [  # (extent_i, extent_j, centre_i, centre_j)
        (dims[i], t, 0.0, (dims[j] - t) / 2.0), (dims[i], t, 0.0, -(dims[j] - t) / 2.0),
        (t, dims[j] - 2 * t, (dims[i] - t) / 2.0, 0.0), (t, dims[j] - 2 * t, -(dims[i] - t) / 2.0, 0.0)]
    bm = bmesh.new()
    for ei, ej, ci, cj in bars:
        ext, ctr = [0.0] * 3, [0.0] * 3
        ext[i], ext[j], ext[ax] = ei, ej, dims[ax]
        ctr[i], ctr[j] = ci, cj
        geom = bmesh.ops.create_cube(bm, size=1.0)
        m = _matrix(np.eye(3), ext)
        m.translation = ctr
        bmesh.ops.transform(bm, matrix=m, verts=geom["verts"])
    return _commit(bm, mesh)
