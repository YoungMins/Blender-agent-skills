"""Seat a hard-surface part on a curved target surface (numpy only): a window or door on a round body, a plaque on a column.

The agent names a spot on the TARGET (the target's own mesh is ray-cast, so a lathe, cylinder or body of any shape works):

* ``{"view": "front", "px": [x, y]}``: the reference pixel, cast along that view's camera axis (the first surface the camera
  sees through that pixel, as :func:`bbr.core.triangulate.pixel_to_plane` places it), or
* ``{"azimuth_deg": a, "height": z}``: cast horizontally at height ``z`` toward the target's vertical axis from azimuth ``a``
  (0 = the front, -Y; +90 = +X, image-right of the front view; 180 = the back).

The hit gives the surface point and its outward normal (barycentric blend of angle-limited vertex normals, so a faceted
round body gives its true curved normal, not the facet's). The part's local ``face`` axis (``-Y`` default: its front, ``+Y``)
turns onto that normal with local Z kept upright, and its back face rests ``standoff`` meters off the surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import BBRError
from . import partgeom
from .calibration import ViewCalibration
from .conventions import AXES
from .triangulate import pixel_to_plane

EPS = 1e-12
SMOOTH_ANGLE_DEG = 40.0       # faces meeting the hit face at less than this blend into its vertex normals
RAY_CHUNK = 200_000
FACES = {"-Y": np.array([0.0, -1.0, 0.0]), "+Y": np.array([0.0, 1.0, 0.0])}


@dataclass
class Placement:
    """Where a part goes: ``location`` / ``rotation_deg`` (Blender XYZ Euler), the surface ``point`` and outward ``normal``."""

    location: np.ndarray
    rotation_deg: list[float]
    point: np.ndarray
    normal: np.ndarray
    back_offset: float
    standoff: float

    def report(self) -> dict[str, Any]:
        """JSON-able summary for ``surface_attach.json``."""
        r = lambda v: [round(float(c), 5) for c in v]  # noqa: E731
        return {"point": r(self.point), "normal": r(self.normal), "location": r(self.location),
                "rotation_deg": r(self.rotation_deg), "back_offset": round(self.back_offset, 5), "standoff": self.standoff}


def raycast(verts: np.ndarray, tris: np.ndarray, origin: Sequence[float], direction: Sequence[float]
            ) -> Optional[tuple[float, int, np.ndarray]]:
    """Nearest hit of the ray (Moeller-Trumbore, both sides): ``(distance, triangle index, barycentric (3,))`` or None."""
    o, d = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    best: Optional[tuple[float, int, np.ndarray]] = None
    for lo in range(0, len(tris), RAY_CHUNK):
        t = verts[tris[lo:lo + RAY_CHUNK]]
        e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        h = np.cross(d, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > EPS
        inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = o - t[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ d)
        dist = inv * np.einsum("ij,ij->i", e2, q)
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (dist > 0)
        if hit.any():
            k = int(np.argmin(np.where(hit, dist, np.inf)))
            if best is None or dist[k] < best[0]:
                best = (float(dist[k]), lo + k, np.array([1.0 - u[k] - v[k], u[k], v[k]]))
    return best


def smooth_normal(verts: np.ndarray, tris: np.ndarray, tri: int, bary: np.ndarray, toward: Sequence[float],
                  angle_deg: float = SMOOTH_ANGLE_DEG) -> np.ndarray:
    """Outward unit normal at barycentric point ``bary`` of triangle ``tri``: its corners' normals from the incident faces within
    ``angle_deg`` of this face (a rim shared with a flat cap does not tilt a curved wall), blended; flipped to face ``toward``
    (the direction to the viewer / ray origin) so a mesh with inward winding still gives the outside."""
    t = verts[tris]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])                  # area-weighted face normals
    unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), EPS)
    n0 = unit[tri]
    cos_lim = math.cos(math.radians(angle_deg))
    corners = []
    for vertex in tris[tri]:
        near = (tris == vertex).any(axis=1) & (unit @ n0 >= cos_lim)
        corners.append(fn[near].sum(axis=0))
    n = np.asarray(bary) @ np.stack(corners)
    n = n / max(float(np.linalg.norm(n)), EPS) if np.linalg.norm(n) > EPS else n0
    return -n if float(n @ np.asarray(toward)) < 0 else n


def facing_rotation(normal: Sequence[float], face: str = "-Y") -> np.ndarray:
    """Rotation matrix (columns = the part's local X, Y, Z in world) taking the part's ``face`` axis (``-Y`` or ``+Y``) onto
    ``normal`` with local Z as upright as possible (world Z projected off the normal; world Y when the normal is vertical).
    A part facing the front (normal -Y) with ``face == "-Y"``, or the back with ``+Y``, gets the identity."""
    if face not in FACES:
        raise BBRError(f"face must be '-Y' or '+Y' (the part's local axis that points out of the surface), got {face!r}")
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    y = n if face == "+Y" else -n
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    z = up - float(up @ y) * y
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)


def _ray_for(at: Mapping[str, Any], verts: np.ndarray, calibs: Mapping[str, ViewCalibration]
             ) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, direction)`` of the ray named by ``at`` (see the module doc)."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    far = 2.0 * float(np.linalg.norm(hi - lo)) + 1.0
    if "view" in at and "px" in at:
        view = str(at["view"])
        if view not in calibs:
            raise BBRError(f"attach_at.view {view!r} is not a calibrated view; views: {sorted(calibs)}")
        if len(at["px"]) != 2:
            raise BBRError(f"attach_at.px must be [x, y] pixels, got {at['px']!r}")
        cam = AXES[calibs[view].camera_axis]
        point = np.nan_to_num(pixel_to_plane(calibs[view], at["px"]))      # the depth coordinate is the one the ray finds
        return point + cam * far, -cam
    if "azimuth_deg" in at and "height" in at:
        a = math.radians(float(at["azimuth_deg"]))
        out = np.array([math.sin(a), -math.cos(a), 0.0])
        centre = (lo + hi) / 2.0
        return np.array([centre[0], centre[1], float(at["height"])]) + out * far, -out
    raise BBRError("attach_at needs {\"view\": V, \"px\": [x, y]} or {\"azimuth_deg\": a, \"height\": z}, "
                   f"got keys {sorted(at)}")


def place_part(verts: np.ndarray, tris: np.ndarray, at: Mapping[str, Any], calibs: Mapping[str, ViewCalibration],
               part_verts: np.ndarray, face: str = "-Y", standoff: float = 0.0) -> Placement:
    """Pose of a part on the surface of the mesh ``(verts, tris)`` (world) at ``at``; ``part_verts`` = the part's own local
    vertices (its back face = the extreme along ``-face``). Raises BBRError when the ray misses the target."""
    verts, tris = np.asarray(verts, dtype=float), np.asarray(tris, dtype=np.int64)
    if len(tris) == 0:
        raise BBRError("the attach target has no geometry")
    origin, direction = _ray_for(at, verts, calibs)
    hit = raycast(verts, tris, origin, direction)
    if hit is None:
        raise BBRError(f"attach_at {dict(at)} misses the target surface: check the pixel / azimuth / height")
    dist, tri, bary = hit
    point = origin + direction * dist
    normal = smooth_normal(verts, tris, tri, bary, -direction)
    rot = facing_rotation(normal, face)
    back = -float((np.asarray(part_verts, dtype=float) @ FACES[face]).min())     # origin to the back plane, along the outward axis
    return Placement(point + normal * (float(standoff) + back), partgeom.euler_xyz_deg(rot), point, normal, back, float(standoff))
