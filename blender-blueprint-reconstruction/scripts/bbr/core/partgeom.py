"""Thin-part geometry (numpy only): bent tapered cones (horns), tapered blades, contour plates (wings, fins), domed / elliptical discs (shields, brims), gems.

Every function returns ``(verts (N,3) float, faces list[tuple[int, ...]])`` in the part's LOCAL frame at true size
(meters), origin at the centre of the part's base / mid-thickness; mesh orientation is chosen by the caller through
``axis`` vectors, so the Blender side only copies the arrays into a mesh. Faces are wound counter-clockwise seen
from outside; the meshes are closed (welded); shade smooth with a sharp-edge angle (see ``mesh.set_smooth_by_angle``).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .. import BBRError

MIN_SIDES = 3
MIN_SEGMENTS = 1
DEFAULT_SIDES = 16
DEFAULT_SEGMENTS = 12
DEFAULT_DISC_SEGMENTS = 48
DEFAULT_DOME_RINGS = 8
DEFAULT_GEM_FACETS = 6
EPS = 1e-9

Faces = list[tuple[int, ...]]


def _unit(v: Sequence[float], what: str) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or n < EPS:
        raise BBRError(f"{what} must be a non-zero 3-vector, got {list(v)}")
    return a / n


def orthonormal_frame(axis: Sequence[float], ref: Optional[Sequence[float]] = None) -> np.ndarray:
    """Right-handed rows ``(u, v, w)`` with ``w`` = unit ``axis`` and ``u`` = ``ref`` (default: the world axis least
    aligned with ``axis``) made perpendicular to it."""
    w = _unit(axis, "axis")
    if ref is None:
        ref = np.eye(3)[int(np.argmin(np.abs(w)))]
    u = np.asarray(ref, dtype=float)
    u = u - np.dot(u, w) * w
    if np.linalg.norm(u) < EPS:
        raise BBRError("reference direction is parallel to the axis")
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(w, u), w])


def _check_sides(sides: int) -> int:
    return max(MIN_SIDES, int(sides))


# --------------------------------------------------------------------------- cone / horn
def cone_path(length: float, axis: Sequence[float], bend_deg: float, bend_dir: Optional[Sequence[float]],
              segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centre line of a bent cone: ``(points (S+1,3), tangents (S+1,3), normal_in_plane (S+1,3), binormal (3,))``.

    The line is a circular arc of arc length ``length`` starting along ``axis`` and turning by ``bend_deg`` degrees in
    total toward ``bend_dir`` (its component perpendicular to ``axis``); ``bend_deg == 0`` gives a straight line.
    """
    if length <= 0:
        raise BBRError(f"length must be > 0, got {length}")
    a = _unit(axis, "axis")
    if abs(bend_deg) > EPS:
        if bend_dir is None:
            raise BBRError("bend needs a bend_dir (direction to curve toward)")
        b = np.asarray(bend_dir, dtype=float)
        b = b - np.dot(b, a) * a
        if np.linalg.norm(b) < EPS:
            raise BBRError("bend_dir must not be parallel to the axis")
        b /= np.linalg.norm(b)
    else:
        b = orthonormal_frame(a)[0]
    s = np.linspace(0.0, length, max(MIN_SEGMENTS, int(segments)) + 1)
    k = math.radians(bend_deg) / length
    if abs(k) > EPS:
        pts = a[None] * (np.sin(k * s) / k)[:, None] + b[None] * ((1.0 - np.cos(k * s)) / k)[:, None]
    else:
        pts = a[None] * s[:, None]
    tang = a[None] * np.cos(k * s)[:, None] + b[None] * np.sin(k * s)[:, None]
    nrm = -a[None] * np.sin(k * s)[:, None] + b[None] * np.cos(k * s)[:, None]
    return pts, tang, nrm, np.cross(a, b)


def cone_mesh(base_radius: float, tip_radius: float, length: float, axis: Sequence[float] = (0, 0, 1),
              bend_deg: float = 0.0, bend_dir: Optional[Sequence[float]] = None, segments: int = DEFAULT_SEGMENTS,
              sides: int = DEFAULT_SIDES, flatten: float = 1.0, weld: bool = True) -> tuple[np.ndarray, Faces]:
    """Tapered tube (horn): radius ``base_radius`` -> ``tip_radius`` over ``length`` along a bent centre line.

    ``tip_radius == 0`` closes the tip to a point. Base (and a non-zero tip) get flat caps.
    ``flatten`` scales the cross-section perpendicular to the bend plane (elliptical horn: < 1 is thinner seen from the
    side). The base centre is the local origin. ``weld=False`` skips merging coincident vertices (faster; enough for
    silhouettes).
    """
    if flatten <= 0:
        raise BBRError(f"flatten must be > 0, got {flatten}")
    if base_radius <= 0 or tip_radius < 0:
        raise BBRError(f"radii must satisfy base_radius > 0 and tip_radius >= 0, got {base_radius}, {tip_radius}")
    sides = _check_sides(sides)
    pts, tang, nrm, binorm = cone_path(length, axis, bend_deg, bend_dir, segments)
    rings = len(pts)
    radius = np.linspace(base_radius, tip_radius, rings)
    phi = np.arange(sides) * (2.0 * np.pi / sides)
    cos, sin = np.cos(phi), np.sin(phi)
    # ring point = centre + r * (cos * binormal + sin * in-plane normal)
    ring = (pts[:, None, :] + radius[:, None, None]
            * (flatten * cos[None, :, None] * binorm[None, None, :] + sin[None, :, None] * nrm[:, None, :]))
    closed_tip = tip_radius <= 0
    body_rings = rings - 1 if closed_tip else rings
    verts = [ring[:body_rings].reshape(-1, 3)]
    faces: Faces = []
    for r in range(body_rings - 1):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append((r * sides + i, r * sides + j, (r + 1) * sides + j, (r + 1) * sides + i))
    n = body_rings * sides
    if closed_tip:
        apex = n
        verts.append(pts[-1][None])
        n += 1
        last = (body_rings - 1) * sides
        for i in range(sides):
            faces.append((last + i, last + (i + 1) % sides, apex))
    for r, top in [(0, False)] + ([] if closed_tip else [(rings - 1, True)]):
        centre = n
        verts.append(pts[r][None])
        verts.append(ring[r])
        for i in range(sides):
            a, b = n + 1 + i, n + 1 + (i + 1) % sides
            faces.append((centre, a, b) if top else (centre, b, a))
        n += 1 + sides
    faces = [tuple(reversed(f)) for f in faces]                  # rings run clockwise seen from outside
    return weld_vertices(np.concatenate(verts), faces) if weld else (np.concatenate(verts), faces)


# --------------------------------------------------------------------------- strand
MIN_STRAND_POINTS = 3
DEFAULT_STRAND_SPAN_SEGMENTS = 8


def _catmull_rom(values: np.ndarray, per_span: int) -> np.ndarray:
    """Uniform Catmull-Rom curve through the rows of ``values`` (N,d), ``per_span`` samples per span, end points included."""
    v = np.asarray(values, dtype=float)
    ext = np.concatenate([2.0 * v[:1] - v[1:2], v, 2.0 * v[-1:] - v[-2:-1]])
    t = np.linspace(0.0, 1.0, per_span, endpoint=False)[:, None]
    out = []
    for i in range(len(v) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
                          + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t ** 3))
    out.append(v[-1:])
    return np.concatenate(out)


def strand_mesh(points: Sequence[Sequence[float]], radii: Sequence[float], flatten: float = 1.0,
                segments: int = DEFAULT_STRAND_SPAN_SEGMENTS, sides: int = DEFAULT_SIDES, weld: bool = True
                ) -> tuple[np.ndarray, Faces]:
    """Tube through ``N >= 3`` ordered ``points`` (Catmull-Rom spline, ``segments`` rings per span), radius ``radii[i]`` at
    point i (a single value or base / tip is interpolated; the taper is the per-point radius). ``flatten`` < 1 thins the
    cross-section along the depth axis: the strand is as wide as its radius seen from the front (local X) and ``flatten`` x
    that deep. A zero end radius closes that end to a point, otherwise it gets a flat cap. The first point is the origin
    of the local frame when ``points[0]`` is (0, 0, 0). ``weld=False`` skips merging coincident vertices and the winding
    check (faster; enough for silhouettes)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < MIN_STRAND_POINTS:
        raise BBRError(f"a strand needs at least {MIN_STRAND_POINTS} points, got {len(pts)}")
    if flatten <= 0:
        raise BBRError(f"flatten must be > 0, got {flatten}")
    rad = np.interp(np.linspace(0.0, 1.0, len(pts)), np.linspace(0.0, 1.0, len(np.atleast_1d(radii))), np.atleast_1d(radii).astype(float))
    if np.any(rad < 0) or rad.max() <= 0:
        raise BBRError(f"radii must be >= 0 with at least one > 0, got {list(radii)}")
    sides = _check_sides(sides)
    seg = max(MIN_SEGMENTS, int(segments))
    path = _catmull_rom(pts, seg)
    r = np.maximum(_catmull_rom(rad[:, None], seg)[:, 0], 0.0)
    tang = np.gradient(path, axis=0)
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(norm, EPS)
    wx = np.array([1.0, 0.0, 0.0]) - tang[:, :1] * tang                 # world X (else Y) made perpendicular to the tangent
    wy = np.array([0.0, 1.0, 0.0]) - tang[:, 1:2] * tang
    use_x = np.linalg.norm(wx, axis=1, keepdims=True) > 0.2
    width = np.where(use_x, wx, wy)
    width = width / np.maximum(np.linalg.norm(width, axis=1, keepdims=True), EPS)
    depth = np.cross(tang, width)
    phi = np.arange(sides) * (2.0 * np.pi / sides)
    cos, sin = np.cos(phi), np.sin(phi)
    ring = path[:, None, :] + r[:, None, None] * (cos[None, :, None] * width[:, None, :] + flatten * sin[None, :, None] * depth[:, None, :])
    rings = len(path)
    closed_tip = r[-1] <= 1e-9
    body = rings - 1 if closed_tip else rings
    verts = [ring[:body].reshape(-1, 3)]
    faces: Faces = []
    for k in range(body - 1):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append((k * sides + i, k * sides + j, (k + 1) * sides + j, (k + 1) * sides + i))
    n = body * sides
    if closed_tip:
        verts.append(path[-1][None])
        last = (body - 1) * sides
        faces += [(last + i, last + (i + 1) % sides, n) for i in range(sides)]
        n += 1
    for k, top in [(0, False)] + ([] if closed_tip else [(rings - 1, True)]):
        verts.append(path[k][None])
        verts.append(ring[k])
        faces += [((n, n + 1 + i, n + 1 + (i + 1) % sides) if top else (n, n + 1 + (i + 1) % sides, n + 1 + i)) for i in range(sides)]
        n += 1 + sides
    if not weld:
        return np.concatenate(verts), faces
    v, f = weld_vertices(np.concatenate(verts), faces)
    if mesh_volume(v, f) < 0:                        # wind outward whatever the ring orientation of the frame
        f = [tuple(reversed(x)) for x in f]
    return v, f


# --------------------------------------------------------------------------- blade
def blade_mesh(length: float, width: float, thickness: float, tip_width: Optional[float] = None,
               tip_thickness: Optional[float] = None, tip_length: float = 0.0, axis: Sequence[float] = (0, 0, 1),
               width_dir: Optional[Sequence[float]] = None) -> tuple[np.ndarray, Faces]:
    """Tapered box along ``axis`` with an optional pointed tip.

    Cross-section ``width x thickness`` at the base (origin = centre of the base end), linearly tapering to
    ``tip_width x tip_thickness`` (default: constant) at ``length - tip_length``; with ``tip_length > 0`` the last
    section closes in a pyramid apex at ``length`` (a sword point), else the end is a flat cap. ``width_dir`` is the
    direction of the width (default: the world axis least aligned with ``axis``). Flat faces: shade flat.
    """
    if length <= 0 or width <= 0 or thickness <= 0:
        raise BBRError(f"length, width and thickness must be > 0, got {length}, {width}, {thickness}")
    tip_width = width if tip_width is None else float(tip_width)
    tip_thickness = thickness if tip_thickness is None else float(tip_thickness)
    if tip_width <= 0 or tip_thickness <= 0:
        raise BBRError("tip_width and tip_thickness must be > 0")
    if not 0.0 <= tip_length < length:
        raise BBRError(f"tip_length must be in [0, length), got {tip_length}")
    u, v, w = orthonormal_frame(axis, width_dir)
    z_end = length - tip_length
    stations = [(0.0, width, thickness), (z_end, tip_width, tip_thickness)]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * 0.5   # ccw seen from +w
    verts = []
    for z, wd, th in stations:
        for cx, cy in corners:
            verts.append(cx * wd * u + cy * th * v + z * w)
    faces: Faces = [(3, 2, 1, 0)]                                        # base cap (normal -w)
    for i in range(4):
        j = (i + 1) % 4
        faces.append((i, j, 4 + j, 4 + i))
    if tip_length > 0:
        verts.append(length * w)
        for i in range(4):
            faces.append((4 + i, 4 + (i + 1) % 4, 8))
    else:
        faces.append((4, 5, 6, 7))
    return np.array(verts), faces


# --------------------------------------------------------------------------- plate
def _subdivide(verts: np.ndarray, tris: np.ndarray, loop: list[int], levels: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Midpoint-subdivide a triangulated polygon ``levels`` times (interior vertices for a bend); the boundary ``loop``
    (vertex indices, in order) follows. Returns ``(verts, tris, loop)``."""
    v = [tuple(p) for p in verts]
    t = [tuple(x) for x in tris]
    for _ in range(levels):
        mid: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(v)
                v.append(tuple((np.asarray(v[a]) + np.asarray(v[b])) / 2.0))
            return mid[key]

        new_t = []
        for a, b, c in t:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_t += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        loop = [x for i, a in enumerate(loop) for x in (a, midpoint(a, loop[(i + 1) % len(loop)]))]
        t = new_t
    return np.array(v, dtype=float), np.array(t, dtype=np.int64).reshape(-1, 3), loop


def plate_mesh(outline: Sequence[Sequence[float]], tris: Sequence[Sequence[int]], thickness: float,
               axis: Sequence[float], normal: Sequence[float], bend_deg: float = 0.0, subdivisions: int = 0
               ) -> tuple[np.ndarray, Faces]:
    """Flat plate of the 2D ``outline`` (``(N, 2)`` meters ``(a, b)`` along ``(u, v)``; ``tris`` = its triangulation,
    :func:`bbr.core.plate.triangulate_polygon`) extruded ``thickness`` along ``normal``: ``u`` = ``axis`` made
    perpendicular to ``normal``, ``v`` = normal x u. Origin = the plane origin (the root edge). Two cap faces from the
    triangulation plus a side wall per outline edge; closed, wound outward whatever the outline's orientation.

    ``bend_deg`` curves the plate along ``u``: an arc of that total turn over the plate's length (``max a``) toward
    ``+normal`` (the face side); ``subdivisions`` > 0 subdivides the caps first so a bend has interior vertices (a bent
    plate gets 2 by default). Shade with a sharp-edge angle."""
    if thickness <= 0:
        raise BBRError(f"thickness must be > 0, got {thickness}")
    pts = np.asarray(outline, dtype=float).reshape(-1, 2)
    tri = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(pts) < 3 or len(tri) < 1:
        raise BBRError("a plate needs an outline of at least 3 points and its triangulation")
    n = _unit(normal, "normal")
    u = np.asarray(axis, dtype=float) - float(np.dot(axis, n)) * n
    if np.linalg.norm(u) < EPS:
        raise BBRError("axis must not be parallel to the normal")
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    loop = list(range(len(pts))) if area > 0 else list(range(len(pts)))[::-1]
    d1, d2 = pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]
    flip = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) < 0
    tri = np.where(flip[:, None], tri[:, [0, 2, 1]], tri)
    if abs(bend_deg) > EPS and subdivisions <= 0:
        subdivisions = 2
    pts, tri, loop = _subdivide(pts, tri, loop, int(subdivisions))
    if abs(bend_deg) > EPS:
        length = max(float(pts[:, 0].max()), 1e-6)
        k = math.radians(bend_deg) / length
        s = pts[:, 0]
        along = np.sin(k * s) / k
        lift = (1.0 - np.cos(k * s)) / k
        centre = along[:, None] * u + pts[:, 1:2] * v + lift[:, None] * n
        top_dir = -np.sin(k * s)[:, None] * u + np.cos(k * s)[:, None] * n
    else:
        centre = pts[:, :1] * u + pts[:, 1:2] * v
        top_dir = np.broadcast_to(n, (len(pts), 3))
    h = thickness / 2.0
    top, bottom = centre + h * top_dir, centre - h * top_dir
    m = len(pts)
    verts = np.concatenate([top, bottom])
    faces: Faces = [tuple(int(i) for i in t) for t in tri]                     # top cap, normal +n
    faces += [(int(c) + m, int(b) + m, int(a) + m) for a, b, c in tri]         # bottom cap, normal -n
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        faces.append((a + m, b + m, b, a))                                      # wall: outward = edge x normal
    return verts, faces


# --------------------------------------------------------------------------- disc / shield
def disc_mesh(radius: float, thickness: float, dome_height: float = 0.0, dome_radius: Optional[float] = None,
              segments: int = DEFAULT_DISC_SEGMENTS, dome_rings: int = DEFAULT_DOME_RINGS,
              axis: Sequence[float] = (0, 0, 1), radius_ratio: float = 1.0) -> tuple[np.ndarray, Faces]:
    """Cylinder (``radius``, ``thickness``, centred on the origin, axis ``axis``) with an optional spherical-cap dome
    of height ``dome_height`` and base radius ``dome_radius`` (default: the whole face) on its +axis face.
    ``radius_ratio`` (default 1) makes it elliptical (a wide-brim hat): the extent along the second frame vector
    (:func:`orthonormal_frame`; world Y for axis Z) is ``radius * radius_ratio``.

    The strips (back face, rim, front annulus, dome) are welded into one closed mesh; shade it with a sharp-edge angle.
    """
    if radius <= 0 or thickness <= 0 or dome_height < 0:
        raise BBRError(f"radius and thickness must be > 0 and dome_height >= 0, got {radius}, {thickness}, {dome_height}")
    a = radius if dome_radius is None else float(dome_radius)
    if dome_height > 0 and not 0 < a <= radius:
        raise BBRError(f"dome_radius must be in (0, radius], got {a}")
    h = thickness / 2.0
    strips: list[list[tuple[float, float]]] = [[(0.0, -h), (radius, -h)], [(radius, -h), (radius, h)]]
    if dome_height > 0:
        if a < radius:
            strips.append([(radius, h), (a, h)])
        big_r = (a * a + dome_height ** 2) / (2.0 * dome_height)
        rs = np.linspace(a, 0.0, max(2, int(dome_rings)) + 1)
        strips.append([(float(r), h + math.sqrt(max(big_r ** 2 - r * r, 0.0)) - (big_r - dome_height)) for r in rs])
    else:
        strips.append([(radius, h), (0.0, h)])
    if radius_ratio <= 0:
        raise BBRError(f"radius_ratio must be > 0, got {radius_ratio}")
    verts, faces = _revolve(strips, int(segments), axis)
    if abs(radius_ratio - 1.0) > EPS:
        v = orthonormal_frame(axis)[1]
        verts = verts + np.outer(verts @ v, v) * (radius_ratio - 1.0)
    return verts, faces


def gem_mesh(radius: float, length: float, facets: int = 6, waist: float = 0.5,
             axis: Sequence[float] = (0, 0, 1)) -> tuple[np.ndarray, Faces]:
    """Faceted crystal: a bipyramid of ``facets`` sides (4 = octahedron) centred on the origin, ``length`` apex to apex
    along ``axis`` (the -axis apex is the ``base``), widest ring of ``radius`` at ``waist`` (0..1 of the length from the
    base apex; 0.5 = symmetric). Triangles only, to be shaded flat."""
    if radius <= 0 or length <= 0 or not 0.0 < waist < 1.0:
        raise BBRError(f"radius and length must be > 0 and waist in (0, 1), got {radius}, {length}, {waist}")
    n = max(MIN_SIDES, int(facets))
    u, v, w = orthonormal_frame(axis)
    phi = np.arange(n) * (2.0 * np.pi / n)
    zw = -length / 2.0 + waist * length
    verts = np.concatenate([(-length / 2.0 * w)[None],
                            radius * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v) + zw * w,
                            (length / 2.0 * w)[None]])
    faces: Faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((0, 1 + j, 1 + i))                # lower facets, seen from below
        faces.append((n + 1, 1 + i, 1 + j))            # upper facets
    return verts, faces


def capsule_mesh(r0: float, r1: float, length: float, segments: int = 16, rings: int = 6,
                 axis: Sequence[float] = (0, 0, 1)) -> tuple[np.ndarray, Faces]:
    """Tapered capsule: two spheres (radius ``r0`` at the -axis end, ``r1`` at the +axis end) joined by their tangent
    cone, centred on the origin, ``length`` end to end (pole to pole). ``r0 == r1`` is the plain capsule."""
    if r0 <= 0 or r1 <= 0 or length <= 0:
        raise BBRError(f"radii and length must be > 0, got {r0}, {r1}, {length}")
    z0, z1 = -length / 2.0 + r0, length / 2.0 - r1                    # sphere centres
    d = max(z1 - z0, EPS)
    alpha = math.asin(max(-0.98, min(0.98, (r0 - r1) / d)))            # tilt of the tangent cone's normal
    rings = max(2, int(rings))
    t0 = np.linspace(-math.pi / 2.0, alpha, rings + 1)
    t1 = np.linspace(alpha, math.pi / 2.0, rings + 1)
    prof = [(r0 * math.cos(t), z0 + r0 * math.sin(t)) for t in t0] + [(r1 * math.cos(t), z1 + r1 * math.sin(t)) for t in t1[1:]]
    prof[0] = (0.0, prof[0][1])
    prof[-1] = (0.0, prof[-1][1])
    return _revolve([prof], int(segments), axis)


def _revolve(strips: Sequence[Sequence[tuple[float, float]]], segments: int, axis: Sequence[float]
             ) -> tuple[np.ndarray, Faces]:
    """Revolve polylines ``(r, z)`` about the local Z axis (then Z -> ``axis``); ``r == 0`` points are single vertices."""
    segments = _check_sides(segments)
    u, v, w = orthonormal_frame(axis)
    phi = np.arange(segments) * (2.0 * np.pi / segments)
    verts: list[np.ndarray] = []
    faces: Faces = []
    n = 0
    for strip in strips:
        rows: list[list[int]] = []
        for r, z in strip:
            if r <= EPS:
                verts.append((z * w)[None])
                rows.append([n] * segments)
                n += 1
            else:
                verts.append(r * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v) + z * w)
                rows.append(list(range(n, n + segments)))
                n += segments
        for a_row, b_row in zip(rows[:-1], rows[1:]):
            for i in range(segments):
                j = (i + 1) % segments
                quad = (a_row[i], a_row[j], b_row[j], b_row[i])
                uniq = tuple(dict.fromkeys(quad))      # collapses the quad to a triangle at the axis
                if len(uniq) >= 3:
                    faces.append(uniq)
    return weld_vertices(np.concatenate(verts), faces)


def weld_vertices(verts: np.ndarray, faces: Faces, decimals: int = 9) -> tuple[np.ndarray, Faces]:
    """Merge coincident vertices (rounded to ``decimals``) and drop faces that collapse; makes the mesh watertight."""
    verts = np.asarray(verts, dtype=float)
    _, first, inverse = np.unique(np.round(verts, decimals), axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)                      # keep the original vertex order
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    remap = rank[inverse.ravel()]
    out: Faces = []
    for f in faces:
        uniq = tuple(dict.fromkeys(int(remap[i]) for i in f))
        if len(uniq) >= 3:
            out.append(uniq)
    return verts[np.sort(first)], out


def polygons_triangles(faces: Faces) -> np.ndarray:
    """Fan-triangulate polygons -> ``(T,3)`` vertex indices."""
    tris = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def mesh_volume(verts: np.ndarray, faces: Faces) -> float:
    """Signed volume of a closed polygon mesh (positive when faces wind outward)."""
    t = np.asarray(verts, dtype=float)[polygons_triangles(faces)]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


# --------------------------------------------------------------------------- hard-surface curved shapes
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
DEFAULT_ARCH_SEGMENTS = 32
DEFAULT_TORUS_MAJOR_SEGMENTS = 48
DEFAULT_TORUS_MINOR_SEGMENTS = 16


def _axis_index(axis: str, what: str = "axis") -> int:
    if str(axis).upper() not in AXIS_INDEX:
        raise BBRError(f"{what} must be one of X, Y, Z, got {axis!r}")
    return AXIS_INDEX[str(axis).upper()]


def _outward(verts: np.ndarray, faces: Faces) -> Faces:
    """``faces`` flipped when they wind inward (a mirrored axis frame flips the winding)."""
    return faces if mesh_volume(verts, faces) >= 0 else [f[::-1] for f in faces]


def arch_mesh(size: Sequence[float], axis: str = "X", bulge: Optional[str] = None, thickness: float = 0.0,
              segments: int = DEFAULT_ARCH_SEGMENTS) -> tuple[np.ndarray, Faces]:
    """Half cylinder (a chest lid, a vault) or, with ``thickness``, a half-ring band (a shell segment).

    ``size`` = full bounding extents ``[x, y, z]``, the mesh centred on the origin; ``axis`` = the long (extrusion) axis;
    ``bulge`` = the direction the dome rises: ``"+Z"`` by default (``"+X"`` for axis Z), or ``"-Y"`` ... any signed axis
    perpendicular to ``axis``. The cross-section is a half ellipse: half the ``size`` along the third axis is its radius
    there, the full ``size`` along ``bulge`` is its height (the flat base sits on the bounding box's opposite face), so
    elliptic radii come straight from ``size``. ``thickness`` > 0 hollows it into a band: a wall that thick along both
    semi-axes, open toward the base, ends closed (a solid shell segment). Closed, outward-wound, ``segments`` over the 180 degrees.
    """
    if len(size) != 3 or any(s <= 0 for s in size):
        raise BBRError(f"size must be three positive extents, got {list(size)}")
    li = _axis_index(axis)
    bulge = bulge or ("+X" if li == 2 else "+Z")
    if len(bulge) != 2 or bulge[0] not in "+-":
        raise BBRError(f"bulge must be +/-X, +/-Y or +/-Z, got {bulge!r}")
    bi, bsign = _axis_index(bulge[1], "bulge axis"), (-1.0 if bulge[0] == "-" else 1.0)
    if bi == li:
        raise BBRError(f"bulge {bulge} must be perpendicular to axis {axis}")
    ui = 3 - li - bi
    length, a, b = float(size[li]), size[ui] / 2.0, float(size[bi])
    t = float(thickness)
    if t < 0 or (t > 0 and t >= min(a, b)):
        raise BBRError(f"thickness must be in [0, min(half width {a:.4f}, height {b:.4f})), got {t}")
    m = max(2, int(segments))
    ang = np.linspace(0.0, np.pi, m + 1)
    outer = np.stack([a * np.cos(ang), b * np.sin(ang)], axis=1)
    if t > 0:
        inner = np.stack([(a - t) * np.cos(ang), (b - t) * np.sin(ang)], axis=1)
        poly = np.concatenate([outer, inner[::-1]])
    else:
        poly = outer
    n = len(poly)
    verts = np.zeros((2 * n, 3))
    for k, z in enumerate((-length / 2.0, length / 2.0)):
        block = verts[k * n:(k + 1) * n]
        block[:, ui], block[:, bi], block[:, li] = poly[:, 0], bsign * (poly[:, 1] - b / 2.0), z
    faces: Faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    if t > 0:    # band ends: quads from outer point k to its inner partner n - 1 - k (+length end first)
        faces += [(n + k, n + k + 1, 2 * n - 2 - k, 2 * n - 1 - k) for k in range(m)]
        faces += [(n - 1 - k, n - 2 - k, k + 1, k) for k in range(m)]
    else:
        faces += [tuple(range(n, 2 * n)), tuple(range(n - 1, -1, -1))]
    return verts, _outward(verts, faces)


def torus_mesh(major_radius: float | Sequence[float], minor_radius: float, axis: str = "Z",
               major_segments: int = DEFAULT_TORUS_MAJOR_SEGMENTS, minor_segments: int = DEFAULT_TORUS_MINOR_SEGMENTS
               ) -> tuple[np.ndarray, Faces]:
    """Ring (handle, hoop, washer): a tube of radius ``minor_radius`` along a circle of radius ``major_radius``, centred
    on the origin, the ring plane perpendicular to ``axis``. ``major_radius`` may be ``[r1, r2]`` for an elliptic ring
    (radii along the two other axes in ascending X, Y, Z order). ``minor_radius`` must stay below the ring's tightest
    curvature radius (``min(r)**2 / max(r)``) so the tube does not fold into itself. Closed, outward-wound quads."""
    radii = [float(major_radius)] * 2 if np.isscalar(major_radius) else [float(r) for r in major_radius]
    if len(radii) != 2 or min(radii) <= 0 or minor_radius <= 0:
        raise BBRError(f"torus needs major_radius (number or [r1, r2]) > 0 and minor_radius > 0, got {major_radius}, {minor_radius}")
    ai = _axis_index(axis)
    limit = min(radii) ** 2 / max(radii)
    if minor_radius >= limit:
        raise BBRError(f"minor_radius {minor_radius} must be below {limit:.4f} (the ring's tightest curvature radius)")
    i, j = [k for k in range(3) if k != ai]
    nu, nv = max(3, int(major_segments)), max(3, int(minor_segments))
    th = np.arange(nu) * (2.0 * np.pi / nu)
    ph = np.arange(nv) * (2.0 * np.pi / nv)
    centre = np.stack([radii[0] * np.cos(th), radii[1] * np.sin(th)], axis=1)
    norm = np.stack([np.cos(th) / radii[0], np.sin(th) / radii[1]], axis=1)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True)
    q = centre[:, None, :] + minor_radius * np.cos(ph)[None, :, None] * norm[:, None, :]       # (nu, nv, 2)
    verts = np.zeros((nu, nv, 3))
    verts[..., i], verts[..., j] = q[..., 0], q[..., 1]
    verts[..., ai] = (minor_radius * np.sin(ph))[None, :]
    faces: Faces = [(a * nv + b, ((a + 1) % nu) * nv + b, ((a + 1) % nu) * nv + (b + 1) % nv, a * nv + (b + 1) % nv)
                    for a in range(nu) for b in range(nv)]
    verts = verts.reshape(-1, 3)
    return verts, _outward(verts, faces)


DEFAULT_LATHE_SEGMENTS = 48


def lathe_mesh(profile: Sequence[Sequence[float]], segments: int = DEFAULT_LATHE_SEGMENTS, depth_ratio: float = 1.0,
               cap_top: bool = True, cap_bottom: bool = True) -> tuple[np.ndarray, Faces]:
    """Body of revolution about the local Z axis (mushroom cap, vase, lamp, bottle, column): ``profile`` = ``[[z, r], ...]``
    in meters, first point to last (either order; it is read bottom to top), ``r`` the radius along X. ``depth_ratio``
    scales the section along Y (elliptic: ``r * depth_ratio``), ``segments`` points around. A ring with ``r == 0`` is one
    apex vertex; ``cap_top`` / ``cap_bottom`` close an end that has a radius with a flat disc (False leaves it open, e.g.
    a hollow mouth). Closed and outward-wound when both ends are closed or on the axis; shade smooth with a sharp-edge angle."""
    pts = np.asarray(profile, dtype=float).reshape(-1, 2) if len(profile) else np.zeros((0, 2))
    if len(pts) < 2 or (pts[:, 1] < 0).any() or not np.isfinite(pts).all() or (pts[:, 1] <= EPS).all():
        raise BBRError(f"lathe profile needs at least two [z, r] points with r >= 0 and some r > 0, got {pts.tolist()}")
    if depth_ratio <= 0:
        raise BBRError(f"depth_ratio must be > 0, got {depth_ratio}")
    if pts[-1, 0] < pts[0, 0]:
        pts = pts[::-1]
    segments = _check_sides(segments)
    phi = np.arange(segments) * (2.0 * np.pi / segments)
    cos, sin = np.cos(phi), depth_ratio * np.sin(phi)
    chunks = [np.stack([r * cos, r * sin, np.full(segments, z)], axis=1) for z, r in pts]
    faces: Faces = []
    for k in range(len(pts) - 1):
        a, b = k * segments, (k + 1) * segments
        faces += [(a + i, a + (i + 1) % segments, b + (i + 1) % segments, b + i) for i in range(segments)]
    n = len(pts) * segments
    for k, top, wanted in ((0, False, cap_bottom), (len(pts) - 1, True, cap_top)):
        if wanted and pts[k, 1] > EPS:
            chunks.append(np.array([[0.0, 0.0, pts[k, 0]]]))
            ring = k * segments
            faces += [(n, ring + i, ring + (i + 1) % segments) if top else (n, ring + (i + 1) % segments, ring + i)
                      for i in range(segments)]
            n += 1
    return weld_vertices(np.concatenate(chunks), faces)


def rounded_box_mesh(size: Sequence[float], radius: Sequence[float], segments: int = 3) -> tuple[np.ndarray, Faces]:
    """Box with a corner radius per axis: ``radius[k]`` = how far the rounding reaches along axis ``k`` (the edge parallel
    to Z gets an elliptic profile of ``radius[0]`` x ``radius[1]``; 0 keeps that direction sharp). Each radius is clamped
    below half the extent; ``segments`` steps per rounded zone. Closed, welded, outward-wound."""
    if len(size) != 3 or any(s <= 0 for s in size) or len(radius) != 3 or any(r < 0 for r in radius):
        raise BBRError(f"size must be 3 positive extents and radius 3 non-negative values, got {list(size)}, {list(radius)}")
    half = np.asarray(size, dtype=float) / 2.0
    r = np.minimum(np.asarray(radius, dtype=float), 0.49 * np.asarray(size, dtype=float))
    h = half - r                                                     # half extent of the flat core
    seg = max(1, int(segments))
    nodes = []
    for k in range(3):
        if r[k] > EPS:
            u = np.linspace(0.0, 1.0, seg + 1)
            nodes.append(np.concatenate([-(h[k] + r[k] * u[::-1]), h[k] + r[k] * u]))
        else:
            nodes.append(np.array([-half[k], half[k]]))
    chunks: list[np.ndarray] = []
    faces: Faces = []
    base = 0
    for axis in range(3):
        i, j = [k for k in range(3) if k != axis]
        for sign in (-1.0, 1.0):
            gi, gj = np.meshgrid(nodes[i], nodes[j], indexing="ij")
            p = np.zeros(gi.shape + (3,))
            p[..., i], p[..., j], p[..., axis] = gi, gj, sign * half[axis]
            c = np.clip(p, -h, h)
            v = np.divide(p - c, r, out=np.zeros_like(p), where=r > EPS)
            nv = np.linalg.norm(v, axis=-1, keepdims=True)
            inf = np.abs(v).max(axis=-1, keepdims=True)                # square -> disc: v * |v|_inf / |v|_2 (flat strips stay put)
            q = c + r * v * np.divide(inf, nv, out=np.zeros_like(nv), where=nv > EPS)
            ni, nj = gi.shape
            chunks.append(q.reshape(-1, 3))
            idx = np.arange(ni * nj).reshape(ni, nj) + base
            quads = [(int(idx[a, b]), int(idx[a + 1, b]), int(idx[a + 1, b + 1]), int(idx[a, b + 1]))
                     for a in range(ni - 1) for b in range(nj - 1)]
            ccw = sign * (-1.0 if axis == 1 else 1.0) > 0           # (i, j) = ascending other axes: i x j = +axis except for Y
            faces += quads if ccw else [q[::-1] for q in quads]
            base += ni * nj
    verts, faces = weld_vertices(np.concatenate(chunks), faces)
    return verts, _outward(verts, faces)


# --------------------------------------------------------------------------- placement helpers
def arc_through_points(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float]
                       ) -> tuple[float, np.ndarray, float, Optional[np.ndarray]]:
    """Circular arc from ``p0`` through ``p1`` to ``p2`` -> ``(length, axis, bend_deg, bend_dir)`` for :func:`cone_mesh`.

    ``axis`` is the tangent at ``p0``, ``bend_deg`` the total turn (may exceed 180 for a hook), ``bend_dir`` the unit
    direction from ``p0`` toward the circle centre. Collinear points give the straight line (``bend_dir`` None).
    """
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    a, b = p1 - p0, p2 - p0
    chord = float(np.linalg.norm(b))
    if chord < EPS:
        raise BBRError("arc end points coincide")
    n = np.cross(a, b)
    nn = float(np.linalg.norm(n))
    if nn < 1e-3 * float(np.linalg.norm(a)) * chord:                 # (nearly) straight
        return chord, b / chord, 0.0, None
    e1 = a / np.linalg.norm(a)
    e3 = n / nn
    e2 = np.cross(e3, e1)
    ax, bx, by = float(a @ e1), float(b @ e1), float(b @ e2)
    x0 = ax / 2.0
    y0 = (bx * bx + by * by - bx * ax) / (2.0 * by)
    radius = math.hypot(x0, y0)
    centre = p0 + x0 * e1 + y0 * e2
    f1 = (p0 - centre) / radius
    f2 = np.cross(e3, f1)

    def angle(p: np.ndarray) -> float:
        v = p - centre
        return math.atan2(float(v @ f2), float(v @ f1)) % (2.0 * math.pi)

    a1, a2 = angle(p1), angle(p2)
    if a1 < a2:
        theta, direction = a2, 1.0                                   # counter-clockwise about e3
    else:
        theta, direction = 2.0 * math.pi - a2, -1.0
    return radius * theta, direction * f2, math.degrees(theta), -f1


def euler_xyz_deg(rot: np.ndarray) -> list[float]:
    """Blender XYZ-Euler degrees ``[x, y, z]`` of a rotation matrix (inverse of ``R = Rz @ Ry @ Rx``)."""
    r = np.asarray(rot, dtype=float)
    y = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(math.cos(y)) > 1e-6:
        x, z = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:                                                            # gimbal lock
        x, z = math.atan2(-r[1, 2], r[1, 1]), 0.0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def rotation_z_to(axis: Sequence[float]) -> np.ndarray:
    """Rotation matrix taking the local +Z axis to the unit vector ``axis``."""
    return orthonormal_frame(axis).T


# --------------------------------------------------------------------------- glove (a hand with fingers)
GLOVE_FINGER_FRAC = 0.34            # share of the hand length the fingers take (cartoon glove: short)
GLOVE_CURL_DEG = 60.0               # how far the fingers bend toward the palm (relaxed, half closed)
GLOVE_THICKNESS = 0.7               # palm thickness over its width
GLOVE_FINGER_RADIUS = 0.56          # finger radius over the width per finger (> 0.5: neighbours touch, a chunky glove)
GLOVE_FINGER_LENGTHS = {1: (1.0,), 2: (1.0, 0.92), 3: (0.95, 1.0, 0.9), 4: (0.9, 1.0, 0.96, 0.82)}


def glove_mesh(length: float, width: float, fingers: int = 4, finger_frac: float = GLOVE_FINGER_FRAC,
               curl_deg: float = GLOVE_CURL_DEG, thickness: Optional[float] = None, thumb: bool = True,
               sides: int = 10) -> tuple[np.ndarray, Faces]:
    """A hand in the frame of a capsule of ``length`` x ``width`` (local +Z from the wrist at ``-length / 2`` to the
    fingertips at ``+length / 2``): a rounded-box palm, ``fingers`` tubes (1-4) along +Z bent ``curl_deg`` toward the palm
    side (+Y) and a thumb on the -X side pointing out, up and toward the palm. Each piece is closed; they overlap (one
    object, no boolean). The palm faces +Y and the thumb is on -X: mirror X for the other hand."""
    if length <= 0 or width <= 0:
        raise BBRError(f"glove length and width must be > 0, got {length}, {width}")
    n = int(fingers)
    if n not in GLOVE_FINGER_LENGTHS:
        raise BBRError(f"fingers must be 1-4, got {fingers}")
    t = float(thickness if thickness is not None else GLOVE_THICKNESS * width)
    z0 = -length / 2.0
    finger_len = finger_frac * length
    palm_len = length - finger_len
    parts = [rounded_box_mesh([width, t, palm_len], [0.3 * width, 0.45 * t, 0.3 * palm_len])]
    parts[0] = (parts[0][0] + np.array([0.0, 0.0, z0 + palm_len / 2.0]), parts[0][1])
    rf = GLOVE_FINGER_RADIUS * width / n
    curl = math.radians(curl_deg)
    for i, scale in enumerate(GLOVE_FINGER_LENGTHS[n]):
        x = -width / 2.0 + width / (2 * n) + i * width / n
        lf = finger_len * scale + rf
        p0 = np.array([x, 0.0, z0 + palm_len - rf])
        d1 = np.array([0.0, math.sin(curl / 2.0), math.cos(curl / 2.0)])
        d2 = np.array([0.0, math.sin(curl), math.cos(curl)])
        pts = [p0, p0 + 0.4 * lf * d1, p0 + 0.4 * lf * d1 + 0.6 * lf * d2]
        parts.append(strand_mesh(pts, [rf, 0.95 * rf, 0.8 * rf], sides=sides))
    if thumb:
        rt = 1.35 * rf
        base = np.array([-0.32 * width, 0.15 * t, z0 + 0.35 * palm_len])
        d = np.array([-0.55, 0.45, 0.7])
        d /= np.linalg.norm(d)
        lt = 0.5 * finger_len + 0.35 * width
        pts = [base, base + 0.5 * lt * d, base + lt * (d + np.array([0.15, 0.1, 0.0]))]
        parts.append(strand_mesh(pts, [rt, rt, 0.85 * rt], sides=sides))
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(np.asarray(v, dtype=float))
        faces += [[int(k) + offset for k in poly] for poly in f]
        offset += len(v)
    return np.concatenate(verts), faces
