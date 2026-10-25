"""Contour plates (numpy only): wings, fins, ear fins, leaves, flat accessories.

A plate is the part's label mask in the view where it is seen face-on (``face_view``), turned into a simplified polygon
(:func:`outline_from_mask`: boundary trace -> Douglas-Peucker), triangulated (:func:`triangulate_polygon`, ear clipping)
and lifted into 3D on a plane (:func:`local_outline`): every outline pixel is moved along the face view's camera axis
onto the plate plane, so the plate looks exactly like its label from the face-on view whatever its tilt, and the OTHER
views (edge-on or oblique) constrain the plane's pose. The mesh itself is :func:`bbr.core.partgeom.plate_mesh`.

Plane pose: origin = the ``base`` keypoint (root edge), in-plane axis ``u`` = base -> tip, normal ``n`` = the face view's
camera direction made perpendicular to ``u`` (:func:`plate_normal`), or, when an edge-on view gives the plane's line,
the normal of that line (:func:`normal_from_lines`).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .. import BBRError
from .calibration import ViewCalibration
from .conventions import AXES, normalize_axis
from .imageops import label_components

DEFAULT_SIMPLIFY_PX = 1.5      # Douglas-Peucker tolerance of the outline polygon (view pixels)
MAX_OUTLINE_VERTICES = 64      # the tolerance grows until the polygon has at most this many vertices
MIN_FACING = 0.2               # |n . camera| below this is treated as 0.2 when lifting the outline (a plate seen edge-on)
EPS = 1e-9


# --------------------------------------------------------------------------- mask -> polygon
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    labels, n = label_components(~mask, full=False)
    if n == 0:
        return mask
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    return mask | (~mask & ~outside)


def trace_outline(mask: np.ndarray) -> np.ndarray:
    """Outer boundary of the largest 8-connected blob of ``mask`` (holes filled) as a closed polygon ``(N, 2)`` of pixel
    corner coordinates ``[x, y]`` (the staircase along the pixel edges, counter-clockwise on screen with y down)."""
    m = np.asarray(mask, dtype=bool)
    labels, n = label_components(m, full=True)
    if n == 0:
        raise BBRError("an empty label mask has no outline")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    m = _fill_holes(labels == int(sizes.argmax()))
    p = np.pad(m, 1)
    h, w = m.shape
    ys, xs = np.nonzero(m)
    nxt: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for dy, dx, a, b in ((-1, 0, (0, 0), (1, 0)), (0, 1, (1, 0), (1, 1)), (1, 0, (1, 1), (0, 1)), (0, -1, (0, 1), (0, 0))):
        edge = ~p[ys + 1 + dy, xs + 1 + dx]                      # the neighbour in this direction is outside
        for x, y in zip(xs[edge], ys[edge]):
            nxt.setdefault((int(x) + a[0], int(y) + a[1]), []).append((int(x) + b[0], int(y) + b[1]))
    best: list[tuple[int, int]] = []
    best_area = 0.0
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for start, outs in nxt.items():
        for first in outs:
            if (start, first) in seen:
                continue
            loop = [start]
            prev, cur = start, first
            seen.add((start, first))
            while cur != start:
                loop.append(cur)
                cands = nxt[cur]
                if len(cands) > 1:                                # a pinch point: keep turning the same way (right-hand rule)
                    d_in = (cur[0] - prev[0], cur[1] - prev[1])
                    cands = sorted(cands, key=lambda c: _turn(d_in, (c[0] - cur[0], c[1] - cur[1])))
                nxt_pt = cands[0]
                seen.add((cur, nxt_pt))
                prev, cur = cur, nxt_pt
                if len(loop) > 4 * (h + 2) * (w + 2):
                    raise BBRError("outline trace did not close")
            area = abs(_signed_area(np.array(loop, dtype=float)))
            if area > best_area:
                best, best_area = loop, area
    if len(best) < 3:
        raise BBRError("the label mask has no usable outline")
    poly = np.array(best, dtype=float)
    return poly if _signed_area(poly) > 0 else poly[::-1].copy()


def _turn(d_in: tuple[int, int], d_out: tuple[int, int]) -> float:
    """Sort key: the sharpest turn toward one side first (cross of the directions, then straight on)."""
    return float(d_in[0] * d_out[1] - d_in[1] * d_out[0]) * -1.0


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _dp(points: np.ndarray, eps: float) -> list[int]:
    """Indices kept by Douglas-Peucker on the open polyline ``points`` (end points always kept)."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        d = points[i + 1:j] - a
        ab = b - a
        norm = float(np.hypot(*ab))
        dist = np.hypot(d[:, 0], d[:, 1]) if norm < EPS else np.abs(d[:, 0] * ab[1] - d[:, 1] * ab[0]) / norm
        k = int(np.argmax(dist))
        if float(dist[k]) > eps:
            keep.add(i + 1 + k)
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


def simplify_closed(poly: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker on a closed polygon: split at the two mutually farthest vertices, simplify both halves."""
    pts = np.asarray(poly, dtype=float)
    if len(pts) <= 3:
        return pts
    a = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    b = int(np.argmax(np.hypot(*(pts - pts[a]).T)))
    i, j = sorted((a, b))
    first = pts[i:j + 1]
    second = np.concatenate([pts[j:], pts[:i + 1]])
    out = [pts[i:j + 1][k] for k in _dp(first, eps)[:-1]] + [second[k] for k in _dp(second, eps)[:-1]]
    return np.array(out, dtype=float)


def outline_from_mask(mask: np.ndarray, eps_px: float = DEFAULT_SIMPLIFY_PX, max_vertices: int = MAX_OUTLINE_VERTICES
                      ) -> np.ndarray:
    """Simplified outline polygon (``(N, 2)`` pixel ``[x, y]``, N >= 3) of a label mask; ``eps_px`` grows until it has at
    most ``max_vertices`` vertices."""
    poly = trace_outline(mask)
    eps = float(eps_px)
    for _ in range(12):
        simple = simplify_closed(poly, eps)
        if len(simple) <= max_vertices and len(simple) >= 3:
            return simple
        eps *= 1.4
    return simplify_closed(poly, eps)


# --------------------------------------------------------------------------- triangulation
def _in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    return not ((d1 < -EPS or d2 < -EPS or d3 < -EPS) and (d1 > EPS or d2 > EPS or d3 > EPS))


def triangulate_polygon(poly: np.ndarray) -> np.ndarray:
    """Ear clipping of a simple polygon ``(N, 2)`` -> ``(N - 2, 3)`` vertex indices, every triangle counter-clockwise in
    the polygon's own axes (y up) whatever the polygon's orientation. Degenerate (zero-area) ears are dropped."""
    pts = np.asarray(poly, dtype=float)
    n = len(pts)
    if n < 3:
        raise BBRError("a polygon needs at least 3 vertices")
    order = list(range(n)) if _signed_area(pts) > 0 else list(range(n))[::-1]
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3 and guard < 4 * n * n:
        guard += 1
        clipped = False
        for k in range(len(order)):
            i0, i1, i2 = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= EPS:                                       # reflex or flat
                continue
            if any(_in_triangle(pts[j], a, b, c) for j in order if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            order.pop(k)
            clipped = True
            break
        if not clipped:                                            # numerically stuck (touching edges): drop the flattest vertex
            k = int(np.argmin([abs((pts[order[m]][0] - pts[order[m - 1]][0]) * (pts[order[(m + 1) % len(order)]][1] - pts[order[m - 1]][1])
                                   - (pts[order[m]][1] - pts[order[m - 1]][1]) * (pts[order[(m + 1) % len(order)]][0] - pts[order[m - 1]][0]))
                               for m in range(len(order))]))
            order.pop(k)
    if len(order) == 3:
        a, b, c = (pts[i] for i in order)
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > EPS:
            tris.append(tuple(order))                              # type: ignore[arg-type]
    if not tris:
        raise BBRError("the outline polygon could not be triangulated (self-intersecting or zero area)")
    return np.array(tris, dtype=np.int64)


# --------------------------------------------------------------------------- plane pose
def camera_dir(calib: ViewCalibration) -> np.ndarray:
    """Unit vector from the origin toward the view's camera."""
    return np.asarray(AXES[normalize_axis(calib.camera_axis)], dtype=float)


def unit(v: Sequence[float], what: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < EPS:
        raise BBRError(f"{what} is zero")
    return a / n


def plate_normal(axis: np.ndarray, face_calib: ViewCalibration) -> np.ndarray:
    """Plate normal: the face view's camera direction minus its component along the plate's in-plane axis ``axis``
    (unit): the plane contains ``axis`` and faces the face view as much as it can."""
    c = camera_dir(face_calib)
    n = c - float(c @ axis) * axis
    if float(np.linalg.norm(n)) < 0.05:                       # the axis points at the camera: any perpendicular will do
        n = np.cross(axis, np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    return unit(n, "plate normal")


def normal_from_lines(lines: Mapping[str, Sequence[Sequence[float]]], calibs: Mapping[str, ViewCalibration],
                      toward: np.ndarray) -> np.ndarray:
    """Plate normal from lines drawn along the plate in edge-on views (``{view: [[x0, y0], [x1, y1]]}`` pixels): the plane
    contains the viewing direction and the line, so ``n`` = camera x line. The mean over views; signed toward ``toward``
    (the face view's camera direction)."""
    ns = []
    for view, (p0, p1) in lines.items():
        if view not in calibs:
            raise BBRError(f"edge_on_views: view {view!r} is not calibrated; views: {sorted(calibs)}")
        calib = calibs[view]
        t = calib.px_to_plane(np.array([p1], float))[0] - calib.px_to_plane(np.array([p0], float))[0]
        if float(np.linalg.norm(t)) < EPS:
            raise BBRError(f"edge_on_views.{view}: the two points coincide")
        n = np.cross(camera_dir(calib), t)
        n = unit(n, "edge-on normal")
        ns.append(n if float(n @ toward) >= 0 else -n)
    if not ns:
        raise BBRError("edge_on_views is empty")
    return unit(np.mean(ns, axis=0), "edge-on normal")


def in_plane_axis(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """``axis`` made perpendicular to ``normal`` (unit)."""
    u = np.asarray(axis, float) - float(np.asarray(axis, float) @ normal) * normal
    if float(np.linalg.norm(u)) < 0.05:
        u = np.cross(normal, np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    return unit(u, "in-plane axis")


def local_outline(outline_px: np.ndarray, face_calib: ViewCalibration, origin: np.ndarray, normal: np.ndarray,
                  axis: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Outline in the plate's own plane: ``(a, b)`` meters along ``(u, v)`` from ``origin`` (``u`` = ``axis`` made
    perpendicular to ``normal``, ``v`` = normal x u), times ``scale``, and ``u``. Each outline pixel is moved along the
    face view's camera axis onto the plane (``|n . camera|`` is clamped to :data:`MIN_FACING`)."""
    n = unit(normal, "normal")
    u = in_plane_axis(axis, n)
    v = np.cross(n, u)
    c = camera_dir(face_calib)
    q = face_calib.px_to_plane(np.asarray(outline_px, dtype=float))
    facing = float(n @ c)
    if abs(facing) < MIN_FACING:
        facing = math.copysign(MIN_FACING, facing if facing != 0 else 1.0)
    t = ((np.asarray(origin, float) - q) @ n) / facing
    p = q + t[:, None] * c[None, :] - np.asarray(origin, float)
    return float(scale) * np.stack([p @ u, p @ v], axis=1), u


def outline_half_width(outline: np.ndarray) -> float:
    """Half the extent of a local outline across its ``v`` axis."""
    b = np.asarray(outline, float)[:, 1]
    return float(b.max() - b.min()) / 2.0
