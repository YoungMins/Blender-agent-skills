"""UV-space rasterizer: triangles + per-corner UVs -> per-texel position / normal / coverage (numpy only).

Row index = V * height (bottom-up, like ``bpy.data.images`` pixels); texel centres sit at ``(x + 0.5, y + 0.5)``.
Rasterization is vectorised over (triangle, texel-in-bbox) pairs in chunks, so 20k-100k triangles at 2048^2
take seconds. Overlapping UV triangles: the last one written wins (a packed unwrap has none).
"""
from __future__ import annotations

import numpy as np

from .. import get_logger
from .islands import vertex_components

log = get_logger(__name__)

EDGE_EPS = 1e-7        # barycentric tolerance: texels exactly on a shared edge belong to a triangle
DEGENERATE_AREA = 1e-12
UV_JOIN_EPS = 1e-6      # UV corners closer than this are the same point (island connectivity)
PAIR_CHUNK = 2_000_000  # (triangle, texel) pairs processed at once


def barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates ``(N, 3)`` of points ``p (N,2)`` in triangles ``a, b, c (N,2)``."""
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    degenerate = np.abs(denom) < DEGENERATE_AREA
    denom = np.where(degenerate, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    bary = np.stack([1.0 - v - w, v, w], axis=1)
    bary[degenerate] = -1.0  # a zero-area triangle covers nothing
    return bary


def pair_chunks(tri_xy: np.ndarray, size: tuple[int, int]):
    """Yield ``(tri, px, py)`` arrays of every (triangle, texel-in-bbox) pair, in chunks of ~``PAIR_CHUNK``.

    ``tri_xy (T,3,2)`` in texel coordinates; ``size`` = (width, height); bboxes are clipped to the image.
    """
    w, h = size
    lo = np.floor(tri_xy.min(axis=1) - 0.5).astype(np.int64)
    hi = np.ceil(tri_xy.max(axis=1) - 0.5).astype(np.int64)
    x0, y0 = np.clip(lo[:, 0], 0, w - 1), np.clip(lo[:, 1], 0, h - 1)
    x1, y1 = np.clip(hi[:, 0], 0, w - 1), np.clip(hi[:, 1], 0, h - 1)
    bw, bh = np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0)
    counts = bw * bh
    ids = np.flatnonzero(counts > 0)
    cum = np.cumsum(counts[ids])
    start = 0
    while start < len(ids):
        base = cum[start - 1] if start else 0
        end = max(int(np.searchsorted(cum, base + PAIR_CHUNK, side="right")), start + 1)
        sel = ids[start:end]
        cnt = counts[sel]
        tri = np.repeat(sel, cnt)
        first = np.cumsum(cnt) - cnt
        local = np.arange(int(cnt.sum())) - np.repeat(first, cnt)
        yield tri, x0[tri] + local % bw[tri], y0[tri] + local // bw[tri]
        start = end


def rasterize_triangles(tri_xy: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Triangle id per texel (``-1`` = empty) for triangles given in texel coordinates ``(T, 3, 2)``.

    ``size`` = (width, height). A texel is covered when its centre lies inside (or on the edge of) a triangle.
    """
    w, h = size
    tri_xy = np.asarray(tri_xy, dtype=float)
    out = np.full((h, w), -1, dtype=np.int32)
    if len(tri_xy) == 0:
        return out
    for tri, px, py in pair_chunks(tri_xy, (w, h)):
        t = tri_xy[tri]
        bary = barycentric(np.stack([px + 0.5, py + 0.5], axis=1), t[:, 0], t[:, 1], t[:, 2])
        inside = (bary >= -EDGE_EPS).all(axis=1)
        out[py[inside], px[inside]] = tri[inside]
    return out


def uv_islands(uvs: np.ndarray) -> np.ndarray:
    """Island id per triangle: triangles that share a UV corner position (``uvs (T,3,2)``) are one island."""
    uvs = np.asarray(uvs, dtype=float)
    if len(uvs) == 0:
        return np.zeros(0, dtype=np.int64)
    _, corner = np.unique(np.round(uvs.reshape(-1, 2) / UV_JOIN_EPS).astype(np.int64), axis=0, return_inverse=True)
    corner = corner.reshape(-1, 3)
    edges = np.concatenate([corner[:, [0, 1]], corner[:, [1, 2]]])
    return vertex_components(int(corner.max()) + 1, edges)[corner[:, 0]]


def rasterize_uv_mesh(verts: np.ndarray, tris: np.ndarray, uvs: np.ndarray, normals: np.ndarray,
                      size: int | tuple[int, int]) -> dict[str, np.ndarray]:
    """Per-texel surface samples of a triangle mesh.

    ``verts (N,3)``, ``tris (T,3)`` vertex indices, ``uvs (T,3,2)`` per-corner UVs in [0,1], ``normals (N,3)``
    per-vertex normals (or ``(T,3,3)`` per-corner normals: a flat-shaded face keeps its own normal all over). Returns ``position (H,W,3)``, ``normal (H,W,3)`` (unit), ``mask (H,W)`` coverage,
    ``tri_index (H,W)`` (-1 empty), ``island (H,W)`` UV island id (-1 empty) and ``bary (H,W,3)``.
    """
    w, h = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    tri_xy = np.asarray(uvs, dtype=float) * np.array([w, h], dtype=float)
    tri_index = rasterize_triangles(tri_xy, (w, h))
    mask = tri_index >= 0
    ys, xs = np.nonzero(mask)
    t = tri_index[ys, xs]
    bary = barycentric(np.stack([xs + 0.5, ys + 0.5], axis=1), tri_xy[t, 0], tri_xy[t, 1], tri_xy[t, 2])
    corner = tris[t]
    pos = np.einsum("nk,nkj->nj", bary, verts[corner])
    normals = np.asarray(normals, dtype=float)
    nor = np.einsum("nk,nkj->nj", bary, normals[t] if normals.ndim == 3 else normals[corner])
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    position = np.zeros((h, w, 3))
    normal = np.zeros((h, w, 3))
    bary_img = np.zeros((h, w, 3))
    position[ys, xs], normal[ys, xs], bary_img[ys, xs] = pos, nor, bary
    island = np.full((h, w), -1, dtype=np.int64)
    island[ys, xs] = uv_islands(uvs)[t]
    log.info("uv raster %dx%d: %d triangles, %d texels covered (%.1f%%)", w, h, len(tris), len(ys),
             100.0 * len(ys) / max(w * h, 1))
    return {"position": position, "normal": normal, "mask": mask, "tri_index": tri_index, "island": island, "bary": bary_img}
