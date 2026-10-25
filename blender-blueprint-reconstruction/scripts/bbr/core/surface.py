"""Naive Surface Nets: smooth quad surface of a scalar field's iso-contour (numpy only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .imageops import label_components

_CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])           # corner c = 4i+2j+k
_EDGES = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                   if int(np.abs(_CORNERS[a] - _CORNERS[b]).sum()) == 1])                  # 12 cube edges


def surface_nets(field: np.ndarray, origin: Sequence[float] = (0.0, 0.0, 0.0), cell: float = 1.0,
                 iso: float = 0.0, min_island_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``field == iso`` (inside = ``field < iso``) as ``(verts (N,3) world, quads (M,4) int)``.

    ``field[i, j, k]`` is the value at world ``origin + (i, j, k) * cell``. One vertex per sign-changing cell
    (mean of the linearly interpolated edge crossings), one quad per sign-changing grid edge joining the four
    cells around it, wound counter-clockwise seen from outside. The field is padded with an outside layer so the
    surface is always closed. Of the 26-connected groups of cells only those with at least ``min_island_frac`` of the
    largest group's cell count are kept (default 1.0 = the largest only; the hull passes less so a silhouette that
    really is disconnected, a staff or a hat floating over a skin-coloured head, keeps its parts).
    """
    f = np.asarray(field, dtype=np.float32) - np.float32(iso)
    pad_val = max(float(f.max()), 1e-6) + 1.0
    f = np.pad(f, 1, constant_values=pad_val)
    origin = np.asarray(origin, dtype=float) - cell
    inside = f < 0
    cs = tuple(n - 1 for n in f.shape)
    corner = [inside[i:i + cs[0], j:j + cs[1], k:k + cs[2]] for i, j, k in _CORNERS]
    n_in = sum(c.astype(np.int8) for c in corner)
    active = (n_in > 0) & (n_in < 8)
    cells = np.argwhere(active)
    if len(cells) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    vals = np.stack([f[cells[:, 0] + i, cells[:, 1] + j, cells[:, 2] + k] for i, j, k in _CORNERS], axis=1)
    a, b = vals[:, _EDGES[:, 0]], vals[:, _EDGES[:, 1]]
    cross = (a < 0) != (b < 0)
    t = np.where(cross, a / np.where(cross, a - b, 1.0), 0.0)                               # crossing fraction a->b
    pa, pb = _CORNERS[_EDGES[:, 0]].astype(np.float32), _CORNERS[_EDGES[:, 1]].astype(np.float32)
    pts = pa[None] + t[..., None] * (pb - pa)[None]                                         # [M, 12, 3]
    cnt = np.maximum(cross.sum(axis=1), 1)[:, None]
    local = (pts * cross[..., None]).sum(axis=1) / cnt
    verts = origin + (cells + local) * cell

    vid = np.full(active.shape, -1, dtype=np.int64)
    vid[tuple(cells.T)] = np.arange(len(cells))
    quads = []
    for axis in range(3):
        b1, b2 = (axis + 1) % 3, (axis + 2) % 3
        sl_a, sl_b = [slice(None)] * 3, [slice(None)] * 3
        sl_a[axis], sl_b[axis] = slice(None, -1), slice(1, None)
        ia, ib = inside[tuple(sl_a)], inside[tuple(sl_b)]
        edges = np.argwhere(ia != ib)                                                       # padded node index of the edge start
        if len(edges) == 0:
            continue
        flip = ~ia[tuple(edges.T)]                                                    # outside -> inside along +axis
        ring = []
        for d1, d2 in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = edges.copy()
            c[:, b1] += d1
            c[:, b2] += d2
            ring.append(vid[tuple(c.T)])
        q = np.stack(ring, axis=1)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    quads = quads[(quads >= 0).all(axis=1)]
    return _largest_component(verts, quads, cells, min_island_frac)


def _largest_component(verts: np.ndarray, quads: np.ndarray, cells: np.ndarray, min_frac: float = 1.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Drop the groups of 26-connected surface cells smaller than ``min_frac`` x the largest (always keeps the largest) and renumber vertices."""
    if len(verts) == 0 or len(quads) == 0:
        return verts, quads
    grid = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    grid[tuple(cells.T)] = True
    labels, n = label_components(grid, full=True)
    if n > 1:
        lab = labels[tuple(cells.T)]
        sizes = np.bincount(lab, minlength=n + 1)
        sizes[0] = 0
        keep = sizes[lab] >= min(min_frac, 1.0) * sizes.max()
        quads = quads[keep[quads].all(axis=1)]
    used, inverse = np.unique(quads, return_inverse=True)
    return verts[used], inverse.reshape(-1, 4).astype(np.int64)
