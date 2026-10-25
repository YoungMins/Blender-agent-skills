"""Shrink-free smoothing and normals of quad meshes (numpy only)."""
from __future__ import annotations

import numpy as np

from .. import get_logger

log = get_logger(__name__)

TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53  # slightly stronger than -lambda so volume is preserved (Taubin 1995)
_EPS = 1e-12


def quad_edges(quads: np.ndarray) -> np.ndarray:
    """Directed edge list ``(E, 2)`` of a quad mesh (each undirected edge appears in both directions)."""
    q = np.asarray(quads)
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    return np.concatenate([e, e[:, ::-1]])


def _neighbour_mean_delta(x: np.ndarray, edges: np.ndarray, degree: np.ndarray) -> np.ndarray:
    n = len(x)
    total = np.stack([np.bincount(edges[:, 0], weights=x[edges[:, 1], k], minlength=n) for k in range(3)], axis=1)
    return total / degree[:, None] - x


def taubin_smooth(verts: np.ndarray, quads: np.ndarray, iterations: int,
                  lam: float = TAUBIN_LAMBDA, mu: float = TAUBIN_MU) -> np.ndarray:
    """Taubin lambda|mu smoothing: each iteration = shrink pass (lam) + inflate pass (mu)."""
    x = np.asarray(verts, dtype=float).copy()
    if iterations <= 0 or len(quads) == 0:
        return x
    edges = quad_edges(quads)
    degree = np.maximum(np.bincount(edges[:, 0], minlength=len(x)), 1).astype(float)
    for _ in range(iterations):
        x += lam * _neighbour_mean_delta(x, edges, degree)
        x += mu * _neighbour_mean_delta(x, edges, degree)
    log.info("taubin smoothing: %d iterations on %d vertices", iterations, len(x))
    return x


def vertex_normals(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals of a quad mesh (outward for CCW winding)."""
    v = np.asarray(verts, dtype=float)
    q = np.asarray(quads)
    face_n = np.cross(v[q[:, 2]] - v[q[:, 0]], v[q[:, 3]] - v[q[:, 1]])
    out = np.zeros_like(v)
    for corner in range(4):
        for k in range(3):
            out[:, k] += np.bincount(q[:, corner], weights=face_n[:, k], minlength=len(v))
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(length, _EPS)


def taubin_smooth_edges(verts: np.ndarray, edges: np.ndarray, iterations: int,
                        lam: float = TAUBIN_LAMBDA, mu: float = TAUBIN_MU) -> np.ndarray:
    """Taubin smoothing over an undirected edge list ``(E, 2)`` (any polygon mesh)."""
    x = np.asarray(verts, dtype=float).copy()
    if iterations <= 0 or len(edges) == 0:
        return x
    e = np.asarray(edges, dtype=np.int64)
    both = np.concatenate([e, e[:, ::-1]])
    degree = np.maximum(np.bincount(both[:, 0], minlength=len(x)), 1).astype(float)
    for _ in range(iterations):
        x += lam * _neighbour_mean_delta(x, both, degree)
        x += mu * _neighbour_mean_delta(x, both, degree)
    return x


def triangle_vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals of a triangle mesh (outward for CCW winding)."""
    v = np.asarray(verts, dtype=float)
    t = np.asarray(tris)
    face_n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    out = np.zeros_like(v)
    for corner in range(3):
        for k in range(3):
            out[:, k] += np.bincount(t[:, corner], weights=face_n[:, k], minlength=len(v))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), _EPS)
