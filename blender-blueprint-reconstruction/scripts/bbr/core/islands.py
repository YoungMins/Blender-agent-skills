"""Connected components of a mesh and small-island selection (numpy only)."""
from __future__ import annotations

import numpy as np

from .. import get_logger

log = get_logger(__name__)


def vertex_components(n_verts: int, edges: np.ndarray) -> np.ndarray:
    """Component label per vertex (labels are the smallest vertex index of the component) from ``edges (E,2)``."""
    label = np.arange(n_verts)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return label
    while True:
        low = np.minimum(label[e[:, 0]], label[e[:, 1]])
        new = label.copy()
        np.minimum.at(new, e[:, 0], low)
        np.minimum.at(new, e[:, 1], low)
        new = new[new]  # pointer jumping: converges in O(log diameter) rounds
        if np.array_equal(new, label):
            return label
        label = new


def small_island_vertices(n_verts: int, edges: np.ndarray, vertex_weight: np.ndarray, min_ratio: float) -> np.ndarray:
    """Bool mask of vertices in islands whose summed ``vertex_weight`` (e.g. area) is below ``min_ratio`` of the total.

    The heaviest island is always kept, so a mesh never becomes empty.
    """
    label = vertex_components(n_verts, edges)
    weight = np.bincount(label, weights=np.asarray(vertex_weight, dtype=float), minlength=n_verts)
    total = float(weight.sum())
    if total <= 0:
        return np.zeros(n_verts, dtype=bool)
    drop = (weight < min_ratio * total) & (weight > 0)
    drop[int(np.argmax(weight))] = False
    return drop[label]
