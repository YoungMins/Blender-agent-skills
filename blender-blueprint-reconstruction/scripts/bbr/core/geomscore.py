"""Mesh sampling and mesh-vs-mesh scores (numpy only): surface samples, Chamfer distance, F-score, OBJ files.

Used by the benchmark (``bench/score.py``) to compare a reconstruction with a ground-truth mesh, and by the
attribution metrics to sample texel-like points on the model surface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_SAMPLES = 40000
DEFAULT_TAUS = (0.01, 0.02)   # F-score thresholds as fractions of the height
NN_CHUNK = 1024


def triangle_areas(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area of every triangle."""
    v = np.asarray(verts, dtype=float)[np.asarray(tris, dtype=np.int64)]
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh (positive when the normals point outward)."""
    if len(tris) == 0:
        return 0.0
    v = np.asarray(verts, dtype=float)[np.asarray(tris, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def sample_surface(verts: np.ndarray, tris: np.ndarray, n: int, rng: Optional[np.random.Generator] = None,
                   orient_outward: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``n`` area-uniform points on a triangle mesh and their unit face normals, ``(n,3)`` each.

    ``orient_outward``: the normals of a mesh with negative signed volume (inverted winding) are flipped so they
    point outward. Empty or zero-area meshes give empty arrays.
    """
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    if len(tris) == 0 or n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    rng = rng or np.random.default_rng(0)
    v = verts[tris]
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    total = norm.sum()
    if total <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    idx = rng.choice(len(tris), size=n, p=norm / total)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)
    pts = np.einsum("nk,nkc->nc", w, v[idx])
    nrm = cross[idx] / np.maximum(norm[idx], 1e-30)[:, None]
    if orient_outward and signed_volume(verts, tris) < 0:
        nrm = -nrm
    return pts, nrm


def nearest_distances(a: np.ndarray, b: np.ndarray, chunk: int = NN_CHUNK) -> np.ndarray:
    """Distance from every point of ``a`` to its nearest point of ``b`` (brute force, in chunks)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    centre = b.mean(axis=0)                                   # centred coordinates keep |a|^2 + |b|^2 - 2ab accurate
    a, b = a - centre, b - centre
    b2 = (b * b).sum(axis=1)
    out = np.empty(len(a))
    for lo in range(0, len(a), chunk):
        blk = a[lo:lo + chunk]
        d2 = (blk * blk).sum(axis=1)[:, None] + b2[None, :] - 2.0 * blk @ b.T
        out[lo:lo + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def chamfer_fscore(pts_a: np.ndarray, pts_b: np.ndarray, scale: float,
                   taus: Sequence[float] = DEFAULT_TAUS) -> dict[str, Any]:
    """Distances between point sets ``a`` (reconstruction) and ``b`` (ground truth), normalised by ``scale``.

    ``chamfer`` = mean of both directed mean distances / ``scale``. For every tau (a fraction of ``scale``):
    ``precision`` = share of ``a`` within tau of ``b``, ``recall`` = share of ``b`` within tau of ``a``,
    ``fscore`` = their harmonic mean (keys ``fscore@1%``...).
    """
    d_ab = nearest_distances(pts_a, pts_b)
    d_ba = nearest_distances(pts_b, pts_a)
    out: dict[str, Any] = {"chamfer": float((d_ab.mean() + d_ba.mean()) / 2.0 / scale),
                           "accuracy": float(d_ab.mean() / scale), "completeness": float(d_ba.mean() / scale)}
    for tau in taus:
        t = tau * scale
        p, r = float((d_ab <= t).mean()), float((d_ba <= t).mean())
        label = f"{tau * 100:g}%"
        out[f"precision@{label}"], out[f"recall@{label}"] = p, r
        out[f"fscore@{label}"] = 0.0 if p + r == 0 else 2.0 * p * r / (p + r)
    return out


def mesh_scores(verts_a: np.ndarray, tris_a: np.ndarray, verts_b: np.ndarray, tris_b: np.ndarray,
                scale: Optional[float] = None, n: int = DEFAULT_SAMPLES, seed: int = 0,
                taus: Sequence[float] = DEFAULT_TAUS) -> dict[str, Any]:
    """Chamfer + F-score of mesh ``a`` (reconstruction) against mesh ``b`` (ground truth).

    ``scale`` defaults to the height (Z extent) of ``b``. Also reports the bbox-centre offset (m) of ``a`` relative to ``b``.
    """
    rng = np.random.default_rng(seed)
    pa, _ = sample_surface(verts_a, tris_a, n, rng, orient_outward=False)
    pb, _ = sample_surface(verts_b, tris_b, n, rng, orient_outward=False)
    lo_b, hi_b = np.asarray(verts_b).min(axis=0), np.asarray(verts_b).max(axis=0)
    height = float(hi_b[2] - lo_b[2]) if scale is None else float(scale)
    out = chamfer_fscore(pa, pb, height, taus)
    centre_a = (np.asarray(verts_a).min(axis=0) + np.asarray(verts_a).max(axis=0)) / 2.0
    out["scale_m"] = height
    out["bbox_center_offset_m"] = [float(x) for x in centre_a - (lo_b + hi_b) / 2.0]
    return out


def write_obj(path: Path | str, verts: np.ndarray, tris: np.ndarray) -> None:
    """Write a triangle mesh as Wavefront OBJ."""
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(verts, dtype=float)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(tris, dtype=np.int64)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_obj(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``v`` / ``f`` lines of an OBJ (polygons are fan-triangulated; ``a/b/c`` and negative indices work)."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            tris.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.asarray(verts, dtype=float).reshape(-1, 3), np.asarray(tris, dtype=np.int64).reshape(-1, 3)
