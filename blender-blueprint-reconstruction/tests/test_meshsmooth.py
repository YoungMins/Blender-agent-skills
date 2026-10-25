"""Tests for core.meshsmooth."""
import numpy as np

from bbr.core.meshsmooth import taubin_smooth, vertex_normals


def _cube_sphere(n=16, noise=0.0):
    """Quad mesh of a unit sphere (cube projected onto it, outward winding), radii jittered by ``noise``."""
    t = np.linspace(-1.0, 1.0, n + 1)
    a, b = np.meshgrid(t, t, indexing="ij")
    pts, quads = [], []
    for axis in range(3):
        for sign in (1, -1):
            p = np.zeros(a.shape + (3,))
            p[..., axis], p[..., (axis + 1) % 3], p[..., (axis + 2) % 3] = sign, a, b
            idx = np.arange(a.size).reshape(a.shape) + sum(len(q) for q in pts)
            q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
            quads.append(q if sign > 0 else q[:, ::-1])
            pts.append(p.reshape(-1, 3))
    pts, quads = np.concatenate(pts), np.concatenate(quads)
    uniq, inv = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    uniq = uniq / np.linalg.norm(uniq, axis=1, keepdims=True)
    uniq *= 1.0 + noise * np.random.default_rng(0).standard_normal((len(uniq), 1))
    return uniq, inv.reshape(-1)[quads]


def test_taubin_keeps_volume_extent_and_reduces_roughness():
    verts, quads = _cube_sphere(noise=0.02)
    smooth = taubin_smooth(verts, quads, iterations=10)
    c = verts.mean(axis=0)
    r0 = np.linalg.norm(verts - c, axis=1)
    r1 = np.linalg.norm(smooth - c, axis=1)
    assert r1.std() < r0.std()
    assert abs(r1.mean() / r0.mean() - 1.0) < 0.03


def test_vertex_normals_point_outward():
    verts, quads = _cube_sphere()
    n = vertex_normals(verts, quads)
    radial = verts - verts.mean(axis=0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert float((n * radial).sum(axis=1).mean()) > 0.8
