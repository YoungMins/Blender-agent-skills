"""F-score / Chamfer / OBJ i/o on identical and offset meshes."""
import numpy as np
import pytest

from bbr.core import geomscore as G
from test_attribution import box


def test_identical_meshes_score_perfectly():
    v, t = box(0, 0, 0.5, 0.4, 0.3, 1.0)
    s = G.mesh_scores(v, t, v, t, n=20000)
    assert s["chamfer"] < 0.015 and s["fscore@1%"] > 0.9 and s["fscore@2%"] > 0.99
    assert s["scale_m"] == pytest.approx(1.0) and np.allclose(s["bbox_center_offset_m"], 0)


def test_offset_mesh_matches_known_distance():
    v, t = box(0, 0, 0.5, 0.4, 0.3, 1.0)
    shifted = v + np.array([0.0, 0.0, 0.015])                # 1.5 % of the height
    s = G.mesh_scores(shifted, t, v, t, n=20000)
    assert s["fscore@2%"] > 0.95 and s["fscore@1%"] < s["fscore@2%"] - 0.05
    far = G.mesh_scores(v + np.array([0.0, 0.0, 0.2]), t, v, t, n=20000)
    assert far["fscore@2%"] < 0.8 and far["fscore@2%"] < s["fscore@2%"] and far["chamfer"] > 0.03
    assert far["bbox_center_offset_m"][2] == pytest.approx(0.2)


def test_nearest_distances_and_sampling():
    a = np.array([[0.0, 0, 0], [1, 0, 0]])
    b = np.array([[0.0, 0, 0.5], [3, 0, 0]])
    assert G.nearest_distances(a, b, chunk=1) == pytest.approx([0.5, np.hypot(1, 0.5)])
    v, t = box(0, 0, 0, 1, 1, 1)
    p, n = G.sample_surface(v, t, 2000, np.random.default_rng(0))
    assert np.all(np.abs(p).max(axis=1) <= 0.5 + 1e-9) and np.allclose(np.linalg.norm(n, axis=1), 1)
    assert np.all((p * n).sum(axis=1) > 0.49)                 # normals point outward
    assert G.sample_surface(v, t[:0], 10)[0].shape == (0, 3)


def test_obj_roundtrip(tmp_path):
    v, t = box(0, 0, 0.5, 0.4, 0.3, 1.0)
    G.write_obj(tmp_path / "m.obj", v, t)
    v2, t2 = G.read_obj(tmp_path / "m.obj")
    assert np.allclose(v, v2, atol=1e-6) and (t == t2).all()
