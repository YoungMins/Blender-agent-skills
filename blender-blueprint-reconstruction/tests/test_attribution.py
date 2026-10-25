"""Misattribution and camera-facing occlusion (pure numpy: software depth buffers, no Blender)."""
import numpy as np
import pytest

from bbr.core import attribution as A
from bbr.core.calibration import ViewCalibration
from bbr.core.geomscore import sample_surface, signed_volume
from bbr.core.zbuffer import depth_buffer

M_PER_PX = 0.01
SIZE = (200, 200)


def calib(view, axis):
    return ViewCalibration(view, axis, SIZE, M_PER_PX, (100.0, 150.0))


def box(cx, cy, cz, sx, sy, sz):
    """Closed box with outward winding: (verts, tris)."""
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([[a, b, c] for c in (-z, z) for b in (-y, y) for a in (-x, x)], float) + (cx, cy, cz)
    t = []
    for a, b, c, d in [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]:
        t += [(a, b, c), (a, c, d)]
    tris = np.array(t)
    return (v, tris) if signed_volume(v, tris) > 0 else (v, tris[:, ::-1])


def scene(parts, views):
    """Merge meshes, sample 4000 points each; return points, normals, owner and the views' depth buffers."""
    verts, tris, offset = [], [], 0
    pts, nrm, owner = [], [], []
    rng = np.random.default_rng(1)
    for k, (v, t) in enumerate(parts):
        verts.append(v)
        tris.append(t + offset)
        offset += len(v)
        p, n = sample_surface(v, t, 4000, rng)
        pts.append(p), nrm.append(n), owner.append(np.full(len(p), k))
    v, t = np.concatenate(verts), np.concatenate(tris)
    return (np.concatenate(pts), np.concatenate(nrm), np.concatenate(owner),
            [depth_buffer(v, t, c) for c in views])


def test_embedded_box_is_occluded_and_flush_box_is_not():
    views = [calib("front", "-Y"), calib("back", "+Y")]
    body = box(0, 0, 0.5, 1.0, 0.5, 1.0)
    embedded = box(0.0, 0.0, 0.5, 0.3, 0.2, 0.3)             # entirely inside the body
    flush = box(0.0, 0.3, 0.5, 0.3, 0.1, 0.3)                # sits on the back face (y 0.25..0.35)
    pts, nrm, owner, zb = scene([body, embedded, flush], views)
    res = A.occlusion_by_object(pts, nrm, owner, 3, views, zb)
    assert res[0]["occlusion_share"] < 0.05                  # the body sees itself
    assert res[1]["occlusion_share"] > 0.95 and res[1]["hidden_share"] > 0.95
    assert res[1]["occlusion_share"] > A.OCCLUSION_WARN
    # the flush part hides its inner face from the front camera (pooled share ~0.5) but the back camera sees it fully
    assert res[2]["occlusion_share"] < 0.05 and res[2]["best_view"] == "back"
    assert res[2]["pooled_share"] == pytest.approx(0.5, abs=0.15)
    assert res[2]["occlusion_share"] <= A.OCCLUSION_WARN


def test_inverted_winding_is_oriented_outward():
    views = [calib("front", "-Y")]
    v, t = box(0, 0, 0.5, 1.0, 0.5, 1.0)
    pts, nrm, owner, zb = scene([(v, t[:, ::-1])], views)       # inside-out mesh
    res = A.occlusion_by_object(pts, nrm, owner, 1, views, zb)
    assert res[0]["per_view"]["front"]["facing"] > 0 and res[0]["occlusion_share"] < 0.05


def test_misattribution_two_objects():
    cal = calib("front", "-Y")
    ids = np.zeros((200, 200), np.uint8)
    ids[:, :100] = 1                                          # object 0 owns the left half of the image
    ids[:, 100:] = 2                                          # object 1 the right half
    # object 0 samples: world x -0.3..0.3 -> pixels 70..130; the right half of them lands on object 1's pixels
    x = np.linspace(-0.3, 0.299, 1000)
    pts = np.stack([x, np.zeros_like(x), np.full_like(x, 0.5)], axis=1)
    owner = np.zeros(1000, int)
    res = A.misattribution_by_object(pts, owner, 2, np.zeros(1000, int), [cal], [ids])
    assert res[0]["share"] == pytest.approx(0.5, abs=0.01) and res[0]["samples"] == 1000
    assert res[0]["top_confusion"]["object"] == 1
    assert res[1]["share"] is None                           # no samples
    # inpainted samples (source -1) are not counted; background source pixels are not mismatches
    res = A.misattribution_by_object(pts, owner, 2, np.full(1000, -1), [cal], [ids])
    assert res[0]["share"] is None
    res = A.misattribution_by_object(pts, owner, 2, np.zeros(1000, int), [cal], [np.zeros_like(ids)])
    assert res[0]["share"] == 0.0
