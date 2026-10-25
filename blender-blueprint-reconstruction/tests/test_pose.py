"""Held-out pose search: finds the true azimuth of an asymmetric object, off the nominal angle."""
import numpy as np
import pytest

from bbr.core import pose as P
from bbr.core.conventions import view_frame
from bbr.core.imageops import mask_bbox


def test_view_basis_matches_axis_frames():
    for az, axis in ((0, "-Y"), (90, "+X"), (180, "+Y"), (270, "-X")):
        d, right, up = P.view_basis(az, 0)
        f = view_frame(axis)
        assert np.allclose(d, -f.forward, atol=1e-9) and np.allclose(right, f.right, atol=1e-9)
        assert np.allclose(up, f.up, atol=1e-9)
        assert P.axis_angles(axis) == (float(az), 0.0)
    d, right, up = P.view_basis(45, 20)
    assert abs(np.dot(d, right)) < 1e-9 and abs(np.dot(d, up)) < 1e-9 and up[2] > 0
    assert np.allclose(P.view_basis(0, 90)[1], view_frame("+Z").right)


def make_points():
    """Asymmetric standing figure: body, one arm out to +X, a long tail toward +Y."""
    rng = np.random.default_rng(3)
    body = rng.uniform([-0.15, -0.1, 0.0], [0.15, 0.1, 0.8], (6000, 3))
    arm = rng.uniform([0.15, -0.04, 0.5], [0.55, 0.04, 0.58], (1500, 3))
    tail = rng.uniform([-0.03, 0.1, 0.1], [0.03, 0.6, 0.18], (2500, 3))
    return np.concatenate([body, arm, tail])


def silhouette(points, az, el, m_per_px, size, centre):
    _, right, up = P.view_basis(az, el)
    r = (points - centre) @ right
    u = (points - centre) @ up
    px = np.floor(r / m_per_px + size[0] / 2).astype(int)
    py = np.floor(-u / m_per_px + size[1] / 2).astype(int)
    m = np.zeros((size[1], size[0]), bool)
    ok = (px >= 0) & (px < size[0]) & (py >= 0) & (py < size[1])
    m[py[ok], px[ok]] = True
    k = 3                                                     # close the splat holes
    pad = np.pad(m, k)
    out = np.zeros_like(m)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= pad[k + dy:k + dy + m.shape[0], k + dx:k + dx + m.shape[1]]
    return out


def test_search_finds_true_azimuth_off_nominal():
    pts = make_points()
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    m0, canvas = 0.004, (400, 400)
    render = lambda az, el: silhouette(pts, az, el, m0, canvas, centre)       # noqa: E731
    # reference: 55 degrees, its own scale (5 mm/px), 3 % larger and a shifted canvas; nominal 45
    ref = silhouette(pts * 1.03, 55.0, 0.0, 0.005, (260, 300), centre * 1.03 + [0.1, 0.0, -0.05])
    res = P.search_pose(ref, render, m0, nominal_azimuth=45.0, azimuth_range=15, azimuth_step=5, scale_range=0.05)
    assert res["azimuth"] == pytest.approx(55.0, abs=5.0)
    assert res["iou"] > 0.9 and res["iou"] > res["nominal_iou"] + 0.02
    assert res["azimuth_step"] == 5.0 and len(res["by_azimuth"]) == 7
    # without a search the nominal azimuth stays
    fixed = P.search_pose(ref, render, m0, nominal_azimuth=45.0, azimuth_range=0, scale_range=0)
    assert fixed["azimuth"] == 45.0 and fixed["iou"] < res["iou"]


def test_exact_view_fits_perfectly_and_warp_is_identity():
    pts = make_points()
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    m0 = 0.004
    canon = silhouette(pts, 0.0, 0.0, m0, (400, 400), centre)
    fit = P.fit_transform(mask_bbox(canon), P.plane_bbox(canon, m0), canon.shape)
    assert fit["k"] == pytest.approx(m0, rel=0.02)
    assert (P.warp_canonical(canon, m0, canon.shape, fit) == canon).mean() > 0.99
