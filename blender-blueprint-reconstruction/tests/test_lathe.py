"""Lathe (body of revolution): mesh geometry, and the profile measured from front / side silhouettes."""
import math

import numpy as np
import pytest

from bbr import BBRError
from bbr.core import lathe, partgeom as g, zbuffer
from bbr.core.calibration import ViewCalibration

W, H, M = 300, 400, 0.005          # 1.5 m x 2 m image, 5 mm per pixel
FRONT = ViewCalibration("front", "-Y", (W, H), M, (150.0, 380.0))
SIDE = ViewCalibration("side", "-X", (W, H), M, (120.0, 380.0))
RATIO = 0.8


def vase_radius(z):
    """Foot, belly, narrow neck, flared lip (metres)."""
    z = np.asarray(z, dtype=float)
    return np.interp(z, [0.0, 0.05, 0.4, 0.9, 1.3, 1.5, 1.7], [0.18, 0.2, 0.42, 0.35, 0.12, 0.1, 0.2])


def _silhouettes():
    """Front / side masks of the analytic vase (axis at x = 0.05, y = -0.1; elliptic section RATIO)."""
    vv, uu = np.mgrid[0:H, 0:W]
    z = (FRONT.origin_px[1] - (vv + 0.5)) * M
    inside_z = (z >= 0.0) & (z <= 1.7)
    r = vase_radius(np.clip(z, 0, 1.7))
    x = (uu + 0.5 - FRONT.origin_px[0]) * M
    front = inside_z & (abs(x - 0.05) <= r)
    ys = -(uu + 0.5 - SIDE.origin_px[0]) * M                                   # side camera -X: image right = -Y
    side = inside_z & (abs(ys + 0.1) <= r * RATIO)
    return front, side


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def _silhouette_of(verts, faces, calib):
    tris = g.polygons_triangles(faces)
    return np.isfinite(zbuffer.depth_buffer(verts, tris, calib))


def test_lathe_mesh_of_a_cylinder_is_closed_with_the_exact_volume():
    v, f = g.lathe_mesh([[0.0, 0.1], [0.5, 0.1]], segments=64)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * 0.1 ** 2 * 0.5, rel=0.01)       # positive: outward winding
    assert v[:, 2].min() == 0.0 and v[:, 2].max() == 0.5


def test_lathe_mesh_reversed_profile_and_elliptic_section():
    a = g.lathe_mesh([[0.0, 0.1], [0.3, 0.2], [0.5, 0.0]], 32, 0.5)
    b = g.lathe_mesh([[0.5, 0.0], [0.3, 0.2], [0.0, 0.1]], 32, 0.5)
    assert g.mesh_volume(*a) == pytest.approx(g.mesh_volume(*b)) and g.mesh_volume(*a) > 0
    assert np.abs(a[0][:, 1]).max() == pytest.approx(0.5 * np.abs(a[0][:, 0]).max(), rel=1e-6)
    assert (np.isclose(a[0][:, 2], 0.5)).sum() == 1                                       # apex = one vertex


def test_lathe_mesh_open_ends_leave_no_cap():
    closed = g.lathe_mesh([[0.0, 0.1], [0.5, 0.1]], 16)
    opened = g.lathe_mesh([[0.0, 0.1], [0.5, 0.1]], 16, cap_top=False, cap_bottom=False)
    assert len(opened[1]) == 16 and len(closed[1]) == 16 * 3


@pytest.mark.parametrize("profile", [[[0, 0.1]], [[0, 0.1], [1, -0.1]], [[0, 0], [1, 0]]])
def test_lathe_mesh_rejects_bad_profiles(profile):
    with pytest.raises(BBRError):
        g.lathe_mesh(profile)


def test_vase_silhouette_revolves_back_to_matching_front_and_side_silhouettes():
    front, side = _silhouettes()
    m = lathe.measure_lathe(front, FRONT, side, SIDE)
    assert m.depth_ratio == pytest.approx(RATIO, abs=0.02)
    assert m.axis_xy == pytest.approx((0.05, -0.1), abs=0.01)
    assert len(m.profile) <= lathe.DEFAULT_MAX_POINTS
    verts, faces = g.lathe_mesh(m.profile, 96, m.depth_ratio)
    verts = verts + np.array([m.axis_xy[0], m.axis_xy[1], 0.0])
    assert _iou(_silhouette_of(verts, faces, FRONT), front) >= 0.98
    assert _iou(_silhouette_of(verts, faces, SIDE), side) >= 0.98


def test_bbox_and_label_regions_pick_one_part_of_a_stack():
    front, side = _silhouettes()
    lower = lathe.measure_lathe(front, FRONT, side, SIDE, bbox_px=[0, 380 - 400 // 2 * 1, W, 380])   # rows 180..380: z 0..1.0
    assert lower.profile[0][0] == pytest.approx(0.0, abs=M) and lower.profile[-1][0] == pytest.approx(1.0, abs=2 * M)
    label = np.zeros_like(front)
    label[200:300] = front[200:300]                                                       # a label mask = rows 200..299
    part = lathe.measure_lathe(label, FRONT)
    assert part.depth_ratio == 1.0 and part.info["depth_ratio_source"] == "default"
    assert [p[0] for p in part.profile] == sorted(p[0] for p in part.profile)             # bottom to top
    assert lathe.measure_lathe(label, FRONT, depth_ratio=0.7).depth_ratio == 0.7


def test_measure_lathe_errors_and_views():
    front, side = _silhouettes()
    with pytest.raises(BBRError, match="pixel rows"):
        lathe.measure_lathe(front, FRONT, bbox_px=[0, 0, 5, 5])
    with pytest.raises(BBRError, match="front/back"):
        lathe.measure_lathe(front, SIDE)
    with pytest.raises(BBRError, match="camera on X"):
        lathe.measure_lathe(front, FRONT, side, FRONT)


def test_simplify_keeps_corners_and_drops_collinear_points():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2]], float)
    assert lathe.simplify(pts, 0.1).tolist() == [[0, 0], [3, 0], [3, 2]]


def test_a_one_sided_cut_or_occluder_does_not_shrink_the_profile():
    front, side = _silhouettes()
    full = lathe.measure_lathe(front, FRONT, side, SIDE)
    cut = front.copy()
    cut[100:250, 190:] = False                                      # a chimney / spout labelled out on the right flank, rows 100..250
    cut_side = side.copy()
    cut_side[100:250, 130:] = False                                 # and on one side of the side view too
    m = lathe.measure_lathe(cut, FRONT, cut_side, SIDE)
    assert m.axis_xy == pytest.approx(full.axis_xy, abs=0.01) and m.depth_ratio == pytest.approx(RATIO, abs=0.03)
    mid = lambda prof, z: float(np.interp(z, [p[0] for p in prof], [p[1] for p in prof]))
    for z in (0.7, 0.9, 1.1, 1.3):                                  # rows 100..250 are z 1.4..0.65
        assert mid(m.profile, z) == pytest.approx(float(vase_radius(z)), abs=0.02), z
    verts, faces = g.lathe_mesh(m.profile, 96, m.depth_ratio)
    verts = verts + np.array([m.axis_xy[0], m.axis_xy[1], 0.0])
    assert _iou(_silhouette_of(verts, faces, FRONT), front) >= 0.98


def test_stray_rows_and_the_rounded_bottom_of_a_base_seen_from_above():
    front, side = _silhouettes()
    base = front.copy()
    base[150:153, 40:250] = True                                     # 3 stray rows (a label edge that kept a chimney's last pixels)
    m = lathe.measure_lathe(base, FRONT, side, SIDE)
    assert max(r for _, r in m.profile) < 0.45                       # no spike wider than the belly (r = 0.42)
    flat = lathe.measure_lathe(front, FRONT, bbox_px=[0, 0, W, 370], bottom_z=0.0)     # rows 370..379 (z < 0.05) cut off, then extended down
    assert flat.profile[0][0] == 0.0 and flat.profile[0][1] == flat.profile[1][1]
    assert lathe.measure_lathe(front, FRONT, bottom_z=-0.5).profile[0] == [-0.5, pytest.approx(0.18, abs=0.01)]
