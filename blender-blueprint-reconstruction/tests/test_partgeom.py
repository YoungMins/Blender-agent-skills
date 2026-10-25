"""Thin-part geometry (cone / blade / disc): volumes, extents, winding."""
import math

import numpy as np
import pytest

from bbr import BBRError
from bbr.core import partgeom as g


def test_straight_cone_volume_and_extents():
    v, f = g.cone_mesh(0.02, 0.005, 0.1, (0, 0, 1), 0, None, 8, 64)
    frustum = math.pi * 0.1 / 3 * (0.02 ** 2 + 0.02 * 0.005 + 0.005 ** 2)
    assert g.mesh_volume(v, f) == pytest.approx(frustum, rel=0.02)      # positive: faces wind outward
    assert v[:, 2].min() == pytest.approx(0.0) and v[:, 2].max() == pytest.approx(0.1)
    assert np.hypot(v[:, 0], v[:, 1]).max() == pytest.approx(0.02, rel=1e-6)


def test_pointed_cone_closes_to_apex():
    v, f = g.cone_mesh(0.02, 0.0, 0.1, (0, 0, 1), 0, None, 8, 16)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * 0.02 ** 2 * 0.1 / 3, rel=0.03)
    assert np.isclose(v[:, 2], 0.1).sum() == 1                            # a single apex vertex


def test_bent_cone_tip_follows_the_arc():
    length, bend = 0.1, 90.0
    v, f = g.cone_mesh(0.01, 0.002, length, (0, 0, 1), bend, (1, 0, 0), 24, 12)
    radius = length / math.radians(bend)                                  # quarter circle: tip at (R, 0, R)
    pts, tang, _, _ = g.cone_path(length, (0, 0, 1), bend, (1, 0, 0), 24)
    assert pts[-1] == pytest.approx([radius, 0, radius], abs=1e-9)
    assert tang[-1] == pytest.approx([1, 0, 0], abs=1e-9)
    assert g.mesh_volume(v, f) > 0
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() == pytest.approx(length, rel=1e-3)


def test_bend_dir_is_projected_perpendicular_to_axis():
    pts, _, _, _ = g.cone_path(0.1, (0, 0, 1), 60, (1, 0, 5), 8)           # z part of bend_dir is ignored
    assert abs(pts[:, 1]).max() < 1e-12 and pts[-1, 0] > 0


def test_cone_errors():
    with pytest.raises(BBRError):
        g.cone_mesh(0.0, 0.0, 0.1)
    with pytest.raises(BBRError):
        g.cone_mesh(0.01, 0.0, 0.1, bend_deg=30)                          # bend without a direction
    with pytest.raises(BBRError):
        g.cone_mesh(0.01, 0.0, 0.1, (0, 0, 1), 30, (0, 0, 1))             # bend_dir parallel to axis


def test_blade_volume_with_and_without_point():
    v, f = g.blade_mesh(0.3, 0.04, 0.02)
    assert g.mesh_volume(v, f) == pytest.approx(0.3 * 0.04 * 0.02)
    v, f = g.blade_mesh(0.3, 0.04, 0.02, tip_length=0.06)
    assert g.mesh_volume(v, f) == pytest.approx(0.24 * 0.04 * 0.02 + 0.06 * 0.04 * 0.02 / 3)
    assert v[:, 2].max() == pytest.approx(0.3)
    assert np.isclose(v[:, 2], 0.3).sum() == 1                            # pointed tip


def test_blade_axis_and_width_dir():
    v, f = g.blade_mesh(0.2, 0.05, 0.01, axis=(1, 0, 0), width_dir=(0, 0, 1))
    assert v[:, 0].max() == pytest.approx(0.2) and np.ptp(v[:, 2]) == pytest.approx(0.05) and np.ptp(v[:, 1]) == pytest.approx(0.01)
    assert g.mesh_volume(v, f) > 0


def test_disc_with_and_without_dome():
    v, f = g.disc_mesh(0.1, 0.02, 0.0, None, 96)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * 0.1 ** 2 * 0.02, rel=0.01)
    v, f = g.disc_mesh(0.1, 0.02, 0.03, 0.04, 96, 16)
    cap = math.pi * 0.03 / 6 * (3 * 0.04 ** 2 + 0.03 ** 2)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * 0.1 ** 2 * 0.02 + cap, rel=0.01)
    assert v[:, 2].max() == pytest.approx(0.01 + 0.03)                    # dome apex above the +axis face
    assert v[:, 2].min() == pytest.approx(-0.01)


def test_disc_axis_points_the_dome():
    v, f = g.disc_mesh(0.07, 0.012, 0.02, 0.03, 32, axis=(0, 1, 0))
    assert v[:, 1].max() == pytest.approx(0.026) and v[:, 1].min() == pytest.approx(-0.006)
    assert g.mesh_volume(v, f) > 0


def test_disc_errors():
    with pytest.raises(BBRError):
        g.disc_mesh(0.1, 0.02, 0.03, 0.2)                                  # dome wider than the disc


def _edge_use(faces):
    count = {}
    for f in faces:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            count[e] = count.get(e, 0) + 1
    return count


@pytest.mark.parametrize("build", [
    lambda: g.cone_mesh(0.02, 0.0, 0.1, (0, 0, 1), 60, (1, 0, 0), 8, 12),
    lambda: g.cone_mesh(0.02, 0.005, 0.1, (0, 0, 1), 0, None, 8, 12),
    lambda: g.blade_mesh(0.3, 0.04, 0.02, tip_length=0.05),
    lambda: g.disc_mesh(0.1, 0.02, 0.03, 0.04, 24),
    lambda: g.disc_mesh(0.1, 0.02),
])
def test_meshes_are_watertight(build):
    v, f = build()
    assert set(_edge_use(f).values()) == {2}                              # every edge shared by exactly two faces
    assert len(np.unique(np.round(v, 9), axis=0)) == len(v)               # no duplicate vertices


def test_cone_flatten_thins_the_cross_section_perpendicular_to_the_bend_plane():
    v, f = g.cone_mesh(0.03, 0.03, 0.1, (0, 0, 1), 0, None, 4, 32, flatten=0.5)
    assert np.ptp(v[:, 1]) == pytest.approx(0.03, rel=1e-6) and np.ptp(v[:, 0]) == pytest.approx(0.06, rel=1e-6)
    bent, _ = g.cone_mesh(0.03, 0.03, 0.1, (0, 0, 1), 60, (1, 0, 0), 12, 32, flatten=0.5)
    assert np.ptp(bent[:, 1]) == pytest.approx(0.03, rel=1e-6)            # y is the binormal of the x-z bend plane
    with pytest.raises(BBRError):
        g.cone_mesh(0.03, 0.0, 0.1, flatten=0.0)


@pytest.mark.parametrize("facets", [3, 4, 6, 8])
def test_gem_is_a_closed_bipyramid_with_the_right_volume(facets):
    v, f = g.gem_mesh(0.02, 0.1, facets)
    assert len(v) == facets + 2 and len(f) == 2 * facets and all(len(t) == 3 for t in f)       # flat triangles only
    assert set(_edge_use(f).values()) == {2}
    base_area = facets / 2.0 * 0.02 ** 2 * math.sin(2 * math.pi / facets)
    assert g.mesh_volume(v, f) == pytest.approx(base_area * 0.1 / 3.0, rel=1e-9)               # positive: faces point outward
    assert v[:, 2].min() == pytest.approx(-0.05) and v[:, 2].max() == pytest.approx(0.05)
    assert np.hypot(v[:, 0], v[:, 1]).max() == pytest.approx(0.02)


def test_gem_waist_axis_and_errors():
    v, f = g.gem_mesh(0.02, 0.1, 4, waist=0.7, axis=(1, 0, 0))
    ring = v[1:-1]
    assert np.allclose(ring[:, 0], -0.05 + 0.07) and v[:, 0].min() == pytest.approx(-0.05)       # apexes along +-X, ring at 70 %
    assert g.mesh_volume(v, f) > 0
    for bad in ((0.0, 0.1, 6, 0.5), (0.02, 0.0, 6, 0.5), (0.02, 0.1, 6, 0.0), (0.02, 0.1, 6, 1.0)):
        with pytest.raises(BBRError):
            g.gem_mesh(*bad)


def test_disc_radius_ratio_makes_an_elliptical_brim():
    v, f = g.disc_mesh(0.1, 0.01, segments=96, radius_ratio=0.6)
    assert np.ptp(v[:, 0]) == pytest.approx(0.2, rel=1e-3) and np.ptp(v[:, 1]) == pytest.approx(0.12, rel=1e-3)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * 0.1 * 0.06 * 0.01, rel=0.01)
    assert set(_edge_use(f).values()) == {2}
    v, f = g.disc_mesh(0.1, 0.01, 0.02, 0.05, 48, axis=(0, 1, 0), radius_ratio=2.0)           # axis Y: ratio acts on the 2nd frame vector
    assert g.mesh_volume(v, f) > 0
    with pytest.raises(BBRError):
        g.disc_mesh(0.1, 0.01, radius_ratio=0.0)


def test_glove_is_closed_pieces_inside_the_capsule_box():
    import numpy as np
    from bbr.core import partgeom as G
    verts, faces = G.glove_mesh(0.08, 0.05)
    edges = {}
    for poly in faces:
        for a, b in zip(poly, poly[1:] + poly[:1]):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    assert all(c == 2 for c in edges.values())                      # every piece closed
    assert G.mesh_volume(verts, faces) > 0                           # outward
    assert verts[:, 2].min() == pytest.approx(-0.04, abs=1e-6)       # wrist end
    assert 0.03 < verts[:, 2].max() <= 0.045                         # fingertips near the far end
    assert verts[:, 0].min() < -0.025                                # thumb sticks out on -X
    assert verts[:, 1].max() > 0.5 * 0.55 * 0.05                     # curled toward the palm (+Y)
    v3, f3 = G.glove_mesh(0.08, 0.05, fingers=3, thumb=False)
    assert len(f3) < len(faces)
    with pytest.raises(Exception):
        G.glove_mesh(0.08, 0.05, fingers=5)
