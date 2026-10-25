"""Seating a part on a curved target surface (cylinder / lathe): ray cast, smooth outward normal, facing rotation, standoff."""
import math

import numpy as np
import pytest

from bbr import BBRError
from bbr.core import partgeom as g, surfaceattach as sa
from bbr.core.calibration import ViewCalibration

R, HEIGHT, CENTRE = 0.3, 1.0, np.array([0.1, 0.3])
M = 0.005
FRONT = ViewCalibration("front", "-Y", (400, 300), M, (200.0, 250.0))


@pytest.fixture(scope="module")
def cylinder():
    v, f = g.lathe_mesh([[0.0, R], [HEIGHT, R]], segments=64)
    v = v + np.array([CENTRE[0], CENTRE[1], 0.0])
    return v, g.polygons_triangles(f)


def _matrix(deg):
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


WINDOW = g.rounded_box_mesh([0.2, 0.04, 0.2], [0.01, 0.01, 0.01])[0]        # a 4 cm thick part, centred on its origin


@pytest.mark.parametrize("azimuth", [0.0, 50.0, 90.0, 200.0, -120.0])
def test_azimuth_places_the_part_on_the_round_surface_along_the_normal(cylinder, azimuth):
    verts, tris = cylinder
    place = sa.place_part(verts, tris, {"azimuth_deg": azimuth, "height": 0.4}, {}, WINDOW, "-Y", standoff=0.005)
    a = math.radians(azimuth)
    radial = np.array([math.sin(a), -math.cos(a), 0.0])                       # azimuth 0 = front (-Y), +90 = +X
    assert _angle(place.normal, radial) < 0.5                                  # the true curved normal, not the facet's
    assert np.hypot(*(place.point[:2] - CENTRE)) == pytest.approx(R, abs=R * (1 - math.cos(math.pi / 64)) + 1e-9)
    assert place.point[2] == pytest.approx(0.4)
    rot = _matrix(place.rotation_deg)
    assert _angle(rot @ np.array([0.0, -1.0, 0.0]), place.normal) < 1e-6        # local -Y points out of the surface
    assert rot @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)    # local Z stays upright
    assert place.back_offset == pytest.approx(0.02)                            # back face = half the 4 cm thickness
    gap = float((place.location - place.point) @ place.normal - place.back_offset)
    assert gap == pytest.approx(0.005)                                         # back face rests 'standoff' off the surface


def test_front_view_pixel_hits_the_surface_the_camera_sees(cylinder):
    verts, tris = cylinder
    x_off, z = 0.1, 0.5                                                        # 0.1 m right of the cylinder axis, 0.5 m up
    px = [FRONT.origin_px[0] + (CENTRE[0] + x_off) / M, FRONT.origin_px[1] - z / M]
    place = sa.place_part(verts, tris, {"view": "front", "px": px}, {"front": FRONT}, WINDOW)
    assert place.point[0] == pytest.approx(CENTRE[0] + x_off, abs=1e-6) and place.point[2] == pytest.approx(z, abs=1e-6)
    assert place.point[1] < CENTRE[1]                                          # the camera-side (front) half, not the far one
    expected = np.array([x_off / R, -math.sqrt(1 - (x_off / R) ** 2), 0.0])
    assert _angle(place.normal, expected) < 0.5
    assert place.location == pytest.approx(place.point + place.normal * place.back_offset)    # default standoff 0


def test_back_part_with_plus_y_face_needs_no_rotation(cylinder):
    verts, tris = cylinder
    place = sa.place_part(verts, tris, {"azimuth_deg": 180.0, "height": 0.5}, {}, WINDOW, "+Y")
    assert place.rotation_deg == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert place.normal == pytest.approx([0.0, 1.0, 0.0], abs=1e-3)
    front = sa.place_part(verts, tris, {"azimuth_deg": 0.0, "height": 0.5}, {}, WINDOW, "-Y")
    assert front.rotation_deg == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)       # facing the front: identity as well


def test_vertical_normal_gives_a_proper_rotation():
    for face in ("-Y", "+Y"):
        rot = sa.facing_rotation([0, 0, 1], face)
        assert np.linalg.det(rot) == pytest.approx(1.0)
        assert rot @ np.array([0.0, -1.0 if face == "-Y" else 1.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_standoff_can_sink_the_part_into_the_surface(cylinder):
    verts, tris = cylinder
    sunk = sa.place_part(verts, tris, {"azimuth_deg": 0.0, "height": 0.5}, {}, WINDOW, standoff=-0.01)
    seated = sa.place_part(verts, tris, {"azimuth_deg": 0.0, "height": 0.5}, {}, WINDOW)
    assert np.linalg.norm(seated.location - sunk.location) == pytest.approx(0.01)


def test_misses_and_bad_arguments_raise(cylinder):
    verts, tris = cylinder
    with pytest.raises(BBRError, match="misses"):
        sa.place_part(verts, tris, {"azimuth_deg": 0.0, "height": 5.0}, {}, WINDOW)
    with pytest.raises(BBRError, match="attach_at needs"):
        sa.place_part(verts, tris, {"azimuth_deg": 0.0}, {}, WINDOW)
    with pytest.raises(BBRError, match="not a calibrated view"):
        sa.place_part(verts, tris, {"view": "front", "px": [1, 1]}, {}, WINDOW)
    with pytest.raises(BBRError, match="face must be"):
        sa.place_part(verts, tris, {"azimuth_deg": 0.0, "height": 0.5}, {}, WINDOW, "+Z")
