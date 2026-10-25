"""Conventions and calibration round trips."""
import numpy as np
import pytest

from bbr import CalibrationError, SpecError
from bbr.core import conventions as C
from bbr.core.calibration import ViewCalibration, calibrate_views, resolve_dimensions
from bbr.core.spec import parse_view_spec
from conftest import rect_mask

# contract section 2 table: camera_axis -> (image_right, image_up)
TABLE = {"-Y": ("+X", "+Z"), "+Y": ("-X", "+Z"), "+X": ("+Y", "+Z"),
         "-X": ("-Y", "+Z"), "+Z": ("+X", "+Y"), "-Z": ("+X", "-Y")}


@pytest.mark.parametrize("axis,expected", TABLE.items())
def test_frame_matches_contract_table(axis, expected):
    f = C.view_frame(axis)
    assert np.array_equal(f.right, C.AXES[expected[0]]) and np.array_equal(f.up, C.AXES[expected[1]])
    assert np.array_equal(f.forward, -C.AXES[axis])
    # right-handed camera: right x up = direction toward the camera
    assert np.allclose(np.cross(f.right, f.up), C.AXES[axis])


def test_default_axes_and_helpers():
    assert C.DEFAULT_CAMERA_AXIS == {"front": "-Y", "back": "+Y", "right": "+X", "left": "-X", "top": "+Z", "bottom": "-Z"}
    assert C.axis_index("-Y") == (1, -1) and C.axis_index("+Z") == (2, 1)
    assert C.normalize_axis(" -y ") == "-Y"
    with pytest.raises(SpecError):
        C.view_frame("Y")


@pytest.mark.parametrize("axis", TABLE)
def test_image_directions(axis):
    cal = ViewCalibration("v", axis, (100, 100), 0.01, (50.0, 80.0))
    f = C.view_frame(axis)
    o = cal.world_to_px(np.zeros(3))[0]
    assert np.allclose(o, (50, 80))
    step = cal.world_to_px(f.right * 0.1)[0] - o
    assert step[0] > 0 and abs(step[1]) < 1e-9          # image right = +u
    step = cal.world_to_px(f.up * 0.1)[0] - o
    assert step[1] < 0 and abs(step[0]) < 1e-9          # image up = -v (v points down)
    away = cal.world_to_px(C.AXES[axis] * 0.3)[0]       # depth direction does not move the pixel
    assert np.allclose(away, o)


@pytest.mark.parametrize("axis", TABLE)
def test_world_px_roundtrip(axis):
    cal = ViewCalibration("v", axis, (64, 48), 0.0125, (30.5, 40.0))
    uv = np.random.default_rng(1).uniform(0, 60, (20, 2))
    assert np.allclose(cal.world_to_px(cal.px_to_plane(uv)), uv)
    f = C.view_frame(axis)
    pts = np.random.default_rng(2).normal(size=(10, 3))
    plane = cal.px_to_plane(cal.world_to_px(pts))
    expected = pts - np.outer(pts @ C.AXES[axis], C.AXES[axis])  # projection onto the view plane
    assert np.allclose(plane, expected)
    assert cal.image_extent_m() == pytest.approx((0.8, 0.6))
    assert np.allclose(plane @ f.forward, 0)


def test_resolve_dimensions_from_height():
    boxes = {"front": (0, 0, 60, 100), "side": (0, 0, 30, 100), "top": (0, 0, 60, 30)}
    axes = {"front": "-Y", "side": "-X", "top": "+Z"}
    dims, prov = resolve_dimensions(boxes, axes, {"height": 2.0})
    assert dims == pytest.approx({"width": 1.2, "depth": 0.6, "height": 2.0})
    assert prov == {"height": "known", "width": "derived:front", "depth": "derived:side"}


def test_resolve_dimensions_default_and_known_width():
    boxes = {"front": (0, 0, 50, 100)}
    dims, prov = resolve_dimensions(boxes, {"front": "-Y"}, {})
    assert prov["height"] == "default" and dims["height"] == 1.0 and dims["width"] == pytest.approx(0.5)
    assert prov["depth"] == "default"
    dims, prov = resolve_dimensions({"front": (0, 0, 50, 100)}, {"front": "-Y"}, {"width": 0.5})
    assert dims["height"] == pytest.approx(1.0) and prov["height"] == "derived:front"


def _spec(**kw):
    views = {"front": {"path": "f.png"}, "side": {"path": "s.png", "camera_axis": "-X"}, "top": {"path": "t.png"}}
    return parse_view_spec({"spec_version": "0.1", "views": views, **kw})


def test_calibrate_views_silhouette_bbox():
    masks = {"front": rect_mask(120, 100, 20, 10, 80, 110),   # 60 x 100 px
             "side": rect_mask(120, 100, 30, 10, 60, 110),
             "top": rect_mask(100, 100, 20, 30, 80, 60)}
    calibs, info = calibrate_views(_spec(known_dimensions={"height": 2.0}), masks)
    assert info["dimensions"]["width"] == pytest.approx(1.2)
    assert calibs["front"].m_per_px == pytest.approx(0.02)
    assert calibs["front"].origin_px == (50.0, 110.0)         # bbox centre x, bbox bottom
    assert calibs["side"].origin_px == (45.0, 110.0)
    assert calibs["top"].m_per_px == pytest.approx(1.2 / 60)  # horizontal extent = width
    assert calibs["top"].origin_px == (50.0, 45.0)            # bbox centre both
    # bbox bottom is world z=0, bbox top is the height
    assert calibs["front"].px_to_plane(np.array([[50.0, 10.0]]))[0] == pytest.approx([0, 0, 2.0])


def test_calibrate_manual_and_errors():
    spec = _spec(known_dimensions={"height": 1.0})
    spec.views["front"].calibration.mode = "manual"
    with pytest.raises(CalibrationError):
        calibrate_views(spec, {"front": rect_mask(20, 20, 5, 5, 10, 10)})
    spec.views["front"].calibration.px_per_meter = 50.0
    spec.views["front"].calibration.origin_px = (10.0, 20.0)
    calibs, _ = calibrate_views(spec, {"front": rect_mask(20, 20, 5, 5, 10, 10)})
    assert calibs["front"].m_per_px == pytest.approx(0.02) and calibs["front"].origin_px == (10.0, 20.0)
    with pytest.raises(CalibrationError):
        calibrate_views(spec, {})
