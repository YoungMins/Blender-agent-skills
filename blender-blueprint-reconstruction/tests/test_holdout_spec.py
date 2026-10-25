"""Holdout role: parsing, azimuth/elevation camera_axis, and isolation from carving / colour / calibration / consistency."""
import pytest

from bbr import SpecError
from bbr.core import consistency, imageio, measure
from bbr.core.calibration import calibrate_views
from bbr.core.config import PipelineConfig
from bbr.core.spec import _semantic_view_errors, parse_view_spec, schema_errors_builtin, validate_view_spec
from conftest import ellipse_mask, paint


def write_views(tmp_path, names):
    for i, name in enumerate(names):
        imageio.save_rgba(tmp_path / f"{name}.png", paint(ellipse_mask(120, 100, 60, 50, 50, 30 + 4 * i), (0.9, 0.2, 0.2)))


def spec_dict(**extra_views):
    views = {"front": {"path": "front.png"}, "side": {"path": "side.png", "camera_axis": "-X"},
             "back": {"path": "back.png"},
             "h45": {"path": "h45.png", "camera_axis": {"azimuth": 45, "elevation": 10}},
             "left": {"path": "left.png", "roles": ["holdout"]}}
    views.update(extra_views)
    return {"spec_version": "0.1", "views": views, "known_dimensions": {"height": 1.0}}


def test_parse_angle_and_axis_holdouts(tmp_path):
    vs = parse_view_spec(spec_dict(), tmp_path)
    assert vs.views["h45"].roles == ["holdout"] and vs.views["h45"].camera_angles == (45.0, 10.0)
    assert vs.views["h45"].camera_axis is None and vs.views["left"].camera_axis == "-X"
    assert [v.key for v in vs.holdout_views()] == ["h45", "left"]
    assert not vs.views["h45"].is_orthographic_usable and not vs.views["left"].is_orthographic_usable
    assert vs.views["front"].is_orthographic_usable
    assert set(vs.view_axes()) == {"front", "side", "back"}
    for role in ("carve", "color", "validate"):
        assert not {"h45", "left"} & {v.key for v in vs.enabled_views(role)}


def test_holdout_is_never_used_to_build(tmp_path):
    write_views(tmp_path, ["front", "side", "back", "h45", "left"])
    vs = parse_view_spec(spec_dict(), tmp_path)
    rgba, masks = measure.load_view_data(vs, PipelineConfig())
    assert set(rgba) == set(masks) == {"front", "side", "back"}
    calibs, info = calibrate_views(vs, masks)
    assert set(calibs) == set(info["bboxes_px"]) == {"front", "side", "back"}
    roles = {k: v.roles for k, v in vs.views.items()}
    result = consistency.analyze(masks, calibs, vs.view_axes(), roles)
    assert not {"h45", "left"} & set(result["views"])
    assert not any("h45" in str(p) or "left" in str(p) for p in result.get("pairs", []))


def test_holdout_errors(tmp_path):
    bad_roles = spec_dict(h45={"path": "h45.png", "camera_axis": {"azimuth": 45}, "roles": ["holdout", "carve"]})
    assert any("holdout" in e for e in _semantic_view_errors(bad_roles))
    with pytest.raises(SpecError):
        parse_view_spec(bad_roles, tmp_path)
    assert validate_view_spec(spec_dict(h45={"path": "h", "camera_axis": {"azimuth": "x"}}))   # whichever validator is active
    angle_carve = spec_dict(h45={"path": "h45.png", "camera_axis": {"azimuth": 45}, "roles": ["carve"]})
    assert any("holdout / preview" in e for e in _semantic_view_errors(angle_carve))
    no_az = spec_dict(h45={"path": "h45.png", "camera_axis": {"elevation": 5}})
    assert any("azimuth" in e for e in _semantic_view_errors(no_az))
    assert any("elevation" in e for e in _semantic_view_errors(
        spec_dict(h45={"path": "h45.png", "camera_axis": {"azimuth": 5, "elevation": 120}})))
    assert any("camera_axis" in e for e in _semantic_view_errors(spec_dict(h45={"path": "h", "camera_axis": "up"})))
    assert any("unknown keys" in e for e in _semantic_view_errors(
        spec_dict(h45={"path": "h", "camera_axis": {"azimuth": 5, "roll": 3}})))
    no_cam = spec_dict(odd={"path": "odd.png", "roles": ["holdout"]})
    assert any("camera_axis" in e for e in _semantic_view_errors(no_cam))
    # the built-in validator (used when jsonschema is absent) accepts the object form too
    assert schema_errors_builtin(spec_dict(), "view_spec.schema.json") == []
    assert any("not in" in e for e in schema_errors_builtin(
        spec_dict(x={"path": "x", "roles": ["nope"]}), "view_spec.schema.json"))
