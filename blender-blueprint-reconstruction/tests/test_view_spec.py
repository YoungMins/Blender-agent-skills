"""ViewSpec parsing, defaults and validation."""
import copy
import json

import pytest

from bbr import SpecError
from bbr.core import spec as S


@pytest.fixture
def data(fixtures_dir):
    return json.loads((fixtures_dir / "view_spec_min.json").read_text())


def test_fixture_is_valid_with_both_validators(data):
    assert S.validate_view_spec(data) == []
    assert S.schema_errors_builtin(data, S.VIEW_SCHEMA_NAME) == []


def test_defaults_and_paths(data, tmp_path):
    vs = S.parse_view_spec(data, tmp_path)
    assert vs.views["front"].camera_axis == "-Y"
    assert vs.views["front"].roles == ["carve", "color", "validate"]
    assert vs.views["side"].camera_axis == "-X"
    assert vs.views["top"].camera_axis == "+Z" and vs.views["top"].roles == ["validate"]
    assert vs.views["front"].path == (tmp_path / "references/front.png").resolve()
    assert vs.views["front"].background.tolerance == 0.12
    assert vs.views["front"].calibration.mode == "silhouette_bbox"
    assert vs.known_dimensions["height"] == 0.72
    assert vs.view_axes() == {"front": "-Y", "side": "-X", "top": "+Z"}  # preview-only view excluded


def test_back_roles_and_preview_view(data, tmp_path):
    data["views"]["back"] = {"path": "b.png"}
    vs = S.parse_view_spec(data, tmp_path)
    assert vs.views["back"].roles == ["color", "validate"] and vs.views["back"].camera_axis == "+Y"
    tq = vs.views["three_quarter_front"]
    assert tq.camera_axis is None and tq.roles == ["preview"] and not tq.is_orthographic_usable


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["views"]["side"].pop("camera_axis"), "side"),
    (lambda d: d["views"].__setitem__("weird", {"path": "x.png"}), "weird"),
    (lambda d: d["views"].__setitem__("weird", {"path": "x.png", "roles": ["carve"]}), "weird"),
    (lambda d: d["views"]["front"].__setitem__("camera_axis", "up"), "camera_axis"),
    (lambda d: d["views"]["front"].__setitem__("roles", ["paint"]), "roles"),
    (lambda d: d["views"]["front"].__setitem__("crop_px", [10, 10, 5, 50]), "crop_px"),
    (lambda d: d["views"]["front"].__setitem__("calibration", {"mode": "manual"}), "manual"),
    (lambda d: d["known_dimensions"].__setitem__("height", -1), "height"),
    (lambda d: d.__setitem__("spec_version", "9"), "spec_version"),
    (lambda d: d.pop("views"), "views"),
    (lambda d: d.__setitem__("bogus", 1), "bogus"),
    (lambda d: [v.__setitem__("enabled", False) for v in d["views"].values()], "enabled"),
])
def test_invalid_specs_report_errors(data, mutate, needle):
    bad = copy.deepcopy(data)
    mutate(bad)
    errors = S.validate_view_spec(bad)
    assert errors and any(needle in e for e in errors), errors
    with pytest.raises(SpecError):
        S.parse_view_spec(bad)


def test_builtin_validator_catches_schema_violations(data):
    data["views"]["front"]["roles"] = ["paint"]
    data["known_dimensions"]["height"] = "tall"
    errors = S.schema_errors_builtin(data, S.VIEW_SCHEMA_NAME)
    assert len(errors) == 2


def test_load_missing_files_raises(fixtures_dir):
    with pytest.raises(SpecError, match="missing image files"):
        S.load_view_spec(fixtures_dir / "view_spec_min.json")
    assert S.load_view_spec(fixtures_dir / "view_spec_min.json", check_files=False).name == "simple_table"


def test_load_bad_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{ // comment\n}")
    with pytest.raises(SpecError, match="invalid JSON"):
        S.load_view_spec(p)
