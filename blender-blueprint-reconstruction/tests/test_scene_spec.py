"""SceneSpec parsing, defaults and validation."""
import copy
import json

import pytest

from bbr import SpecError
from bbr.core import spec as S
from bbr.core.config import PipelineConfig


@pytest.fixture
def data(fixtures_dir):
    return json.loads((fixtures_dir / "scene_spec_min.json").read_text())


def test_fixture_valid_with_both_validators(data):
    assert S.validate_scene_spec(data) == []
    assert S.schema_errors_builtin(data, S.SCENE_SCHEMA_NAME) == []


def test_defaults_applied(data):
    sc = S.parse_scene_spec(data)
    leg = sc.get_component("leg_fl")
    assert leg.stage == "primary" and leg.confidence == "observed" and leg.rotation_deg == [0, 0, 0]
    assert leg.mirror == {"x": True, "y": True, "z": False}
    assert sc.style["bevel_segments"] == 2 and sc.workflow["max_iterations"] == 3
    assert sc.agent_id("top") == "simple_table:top"


def test_stage_order_is_stable(data):
    data["components"].insert(0, {"id": "late", "type": "box", "stage": "detail", "size": [1, 1, 1]})
    data["components"].append({"id": "early", "type": "box", "stage": "blockout", "size": [1, 1, 1]})
    ids = [c.id for c in S.parse_scene_spec(data).components_by_stage()]
    assert ids == ["early", "top", "leg_fl", "shelf", "late"]


def test_roundtrip(data):
    sc = S.parse_scene_spec(data)
    again = S.parse_scene_spec(json.loads(json.dumps(sc.to_dict())))
    assert again.to_dict() == sc.to_dict()


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["components"].append(dict(d["components"][0])), "duplicate"),
    (lambda d: d["components"][0].__setitem__("material", "unobtanium"), "unobtanium"),
    (lambda d: d["components"][0].__setitem__("boolean", {"target": "nope"}), "nope"),
    (lambda d: d["components"][0].__setitem__("boolean", {"target": "top"}), "itself"),
    (lambda d: d["components"][0].pop("size"), "size is required"),
    (lambda d: d["components"][0].__setitem__("type", "teapot"), "type"),
    (lambda d: d["components"][0].__setitem__("size", [1, 0, 1]), "size"),
    (lambda d: d["components"][0].__setitem__("location", [0, 0]), "location"),
    (lambda d: d["components"][0].__setitem__("stage", "final"), "stage"),
    (lambda d: d["dimensions"].__setitem__("width", 0), "width"),
    (lambda d: d.pop("components"), "components"),
])
def test_invalid(data, mutate, needle):
    bad = copy.deepcopy(data)
    mutate(bad)
    errors = S.validate_scene_spec(bad)
    assert errors and any(needle in e for e in errors), errors
    with pytest.raises(SpecError):
        S.parse_scene_spec(bad)


def test_hull_needs_no_size(data):
    data["components"].append({"id": "hull", "type": "silhouette_hull", "params": {"resolution": 96}})
    assert S.validate_scene_spec(data) == []


def test_load_scene_spec(fixtures_dir):
    assert len(S.load_scene_spec(fixtures_dir / "scene_spec_min.json").components) == 3


def test_pipeline_config():
    cfg = PipelineConfig.from_dict({"max_iterations": 5})
    assert cfg.max_iterations == 5 and cfg.hull_resolution == 320
    with pytest.raises(SpecError):
        PipelineConfig.from_dict({"nope": 1})


@pytest.mark.parametrize("ptype,params,ok", [
    ("cone", {"flatten": 0.5}, True), ("cone", {"flatten": 1}, True), ("cone", {"flatten": 0}, False), ("cone", {"flatten": -0.2}, False),
    ("disc", {"thickness": 0.01, "radius_ratio": 1.5}, True), ("disc", {"thickness": 0.01, "radius_ratio": 0}, False),
    ("gem", {"waist": 0.5, "facets": 6}, True), ("gem", {"waist": 0}, False), ("gem", {"waist": 1}, False), ("gem", {"facets": 2}, False),
    ("blade", {"thickness": 0.01, "tip_length": 0}, True), ("blade", {"thickness": 0.01, "tip_length": -1}, False),
    ("disc", {"thickness": 0.01, "dome_height": -0.01}, False), ("disc", {"thickness": 0.01, "dome_radius": 0}, False)])
def test_part_parameter_ranges_are_in_the_schema(ptype, params, ok):
    data = {"spec_version": "0.1", "name": "p", "components": [{"id": "a", "type": ptype, "size": [0.02, 0.02, 0.05], "params": params}]}
    assert (S.validate_scene_spec(data) == []) is ok
    assert (S.schema_errors_builtin(data, S.SCENE_SCHEMA_NAME) == []) is ok      # the built-in validator agrees (exclusiveMaximum too)


def _add(data, **comp):
    data["components"].append({"stage": "primary", **comp})
    return data


def test_lathe_needs_a_profile_or_profile_from_and_no_size(data):
    ok = _add(copy.deepcopy(data), id="cap", type="lathe", params={"profile_from": {"view": "front", "bbox_px": [0, 0, 9, 9]}})
    assert S.validate_scene_spec(ok) == [] and S.schema_errors_builtin(ok, S.SCENE_SCHEMA_NAME) == []
    explicit = _add(copy.deepcopy(data), id="vase", type="lathe", params={"profile": [[0, 0.1], [0.5, 0.2]], "cap_top": False})
    assert S.validate_scene_spec(explicit) == []
    bare = _add(copy.deepcopy(data), id="cap", type="lathe")
    assert any("lathe needs params.profile" in e for e in S.validate_scene_spec(bare))


def test_attach_at_validation(data):
    at = {"azimuth_deg": 0, "height": 0.3}
    good = _add(copy.deepcopy(data), id="win", type="torus", params={"major_radius": 0.1, "minor_radius": 0.01, "attach_to": "top", "attach_at": at})
    assert S.validate_scene_spec(good) == [] and S.schema_errors_builtin(good, S.SCENE_SCHEMA_NAME) == []
    no_target = _add(copy.deepcopy(data), id="win", type="box", size=[1, 1, 1], params={"attach_at": at})
    assert any("needs params.attach_to" in e for e in S.validate_scene_spec(no_target))
    bad = _add(copy.deepcopy(data), id="win", type="box", size=[1, 1, 1], params={"attach_to": "top", "attach_at": {"view": "front"}})
    assert any("attach_at must be" in e for e in S.validate_scene_spec(bad))
    both = _add(copy.deepcopy(data), id="win", type="sphere", params={"attach_to": "top", "attach_at": at, "anchors_px": {"front": {"center": [1, 1], "radius_px": 2}}})
    assert any("use one or the other" in e for e in S.validate_scene_spec(both))
