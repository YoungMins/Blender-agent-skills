"""PBR material classes: preset + override resolution, aliases, column choice (numpy-free)."""
import json
from pathlib import Path

import pytest

from bbr import SpecError
from bbr.core import pbr

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "scene_spec.schema.json"


def test_class_preset_per_column():
    assert pbr.resolve({"class": "metal_steel"}, "physical") == {"metallic": 1.0, "roughness": 0.4}
    assert pbr.resolve({"class": "metal_steel"}, "stylized") == {"metallic": 0.6, "roughness": 0.45}


def test_explicit_keys_override_the_preset():
    out = pbr.resolve({"class": "metal_gold", "roughness": 0.1, "coat": 0.5}, "physical")
    assert out == {"metallic": 1.0, "roughness": 0.1, "coat": 0.5}


def test_bare_class_name_and_alias():
    assert pbr.resolve("gold", "physical") == pbr.resolve({"class": "metal_gold"}, "physical")
    assert pbr.canonical("iron") == pbr.canonical("steel") == "metal_steel"
    assert pbr.canonical("fabric") == "cloth" and pbr.canonical("paint") == "painted"


def test_no_class_returns_only_explicit_keys():
    assert pbr.resolve(None, "physical") == {}
    assert pbr.resolve({}, "stylized") == {}
    assert pbr.resolve({"base_color": [1, 0, 0], "roughness": 0.3}, "physical") == {"roughness": 0.3}


def test_unknown_class_and_column_are_rejected():
    with pytest.raises(SpecError, match="unobtanium"):
        pbr.resolve({"class": "unobtanium"}, "physical")
    with pytest.raises(SpecError):
        pbr.resolve({"class": "wood"}, "cartoon")
    assert not pbr.is_class("unobtanium") and pbr.is_class("gold") and pbr.is_class("visor")


@pytest.mark.parametrize("style,projected,column", [
    (None, True, "stylized"), (None, False, "physical"),
    ({"pbr": "auto"}, True, "stylized"), ({}, False, "physical"),
    ({"pbr": "physical"}, True, "physical"), ("stylized", False, "stylized"),
])
def test_column_for(style, projected, column):
    assert pbr.column_for(projected, style) == column


def test_column_for_rejects_bad_mode():
    with pytest.raises(SpecError):
        pbr.column_for(True, {"pbr": "toon"})


def test_entry_for():
    mats = {"blade": {"class": "metal_steel", "roughness": 0.2}}
    assert pbr.entry_for("blade", mats) == {"class": "metal_steel", "roughness": 0.2}
    assert pbr.entry_for("visor", mats) == {"class": "visor"}
    assert pbr.entry_for("custom_unknown", mats) == {}
    assert pbr.entry_for(None, mats) == {}


def test_every_preset_stays_in_principled_range():
    for name, columns in pbr.CLASSES.items():
        for col in columns:
            assert set(col) <= set(pbr.PBR_KEYS), name
            for key, val in col.items():
                assert val >= 0 and (key in ("emission_strength", "ior") or val <= 1), (name, key)


def test_schema_enum_matches_the_classes():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    enum = schema["properties"]["materials"]["additionalProperties"]["properties"]["class"]["enum"]
    assert enum == pbr.class_names()
