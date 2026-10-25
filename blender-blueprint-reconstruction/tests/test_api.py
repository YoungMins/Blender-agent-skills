"""api tool wrapper: argument validation and the atomic update_component (Blender modules are stubbed)."""
import json
import sys
import types
from pathlib import Path

import pytest

from bbr import api

SPEC = json.loads((Path(__file__).resolve().parents[1] / "examples" / "simple_table" / "scene_spec.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("call", [
    lambda: api.save_checkpoint("", "out"),
    lambda: api.save_checkpoint("  ", "out"),
    lambda: api.save_checkpoint(None, "out"),
    lambda: api.save_checkpoint("cp", 5),
    lambda: api.restore_checkpoint("cp", ""),
    lambda: api.run_reconstruction("", "scene.json", "out"),
    lambda: api.run_reconstruction("view.json", ["x"], "out"),
    lambda: api.build_from_scene_spec(""),
    lambda: api.build_from_scene_spec("scene.json", view_spec_path=""),
    lambda: api.measure_points(None, {}),
])
def test_bad_text_arguments_return_an_error_dict(call):
    res = call()
    assert res["ok"] is False and "non-empty string" in res["error"]


def test_optional_arguments_may_stay_none(monkeypatch):
    monkeypatch.setattr(api, "_jsonable", lambda v: v)
    reached = {}
    fake = types.SimpleNamespace(save_checkpoint=lambda name, out_dir: reached.setdefault("args", (name, out_dir)))
    monkeypatch.setitem(sys.modules, "bbr.blender.checkpoint", fake)
    monkeypatch.setattr("bbr.blender.checkpoint", fake, raising=False)
    assert api.save_checkpoint("cp", "out")["ok"] is True
    assert reached["args"] == ("cp", "out")


def test_update_component_restores_the_spec_when_the_rebuild_fails(tmp_path, monkeypatch):
    path = tmp_path / "scene_spec.json"
    original = json.dumps(SPEC, indent=4)   # deliberately not the format update_component writes
    path.write_text(original, encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("rebuild failed")

    fake = types.ModuleType("bbr.blender.reconstruction")
    fake.load_inputs = boom
    monkeypatch.setitem(sys.modules, "bbr.blender.reconstruction", fake)
    monkeypatch.setattr("bbr.blender.reconstruction", fake, raising=False)
    res = api.update_component(str(path), "top", {"size": [2, 2, 2]})
    assert res["ok"] is False and "rebuild failed" in res["error"]
    assert path.read_text(encoding="utf-8") == original
