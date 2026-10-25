"""Pixel -> world measuring against the calibration module, and the CLI."""
import json
import sys

import pytest

from bbr import BBRError
from bbr.core import imageio, measure
from bbr.core.calibration import calibrate_views
from bbr.core.config import PipelineConfig
from bbr.core.spec import load_view_spec
from conftest import SK, paint, rect_mask

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402


@pytest.fixture
def view_spec(tmp_path):
    """Front (100x200 px box at 20..120 / 30..230 in a 300x260 image) + side views of a 0.5 m tall object."""
    front = rect_mask(260, 300, 20, 30, 120, 230)
    side = rect_mask(260, 300, 100, 30, 160, 230)
    imageio.save_rgba(tmp_path / "front.png", paint(front, (0.9, 0.9, 0.9)))
    imageio.save_rgba(tmp_path / "side.png", paint(side, (0.9, 0.9, 0.9)))
    spec = {"spec_version": "0.1", "name": "t", "views": {"front": {"path": "front.png"},
            "side": {"path": "side.png", "camera_axis": "-X"}}, "known_dimensions": {"height": 0.5},
            "symmetry": {"x": False, "y": False, "z": False}}
    (tmp_path / "view_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path / "view_spec.json"


def _calibs(path):
    spec = load_view_spec(path)
    _, masks = measure.load_view_data(spec, PipelineConfig())
    return calibrate_views(spec, masks)[0]


def test_pixels_map_through_the_pipeline_calibration(view_spec):
    calibs = _calibs(view_spec)
    res = measure.measure_reference(view_spec, "front", points=[[70, 230], [20, 30], [120, 130]])
    assert res["plane"] == ["x", "z"] and res["depth_axis"] == "y"
    m = 0.5 / 200
    assert res["points"][0]["world"] == {"x": pytest.approx(0.0), "z": pytest.approx(0.0)}   # bottom centre = origin
    assert res["points"][1]["world"] == {"x": pytest.approx(-50 * m, abs=1e-5), "z": pytest.approx(0.5)}
    assert res["points"][2]["world"]["z"] == pytest.approx(100 * m, abs=1e-5)
    pts = measure.pixel_to_world(calibs["front"], [[20, 30]])[0]
    assert pts[0] == pytest.approx(res["points"][1]["world"]["x"], abs=1e-5)
    back = calibs["front"].world_to_px(measure.pixel_to_world(calibs["front"], [[33.5, 77.25]]))
    assert back[0] == pytest.approx([33.5, 77.25])                          # round trip


def test_side_view_axes_and_bbox(view_spec):
    res = measure.measure_reference(view_spec, "side", points=[[130, 130]], bbox=[100, 30, 160, 230])
    assert res["plane"] == ["y", "z"] and res["depth_axis"] == "x"
    assert res["points"][0]["world"]["y"] == pytest.approx(0.0, abs=1e-6)  # -X camera: image right = -Y, bbox centre = 0
    b = res["bbox"]
    assert b["size"]["z"] == pytest.approx(0.5) and b["size"]["y"] == pytest.approx(60 * 0.5 / 200, abs=1e-5)
    assert b["min"]["z"] == pytest.approx(0.0, abs=1e-6)


def test_measure_errors(view_spec):
    with pytest.raises(BBRError):
        measure.measure_reference(view_spec, "top", points=[[1, 1]])
    with pytest.raises(BBRError):
        measure.measure_reference(view_spec, "front")
    with pytest.raises(BBRError):
        measure.measure_reference(view_spec, "front", bbox=[1, 2, 3])


def test_cli_prints_world_coordinates(view_spec, capsys):
    assert cli.main(["measure", "--view-spec", str(view_spec), "--view", "front", "--px", "70,230", "--px", "20,30",
                     "--bbox", "20,30,120,230"]) == 0
    out = capsys.readouterr().out
    assert "x=+0.0000 z=+0.0000" in out and "z [+0.0000, +0.5000]" in out
    assert cli.main(["measure", "--view-spec", str(view_spec), "--view", "front", "--px", "70,230", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["points"][0]["world"]["z"] == pytest.approx(0.0)
    assert cli.main(["measure", "--view-spec", str(view_spec), "--view", "front", "--px", "oops"]) == 1


def test_points_file_batch_matches_single_measurements_and_states_the_formula(view_spec, tmp_path, capsys):
    named = {"front": {"top_left": [20, 30], "centre": [70, 130], "box": [20, 30, 120, 230]}, "side": {"mid": [130, 130]}}
    res = measure.measure_points(view_spec, named)
    single = measure.measure_reference(view_spec, "front", points=[[20, 30], [70, 130]], bbox=[20, 30, 120, 230])
    assert res["units"] == "m"
    assert res["views"]["front"]["top_left"] == single["points"][0]["world"]
    assert res["views"]["front"]["centre"] == single["points"][1]["world"]
    assert res["views"]["front"]["box"] == {k: single["bbox"][k] for k in ("min", "max", "size")}
    assert res["views"]["side"]["mid"] == {"y": pytest.approx(0.0, abs=1e-6), "z": pytest.approx(0.25, abs=1e-5)}
    cal = res["calibration"]["front"]                                  # the formula reproduces the numbers by hand
    assert cal["m_per_px"] == pytest.approx(0.0025) and cal["origin_px"] == [70.0, 230.0]
    assert cal["formula"] == {"x": "(u - 70) * 0.0025000", "z": "(230 - v) * 0.0025000"}
    u, v = 20, 30
    assert res["views"]["front"]["top_left"] == {"x": pytest.approx((u - 70) * 0.0025), "z": pytest.approx((230 - v) * 0.0025)}
    assert res["calibration"]["side"]["formula"]["y"] == "(130 - u) * 0.0025000"    # -X camera: image right = -Y
    f = tmp_path / "pts.json"
    f.write_text(json.dumps(named), encoding="utf-8")
    out = tmp_path / "metres.json"
    assert cli.main(["measure", "--view-spec", str(view_spec), "--points-file", str(f), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["views"]["front"]["centre"]["z"] == pytest.approx(0.25, abs=1e-5)
    assert json.loads(out.read_text())["views"] == json.loads(json.dumps(res["views"]))


def test_points_file_errors(view_spec, tmp_path):
    with pytest.raises(BBRError, match="not calibrated"):
        measure.measure_points(view_spec, {"top": {"a": [1, 1]}})
    with pytest.raises(BBRError, match=r"need \[x, y\]"):
        measure.measure_points(view_spec, {"front": {"a": [1, 2, 3]}})
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert cli.main(["measure", "--view-spec", str(view_spec), "--points-file", str(bad)]) == 1
    assert cli.main(["measure", "--view-spec", str(view_spec), "--px", "1,1"]) == 1          # no --view, no --points-file
