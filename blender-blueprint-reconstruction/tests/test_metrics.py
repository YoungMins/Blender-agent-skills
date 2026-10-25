"""Metrics and correction (error -> axis diagnosis)."""
import numpy as np
import pytest

from bbr.core import correction as K
from bbr.core import metrics as M
from bbr.core.calibration import ViewCalibration
from conftest import ellipse_mask, rect_mask

M_PER_PX = 0.01


def calib(view="front", axis="-Y", size=(100, 100)):
    return ViewCalibration(view, axis, size, M_PER_PX, (50.0, 90.0))


def test_iou_and_edges():
    a = rect_mask(50, 50, 10, 10, 30, 30)
    b = rect_mask(50, 50, 20, 10, 40, 30)
    assert M.iou(a, a) == 1.0 and M.iou(a, b) == pytest.approx(200 / 600)
    assert M.iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0
    assert M.edge_distance(a, a) == 0.0
    assert M.edge_distance(a, rect_mask(50, 50, 13, 10, 33, 30)) == pytest.approx(1.5, abs=1.0)  # 3 px shift on 2 of 4 sides
    assert M.edge_distance(a, np.zeros_like(a), cap=7) == 7.0


def test_bbox_delta_sign():
    ref = rect_mask(50, 50, 10, 10, 30, 30)
    out = rect_mask(50, 50, 8, 10, 35, 28)
    d = M.bbox_delta_px(ref, out)
    assert d == {"left": 2.0, "right": 5.0, "top": 0.0, "bottom": -2.0}


def test_symmetry_score():
    m = ellipse_mask(60, 100, 30, 50, 20, 30)
    assert M.symmetry_score(m, 50.0) == 1.0
    lopsided = m | rect_mask(60, 100, 90, 25, 95, 35)
    assert M.symmetry_score(lopsided, 50.0) < 0.99
    assert M.symmetry_score(rect_mask(60, 100, 10, 10, 30, 30), 50.0) == 0.0


def test_compare_view_and_region():
    ref = rect_mask(100, 100, 20, 20, 80, 90)
    out = rect_mask(100, 100, 20, 20, 80, 90).copy()
    out[20:40, 60:80] = False                     # missing top-right corner block
    res = M.compare_view(ref, out, calib())
    assert res.iou == pytest.approx(1 - 400 / 4200) and res.error == pytest.approx(400 / 4200)
    reg = res.largest_mismatch_region
    assert reg["kind"] == "missing" and reg["bbox_px"] == [60, 20, 80, 40]
    assert reg["center_world"][1] is None                     # front view: Y is the line of sight
    assert reg["center_world"][0] == pytest.approx(0.20) and reg["center_world"][2] == pytest.approx(0.60)
    assert reg["world_bbox"]["min"] == [pytest.approx(0.10), None, pytest.approx(0.50)]
    assert M.compare_view(ref, ref, calib()).largest_mismatch_region is None
    empty = M.compare_view(ref, np.zeros_like(ref), calib())
    assert empty.error == 1.0 and empty.out_bbox is None and empty.width_ratio == 0.0
    with pytest.raises(Exception):
        M.compare_view(ref, ref[:50], calib())


def test_aggregate_and_format():
    ref = rect_mask(100, 100, 20, 20, 80, 90)
    res = {"front": M.compare_view(ref, ref, calib()),
           "side": M.compare_view(ref, rect_mask(100, 100, 20, 20, 80, 80), calib("side", "-X"))}
    agg = M.aggregate(res, {"front": 1, "side": 3}, threshold=0.05)
    assert agg["total_error"] == pytest.approx(0.75 * res["side"].error) and agg["passed"] is False
    assert M.aggregate(res, threshold=0.5)["passed"] is True
    assert M.format_error_line("front", 0.043) == "Front Error : 4.3%"
    assert agg["per_view"]["front"]["line"] == "Front Error : 0.0%"


# ------------------------------------------------------------------ correction
def _result(view, axis, ref, out):
    return M.compare_view(ref, out, calib(view, axis))


def test_diagnose_maps_to_world_axes():
    ref = rect_mask(100, 100, 30, 20, 70, 90)                    # 40 x 70 px
    wide = rect_mask(100, 100, 28, 20, 72, 90)                   # +4 px width (centred), front view
    res = {"front": _result("front", "-Y", ref, wide)}
    items = K.diagnose(res, {"front": calib()})
    assert len(items) == 1 and items[0]["axis"] == "X" and items[0]["kind"] == "size"
    assert items[0]["recommendation"] == "Front view: width too large by 4.0 cm → X axis"
    items = K.diagnose({"right": _result("right", "+X", ref, wide)})
    assert items[0]["axis"] == "Y" and items[0]["dimension"] == "depth"
    items = K.diagnose({"top": _result("top", "+Z", ref, rect_mask(100, 100, 30, 20, 70, 70))})
    assert items[0]["axis"] == "Y" and "too small by 20.0 cm" in items[0]["recommendation"]
    # shift: output displaced 5 px to the image right in the back view (image right = -X)
    shifted = rect_mask(100, 100, 35, 20, 75, 90)
    sh = K.diagnose({"back": _result("back", "+Y", ref, shifted)})
    assert sh[0]["kind"] == "shift" and sh[0]["direction"] == "-X"
    assert K.diagnose({"front": _result("front", "-Y", ref, ref)}) == []


def test_axis_scale_factors_average_views():
    ref = rect_mask(100, 100, 30, 20, 70, 90)
    front = _result("front", "-Y", ref, rect_mask(100, 100, 30, 20, 70, 90).copy())
    front.width_ratio, front.height_ratio = 1.25, 1.10           # model too wide/tall
    side = _result("right", "+X", ref, ref)
    side.width_ratio, side.height_ratio = 0.8, 1.30
    f = K.axis_scale_factors({"front": front, "right": side})
    assert f["X"] == pytest.approx(0.8)
    assert f["Y"] == pytest.approx(1.25)
    assert f["Z"] == pytest.approx((1 / 1.10 + 1 / 1.30) / 2)
    assert "Y" not in K.axis_scale_factors({"front": front})


def test_apply_axis_scale():
    spec = {"dimensions": {"width": 1.0, "depth": None, "height": 2.0},
            "components": [
                {"id": "a", "location": [1, 2, 3], "size": [1, 2, 3], "rotation_deg": [0, 0, 0]},
                {"id": "b", "location": [0, 1, 0.5], "size": [1, 2, 3], "rotation_deg": [0, 0, 90]},
                {"id": "c", "location": [0, 0, 0], "size": None}]}
    out = K.apply_axis_scale(spec, {"X": 2.0, "Z": 0.5})
    assert out["dimensions"] == {"width": 2.0, "depth": None, "height": 1.0}
    assert out["components"][0]["location"] == [2, 2, 1.5] and out["components"][0]["size"] == [2, 2, 1.5]
    assert out["components"][1]["size"] == [1 * 1.0, 2 * 2.0, 3 * 0.5]   # rotated 90 deg about Z: local x -> world Y
    assert spec["components"][0]["location"] == [1, 2, 3]                 # input untouched


def test_components_in_region():
    spec = {"components": [
        {"id": "top", "location": [0, 0, 0.7], "size": [1.2, 0.6, 0.04]},
        {"id": "leg", "location": [0.5, -0.25, 0.3], "size": [0.05, 0.05, 0.6], "mirror": {"x": True}},
        {"id": "hull", "size": None}]}
    region = {"min": [-0.6, None, 0.65], "max": [-0.4, None, 0.75]}
    assert K.components_in_region(spec, region) == ["top"]
    low = {"min": [-0.55, None, 0.0], "max": [-0.45, None, 0.2]}
    assert K.components_in_region(spec, low) == ["leg"]                    # via x mirror image
    assert K.components_in_region(spec, {"min": [2, 2, 2], "max": [3, 3, 3]}) == []


def _one_view(ref, out, view="front", axis="-Y"):
    return M.compare_view(ref, out, calib(view, axis))


def test_extent_gate_bbox_fallback():
    ref = ellipse_mask(100, 100, 50, 50, 40, 20)
    box = rect_mask(100, 100, 30, 10, 70, 90)  # same bbox as the ellipse
    ok = M.extent_gate({"front": _one_view(ref, box)}, 0.12)
    assert ok["passed"] and ok["mode"] == "bbox_fallback" and ok["error"] < 0.05
    wide = rect_mask(100, 100, 10, 10, 90, 90)  # 2x too wide
    bad = M.extent_gate({"front": _one_view(ref, wide)}, 0.12)
    assert not bad["passed"] and bad["error"] == pytest.approx(1.0, abs=0.05)
    assert not M.extent_gate({}, 0.12)["passed"]


def test_describe_view_results_names_components():
    ref = rect_mask(100, 100, 30, 10, 70, 90)
    out = rect_mask(100, 100, 30, 10, 70, 60)  # missing the lower part
    spec = {"components": [{"id": "leg", "size": [0.1, 0.1, 0.3], "location": [0, 0, 0.1]},
                           {"id": "far", "size": [0.1, 0.1, 0.1], "location": [5, 0, 5]}]}
    info = K.describe_view_results({"front": _one_view(ref, out)}, spec)["front"]
    assert set(info) == {"bbox_delta_m", "largest_mismatch_region", "components_in_region"}
    assert info["largest_mismatch_region"]["world_bbox"]["min"][1] is None
    assert K.describe_view_results({"front": _one_view(ref, ref)}, spec)["front"]["components_in_region"] == []


def test_with_warnings():
    assert K.with_warnings("OK.", []) == "OK."
    assert K.with_warnings("Fits.", ["non_manifold (14 edges)"]) == "Fits; warnings: non_manifold (14 edges)"
