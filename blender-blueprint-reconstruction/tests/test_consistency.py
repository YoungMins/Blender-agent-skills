"""Reference consistency gate and sheet-mode calibration on synthetic silhouettes."""
import json
import sys

import numpy as np
import pytest

from bbr.core import consistency as C
from bbr.core import imageio
from bbr.core.calibration import ViewCalibration, calibrate_views
from bbr.core.config import PipelineConfig
from bbr.core.spec import parse_view_spec
from conftest import SK, ellipse_mask, paint, rect_mask

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402

H, W = 320, 240
MPP = 0.002          # 2 mm per pixel -> a 0.6 m figure
FEET = 300           # sole row
CX = 100


def figure(nose=True, accessory=False, dy=0, scale=1.0):
    """Asymmetric standing figure (head with a nose on the image right, one raised arm); optional back disc."""
    m = np.zeros((H, W), bool)
    m |= rect_mask(H, W, CX - 30, 110, CX + 30, 220)                 # torso
    m |= rect_mask(H, W, CX - 28, 220, CX - 6, FEET) | rect_mask(H, W, CX + 6, 220, CX + 28, FEET)  # legs
    m |= ellipse_mask(H, W, 80, CX, 30, 30)                          # head
    m |= rect_mask(H, W, CX + 30, 110, CX + 70, 135)                 # arm to the right
    if nose:
        m |= rect_mask(H, W, CX + 30, 74, CX + 46, 88)
    if accessory:
        m |= ellipse_mask(H, W, 150, CX - 55, 30, 12)                # disc on the image left
    if dy:
        m = np.roll(m, dy, axis=0)
    if scale != 1.0:                                                  # scale about the foot centre
        yy, xx = np.mgrid[0:H, 0:W]
        sy = np.clip(np.floor(FEET + (yy + 0.5 - FEET) / scale).astype(int), 0, H - 1)
        sx = np.clip(np.floor(CX + (xx + 0.5 - CX) / scale).astype(int), 0, W - 1)
        m = m[sy, sx]
    return m


def calib(view, axis, flip=False):
    return ViewCalibration(view, axis, (W, H), MPP, ((W - CX) if flip else CX, float(FEET)))


def views(front=None, back=None, side=None, right=None):
    """Shared-scale views; opposite views are mirror images of the same object."""
    front = figure() if front is None else front
    masks = {"front": front, "back": front[:, ::-1] if back is None else back}
    axes = {"front": "-Y", "back": "+Y"}
    calibs = {"front": calib("front", "-Y"), "back": calib("back", "+Y", flip=True)}
    if side is not None:
        masks["side"], axes["side"], calibs["side"] = side, "-X", calib("side", "-X")
        masks["right"] = side[:, ::-1] if right is None else right
        axes["right"], calibs["right"] = "+X", calib("right", "+X", flip=True)
    return masks, calibs, axes


def types(result):
    return {(f["type"], tuple(f["views"])) for f in result["flags"]}


def test_identical_mirror_pair_is_perfect():
    res = C.analyze(*views(side=figure(nose=False)))
    assert res["pairs"][0]["iou_raw"] > 0.99 and res["pairs"][0]["floor"] < 0.01
    assert res["passed"] and not res["flags"]
    assert all(v["verdict"] == "ok" for v in res["views"].values())


def test_scale_drift_detected():
    masks, calibs, axes = views(back=figure(scale=1.03)[:, ::-1], side=figure(nose=False))
    res = C.analyze(masks, calibs, axes)
    assert ("scale_drift", ("back",)) in types(res)
    assert res["pairs"][0]["iou_aligned"] > res["pairs"][0]["iou_raw"]
    assert not res["passed"]


def test_vertical_shift_detected():
    masks, calibs, axes = views(back=figure(dy=10)[:, ::-1], side=figure(nose=False))
    res = C.analyze(masks, calibs, axes)
    assert ("vertical_shift", ("back",)) in types(res)


def thin_legged(legs=10):
    """Figure whose legs are thinner than the core opening (2 * 3 % of the height): the opened core loses its feet."""
    m = figure()
    m[220:, :] = False
    m |= rect_mask(H, W, CX - 12 - legs, 220, CX - 12, FEET) | rect_mask(H, W, CX + 12, 220, CX + 12 + legs, FEET)
    return m


def test_thin_legs_do_not_fake_scale_drift_when_bbox_heights_agree():
    """The opened core of front/back (thin legs) is shorter than the side's, yet all three span the same rows."""
    masks, calibs, axes = views(front=thin_legged(6), side=figure(nose=False))
    res = C.analyze(masks, calibs, axes)
    cores = [v["core_height"] for v in res["views"].values()]
    assert max(cores) / min(cores) - 1.0 > 0.05                       # the core estimate alone would raise an alarm
    assert res["bbox_heights_agree"]
    assert not {f["type"] for f in res["flags"]} & {"scale_drift", "vertical_shift"}
    assert all(abs(v["bbox_height"] / res["views"]["front"]["bbox_height"] - 1) < 0.015 for v in res["views"].values())


def test_real_height_difference_still_flagged_with_thin_legs():
    masks, calibs, axes = views(front=thin_legged(6), back=thin_legged(6)[:, ::-1], side=figure(nose=False, scale=1.08))
    res = C.analyze(masks, calibs, axes)
    assert not res["bbox_heights_agree"]
    assert any(f["type"] == "scale_drift" and f["views"] == ["side"] for f in res["flags"])


def test_ignore_flags_and_roles_override_are_recorded():
    masks, calibs, axes = views(back=figure(scale=1.03)[:, ::-1], side=figure(nose=False))
    res = C.analyze(masks, calibs, axes, ignore_flags=["scale_drift", "vertical_shift:back", "vertical_shift"])
    assert not any(f["type"] in ("scale_drift", "vertical_shift") for f in res["flags"])
    assert {f["type"] for f in res["ignored_flags"]} >= {"scale_drift"}
    assert "ignored (judged false alarm) scale_drift" in C.format_consistency(res)
    only_view = C.analyze(masks, calibs, axes, ignore_flags=["scale_drift:front"])
    assert ("scale_drift", ("back",)) in types(only_view)          # "type:view" only silences that view
    roles = {"front": ["carve", "color", "validate"], "back": ["color", "validate"]}
    bad = C.analyze(*views(back=figure(accessory=True)[:, ::-1]), roles, C.ConsistencyThresholds(color_validate_min_iou=0.999))
    assert "back" in bad["roles_to_change"] or "front" in bad["roles_to_change"]
    ok = C.analyze(*views(back=figure(accessory=True)[:, ::-1]), roles, C.ConsistencyThresholds(color_validate_min_iou=0.999),
                   roles_override={"front": roles["front"], "back": roles["back"]})
    assert ok["roles_to_change"] == [] and ok["overrides"].keys() == {"front", "back"}
    assert ok["views"]["back"]["verdict"] == "override"


def test_missing_accessory_detected():
    masks, calibs, axes = views(front=figure(accessory=True), back=figure()[:, ::-1])
    res = C.analyze(masks, calibs, axes)
    assert ("missing_accessory", ("back",)) in types(res)
    flag = next(f for f in res["flags"] if f["type"] == "missing_accessory")
    assert flag["value"] > 0.01


def test_mirrored_side_detected():
    side = figure()
    masks, calibs, axes = views(side=side, right=side)  # right is NOT mirrored -> faces the wrong way
    calibs["right"] = calib("right", "+X")
    res = C.analyze(masks, calibs, axes)
    assert ("mirrored_view", ("right",)) in types(res)
    assert not res["passed"]


def test_roles_follow_thresholds():
    thr = C.ConsistencyThresholds()
    roles = ["carve", "color", "validate"]
    assert C.recommend_roles(0.96, roles, thr) == roles
    assert C.recommend_roles(0.94, roles, thr) == ["color", "validate"]
    assert C.recommend_roles(0.90, roles, thr) == []
    assert C.recommend_roles(0.90, ["color", "preview"], thr) == ["preview"]
    assert C.recommend_roles(0.90, roles, thr, blamed=False) == roles
    assert C.recommend_roles(None, roles, thr) == roles
    strict = C.ConsistencyThresholds(carve_min_iou=0.99)
    assert C.recommend_roles(0.96, roles, strict) == ["color", "validate"]


def test_failed_pair_drops_the_lower_priority_view_only():
    side = figure()
    right = figure(accessory=True, nose=False)[:, ::-1]
    roles = {"front": ["carve"], "back": ["color"], "side": ["carve", "validate"], "right": ["color"]}
    masks, calibs, axes = views(side=side, right=right)
    res = C.analyze(masks, calibs, axes, roles, C.ConsistencyThresholds(color_validate_min_iou=0.999))
    assert res["views"]["right"]["verdict"] == "drop" and res["views"]["right"]["recommended_roles"] == []
    assert res["views"]["side"]["recommended_roles"] == ["carve", "validate"]


def test_floor_and_excess():
    masks, calibs, axes = views(back=figure(accessory=True)[:, ::-1])
    res = C.analyze(masks, calibs, axes, {"back": ["validate"]})
    floor = res["views"]["back"]["floor"]
    assert 0.0 < floor < 0.3 and floor == pytest.approx(1 - res["pairs"][0]["iou_raw"], abs=1e-4)
    C.add_excess(res, {"back": floor + 0.02})
    assert res["views"]["back"]["excess_error"] == pytest.approx(0.02, abs=1e-4)
    text = C.format_consistency(res)
    assert "front<->back" in text and "excess" in text


def test_large_images_are_downscaled_fast():
    big = np.kron(figure(), np.ones((4, 4), bool))                   # 1280 x 960
    cal = ViewCalibration("front", "-Y", big.shape[::-1], MPP / 4, (CX * 4, FEET * 4))
    cal_b = ViewCalibration("back", "+Y", big.shape[::-1], MPP / 4, ((W - CX) * 4, FEET * 4))
    res = C.analyze({"front": big, "back": big[:, ::-1]}, {"front": cal, "back": cal_b}, {"front": "-Y", "back": "+Y"})
    assert res["pairs"][0]["iou_raw"] > 0.99


# --------------------------------------------------------------------------- sheet calibration
def spec_for(modes, **extra):
    data = {"spec_version": "0.1", "name": "t", "views": {
        k: {"path": f"{k}.png", "camera_axis": a, "calibration": {"mode": m}}
        for (k, a), m in zip((("front", "-Y"), ("side", "-X"), ("back", "+Y")), modes)},
        "known_dimensions": {"height": 0.6}}
    data.update(extra)
    return parse_view_spec(data, ".")


def sheet_masks():
    front = figure()
    front[2:12, CX - 2:CX + 2] = True                                # thin horn: front bbox 3 % taller
    side = figure(nose=False)
    side[250:256, 150:220] = True                                    # sword poking out sideways, low
    return {"front": front, "side": side, "back": figure()[:, ::-1]}


def test_bbox_mode_drifts_sheet_mode_shares_scale():
    masks = sheet_masks()
    bbox, _ = calibrate_views(spec_for(["silhouette_bbox"] * 3), masks)
    assert bbox["front"].m_per_px != pytest.approx(bbox["back"].m_per_px, rel=1e-3)
    calibs, info = calibrate_views(spec_for(["sheet"] * 3), masks)
    assert len({round(c.m_per_px, 9) for c in calibs.values()}) == 1
    assert len({c.origin_px[1] for c in calibs.values()}) == 1 and calibs["front"].origin_px[1] == FEET
    assert info["sheet"]["views"] == ["front", "side", "back"]
    # the horizontal origin is the body core, not the bbox centre dragged out by the sword
    assert calibs["side"].origin_px[0] == pytest.approx(CX, abs=1.5)
    assert bbox["side"].origin_px[0] > CX + 5


def test_top_level_sheet_mode_is_a_default_and_views_can_override():
    spec = spec_for(["silhouette_bbox"] * 3, calibration={"mode": "sheet"})
    assert {v.calibration.mode for v in spec.views.values()} == {"silhouette_bbox"}   # explicit view setting wins
    data = {"spec_version": "0.1", "views": {"front": {"path": "f.png"}, "back": {"path": "b.png", "calibration": {"mode": "silhouette_bbox"}}},
            "calibration": {"mode": "sheet"}}
    spec = parse_view_spec(data, ".")
    assert spec.views["front"].calibration.mode == "sheet" and spec.views["back"].calibration.mode == "silhouette_bbox"


def test_pipeline_config_gate_values():
    assert PipelineConfig().consistency_gate == "warn"
    assert PipelineConfig.from_dict({"consistency_gate": "strict"}).consistency_gate == "strict"
    with pytest.raises(Exception):
        PipelineConfig.from_dict({"consistency_gate": "fail"})


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def refs(tmp_path):
    f = figure()
    for name, m in (("front", f), ("back", f[:, ::-1]), ("side", figure(nose=False)),
                    ("right", figure(nose=False, accessory=True)[:, ::-1])):
        imageio.save_rgba(tmp_path / f"{name}.png", paint(m, (0.9, 0.9, 0.9)))
    return tmp_path


def test_check_cli_writes_overlays(refs, capsys):
    assert cli.main(["draft-view-spec", "--dir", str(refs), "--height", "0.6", "--side-axis", "-X"]) == 0
    spec = refs / "view_spec.json"
    out = refs / "xor"
    assert cli.main(["check", "--view-spec", str(spec), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "front<->back" in text and "side<->right" in text and "missing_accessory" in text
    assert (out / "xor_front_back.png").is_file() and (out / "xor_side_right.png").is_file()
    assert cli.main(["check", "--view-spec", str(spec), "--out-dir", str(out), "--strict"]) in (0, 2)
    capsys.readouterr()
    assert cli.main(["check", "--view-spec", str(spec), "--out-dir", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pairs"]
    assert cli.main(["check", "--view-spec", str(spec), "--out-dir", str(out), "--json", "--ignore-flag", "missing_accessory",
                     "--roles-override", "back=color,validate"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert not any(f["type"] == "missing_accessory" for f in res["flags"]) and res["ignored_flags"]
    assert res["overrides"] == {"back": ["color", "validate"]}


def test_split_sheet_marks_views_as_one_sheet(tmp_path):
    h, w = 200, 600
    m = ellipse_mask(h, w, 100, 100, 80, 40) | ellipse_mask(h, w, 100, 300, 80, 30) | ellipse_mask(h, w, 100, 500, 80, 40)
    sheet = tmp_path / "sheet.png"
    imageio.save_rgba(sheet, paint(m, (0.9, 0.9, 0.9)))
    out = tmp_path / "refs"
    assert cli.main(["split-sheet", "--image", str(sheet), "--out-dir", str(out), "--names", "front,side,back"]) == 0
    assert cli.main(["draft-view-spec", "--dir", str(out), "--height", "1.0"]) == 0
    spec = json.loads((out / "view_spec.json").read_text())
    assert all(v["calibration"] == {"mode": "sheet"} for v in spec["views"].values())


def test_summary_prints_consistency_block_and_excess():
    from bbr.core.summary import format_summary
    masks, calibs, axes = views(back=figure(accessory=True)[:, ::-1])
    res = C.analyze(masks, calibs, axes, {"back": ["validate"]})
    C.add_excess(res, {"back": 0.09})
    floor = res["views"]["back"]["floor"]
    per_view = {"back": {"error": 0.09, "floor": floor, "excess_error": 0.09 - floor}}
    text = format_summary({"name": "t", "status": "passed", "reference_match": {"per_view": per_view}, "consistency": res})
    assert "excess" in text and "Consistency (carve" in text and "front<->back" in text
