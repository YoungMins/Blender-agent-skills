"""Dimension tolerance by reference quality, failure attribution, parts-only result lines, warning noise (numpy only)."""
import numpy as np
import pytest

from bbr.core import dimcheck as DC, measure, summary
from bbr.core.calibration import ViewCalibration


def box(lo, hi):
    return np.array(lo, float), np.array(hi, float)


# --------------------------------------------------------------------------- tolerance
def test_tolerance_follows_the_reference_quality():
    assert DC.resolve_tolerance(None, {"orthographic": True, "synthetic": True}, {}) == (0.03, "synthetic / blueprint references")
    assert DC.resolve_tolerance(None, {"orthographic": True}, {})[0] == 0.03                       # a blueprint
    assert DC.resolve_tolerance(None, {"orthographic": True, "ai_generated": True}, {})[0] == 0.05
    assert DC.resolve_tolerance(None, {"orthographic": False}, {})[0] == 0.05                      # a perspective sheet
    assert DC.resolve_tolerance(None, {"orthographic": True}, {"front": "sheet", "side": "sheet"})[0] == 0.05   # cut from one sheet
    assert DC.resolve_tolerance(None, {"ai_generated": False}, {"front": "sheet"})[0] == 0.03      # an explicit statement wins
    assert DC.resolve_tolerance(0.04, {"ai_generated": True}, {}) == (0.04, "workflow.dimension_tolerance")


def test_the_workflow_key_reaches_the_config_and_the_schema():
    from bbr.core.config import PipelineConfig
    from bbr.core.spec import validate_scene_spec
    assert PipelineConfig.from_dict({"dimension_tolerance": 0.06}).dimension_tolerance == 0.06
    assert PipelineConfig().dimension_tolerance is None
    spec = {"spec_version": "0.1", "name": "x", "components": [{"id": "a", "type": "box", "size": [1, 1, 1]}],
            "workflow": {"dimension_tolerance": 0.05}}
    assert validate_scene_spec(spec) == []
    spec["workflow"]["dimension_tolerance"] = "five"
    assert validate_scene_spec(spec)


def test_draft_view_spec_marks_a_sheet_as_ai_generated(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from PIL import Image
    sk = Path(__file__).resolve().parents[1]
    refs = tmp_path / "references"
    refs.mkdir()
    for name in ("front", "side"):
        img = np.zeros((60, 40, 3), np.uint8) + np.array([0, 255, 0], np.uint8)
        img[10:50, 10:30] = (200, 40, 40)
        Image.fromarray(img).save(refs / f"{name}.png")
    (refs / "sheet_source.json").write_text(json.dumps({"names": ["front", "side"]}))
    subprocess.run([sys.executable, str(sk / "scripts" / "prepare_references.py"), "draft-view-spec", "--dir", str(refs), "--height", "0.5"],
                   check=True, capture_output=True)
    rq = json.loads((refs / "view_spec.json").read_text())["reference_quality"]
    assert rq["ai_generated"] is True and rq["orthographic"] is True


# --------------------------------------------------------------------------- attribution
BOXES = {"body": box((-0.15, -0.2, 0.0), (0.15, 0.2, 0.55)),
         "fin_r": box((-0.1855, -0.05, 0.2), (-0.12, 0.05, 0.3)),
         "fin_l": box((0.12, -0.05, 0.2), (0.17, 0.05, 0.3)),            # the left fin was shrunk less (and mirrors worse)
         "horn_r": box((-0.12, -0.03, 0.5), (-0.08, 0.03, 0.62))}
REFERENCE = {0: (-0.2, 0.2), 1: (-0.22, 0.22), 2: (0.0, 0.62)}


def test_a_failed_width_names_the_part_that_sets_the_short_edge():
    width = (0.17 + 0.1855) / 0.4 - 1.0
    out = DC.attribute({"width": width, "depth": 0.0, "height": 0.0}, 0.05 - 0.04, BOXES, REFERENCE, {"body"})
    assert len(out) == 1
    a = out[0]
    assert a["dimension"] == "width" and a["part"] == "fin_l" and a["side"] == "+X" and a["direction"] == "short"
    assert a["off_m"] == pytest.approx(0.03, abs=1e-6)
    assert "model extreme set by fin_l" in a["text"] and "30 mm short of the reference edge" in a["text"]
    assert "consider fit_locked on fin_l position" in a["text"] and "(limit 1%)" in a["text"]


def test_the_message_has_the_shape_the_agent_needs():
    boxes = dict(BOXES, fin_r=box((-0.186, -0.05, 0.2), (-0.12, 0.05, 0.3)), fin_l=box((0.12, -0.05, 0.2), (0.1995, 0.05, 0.3)))
    out = DC.attribute({"width": -0.037, "depth": 0.0, "height": 0.0}, 0.03, boxes, REFERENCE)
    assert len(out) == 1 and out[0]["part"] == "fin_r" and out[0]["side"] == "-X" and out[0]["off_m"] == pytest.approx(0.014)
    hard = {"dimensions": {"passed": False, "relative_error": {"width": -0.037, "depth": 0.0, "height": 0.0}, "tolerance": 0.03,
                           "limiting": out}}
    text = summary.failed_checks(hard, {"passed": True})[0]
    assert text == ("dimensions (width -3.7% (limit 3%); model extreme set by fin_r (-X side), 14 mm short of the reference edge; "
                    "consider fit_locked on fin_r position)")


def test_a_hull_that_sets_the_edge_points_at_the_silhouette_not_fit_locked():
    boxes = {"body": box((-0.18, -0.2, 0.0), (0.2, 0.2, 0.6))}
    out = DC.attribute({"width": -0.045}, 0.03, boxes, {0: (-0.2, 0.2)}, {"body"})
    assert out[0]["part"] == "body" and "fit_locked" not in out[0]["text"] and "silhouette" in out[0]["text"]


def test_a_part_sticking_out_is_named_for_an_oversized_model():
    boxes = dict(BOXES, tail=box((-0.05, 0.1, 0.0), (0.05, 0.27, 0.1)))
    out = DC.attribute({"depth": 0.05 / 0.44 * 1.0 + 0.0}, 0.03, boxes, REFERENCE)
    assert out and out[0]["part"] == "tail" and out[0]["direction"] == "over" and out[0]["side"] == "+Y"
    assert "beyond the reference edge" in out[0]["text"]


def test_no_attribution_inside_the_tolerance_or_without_parts():
    assert DC.attribute({"width": -0.02}, 0.03, BOXES, REFERENCE) == []
    assert DC.attribute({"width": -0.2}, 0.03, {}, REFERENCE) == []


def test_reference_extents_from_masks():
    calibs = {"front": ViewCalibration("front", "-Y", (100, 120), 0.01, (50.0, 100.0)),
              "side": ViewCalibration("side", "-X", (100, 120), 0.01, (50.0, 100.0))}
    m = np.zeros((120, 100), bool)
    m[20:100, 30:70] = True                                                  # x -0.2 .. 0.2, z 0 .. 0.8
    ext = DC.reference_extents({"front": m, "side": m}, calibs)
    assert ext[0] == pytest.approx((-0.2, 0.2)) and ext[2] == pytest.approx((0.0, 0.8))
    assert ext[1] == pytest.approx((-0.2, 0.2))


def test_parts_only_lines_and_next_action():
    dim = {"passed": False, "relative_error": {"width": -0.037, "depth": 0.0, "height": 0.0}, "tolerance": 0.05,
           "target": {"width": 0.4, "depth": 0.44, "height": 0.6},
           "limiting": [{"dimension": "width", "text": "width -3.7% (limit 5%); model extreme set by fin_r (-X side), 14 mm short of the reference edge; consider fit_locked on fin_r position"}]}
    report = {"name": "d", "status": "stopped_after_secondary", "dimension_check": dim, "outputs": {},
              "floating_parts": [{"part": "wing_l", "gap_m": 0.02, "tol_m": 0.01, "nearest": "body"}], "part_gaps": []}
    text = summary.format_summary(report)
    assert "Dimensions : width -3.7%" in text and "FAILED" in text and "model extreme set by fin_r" in text
    assert "WARNING floating_part: wing_l" in text
    assert "next: parts-only checks found: dimensions (width -3.7% (limit 5%)" in text and "floating_part wing_l" in text
    ok = dict(report, dimension_check={"passed": True, "relative_error": {"width": 0.0}, "target": {"width": 0.4}, "tolerance": 0.03},
              floating_parts=[])
    assert "next: parts-only checks passed" in summary.format_summary(ok)


# --------------------------------------------------------------------------- warning noise
def test_base_tip_swap_is_not_reported_for_a_stubby_shape():
    stub = {"front": {"base": [100, 100], "tip": [100, 130], "radius_px": 20.0},               # axis 30 px, width 40: a ball
            "side": {"base": [100, 130], "tip": [100, 100], "radius_px": 20.0}}                # the other way round
    # an upright view's z comes from calibs: build front / side calibrations
    calibs = {"front": ViewCalibration("front", "-Y", (200, 200), 0.001, (100.0, 190.0)),
              "side": ViewCalibration("side", "-X", (200, 200), 0.001, (100.0, 190.0))}
    assert measure._elongation_of(stub) < measure.SYMMETRIC_ELONGATION
    assert not any("swapped between views" in w for w in measure.check_anchor_views(stub, calibs)["warnings"])
    rod = {"front": {"base": [100, 100], "tip": [100, 30], "radius_px": 8.0},
           "side": {"base": [100, 30], "tip": [100, 100], "radius_px": 8.0}}
    assert measure._elongation_of(rod) > measure.SYMMETRIC_ELONGATION
    assert any("swapped between views" in w for w in measure.check_anchor_views(rod, calibs)["warnings"])


def test_a_mirror_partner_without_a_face_on_anchor_is_a_quiet_case():
    from bbr import BBRError
    from bbr.core import partfit
    assert issubclass(partfit.MissingFaceView, BBRError)


def test_check_says_a_single_part_may_differ_when_the_views_differ_in_one_place():
    from bbr.core import consistency as C
    calibs = {"front": ViewCalibration("front", "-Y", (200, 300), 0.002, (100.0, 280.0)),
              "back": ViewCalibration("back", "+Y", (200, 300), 0.002, (100.0, 280.0))}
    base = np.zeros((300, 200), bool)
    base[40:280, 60:140] = True
    with_tail = base.copy()
    with_tail[230:280, 140:190] = True                                     # one extra blob: a tail swung out in one view
    res = C.analyze({"front": base, "back": with_tail}, calibs, {"front": "-Y", "back": "+Y"}, {"front": ["carve"], "back": ["color"]})
    assert res["pairs"][0]["single_part_differs"]
    assert any("a single part may differ (e.g. tail pose)" in f["detail"] for f in res["flags"] if f["type"] == "missing_accessory")
