"""Whole-model review: per-label colour errors, scorecards, ranked issues (numpy only)."""
import numpy as np
import pytest

from bbr.core import review as R


def test_label_color_errors_per_label_and_body():
    ref = np.zeros((20, 20, 3)); model = np.zeros((20, 20, 3))
    ref_mask = np.zeros((20, 20), bool); ref_mask[2:18, 2:18] = True
    hat = np.zeros((20, 20), bool); hat[2:10, 2:18] = True
    model[2:10, 2:18] = 0.5                                     # the hat is off by 0.5, the body exact
    out = R.label_color_errors(ref, model, ref_mask, ref_mask, {"hat": hat, "_x": hat, "tiny": hat & (np.arange(20) < 3)})
    assert out["hat"] == {"error": 0.5, "px": 128}
    assert out["body"]["error"] == 0.0 and out["body"]["px"] == 128
    assert "tiny" not in out and "_x" not in out


def test_build_and_rank():
    report = {"part_iou": {"parts": {"horn": {"mean_label_iou": 0.4}}},
              "attribution": {"objects": {"horn": {"misattribution": 0.05, "occlusion_share": 0.5}}},
              "hull": {"body": {"texture": {"inpainted_share": 0.2}}},
              "face_detail": {"layers": {"face": {"profile_relief_m": 0.01, "relief_applied": False}}},
              "reference_match": {"per_view": {"side": {"excess_error": 0.05, "components_in_region": ["horn"]}}}}
    colors = {"front": {"horn": {"error": 0.16, "px": 400}, "body": {"error": 0.03, "px": 9000}}}
    cards = R.build({"body": "body", "horn": "horn", "face": "face"}, report, colors, {"face": 0.8})
    assert cards["horn"] == {"color_error": {"front": 0.16}, "label_iou": 0.4, "misattribution": 0.05, "occlusion": 0.5,
                             "silhouette_excess": {"side": 0.05}}
    issues = R.rank(cards)
    assert [(i["component"], i["check"]) for i in issues] == [
        ("face", "face_ridge"), ("horn", "silhouette_excess"), ("horn", "color_error"), ("horn", "occlusion"),
        ("horn", "label_iou"), ("face", "texels_per_px")]
    assert issues[2]["view"] == "front" and issues[2]["severity"] == pytest.approx(2.0)
    assert R.format_issue(issues[2]).startswith("horn color_error (front) 0.16 (limit 0.08): its colour is off")


def test_summary_review_lines_and_next():
    from bbr.core.summary import format_review, next_action
    issues = [{"component": f"p{i}", "check": "occlusion", "view": None, "value": 0.5, "threshold": 0.3, "severity": 1.7,
               "action": "hidden"} for i in range(10)]
    lines = format_review({"issues": issues})
    assert lines[0] == "review 1: p0 occlusion 0.5 (limit 0.3): hidden" and lines[-1] == "review: 2 more in report.review.issues"
    assert "write a plan" in next_action({"passed": True, "status": "passed", "review": {"issues": issues}}, [])
    assert "no review issue" in next_action({"passed": True, "status": "passed", "review": {"issues": []}}, [])
    assert next_action({"passed": False, "status": "failed_validation"}, ["x"]).startswith("next: FAILED")
