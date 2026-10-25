"""bench/score.py on a fake run (no Blender): training excess, held-out IoU / PSNR, F-score / Chamfer, Markdown row;
plus the report formatting of the new measurement blocks."""
import json
import sys

import pytest

from bbr.core import attribution as A
from bbr.core import geomscore, imageio
from bbr.core.summary import format_summary
from conftest import SK, ellipse_mask, paint
from test_attribution import box

sys.path.insert(0, str(SK / "bench"))
import score  # noqa: E402


def make_run(tmp_path, shift_z=0.0, recon_mask_dx=0):
    case_dir, run = tmp_path / "case", tmp_path / "run"
    (case_dir / "holdout").mkdir(parents=True)
    (run / "holdout_renders").mkdir(parents=True)
    v, t = box(0, 0, 0.5, 0.4, 0.3, 1.0)
    geomscore.write_obj(case_dir / "gt.obj", v, t)
    geomscore.write_obj(run / "recon.obj", v + [0, 0, shift_z], t)
    for az in (0, 90):
        gt = paint(ellipse_mask(100, 100, 50, 50, 30, 20), (0.8, 0.2, 0.2), bg=(0, 0, 0))
        gt[..., 3] = ellipse_mask(100, 100, 50, 50, 30, 20)
        rec = paint(ellipse_mask(100, 100, 50, 50 + recon_mask_dx, 30, 20), (0.8, 0.2, 0.2), bg=(0, 0, 0))
        rec[..., 3] = ellipse_mask(100, 100, 50, 50 + recon_mask_dx, 30, 20)
        imageio.save_rgba(case_dir / "holdout" / f"az_{az:03d}.png", gt)
        imageio.save_rgba(run / "holdout_renders" / f"az_{az:03d}.png", rec)
    case = {"name": "unit", "kind": "synthetic", "view_spec": "view_spec.json", "scene_spec": "scene_spec.json",
            "gt_obj": "gt.obj", "holdout_dir": "holdout", "height_m": 1.0,
            "holdout_camera": {"azimuths": [0, 90]}, "perturbation": None}
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    report = {"status": "passed", "passed": True, "elapsed_s": 12.0,
              "reference_match": {"total_error": 0.04, "threshold": 0.05, "per_view": {
                  "front": {"iou": 0.95, "error": 0.05, "floor": 0.02, "excess_error": 0.03},
                  "side": {"iou": 0.97, "error": 0.03}}},
              "attribution": {"misattribution_max": {"object": "hand", "share": 0.08},
                              "occlusion_max": {"object": "shield", "share": 0.5},
                              "occluded_objects": ["shield"], "misattributed_objects": ["hand"],
                              "occlusion_warn": 0.3, "objects": {}, "views": ["front"]},
              "holdout": {"views": {"h45": {"iou": 0.9, "iou_nominal": 0.88, "camera": {"azimuth": 45, "elevation": 0},
                                            "best_azimuth": 50, "azimuth_searched": True, "color_error": 0.04,
                                            "pose_search": {"azimuth_step_deg": 5.0}}}, "mean_iou": 0.9},
              "part_iou": {"parts": {"horn": {"mean_iou": 0.5, "min_precision": 0.9}}}}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return case_dir, run, report


def test_identical_run_scores_perfect(tmp_path):
    case_dir, run, _ = make_run(tmp_path)
    res = score.score_case(case_dir, run, n_samples=8000)
    assert res["status"] == "passed" and res["train"]["mean_excess_error"] == pytest.approx(0.03)
    assert res["train"]["views"]["side"]["excess_error"] is None
    assert res["heldout"]["mean_iou"] == 1.0 and res["heldout"]["min_iou"] == 1.0 and res["heldout"]["mean_psnr"] == 99.0
    assert res["geometry"]["fscore@2%"] > 0.95 and res["geometry"]["chamfer"] < 0.02
    row = res["row"]
    assert row.startswith("| unit | passed | 4.0 | 3.0 | 100.0 / 100.0 |") and "occ:shield" in row and "mis:hand" in row
    assert row.count("|") == score.HEADER.count("|")


def test_offset_run_is_penalised(tmp_path):
    case_dir, run, _ = make_run(tmp_path, shift_z=0.05, recon_mask_dx=6)
    res = score.score_case(case_dir, run, n_samples=8000)
    assert res["geometry"]["fscore@1%"] < 0.75 and res["geometry"]["chamfer"] > 0.01
    assert res["geometry"]["bbox_center_offset_m"][2] == pytest.approx(0.05)
    assert res["heldout"]["mean_iou"] < 0.8


def test_missing_report_and_regression_case(tmp_path):
    case_dir, run, report = make_run(tmp_path)
    (run / "report.json").unlink()
    assert score.score_case(case_dir, run)["status"] == "missing_report"
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    case = json.loads((case_dir / "case.json").read_text())
    for key in ("gt_obj", "holdout_dir", "holdout_camera"):
        case.pop(key)
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    res = score.score_case(case_dir, run)
    assert res["heldout"] is None and res["geometry"] is None
    assert "| n/a | n/a | n/a | n/a |" in res["row"]


def test_warning_details_and_summary_lines():
    attr = {"objects": {"shield": {"occlusion_share": 0.38, "best_view": "right", "misattribution": 0.06,
                                   "top_confusion": {"object": "body", "share": 0.06}}},
            "occluded_objects": ["shield"], "misattributed_objects": ["shield"]}
    parts = {"parts": {"shield": {"min_precision": 0.68}}, "parts_outside_reference": ["shield"]}
    d = A.warning_details(attr, parts)
    assert d[0].startswith("occluded_part shield (38%") and "body" in d[1] and d[2].startswith("part_outside_reference shield")
    assert A.warning_details(None, None) == []
    _, _, report = make_run(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    text = format_summary({**report, "name": "x"})
    assert "Holdout h45 IoU : 90.0% (nominal 88.0%, azimuth 45 -> 50 (step 5), color 4.0%)" in text
    assert "Misattribution max : 8.0% (hand)" in text and "Occlusion share max : 50% (shield); flagged shield" in text
    assert "Part horn : on-reference 90% (min), region IoU 50%" in text and "label IoU" not in text


def test_ring_scores_in_row_and_flag(tmp_path):
    case_dir, run, report = make_run(tmp_path)
    report["hull"] = {"body": {"rings": {"slice_coherence": 0.0, "ring_pct": 3.0}},
                      "cap": {"rings": {"ring_pct": 0.5}}, "layer": {"cell_m": 0.001}}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    res = score.score_case(case_dir, run, n_samples=2000)
    assert res["rings"]["max_pct"] == 3.0 and res["rings"]["flagged"] == ["body"]
    assert "| 3.0 |" in res["row"] and "ring:body" in res["row"] and res["row"].count("|") == score.HEADER.count("|")
