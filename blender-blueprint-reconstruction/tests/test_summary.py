"""format_summary regression: color-only views carry just ``color_error`` in per_view."""
from bbr.core.summary import format_summary


def _report():
    return {
        "name": "sd_warrior", "status": "passed", "passed": True, "elapsed_s": 1.0, "outputs": {},
        "reference_match": {
            "total_error": 0.05, "threshold": 0.08,
            "per_view": {
                "front": {"error": 0.013, "error_pct": 1.3, "color_error": 0.047},
                "right": {"color_error": 0.07},                      # roles ["color"]: no silhouette error
            }},
        "hull": {"body": {"texture": {"views": {"front": 0.5, "right": 0.5}, "blend_share": 0.0, "inpainted_share": 0.0,
                                       "conflict_ratio": 0.0, "background_leak_ratio": 0.0}}},
    }


def test_format_summary_handles_a_color_only_view():
    text = format_summary(_report())
    assert "Front Error" in text and "Right Color Error" in text and "Total Error : 5.0%" in text
    assert "Right Error" not in text                                    # no silhouette line for the color-only view
    assert "texture body:" in text


def test_format_summary_without_reference_match_does_not_crash():
    assert "status: error" in format_summary({"name": "x", "status": "error", "error": "boom"})


def test_part_summary_names_the_label_iou_and_the_region_iou():
    from bbr.core.summary import merge_label_iou
    report = _report()
    report["part_iou"] = {"parts": {"staff": {"mean_iou": 0.12, "min_precision": 0.85, "views": {}},
                                    "hat": {"mean_iou": 0.5, "min_precision": 0.9, "views": {}}}}
    merge_label_iou(report["part_iou"], {"staff": {"iou": {"front": {"iou": 0.5, "iou_vis": 0.6}, "side": {"iou": 0.5, "iou_vis": 0.8}}}})
    assert report["part_iou"]["parts"]["staff"]["mean_label_iou"] == 0.7
    assert "mean_label_iou" not in report["part_iou"]["parts"]["hat"]          # no partfit entry: nothing invented
    lines = [ln for ln in format_summary(report).splitlines() if ln.startswith("Part ")]
    staff = next(ln for ln in lines if ln.startswith("Part staff"))
    assert "label IoU 70% (front 60 side 80)" in staff and "region IoU 12%" in staff and "not a defect" in staff
    assert "label IoU" not in next(ln for ln in lines if ln.startswith("Part hat"))


def test_failed_checks_name_the_number_that_failed():
    from bbr.core.summary import failed_checks
    hard = {"dimensions": {"passed": False, "relative_error": {"width": -0.097, "depth": 0.01, "height": 0.0}, "tolerance": 0.03},
            "scale": {"passed": True}, "materials": {"passed": False, "without_material": ["horn"]}}
    assert failed_checks(hard, {"passed": True}) == ["dimensions (width -9.7% (limit 3%))", "materials (no material: horn)"]
    match = {"passed": False, "total_error": 0.091, "threshold": 0.08, "side_error": 0.12, "front_error": 0.03,
             "per_view": {"front": {}, "side": {}}}
    assert failed_checks({}, match) == ["reference_match (total error 9.1% > threshold 8.0%; side 12.0%)"]
    assert failed_checks({"dimensions": {"passed": True}}, {"passed": True}) == []


def test_summary_validation_line_and_total_under_threshold_failure():
    """The fox case: total error 6.7% < 8% but the run failed on the model width."""
    report = _report()
    report.update(status="failed_validation", passed=False, validation={
        "geometry": {"checks": {"dimensions": {"passed": False, "relative_error": {"width": -0.097, "depth": 0.0, "height": 0.0}, "tolerance": 0.03}}},
        "reference_match": {"passed": True, "total_error": 0.067, "threshold": 0.08}})
    text = format_summary(report)
    assert "validation: FAILED: dimensions (width -9.7% (limit 3%))" in text
    assert "Total Error : 5.0%" in text
