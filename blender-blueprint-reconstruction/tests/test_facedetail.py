"""Face detail check (numpy only)."""
import numpy as np
import pytest

from bbr.core import facedetail as F


def test_texels_per_px_counts_front_facing_faces_only():
    # one front quad of 0.01 x 0.01 m holding a quarter of a 1024 texture, one side quad ignored
    tpp = F.texels_per_px(np.array([0.25, 0.5]), np.array([1e-4, 1e-4]), np.array([1.0, 0.1]), 1024, 0.001)
    assert tpp == pytest.approx(np.sqrt(0.25 * 1024 ** 2 / 100.0))
    assert F.texels_per_px(np.array([0.1]), np.array([1e-4]), np.array([0.2]), 1024, 0.001) == 0.0


def test_color_error_over_the_region():
    ref = np.zeros((4, 4, 3)); model = np.zeros((4, 4, 3)); model[0, 0] = 1.0
    region = np.zeros((4, 4), bool); region[0, :2] = True
    assert F.color_error(ref, model, region) == pytest.approx(0.5)
    assert F.color_error(ref, model, np.zeros((4, 4), bool)) == 0.0


def test_warnings():
    block = {"layers": {"face": {"profile_relief_m": 0.012, "relief_applied": False, "texels_per_px": 0.9},
                        "ok": {"profile_relief_m": 0.012, "relief_applied": True, "texels_per_px": 3.0},
                        "flat": {"profile_relief_m": 0.001, "relief_applied": False, "texels_per_px": 2.0},
                        "broken": {"error": "x"}}}
    w = F.warnings(block)
    assert [x.split(" ", 2)[:2] for x in w] == [["face_ridge", "face"], ["face_texture", "face"]]
    assert F.warnings({}) == [] and F.warnings({"skipped": "no face layer"}) == []


def test_summary_lines():
    from bbr.core.summary import format_face_detail
    lines = format_face_detail({"layers": {"face": {"profile_relief_m": 0.0159, "relief_applied": True, "texels_per_px": 2.4,
                                                    "color_error": 0.051, "shells": ["visor"]}, "bad": {"error": "boom"}}})
    assert lines == ["face detail face: profile relief 16 mm on the midline, 2.4 texels/px, colour error 5.1%, glass visor",
                     "face detail bad: error boom"]
    assert format_face_detail(None) == [] and format_face_detail({"skipped": "x"}) == []
