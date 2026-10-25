"""Symmetry explanation with labels, anchors --from-px, view-pixel rulers on the label previews."""
import json
import sys

import numpy as np
import pytest

from bbr.core import imageio, symmetry as S
from bbr.core.calibration import ViewCalibration
from conftest import SK, ellipse_mask, paint, rect_mask

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402

H, W = 120, 160
FRONT = ViewCalibration("front", "-Y", (W, H), 0.01, (80.0, 115.0))


def penguin(h=H, w=W):
    """A symmetric body, a posed flipper (image right) and a held spatula (further right); a 3 px spike on the left keeps
    the silhouette's bbox centred on the body, as the body-core calibration of a real sheet does."""
    body = ellipse_mask(h, w, 60, 80, 40, 30)
    flipper = rect_mask(h, w, 110, 50, 130, 70)
    spatula = rect_mask(h, w, 132, 30, 144, 80)
    spike = rect_mask(h, w, 16, 55, 19, 65)
    return body | flipper | spatula | spike, {"spatula_head": spatula, "flipper_r": flipper, "body": body}


def test_the_explanation_names_the_parts_whose_blanking_explains_the_low_iou():
    mask, labels = penguin()
    raw = S.front_symmetry(mask, FRONT)
    assert not raw["suggest_x"] and raw["suggested_blank"] == []
    assert "pass labels" in raw["explanation"]
    declared = S.front_symmetry(mask, FRONT, labels, ["spatula_*"])
    assert declared["iou"] < declared["iou_blanked"] < S.SUGGEST_MIN_IOU           # the spatula alone is not the whole story
    assert declared["suggested_blank"] == ["flipper_r"] and declared["iou_suggested"] > S.SUGGEST_MIN_IOU and declared["suggest_x"]
    text = declared["explanation"]
    assert text.startswith(f"{declared['iou']:.3f} raw; {declared['iou_blanked']:.3f} with spatula_* blanked; ")
    assert "with spatula_*, flipper_r blanked" in text and "suggest symmetry.x = true" in text and "asymmetric_parts" in text
    bare = S.front_symmetry(mask, FRONT, labels)                                   # nothing declared: the search finds both
    assert set(bare["suggested_blank"]) == {"spatula_head", "flipper_r"} and bare["iou_blanked"] is None


def test_an_asymmetric_body_is_not_explained_away():
    body = ellipse_mask(H, W, 60, 80, 40, 30) | ellipse_mask(H, W, 40, 120, 20, 20)    # a lump that no label covers
    sym = S.front_symmetry(body, FRONT, {"arm": rect_mask(H, W, 0, 0, 5, 5), "body": body})
    assert not sym["suggest_x"] and sym["suggested_blank"] == []
    assert "the body itself is asymmetric" in sym["explanation"]


@pytest.fixture
def penguin_dir(tmp_path):
    folder = tmp_path / "pen" / "references"
    folder.mkdir(parents=True)
    mask, labels = penguin(200, 200)
    shift = 20                                                                     # keep the body centred in the 200 px image
    mask = np.roll(mask, shift, axis=1)
    for key in ("front", "side", "back"):
        shown = {"front": mask, "back": mask[:, ::-1], "side": ellipse_mask(200, 200, 60, 100, 40, 30)}[key]
        imageio.save_rgba(folder / f"{key}.png", paint(shown, (0.9, 0.5, 0.2)))
    regions = tmp_path / "regions" / "front"
    regions.mkdir(parents=True)
    id_map = np.zeros((200, 200), np.int32)
    id_map[np.roll(labels["flipper_r"], shift, axis=1)] = 1
    id_map[np.roll(labels["spatula_head"], shift, axis=1)] = 2
    np.save(regions / "id_map.npy", id_map)
    (tmp_path / "labels.json").write_text(json.dumps({"_regions": "regions", "front": {"flipper_r": [1], "spatula_head": [2]}}))
    return tmp_path, folder


def test_check_and_draft_use_the_labels_and_labels_prints_the_line(penguin_dir, capsys):
    tmp_path, folder = penguin_dir
    spec_path = folder / "view_spec.json"
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "0.6"]) == 0
    assert json.loads(spec_path.read_text())["symmetry"]["x"] is False
    capsys.readouterr()
    labels = str(tmp_path / "labels.json")
    assert cli.main(["check", "--view-spec", str(spec_path), "--out-dir", str(tmp_path / "chk"), "--labels", labels,
                     "--asymmetric", "spatula_*"]) == 0
    out = capsys.readouterr().out
    assert "with spatula_* blanked" in out and "with spatula_*, flipper_r blanked" in out and "suggest symmetry.x = true" in out
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "0.6", "--labels", labels]) == 0
    assert json.loads(spec_path.read_text())["symmetry"]["x"] is True              # the labelled props explain the low IoU
    assert "blanked -> suggest symmetry.x = true" in capsys.readouterr().out
    assert cli.main(["labels", "--view-spec", str(spec_path), "--labels", labels, "--out-dir", str(tmp_path / "lc")]) == 0
    text = capsys.readouterr().out
    assert "symmetry front: left/right mirror IoU" in text and "1x the view, rulers in view px" in text


def test_label_previews_are_1x_with_view_pixel_rulers(penguin_dir, capsys):
    tmp_path, folder = penguin_dir
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "0.6"]) == 0
    spec = str(folder / "view_spec.json")
    labels = str(tmp_path / "labels.json")
    assert cli.main(["labels", "--view-spec", spec, "--labels", labels, "--out-dir", str(tmp_path / "lc")]) == 0
    img = imageio.load_rgba(tmp_path / "lc" / "preview_front.png")
    assert img.shape[:2] == (200, 200)                                             # 1x: pixels are view pixels
    assert np.allclose(img[1, 100, :3], (1, 1, 0), atol=0.02)                      # yellow ruler tick at view x = 100
    assert cli.main(["labels", "--view-spec", spec, "--labels", labels, "--out-dir", str(tmp_path / "lc2"), "--scale", "2"]) == 0
    assert imageio.load_rgba(tmp_path / "lc2" / "preview_front.png").shape[:2] == (400, 400)
    assert "2x the view" in capsys.readouterr().out


def test_anchors_from_px_emits_the_snippet_for_typed_points(capsys):
    args = ["anchors", "--from-px", "front:base=118,543", "--from-px", "front:tip=147,443", "--from-px", "front:radius_px=8",
            "--from-px", "side:base=280,560", "--from-px", "side:tip=236,470", "--id", "spatula_handle"]
    assert cli.main(args) == 0
    cap = capsys.readouterr()
    snippet = json.loads(cap.out)
    assert snippet == {"spatula_handle": {"params": {"anchors_px": {
        "front": {"base": [118, 543], "tip": [147, 443], "radius_px": 8}, "side": {"base": [280, 560], "tip": [236, 470]}}}}}
    assert "omit a point in a view where it is hidden" in cap.err
    assert cli.main(["anchors", "--from-px", "front:radius_px=8,10", "--from-px", "back:center=1,2"]) == 0
    assert json.loads(capsys.readouterr().out)["params"]["anchors_px"] == {"front": {"radius_px": [8, 10]}, "back": {"center": [1, 2]}}
    assert cli.main(["anchors", "--from-px", "front:nose=1,2"]) == 1
    assert cli.main(["anchors", "--from-px", "front:base=1"]) == 1
    assert cli.main(["anchors", "--view", "front"]) == 1                                  # neither --from-px nor a view spec


def test_every_overlay_carries_view_pixel_rulers(tmp_path):
    from bbr.core import proposals
    rgb = np.full((300, 300, 3), 0.4, np.float32)
    out = proposals.add_rulers(rgb, 3, 50, 50, grid=True)
    assert out.shape == (300, 300, 4)
    assert (out[:, :, 0] > out[:, :, 2] + 0.3).any()                                       # yellow marks present


def test_an_origin_dragged_off_the_body_axis_is_reported_instead_of_a_bare_low_iou():
    mask, labels = penguin()
    off = ViewCalibration("front", "-Y", (W, H), 0.01, (70.0, 115.0))                 # the body axis is at 80, not 70
    sym = S.front_symmetry(mask, off, labels, ["spatula_*"])
    assert sym["axis_shift_px"] == 10 and sym["suggested_blank"] == ["flipper_r"] and not sym["suggest_x"]
    assert "the mirror axis moved 10 px right" in sym["explanation"] and "keep symmetry.x false" in sym["explanation"]
    assert S.front_symmetry(mask, FRONT, labels, ["spatula_*"])["axis_shift_px"] == 0
