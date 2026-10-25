"""``anchors --kind auto``: which primitive a part mask is, where its base is, one type for all views; `_l` / `_r` side warnings."""
import json
import math
import sys

import numpy as np
import pytest

from bbr.core import imageio, labels, measure, partshape
from bbr.core.calibration import ViewCalibration
from conftest import SK, paint

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402

H = W = 300
YY, XX = np.mgrid[0:H, 0:W]
PX, PY = XX + 0.5, YY + 0.5
BODY = PY >= 220
TOUCH = BODY | ((PY >= 215) & (PY < 221))                  # what the part is attached to: the body label


def _auto(mask, rest=None, **kw):
    return measure.propose_anchors(mask, [0, 0, W, H], "auto", rest=rest, **kw)


def test_triangle_ear_is_a_cone_based_on_the_body():
    ear = (PY <= 220) & (PY >= 60) & (np.abs(PX - 150) <= (PY - 60) / 160 * 50)      # apex up, 100 px base on the body
    res = _auto(ear, TOUCH)
    assert res["suggested_type"] == "cone" and res["confidence"] >= 0.85
    assert np.abs(np.array(res["base"]) - [150, 220]).max() <= 3 and np.abs(np.array(res["tip"]) - [150, 62]).max() <= 8
    assert res["radius_px"][0] == pytest.approx(48, abs=6) and res["radius_px"][1] < 8
    flipped = _auto(ear[::-1], BODY[::-1] | ((PY[::-1] < 85) & (PY[::-1] > 79)))      # apex down, body above: still base = body side
    assert flipped["base"][1] < flipped["tip"][1]
    assert measure.propose_anchors(ear, [0, 0, W, H], "auto", rest=None)["suggested_type"] == "cone"   # no body: the wider end
    hint = _auto(ear, TOUCH, base_hint=[150, 62])
    assert hint["base"][1] < hint["tip"][1]                                          # the hint beats the contact rule


def test_ball_is_a_sphere_with_centre_and_radius():
    ball = (PX - 150) ** 2 + (PY - 130) ** 2 <= 40 ** 2
    res = _auto(ball)
    assert res["suggested_type"] == "sphere" and "base" not in res
    assert np.abs(np.array(res["center"]) - [150, 130]).max() <= 2 and res["radius_px"] == pytest.approx(40, abs=2)


def test_stubby_bar_is_a_capsule_or_cylinder_and_a_curl_is_a_curved_cone():
    bar = (np.abs(PX - 150) <= 30) & (PY >= 100) & (PY <= 220)
    res = _auto(bar, TOUCH)
    assert res["suggested_type"] in ("capsule", "cylinder") and res["radius_px"] == pytest.approx(30, abs=4)
    assert res["base"][1] > res["tip"][1]
    r, ang = np.hypot(PX - 60, PY - 130), np.arctan2(PY - 130, PX - 60)
    curl = (np.abs(r - 90) <= 18) & (ang >= -0.3) & (ang <= math.pi * 0.55)            # a J: thick quarter ring
    res = _auto(curl, TOUCH)
    assert res["suggested_type"] == "cone" and res["curled"] and "mid" in res
    chord = (np.array(res["tip"]) + np.array(res["base"])) / 2
    assert np.linalg.norm(np.array(res["mid"]) - chord) > 0.1 * res["length_px"]        # mid sits on the curve, off the chord


def test_force_type_gives_the_same_primitive_in_every_view():
    ball = (PX - 150) ** 2 + (PY - 130) ** 2 <= 40 ** 2
    res = _auto(ball, force_type="cone")
    assert res["suggested_type"] == "cone" and {"base", "mid", "tip"} <= set(res)
    res = _auto(ball, force_type="capsule")
    assert res["suggested_type"] == "capsule" and "base" in res and "mid" not in res


def test_harmonize_flips_the_view_whose_base_runs_the_other_way():
    calibs = {"front": ViewCalibration("front", "-Y", (W, H), 0.001, (150.0, 280.0)),
              "side": ViewCalibration("side", "-X", (W, H), 0.001, (150.0, 280.0))}
    ear = {"base": [150.0, 220.0], "tip": [150.0, 60.0], "radius_px": [40.0, 4.0], "confidence": 0.9, "base_by": "attach"}
    wrong = {"base": [150.0, 60.0], "tip": [150.0, 220.0], "radius_px": [4.0, 40.0], "confidence": 0.7, "base_by": "width"}
    found = {"front": dict(ear), "side": dict(wrong)}
    assert measure.harmonize_base_tip(found, calibs) == ["side"]
    assert found["side"]["base"] == [150.0, 220.0] and found["side"]["radius_px"] == [40.0, 4.0]
    assert found["front"]["base"] == [150.0, 220.0]
    assert measure.harmonize_base_tip({"front": dict(ear), "side": dict(ear)}, calibs) == []


def test_unknown_type_is_rejected():
    with pytest.raises(Exception, match="type must be"):
        partshape.suggest(np.ones((20, 20), bool), (0, 0), force_type="torus")


# --------------------------------------------------------------------------- _l / _r on the right side
def _calib(axis):
    return ViewCalibration("v", axis, (W, H), 0.002, (150.0, 280.0))


def test_side_warning_follows_the_characters_own_left_and_right():
    blob = np.zeros((H, W), bool)
    blob[100:140, 40:80] = True                                      # image-left of centre: x = -0.12 m in the front view
    front, back = _calib("-Y"), _calib("+Y")
    assert labels.side_warning("ear_r", "front", blob, front, 0.6) is None      # character right = -X = image-left in the front view
    msg = labels.side_warning("ear_l", "front", blob, front, 0.6)
    assert msg and "image-right in the front view" in msg and "CHARACTER" in msg
    assert labels.side_warning("ear_l", "back", blob, back, 0.6) is None        # image-left in the back view = +X = left
    assert "image-right in the back view" in labels.side_warning("ear_r", "back", blob, back, 0.6)
    assert labels.side_warning("tail", "front", blob, front, 0.6) is None       # no side in the name
    assert labels.side_warning("ear_l", "side", blob, _calib("-X"), 0.6) is None   # side views are not judged
    centred = np.zeros((H, W), bool)
    centred[100:140, 145:155] = True
    assert labels.side_warning("ear_l", "front", centred, front, 0.6) is None   # inside the dead zone
    assert labels.side_warnings({"front": {"ear_l": blob, "ear_r": blob}}, {"front": front}, 0.6) == [msg]


def test_character_side_tokens():
    from bbr.core.symmetry import character_side
    assert [character_side(n) for n in ("ear_l", "arm_L_upper", "Left_hand", "hand_r", "tail", "shield", "label_x")] == [
        "l", "l", "l", "r", None, None, None]


def test_anchors_and_labels_cli_warn_and_suggest(tmp_path, capsys):
    """A cone-shaped `ear_l` at image-LEFT of the front view: both commands warn; anchors print the suggested type."""
    body = (PY >= 200) & (np.abs(PX - 150) <= 90)
    ear = (PY < 200) & (PY >= 60) & (np.abs(PX - 100) <= (PY - 60) / 140 * 30)
    imageio.save_rgba(tmp_path / "front.png", paint(body | ear, (0.9, 0.9, 0.9)))
    spec = {"spec_version": "0.1", "name": "t", "views": {"front": {"path": "front.png"}}, "known_dimensions": {"height": 0.5}}
    (tmp_path / "view_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    id_map = np.zeros((H, W), np.uint8)
    id_map[ear] = 1
    id_map[body] = 2
    (tmp_path / "regions" / "front").mkdir(parents=True)
    np.save(tmp_path / "regions" / "front" / "id_map.npy", id_map)
    (tmp_path / "labels.json").write_text(json.dumps({"_regions": "regions", "front": {"ear_l": [1], "body": [2]}}), encoding="utf-8")
    base = ["--view-spec", str(tmp_path / "view_spec.json"), "--labels", str(tmp_path / "labels.json")]
    assert cli.main(["anchors", *base, "--view", "front", "--part", "ear_l"]) == 0
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert out["ear_l"]["suggested_type"] == "cone" and set(out["ear_l"]["params"]["anchors_px"]["front"]) == {"base", "mid", "tip", "radius_px"}
    assert "suggested_type cone" in cap.err and "WARNING front: 'ear_l'" in cap.err and "CHARACTER" in cap.err
    assert out["ear_l"]["params"]["anchors_px"]["front"]["base"][1] > out["ear_l"]["params"]["anchors_px"]["front"]["tip"][1]
    assert cli.main(["labels", *base, "--out-dir", str(tmp_path / "lc")]) == 0
    assert "WARNING front: 'ear_l'" in capsys.readouterr().err
