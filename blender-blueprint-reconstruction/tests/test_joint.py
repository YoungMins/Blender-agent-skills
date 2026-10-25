"""params.joint (a part pinned to a partner keypoint) and anchors omitted where a point is hidden."""
import numpy as np
import pytest

from bbr import BBRError
from bbr.core import gapcheck, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.spec import parse_scene_spec

MM = 0.001


@pytest.fixture
def calibs():
    return {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
            "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0)),
            "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 450.0))}


def px(calibs, view, point):
    return [float(v) for v in np.round(calibs[view].world_to_px([point])[0])]


def spatula(calibs, joint=True, tip_in_side=True):
    """The head from (0.05, 0.02, 0.30) down to z = 0.22; the handle's typed base is 24 mm above the head's base."""
    head_pts = {"base": [0.05, 0.02, 0.30], "tip": [0.05, 0.02, 0.22]}
    head = T.solve_part("head", "blade", {"thickness": 0.004, "anchors_px": {
        v: {"base": px(calibs, v, head_pts["base"]), "tip": px(calibs, v, head_pts["tip"]), **({"width_px": 40} if v == "front" else {})}
        for v in ("front", "side")}}, calibs)
    handle_pts = {"base": [0.05, 0.02, 0.324], "tip": [0.05, 0.02, 0.45]}
    anchors = {v: {"base": px(calibs, v, handle_pts["base"]), "tip": px(calibs, v, handle_pts["tip"]), "radius_px": 8}
               for v in ("front", "side")}
    if not tip_in_side:
        del anchors["side"]["tip"]
    params = {"anchors_px": anchors, **({"joint": {"part": "head", "at": "base", "self": "base"}} if joint else {})}
    return head, T.solve_part("handle", "capsule", params, calibs, joint_for=lambda pid: head)


def test_joint_pins_the_keypoint_to_the_partner_and_the_gap_disappears(calibs):
    head, loose = spatula(calibs, joint=False)
    _, joined = spatula(calibs, joint=True)
    assert loose.keypoints["base"][2] - head.keypoints["base"][2] == pytest.approx(0.024, abs=2 * MM)
    assert joined.keypoints["base"] == pytest.approx(head.keypoints["base"])
    assert joined.sources["base"] == {"x": "joint:head", "y": "joint:head", "z": "joint:head"} and not joined.inferred
    assert any("mm from the joint with head.base" in w for w in joined.warnings)        # the typed anchors disagree: said so
    assert joined.keypoints["tip"] == pytest.approx(loose.keypoints["tip"])             # the other end is untouched
    pair = [("head", "handle")]
    assert gapcheck.find_gaps({"head": head.report(), "handle": joined.report()}, pair) == []
    gaps = gapcheck.find_gaps({"head": head.report(), "handle": loose.report()}, pair)     # the original spec: a floating head
    assert len(gaps) == 1 and gaps[0]["gap_m"] > 0.01


def test_joint_needs_no_anchor_for_the_pinned_keypoint_and_validates_its_arguments(calibs):
    head, _ = spatula(calibs)
    anchors = {v: {"tip": px(calibs, v, [0.05, 0.02, 0.45]), "radius_px": 8} for v in ("front", "side")}
    sol = T.solve_part("handle", "capsule", {"anchors_px": anchors, "joint": {"part": "head", "at": "base"}}, calibs,
                       joint_for=lambda pid: head)
    assert sol.keypoints["base"] == pytest.approx(head.keypoints["base"]) and not sol.inferred
    with pytest.raises(BBRError, match="joint.at"):
        T.solve_part("handle", "capsule", {"anchors_px": anchors, "joint": {"part": "head", "at": "mid"}}, calibs,
                     joint_for=lambda pid: head)
    with pytest.raises(BBRError, match="solution"):
        T.solve_part("handle", "capsule", {"anchors_px": anchors, "joint": {"part": "head"}}, calibs)
    comps = [{"id": "head", "type": "blade", "params": {"thickness": 0.004, "anchors_px": {"front": {"base": [1, 1], "tip": [2, 2], "width_px": 4}}}},
             {"id": "handle", "type": "capsule", "params": {"joint": {"part": "head"}, "anchors_px": {"front": {"tip": [1, 1], "radius_px": 3}}}}]
    spec = {"spec_version": "0.1", "name": "t", "components": comps}
    assert parse_scene_spec(spec).components[1].params["joint"]["part"] == "head"
    comps[1]["params"]["joint"] = {"part": "nobody"}
    with pytest.raises(Exception, match="params.joint"):
        parse_scene_spec(spec)


def test_a_point_hidden_in_a_view_is_omitted_and_the_part_is_marked_inferred(calibs):
    _, full = spatula(calibs, joint=False)
    _, partial = spatula(calibs, joint=False, tip_in_side=False)
    assert not full.inferred
    tip = partial.keypoints["tip"]
    assert tip[0] == pytest.approx(0.05, abs=MM) and tip[2] == pytest.approx(0.45, abs=MM)      # x, z from the front view
    assert tip[1] == pytest.approx(partial.keypoints["base"][1])                                  # y: the part's measured base
    assert partial.sources["tip"] == {"x": "views:front", "z": "views:front", "y": "inferred:base"}
    assert partial.inferred and partial.report()["inferred"] is True
    assert any("tip: no view measures y" in w for w in partial.warnings)


def test_depth_m_beats_the_fallback_for_a_hidden_point_and_one_view_alone_still_needs_a_source(calibs):
    anchors = {"front": {"base": px(calibs, "front", [0.05, 0.02, 0.2]), "tip": px(calibs, "front", [0.05, 0.02, 0.4]), "radius_px": 8},
               "side": {"base": px(calibs, "side", [0.05, 0.02, 0.2]), "radius_px": 8}}
    sol = T.solve_part("rod", "capsule", {"anchors_px": anchors, "depth_m": {"tip": 0.06}}, calibs)
    assert sol.keypoints["tip"][1] == pytest.approx(0.06) and sol.sources["tip"]["y"] == "depth_m"
    alone = T.solve_part("rod", "capsule", {"anchors_px": {"front": anchors["front"]}}, calibs)   # no keypoint measures y at all
    assert alone.sources["base"]["y"] == "default:body_centre" and alone.keypoints["base"][1] == 0.0   # never a raw default: the centre line, warned
