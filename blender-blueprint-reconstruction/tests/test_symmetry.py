"""Symmetry: mirrored virtual colour view, L/R label swap, symmetry suggestion, paired-fitting tolerance (numpy only)."""
import numpy as np
import pytest

from bbr import SpecError
from bbr.core import consistency as C, partfit as P, symmetry as S, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.partmask import triangles_mask
from bbr.core.texelproj import project_texel_colors
from conftest import ellipse_mask, paint, rect_mask
from test_partfit import calibs, cone_truth, px  # noqa: F401  (fixture + helpers)

H, W = 120, 160
SIDE = ViewCalibration("side", "-X", (W, H), 0.01, (70.0, 100.0))


def gradient_rgba():
    """Every pixel a distinct colour, so a sample identifies the pixel it came from."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.empty((H, W, 4), np.float32)
    img[..., 0], img[..., 1], img[..., 2], img[..., 3] = xx / (W - 1), yy / (H - 1), (xx + 2 * yy) / (W + 2 * H), 1.0
    return img


def test_virtual_view_maps_a_plus_x_texel_to_the_mirrored_side_pixel():
    rgba, mask = gradient_rgba(), np.ones((H, W), bool)
    mv = S.mirror_view(rgba, mask, SIDE)
    assert mv.calib.camera_axis == "+X" and mv.key == "mirror_side"
    # a texel on the +X flank at (x=+0.30, y=0.12, z=0.40) seen head-on from +X ...
    point = np.array([[0.30, 0.12, 0.40]])
    rgb, has, _ = project_texel_colors(point, np.array([[1.0, 0.0, 0.0]]), [(mv.rgba, mv.mask, mv.calib)])
    assert has[0]
    # ... is the side pixel of the same texel mirrored to X = -0.30 (x does not move the side pixel, y and z do)
    u, v = SIDE.world_to_px(np.array([[-0.30, 0.12, 0.40]]))[0]
    np.testing.assert_allclose(rgb[0], rgba[int(np.floor(v)), int(np.floor(u)), :3], atol=0.02)
    # and a texel facing -X is not seen by the +X virtual camera
    assert not project_texel_colors(point, np.array([[-1.0, 0.0, 0.0]]), [(mv.rgba, mv.mask, mv.calib)])[1][0]


def test_virtual_calibration_is_the_exact_flip():
    mv = S.mirror_view(gradient_rgba(), np.ones((H, W), bool), SIDE)
    p = np.array([[0.2, 0.31, 0.5], [-0.1, -0.2, 0.9]])
    mirrored = p * [-1, 1, 1]
    np.testing.assert_allclose(mv.calib.world_to_px(p)[:, 0], W - SIDE.world_to_px(mirrored)[:, 0])
    np.testing.assert_allclose(mv.calib.world_to_px(p)[:, 1], SIDE.world_to_px(mirrored)[:, 1])
    with pytest.raises(SpecError):
        S.mirror_view(gradient_rgba(), np.ones((H, W), bool), ViewCalibration("front", "-Y", (W, H), 0.01, (80.0, 100.0)))


def test_label_swap_through_pairs():
    pairs = S.mirror_pairs(["horn_l", "horn_r", "arm_l_upper", "arm_r_upper", "body", "sword_blade", "Hand_L", "cape"],
                           explicit={"cape_a": "cape_b"})
    assert pairs["horn_l"] == "horn_r" and pairs["horn_r"] == "horn_l"
    assert pairs["arm_l_upper"] == "arm_r_upper" and pairs["Hand_L"] == "Hand_R"
    assert "body" not in pairs and "cape" not in pairs and pairs["cape_b"] == "cape_a"
    assert S.mirror_pairs([], mirror_of={"wing_x": "wing_y"})["wing_y"] == "wing_x"
    horn_l, arm = rect_mask(H, W, 10, 20, 30, 40), rect_mask(H, W, 100, 50, 130, 60)
    labels = {"horn_l": horn_l, "arm_l_upper": arm, "body": ~(horn_l | arm)}
    mv = S.mirror_view(gradient_rgba(), np.ones((H, W), bool), SIDE, labels, pairs)
    assert set(mv.labels) == {"horn_r", "arm_r_upper", "body"}
    np.testing.assert_array_equal(mv.labels["horn_r"], horn_l[:, ::-1])
    assert not (mv.labels["horn_r"] & mv.labels["body"]).any()


def test_asymmetric_parts_are_never_sampled_from_the_virtual_view():
    sword = rect_mask(H, W, 20, 30, 50, 90)
    mv = S.mirror_view(gradient_rgba(), np.ones((H, W), bool), SIDE, {"sword_blade": sword, "body": ~sword}, {},
                       asymmetric=["sword_*"])
    flipped = sword[:, ::-1]
    assert "sword_blade" not in mv.labels
    assert not (mv.mask & flipped).any() and not (mv.labels["body"] & flipped).any()   # cut out of mask and labels
    # a texel that lands in the middle of the flipped sword gets no colour from this view, even with snapping
    pt = mv.calib.px_to_plane(np.array([[W - 35.0, 60.0]]))
    assert not project_texel_colors(pt, np.array([[1.0, 0.0, 0.0]]), [(mv.rgba, mv.mask, mv.calib)])[1][0]
    far = mv.calib.px_to_plane(np.array([[W - 120.0, 60.0]]))
    assert project_texel_colors(far, np.array([[1.0, 0.0, 0.0]]), [(mv.rgba, mv.mask, mv.calib)])[1][0]
    assert S.is_asymmetric("sword_grip", ["sword_*"]) and not S.is_asymmetric("horn_l", ["sword_*"])


def test_mirrored_view_is_a_tie_break_against_real_views():
    full = np.ones((20, 20), bool)
    side = (paint(full, (0.0, 0.0, 1.0)), full, ViewCalibration("side", "-X", (20, 20), 0.1, (10.0, 10.0)))
    front = (paint(full, (1.0, 0.0, 0.0)), full, ViewCalibration("front", "-Y", (20, 20), 0.1, (10.0, 10.0)))
    virt = S.mirror_view(*side)
    views = [front, (virt.rgba, virt.mask, virt.calib)]
    a10, a45 = np.radians(10), np.radians(45)
    # texel 1: +X flank, the front sees it at a grazing angle; texel 2: 45 degrees, front and mirror face it equally
    normals = np.array([[np.cos(a10), -np.sin(a10), 0.0], [np.cos(a45), -np.sin(a45), 0.0]])
    rgb, _, _ = project_texel_colors(np.zeros((2, 3)), normals, views, seam_blend_width=0.0, view_penalty=[0.0, 0.3])
    np.testing.assert_allclose(rgb[0], [0, 0, 1], atol=1e-6)         # no real view sees it well: the mirrored view colours it
    np.testing.assert_allclose(rgb[1], [1, 0, 0], atol=1e-6)         # a real view sees it as well: the real one wins


def warrior_mask(sword=True):
    m = ellipse_mask(H, W, 60, 80, 40, 30) | rect_mask(H, W, 60, 95, 100, 115)
    if sword:
        m |= rect_mask(H, W, 124, 70, 136, 115)                       # sword band on the image right only
    return m


def test_symmetry_suggestion_blanks_labelled_asymmetric_parts():
    calib = ViewCalibration("front", "-Y", (W, H), 0.01, (80.0, 115.0))
    sword = rect_mask(H, W, 124, 70, 136, 115)
    plain = S.front_symmetry(warrior_mask(), calib)
    assert plain["iou"] < S.SUGGEST_MIN_IOU and not plain["suggest_x"]
    blanked = S.front_symmetry(warrior_mask(), calib, {"sword_blade": sword, "body": ~sword}, ["sword_*"])
    assert blanked["iou_blanked"] > 0.99 and blanked["suggest_x"] and blanked["blanked_frac"] > 0
    assert S.front_symmetry(warrior_mask(False), calib)["suggest_x"]
    assert S.mirror_iou(warrior_mask(False), calib) == pytest.approx(1.0)


def test_consistency_report_carries_the_symmetry_suggestion():
    front = ViewCalibration("front", "-Y", (W, H), 0.01, (80.0, 115.0))
    back = ViewCalibration("back", "+Y", (W, H), 0.01, (80.0, 115.0))
    m = warrior_mask(False)
    res = C.analyze({"front": m, "back": m[:, ::-1]}, {"front": front, "back": back}, {"front": "-Y", "back": "+Y"})
    assert [s["view"] for s in res["symmetry"]] == ["front", "back"] and all(s["suggest_x"] for s in res["symmetry"])
    assert "suggest symmetry.x" in C.format_consistency(res)
    lopsided = C.analyze({"front": warrior_mask()}, {"front": front}, {"front": "-Y"})
    assert not lopsided["symmetry"][0]["suggest_x"]


def test_pair_hints_parse_and_validate():
    h = S.pair_hints([["a", "b"], ["c", "d", 0.01], {"a": "e", "b": "f", "tol_m": 0.02}])
    assert h == {"b": ("a", 0.0), "d": ("c", 0.01), "f": ("e", 0.02)}
    assert S.pair_hints(None) == {}
    for bad in ([["a"]], [["a", "a"]], [["a", "b", -1]]):
        with pytest.raises(SpecError):
            S.pair_hints(bad)


def test_paired_fit_tolerance_lets_a_disagreeing_partner_keep_its_own_data(calibs):  # noqa: F811
    mirror = np.array([-1.0, 1.0, 1.0])
    base, mid, tip = np.array([0.04, 0.01, 0.12]), np.array([0.05, 0.012, 0.2]), np.array([0.08, 0.0, 0.27])
    tip_r = tip + [0.0, 0.0, 0.012]                      # the right horn's reference is 12 mm longer than the left one's
    vl, tl = cone_truth(base, mid, tip, 0.016, 0.004)
    vr, tr = cone_truth(base * mirror, mid * mirror, tip_r * mirror, 0.016, 0.004)
    left = {v: triangles_mask(vl, tl, calibs[v]) for v in ("front", "side")}
    right = {v: triangles_mask(vr, tr, calibs[v]) for v in ("front", "back")}

    def params(views, b, m, t):
        return {"anchors_px": {v: {"base": px(calibs, v, b), "mid": px(calibs, v, m), "tip": px(calibs, v, t),
                                   **({"radius_px": [16, 4]} if v == "front" else {})} for v in views}}

    p_l = params(("front", "side"), base, mid, tip)
    p_r = {**params(("front", "back"), base * mirror, mid * mirror, tip_r * mirror), "mirror_of": "l"}   # depth (y) from the sibling
    fitted_l, _ = P.fit_part("l", "cone", T.solve_part("l", "cone", p_l, calibs), p_l, calibs, left, [P.Target("l", False, left)])
    sol_r = T.solve_part("r", "cone", p_r, calibs, sibling_for=lambda _: fitted_l)

    def fit_r(tol):
        new, rep = P.fit_part("r", "cone", sol_r, p_r, calibs, right, [P.Target("r", False, right)],
                              mirror_sibling=fitted_l, mirror_tol=tol, mirror_weight=5.0)
        return rep, min(v["iou"] for v in P.part_iou("cone", new, p_r, calibs, right, right).values())

    strict, iou_strict = fit_r(0.0)
    loose, iou_loose = fit_r(0.03)
    assert strict["mirror_dev_max"] < loose["mirror_dev_max"]        # the dead zone frees the partner ...
    assert iou_loose > iou_strict                                    # ... to follow its own mask
    assert loose["mirror_tol"] == 0.03
