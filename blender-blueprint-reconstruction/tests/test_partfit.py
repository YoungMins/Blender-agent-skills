"""Automatic per-part fitting: recovery of known synthetic parts from label masks (numpy only)."""
import numpy as np
import pytest

from bbr import BBRError
from bbr.core import partfit as P, partgeom, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.partmask import triangles_mask

MM = 0.001


@pytest.fixture
def calibs():
    return {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
            "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0)),
            "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 450.0))}


def px(calibs, view, point):
    return [float(v) for v in np.round(calibs[view].world_to_px([point])[0])]


def cone_truth(base, mid, tip, rb, rt, sides=16, segments=20):
    length, axis, bend, bend_dir = partgeom.arc_through_points(base, mid, tip)
    verts, faces = partgeom.cone_mesh(rb, rt, length, axis, bend, bend_dir, segments, sides, 1.0)
    return verts + base, partgeom.polygons_triangles(faces)


def centreline(base, mid, tip, n=40):
    length, axis, bend, bend_dir = partgeom.arc_through_points(base, mid, tip)
    pts, *_ = partgeom.cone_path(length, axis, bend, bend_dir, n)
    return pts + base


def test_recovers_a_cone_from_perturbed_anchors(calibs):
    base, mid, tip = np.array([0.03, 0.01, 0.12]), np.array([0.04, 0.015, 0.2]), np.array([0.07, 0.0, 0.28])
    verts, tris = cone_truth(base, mid, tip, 0.018, 0.004)
    masks = {v: triangles_mask(verts, tris, calibs[v]) for v in ("front", "side")}
    rng = np.random.default_rng(3)

    def jit(view, p):
        return list(np.array(px(calibs, view, p)) + rng.normal(0, 5, 2))        # ~5 px anchor noise

    params = {"segments": 20, "sides": 16, "anchors_px": {
        "front": {"base": jit("front", base), "mid": jit("front", mid), "tip": jit("front", tip), "radius_px": [24, 6]},
        "side": {"base": jit("side", base), "mid": jit("side", mid), "tip": jit("side", tip)}}}
    sol = T.solve_part("c", "cone", params, calibs)
    new, report = P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)])
    length = float(np.linalg.norm(tip - base))
    err = np.linalg.norm(centreline(new.keypoints["base"], new.keypoints["mid"], new.keypoints["tip"])
                         - centreline(base, mid, tip), axis=1).max()
    assert err <= 0.02 * length
    rb, rt = new.scalars["radius"]
    # the 4 px tip radius is held to half a pixel: a relative bound is below mask quantisation and flips across platforms
    assert rb == pytest.approx(0.018, rel=0.05) and rt == pytest.approx(0.004, abs=0.5 * MM)
    assert report["loss_fit"] < report["loss_init"] / 5
    iou = P.part_iou("cone", new, params, calibs, masks, masks)
    assert min(v["iou"] for v in iou.values()) > 0.97


def test_fit_is_deterministic(calibs):
    verts, tris = cone_truth(np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.17]), np.array([0.0, 0.0, 0.25]), 0.012, 0.012)
    masks = {v: triangles_mask(verts, tris, calibs[v]) for v in ("front", "side")}
    params = {"anchors_px": {"front": {"base": px(calibs, "front", [0, 0, 0.1]), "tip": px(calibs, "front", [0.005, 0, 0.25]),
                                       "radius_px": [10, 10]},
                             "side": {"base": px(calibs, "side", [0, 0, 0.1]), "tip": px(calibs, "side", [0, 0.004, 0.25])}}}
    sol = T.solve_part("c", "cone", params, calibs)
    a, _ = P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)])
    b, _ = P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)])
    assert np.array_equal(a.keypoints["tip"], b.keypoints["tip"])


def test_loss_ignores_pixels_inside_the_body_but_outside_the_label():
    sil = np.zeros((60, 60), bool)
    sil[10:50, 10:50] = True                                   # body silhouette
    label = np.zeros_like(sil)
    label[20:40, 20:30] = True                                 # the visible part of the part
    exact = label.copy()
    occluded = label.copy()
    occluded[20:40, 30:40] = True                              # also covers pixels inside the body (may be occluded)
    spilled = label.copy()
    spilled[20:40, 40:56] = True                               # leaves the silhouette
    base = P.view_loss(exact, label, sil, w_inside=0.0)
    assert base["total"] == 0.0
    assert P.view_loss(occluded, label, sil, w_inside=0.0)["total"] == 0.0
    assert P.view_loss(spilled, label, sil, w_inside=0.0)["outside"] > 0.3
    tie = P.view_loss(occluded, label, sil)["total"]
    assert 0.0 < tie < 0.2 * P.view_loss(spilled, label, sil)["total"]       # a tie-breaker, not a penalty
    miss = np.zeros_like(label)
    assert P.view_loss(miss, label, sil)["uncovered"] == 1.0


def test_small_and_large_parts_weigh_the_same():
    sil = np.ones((200, 200), bool)
    small, big = np.zeros_like(sil), np.zeros_like(sil)
    small[10:20, 10:20] = True
    big[50:150, 50:150] = True
    half_small, half_big = small.copy(), big.copy()
    half_small[:, 15:] = False
    half_big[:, 100:] = False
    assert P.view_loss(half_small, small, sil)["uncovered"] == pytest.approx(P.view_loss(half_big, big, sil)["uncovered"])


def test_mirror_pair_stays_exactly_mirrored(calibs):
    base, mid, tip = np.array([0.04, 0.01, 0.12]), np.array([0.05, 0.012, 0.2]), np.array([0.08, 0.0, 0.27])
    mirror = np.array([-1.0, 1.0, 1.0])
    vl, tl = cone_truth(base, mid, tip, 0.016, 0.004)
    vr, tr = cone_truth(base * mirror, mid * mirror, tip * mirror, 0.016, 0.004)
    left = {v: triangles_mask(vl, tl, calibs[v]) for v in ("front", "side")}
    right = {v: triangles_mask(vr, tr, calibs[v]) for v in ("front", "back")}
    sil = {v: np.zeros(calibs[v].image_size[::-1], bool) for v in calibs}
    for v in ("front", "side"):
        sil[v] |= left[v]
    for v in ("front", "back"):
        sil[v] |= right[v]
    params = {"anchors_px": {
        "front": {"base": px(calibs, "front", base + [0.004, 0, 0.004]), "mid": px(calibs, "front", mid), "tip": px(calibs, "front", tip + [0, 0, -0.006]),
                  "radius_px": [20, 6]},
        "side": {"base": px(calibs, "side", base), "mid": px(calibs, "side", mid), "tip": px(calibs, "side", tip)}}}
    sol = T.solve_part("l", "cone", params, calibs)
    fitted, report = P.fit_part("l", "cone", sol, params, calibs, sil,
                                [P.Target("l", False, left), P.Target("r", True, right)])
    assert report["joint_with"] == ["r"]
    partner = T.solve_part("r", "cone", {"mirror_of": "l"}, calibs, sibling_for=lambda _: fitted)
    for name in ("base", "mid", "tip"):
        assert np.allclose(partner.keypoints[name], fitted.keypoints[name] * mirror)
    assert partner.scalars["radius"] == fitted.scalars["radius"]
    # the partner's own back / front masks are explained by the mirrored geometry
    iou = P.part_iou("cone", partner, {}, calibs, sil, right)
    assert min(v["iou"] for v in iou.values()) > 0.9


def test_disc_rests_on_its_contact_surface(calibs):
    """A disc seen face-on from the back has no measured depth: the contact term puts it on the (curved) target."""
    # target: a slab whose back surface is the plane y = 0.05 (heights along +Y for the back view)
    zbuf = np.full((500, 400), -np.inf)
    zbuf[100:450, 50:350] = 0.05
    centre, radius, thick = np.array([0.02, 0.0, 0.1]), 0.03, 0.01
    truth_y = 0.05 + thick / 2.0
    verts, faces = partgeom.disc_mesh(radius, thick, 0.0, None, 48, 3, (0, 1, 0))
    mask = triangles_mask(verts + [centre[0], truth_y, centre[2]], partgeom.polygons_triangles(faces), calibs["back"])
    sil = {"back": mask | (zbuf > -np.inf)}
    params = {"thickness": thick, "attach_to": "slab", "attach_side": "back", "standoff": 0.0, "anchors_px": {
        "back": {"center": [float(v) + 3 for v in px(calibs, "back", centre)], "radius_px": 24}}}
    sol = T.solve_part("shield", "disc", params, calibs, depth_for=lambda view: zbuf)
    assert sol.keypoints["center"][1] == pytest.approx(truth_y, abs=1e-6)          # the init already rests on the surface
    new, report = P.fit_part("shield", "disc", sol, params, calibs, sil, [P.Target("shield", False, {"back": mask})],
                             depth_for=lambda view: zbuf)
    c = new.keypoints["center"]
    assert c[0] == pytest.approx(centre[0], abs=0.002) and c[2] == pytest.approx(centre[2], abs=0.002)
    assert new.scalars["radius"] == pytest.approx(radius, rel=0.05)
    assert c[1] - new.scalars["thickness"] / 2.0 == pytest.approx(0.05, abs=0.0015)   # inner face on the surface, no sinking
    assert report["locked"] == [] or "center" not in report["locked"]


def test_unobserved_coordinate_stays_at_its_initial_value(calibs):
    base, tip = np.array([0.0, 0.03, 0.1]), np.array([0.0, 0.03, 0.2])
    verts, faces = partgeom.cone_mesh(0.01, 0.01, 0.1, (0, 0, 1), 0.0, None, 12, 12, 1.0)
    mask = triangles_mask(verts + base, partgeom.polygons_triangles(faces), calibs["front"])
    params = {"depth_m": 0.03, "anchors_px": {"front": {"base": px(calibs, "front", base), "tip": px(calibs, "front", tip),
                                                         "radius_px": [10, 10]}}}
    sol = T.solve_part("rod", "cone", params, calibs)
    new, report = P.fit_part("rod", "cone", sol, params, calibs, {"front": mask}, [P.Target("rod", False, {"front": mask})])
    assert new.keypoints["base"][1] == pytest.approx(0.03) and new.keypoints["tip"][1] == pytest.approx(0.03)
    assert "base.y" not in report["free"] and "base.x" in report["free"]


def test_fit_locked_freezes_parameters(calibs):
    verts, tris = cone_truth(np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.17]), np.array([0.0, 0.0, 0.25]), 0.012, 0.012)
    masks = {v: triangles_mask(verts, tris, calibs[v]) for v in ("front", "side")}
    params = {"anchors_px": {"front": {"base": px(calibs, "front", [0, 0, 0.1]), "tip": px(calibs, "front", [0, 0, 0.25]),
                                       "radius_px": [8, 8]},
                             "side": {"base": px(calibs, "side", [0, 0, 0.1]), "tip": px(calibs, "side", [0, 0, 0.25])}}}
    sol = T.solve_part("c", "cone", params, calibs)
    new, report = P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)], locked=["rb", "tip.z"])
    assert new.scalars["radius"][0] == pytest.approx(8 * MM)
    assert new.keypoints["tip"][2] == pytest.approx(sol.keypoints["tip"][2])
    assert "rb" not in report["free"] and "tip.z" not in report["free"]
    with pytest.raises(BBRError, match="fit_locked"):
        P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)], locked=["nope"])


def test_tapered_capsule_silhouette_matches_its_mesh(calibs):
    verts, faces = partgeom.capsule_mesh(0.02, 0.01, 0.2, 48, 12)
    centre = np.array([0.0, 0.0, 0.2])
    mesh_mask = triangles_mask(verts + centre, partgeom.polygons_triangles(faces), calibs["front"])
    shape = ("capsule", centre + [0, 0, -0.1 + 0.02], 0.02, centre + [0, 0, 0.1 - 0.01], 0.01)
    analytic = P.render(shape, calibs["front"])
    assert P.iou(analytic, mesh_mask) > 0.99
    pose = P.Pose(list(centre), [0, 0, 0], [0.04, 0.04, 0.2], {"axis": "Z", "tip_radius": 0.01})
    assert P.iou(P.render(P.shape_of("capsule", pose), calibs["front"]), mesh_mask) > 0.99


def test_joint_pin_bounds_a_keypoint_but_the_part_still_follows_its_labels(calibs):
    truth = (np.array([0.0, 0.0, 0.10]), np.array([0.0, 0.0, 0.25]))
    verts, tris = cone_truth(truth[0], (truth[0] + truth[1]) / 2, truth[1], 0.012, 0.012)
    masks = {v: triangles_mask(verts, tris, calibs[v]) for v in ("front", "side")}
    anchors = {v: {"base": px(calibs, v, truth[0]), "tip": px(calibs, v, truth[1]), "radius_px": [12, 12]} for v in masks}
    partner = T.solve_part("p", "cone", {"anchors_px": {v: {"base": px(calibs, v, [0.02, 0.0, 0.10]), "tip": px(calibs, v, [0.02, 0.0, 0.05]),
                                                              "radius_px": [5, 5]} for v in masks}}, calibs)
    params = {"anchors_px": {v: {"tip": a["tip"], "radius_px": a["radius_px"]} for v, a in anchors.items()},
              "joint": {"part": "p", "at": "base", "self": "base"}}           # the pin is 20 mm from where the label says
    sol = T.solve_part("c", "cone", params, calibs, joint_for=lambda pid: partner)
    pin = sol.keypoints["base"].copy()
    ious = {}
    for name, kw in (("locked", {"locked": ["base"]}), ("joint", {"joint_point": "base"})):
        new, _ = P.fit_part("c", "cone", sol, params, calibs, masks, [P.Target("c", False, masks)], **kw)
        ious[name] = min(v["iou"] for v in P.part_iou("cone", new, params, calibs, masks, masks).values())
        if name == "locked":
            assert new.keypoints["base"] == pytest.approx(pin)
        else:
            assert np.linalg.norm(new.keypoints["base"] - pin) <= P.JOINT_SLACK_M * 1.2
    assert ious["joint"] > ious["locked"] + 0.05


# --------------------------------------------------------------------------- strand
STRAND_TRUTH = np.array([[0.05, 0.0, 0.38], [0.085, 0.01, 0.30], [0.075, 0.02, 0.22], [0.045, 0.015, 0.14], [0.06, 0.0, 0.06], [0.09, -0.01, -0.01]])
STRAND_RADII = [0.022, 0.024, 0.026, 0.024, 0.018, 0.008]


def strand_case(calibs, views=("front", "side"), seed=3, noise_px=6.0, truth=STRAND_TRUTH):
    verts, faces = partgeom.strand_mesh(truth - truth[0], STRAND_RADII, 0.8, 8, 16)
    tris = partgeom.polygons_triangles(faces)
    masks = {v: triangles_mask(verts + truth[0], tris, calibs[v]) for v in views}
    rng = np.random.default_rng(seed)
    params = {"flatten": 0.8, "anchors_px": {v: {**{f"p{i}": list(np.array(px(calibs, v, p)) + rng.normal(0, noise_px, 2)) for i, p in enumerate(truth)},
                                                  **({"radius_px": [r / MM * 1.15 for r in STRAND_RADII]} if v == "front" else {})} for v in views}}
    return params, masks


def strand_centreline(points):
    return partgeom._catmull_rom(np.asarray(points), 10)


@pytest.fixture(scope="module")
def strand_fit():
    calibs = {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
              "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0))}
    params, masks = strand_case(calibs)
    sol = T.solve_part("s", "strand", params, calibs)
    new, report = P.fit_part("s", "strand", sol, params, calibs, masks, [P.Target("s", False, masks)])
    return calibs, params, masks, sol, new, report


def test_recovers_a_strand_from_perturbed_anchors(strand_fit):
    calibs, params, masks, sol, new, report = strand_fit
    pts = lambda s: np.array([s.keypoints[f"p{i}"] for i in range(len(STRAND_TRUTH))])
    length = float(np.linalg.norm(np.diff(STRAND_TRUTH, axis=0), axis=1).sum())
    err = lambda s: np.linalg.norm(strand_centreline(pts(s)) - strand_centreline(STRAND_TRUTH), axis=1).max() / length
    assert err(sol) > 0.03 > err(new)                       # the anchors were off by more than the tolerance
    assert report["loss_fit"] < report["loss_init"] / 3 and report["seconds"] < P.STRAND_BUDGET_S * 1.2
    iou = P.part_iou("strand", new, params, calibs, masks, masks)
    assert min(v["iou"] for v in iou.values()) > 0.9
    assert min(new.params["radii"]) < 0.4 * max(new.params["radii"])        # the taper survived


def test_strand_attachment_end_stays_attached(strand_fit):
    _, _, _, sol, new, _ = strand_fit
    length = float(np.linalg.norm(np.diff(STRAND_TRUTH, axis=0), axis=1).sum())
    assert np.linalg.norm(new.keypoints["p0"] - sol.keypoints["p0"]) <= P.ANCHOR_BOUND * 0.06 * length * np.sqrt(3)
    assert np.linalg.norm(new.keypoints["p0"] - STRAND_TRUTH[0]) < 0.03 * length
    assert new.location == pytest.approx(list(new.keypoints["p0"]))        # the mesh origin is p0: the strand stays attached there


def test_strand_p0_resists_a_label_that_pulls_it_away(calibs):
    params, masks = strand_case(calibs, noise_px=0.0)
    shifted = {v: np.roll(m, 60, axis=0) for v, m in masks.items()}        # labels 6 cm lower: only p0's prior says otherwise
    sil = {v: m | s for (v, m), s in zip(masks.items(), shifted.values())}
    sol = T.solve_part("s", "strand", params, calibs)
    new, _ = P.fit_part("s", "strand", sol, params, calibs, sil, [P.Target("s", False, shifted)], time_budget=6.0)
    free_pull = np.linalg.norm(new.keypoints["p3"] - sol.keypoints["p3"])
    assert np.linalg.norm(new.keypoints["p0"] - sol.keypoints["p0"]) < 0.6 * free_pull


def test_strand_smoothness_penalises_kinks_and_bunching():
    line = np.linspace(0, 1, 6)[:, None] * [0.0, 0.0, 1.0]
    wave = line + np.array([0.04, 0, 0]) * np.sin(np.arange(6))[:, None]
    kink = line.copy()
    kink[3] += [0.3, 0.0, 0.0]
    bunched = line.copy()
    bunched[3] = bunched[2] + [0.0, 0.0, 0.01]
    assert P.strand_kink(line) == pytest.approx(0.0, abs=1e-12)
    assert P.strand_kink(wave) < 0.2 < P.strand_kink(kink)
    assert P.strand_kink(bunched) > P.strand_kink(wave)


def test_strand_rasterizer_matches_the_builder_mesh(calibs):
    points, radii = STRAND_TRUTH - STRAND_TRUTH[0], STRAND_RADII
    verts, faces = partgeom.strand_mesh(points, radii, 0.8, 8, 16)           # what the Blender builder copies into a mesh
    for view in ("front", "side"):
        truth = triangles_mask(verts + STRAND_TRUTH[0], partgeom.polygons_triangles(faces), calibs[view])
        pose = P.Pose(list(STRAND_TRUTH[0]), [0, 0, 0], None, {"points": points.tolist(), "radii": radii, "flatten": 0.8})
        assert P.iou(P.render(P.shape_of("strand", pose), calibs[view]), truth) > 0.995
        assert P.iou(P.render(P.shape_of("strand", pose, coarse=True), calibs[view]), truth) > 0.97
    wv, wf = partgeom.strand_mesh(points, radii, 0.8, 8, 16, weld=False)
    assert abs(partgeom.mesh_volume(*partgeom.weld_vertices(wv, wf)) - partgeom.mesh_volume(verts, faces)) < 1e-9


def test_strand_mirror_pair_follows_the_symmetry_term(calibs, monkeypatch):
    monkeypatch.setattr(P, "STRAND_STAGES", ((True, 0.3, 600, 1),))
    monkeypatch.setattr(P, "STRAND_POLISH", None)
    mirror = np.array([-1.0, 1.0, 1.0])
    pr, mr = strand_case(calibs, ("front", "side"), seed=1)
    pl, ml = strand_case(calibs, ("front", "back"), seed=2, truth=STRAND_TRUTH * mirror)
    sr = T.solve_part("r", "strand", pr, calibs)
    sl = T.solve_part("l", "strand", pl, calibs)
    sil = {v: m for v, m in ml.items()}
    new, report = P.fit_part("l", "strand", sl, pl, calibs, sil, [P.Target("l", False, ml)], mirror_sibling=sr, mirror_tol=0.002)
    assert "mirror_dev_max" in report and report["fitted"]
    state_l = P.state_from_solution("strand", new, pl)
    assert set(state_l) == set(P.state_from_solution("strand", sr, pr))
    assert P.mirror_state(state_l)["p1"][0] == pytest.approx(-state_l["p1"][0])
