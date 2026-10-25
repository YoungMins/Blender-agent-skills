"""Anchor-based placement: triangulation, contact solve, mirror_of, anchor proposals (synthetic data, numpy only)."""
import json
import math
import sys

import numpy as np
import pytest

from bbr import BBRError
from bbr.core import imageio, measure, partgeom, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.spec import parse_scene_spec
from bbr.core.zbuffer import depth_buffer
from conftest import SK, paint

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402

MM = 0.001


@pytest.fixture
def calibs():
    return {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
            "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0)),
            "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 450.0))}


def px(calibs, view, point):
    return [float(v) for v in np.round(calibs[view].world_to_px([point])[0])]       # whole-pixel anchors


def test_point_from_front_and_side_within_1mm(calibs):
    p = [0.05, 0.03, 0.2]
    pt, spread, count, _ = T.triangulate_point(calibs, {"front": px(calibs, "front", p), "side": px(calibs, "side", p)})
    assert np.abs(pt - p).max() < 1 * MM
    assert list(count) == [1, 1, 2]                      # z measured twice


def test_back_view_is_mirrored_and_averaged(calibs):
    p = [-0.04, 0.02, 0.3]
    obs = {v: px(calibs, v, p) for v in ("front", "back", "side")}
    pt, spread, count, used = T.triangulate_point(calibs, obs)
    assert np.abs(pt - p).max() < 1 * MM and list(count) == [2, 1, 3]
    assert spread[2] < 1.5 * MM
    obs["back"] = [obs["back"][0], obs["back"][1] + 10]                  # 10 px off in z: reported, averaged
    _, spread, _, _ = T.triangulate_point(calibs, obs)
    assert spread[2] == pytest.approx(10 * MM, abs=1.5 * MM)


def test_single_view_needs_a_depth_source(calibs):
    params = {"anchors_px": {"front": {"center": [250, 250], "radius_px": 20}}}
    unmeasured = T.solve_part("ball", "sphere", params, calibs)           # no source at all: never a raw default, the body centre line
    assert unmeasured.keypoints["center"][1] == 0.0 and unmeasured.sources["center"]["y"] == "default:body_centre"
    assert unmeasured.inferred and any("centre line" in w for w in unmeasured.warnings)
    sol = T.solve_part("ball", "sphere", {**params, "depth_m": 0.07}, calibs)
    assert sol.keypoints["center"][1] == pytest.approx(0.07) and sol.inferred
    assert sol.size == pytest.approx([0.04, 0.04, 0.04])


def test_sphere_radius_from_two_views_and_axis_mean(calibs):
    sol = T.solve_part("hand", "sphere", {"anchors_px": {
        "front": {"center": [250, 250], "radius_px": [30, 40]}, "side": {"center": [120, 250], "radius_px": [20, 40]}}}, calibs)
    assert sol.size == pytest.approx([0.06, 0.04, 0.08])                 # x: front h, y: side h, z: mean of the two v
    assert not sol.inferred


def sphere_mesh(radius, n=64):
    th, ph = np.meshgrid(np.linspace(0, math.pi, n), np.linspace(0, 2 * math.pi, 2 * n, endpoint=False), indexing="ij")
    v = radius * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
    t = []
    m = 2 * n
    for i in range(n - 1):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            t += [(a, b, c), (b, d, c)]
    return v, np.array(t)


def test_disc_rests_on_the_back_of_a_sphere(calibs):
    calibs = {**calibs, "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 250.0))}
    v, t = sphere_mesh(0.1)
    zbuf = depth_buffer(v, t, calibs["back"])
    params = {"thickness": 0.01, "attach_to": "ball", "attach_side": "back", "standoff": 0.002,
              "anchors_px": {"back": {"center": [200, 250], "radius_px": 30}}}
    sol = T.solve_part("shield", "disc", params, calibs, depth_for=lambda view: zbuf)
    # inner face on the sphere's back (y = +0.1 at the centre) + standoff, centre half a thickness further out
    assert sol.keypoints["center"][1] == pytest.approx(0.1 + 0.002 + 0.005, abs=2 * MM)
    assert sol.params["axis"] == pytest.approx([0, 1, 0], abs=0.03)       # facing the back camera
    assert sol.sources["center"]["y"].startswith("contact:ball") and sol.inferred
    # off-centre on the sphere the disc tilts with the surface
    params["anchors_px"]["back"]["center"] = [200 - 60, 250]           # x = +0.06 (the back view mirrors x)
    sol = T.solve_part("shield", "disc", params, calibs, depth_for=lambda view: zbuf)
    assert sol.params["axis"][0] > 0.3 and sol.params["axis"][1] > 0.7
    assert sol.keypoints["center"][1] < 0.1                              # lower than the sphere's peak, still outside it
    assert np.linalg.norm(sol.keypoints["center"]) > 0.1


def test_attach_without_surface_is_an_error(calibs):
    zbuf = np.full((500, 400), -np.inf)
    params = {"thickness": 0.01, "attach_to": "t", "attach_side": "back", "anchors_px": {"back": {"center": [200, 250], "radius_px": 30}}}
    with pytest.raises(BBRError, match="no surface"):
        T.solve_part("d", "disc", params, calibs, depth_for=lambda v: zbuf)


def horn_params():
    return {"anchors_px": {"front": {"base": [150, 400], "mid": [120, 330], "tip": [170, 260], "radius_px": [20, 3]},
                           "side": {"base": [150, 400], "mid": [135, 330], "tip": [140, 260]}}}


def test_mirror_of_is_the_exact_mirror(calibs):
    left = T.solve_part("horn_l", "cone", horn_params(), calibs)
    right = T.solve_part("horn_r", "cone", {"mirror_of": "horn_l"}, calibs, sibling_for=lambda i: left)
    for name, p in left.keypoints.items():
        assert right.keypoints[name] == pytest.approx(p * [-1, 1, 1])
    assert right.inferred and not left.inferred

    def world(sol):
        g = sol.params
        v, f = partgeom.cone_mesh(g["base_radius"], g["tip_radius"], g["length"], g["axis"], g["bend"], g.get("bend_dir"), 12, 16)
        assert partgeom.mesh_volume(v, f) > 0                              # outward winding
        return v + sol.location

    a, b = world(left) * [-1, 1, 1], world(right)
    key = lambda m: m[np.lexsort(np.round(m, 7).T[::-1])]
    assert np.allclose(key(a), key(b), atol=1e-6)


def test_mirror_fills_only_the_missing_axis(calibs):
    left = T.solve_part("arm_l", "capsule", {"anchors_px": {"front": {"base": [150, 300], "tip": [120, 380], "radius_px": 20},
                                                           "side": {"base": [140, 300], "tip": [145, 380]}}}, calibs)
    own = {"mirror_of": "arm_l", "anchors_px": {"front": {"base": [250, 300], "tip": [280, 380], "radius_px": 20}}}
    right = T.solve_part("arm_r", "capsule", own, calibs, sibling_for=lambda i: left)
    assert right.sources["base"]["x"] == "views:front" and right.sources["base"]["y"] == "mirror:arm_l"
    assert right.keypoints["base"][1] == pytest.approx(left.keypoints["base"][1])


def test_horn_arc_goes_through_the_mid_anchor(calibs):
    sol = T.solve_part("horn", "cone", horn_params(), calibs)
    g = sol.params
    pts, _, _, _ = partgeom.cone_path(g["length"], g["axis"], g["bend"], g.get("bend_dir"), 2000)
    base = sol.keypoints["base"]
    assert base + pts[-1] == pytest.approx(sol.keypoints["tip"], abs=1e-6)
    assert np.linalg.norm(base + pts - sol.keypoints["mid"], axis=1).min() < 1e-4    # the arc passes through the mid anchor
    assert g["base_radius"] == pytest.approx(0.02) and g["tip_radius"] == pytest.approx(0.003)


def test_arc_through_points_quarter_circle():
    length, axis, bend, bend_dir = partgeom.arc_through_points([0, 0, 0], [math.sin(math.pi / 4), 0, 1 - math.cos(math.pi / 4)], [1, 0, 1])
    assert bend == pytest.approx(90) and length == pytest.approx(math.pi / 2)
    assert list(axis) == pytest.approx([1, 0, 0], abs=1e-9) and list(bend_dir) == pytest.approx([0, 0, 1], abs=1e-9)
    assert partgeom.arc_through_points([0, 0, 0], [1, 0, 0], [2, 0, 0])[2] == 0.0


def test_euler_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=3)
        r = partgeom.rotation_z_to(a)
        ex, ey, ez = np.radians(partgeom.euler_xyz_deg(r))
        rx = np.array([[1, 0, 0], [0, math.cos(ex), -math.sin(ex)], [0, math.sin(ex), math.cos(ex)]])
        ry = np.array([[math.cos(ey), 0, math.sin(ey)], [0, 1, 0], [-math.sin(ey), 0, math.cos(ey)]])
        rz = np.array([[math.cos(ez), -math.sin(ez), 0], [math.sin(ez), math.cos(ez), 0], [0, 0, 1]])
        assert rz @ ry @ rx @ [0, 0, 1] == pytest.approx(a / np.linalg.norm(a), abs=1e-9)


def test_blade_width_lies_in_the_image_plane(calibs):
    sol = T.solve_part("sword", "blade", {"thickness": 0.01, "anchors_px": {
        "front": {"base": [200, 300], "tip": [240, 400], "width_px": 30}, "side": {"base": [150, 300], "tip": [150, 400]}}}, calibs)
    assert sol.params["width"] == pytest.approx(0.03) and abs(sol.params["width_dir"][1]) < 1e-9   # no depth component
    assert np.dot(sol.params["width_dir"], sol.params["axis"]) == pytest.approx(0, abs=1e-9)


def test_unknown_anchor_name_is_reported(calibs):
    with pytest.raises(BBRError, match="takes anchors"):
        T.solve_part("x", "cone", {"anchors_px": {"front": {"base": [1, 2], "tip": [3, 4], "center": [1, 1]}}}, calibs)


# --------------------------------------------------------------------------- anchor proposals
def cone_mask(h=300, w=300, tip=(150, 40), base_y=190, half=30, angle=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx + 0.5 - tip[0], yy + 0.5 - tip[1]
    c, s = math.cos(angle), math.sin(angle)
    along, across = u * s + v * c, u * c - v * s
    t = along / (base_y - tip[1])
    return (t >= 0) & (t <= 1) & (np.abs(across) <= half * t)


@pytest.mark.parametrize("box", [[100, 30, 200, 190], [100, 20, 200, 215]])
def test_anchors_of_a_cone_within_3px(box):
    res = measure.propose_anchors(cone_mask(), box, "tube")
    assert np.abs(np.array(res["base"]) - [150, 190]).max() <= 3
    assert np.abs(np.array(res["tip"]) - [150, 40]).max() <= 3
    assert np.abs(np.array(res["mid"]) - [150, 115]).max() <= 3
    assert res["radius_px"][0] == pytest.approx(30, abs=3) and res["radius_px"][1] < 4


def test_anchors_of_a_tilted_cone_and_base_hint():
    angle = math.radians(25)
    m = cone_mask(angle=angle)
    tip = np.array([150.0, 40.0])
    base = tip + 150 * np.array([math.sin(angle), math.cos(angle)])
    res = measure.propose_anchors(m, [0, 0, 300, 300], "tube")
    assert np.abs(np.array(res["tip"]) - tip).max() <= 3 and np.abs(np.array(res["base"]) - base).max() <= 3
    swapped = measure.propose_anchors(m, [0, 0, 300, 300], "tube", base_hint=tip)
    assert swapped["base"] == res["tip"]


def test_anchors_ignore_other_components_in_the_box():
    m = cone_mask() | (np.arange(300)[:, None] > 250) & (np.arange(300)[None, :] > 220)
    res = measure.propose_anchors(m, [100, 30, 299, 299], "tube")
    assert np.abs(np.array(res["tip"]) - [150, 40]).max() <= 3


def test_anchors_capsule_blade_disc():
    yy, xx = np.mgrid[0:200, 0:200]
    cap = (np.abs(xx + 0.5 - 100) <= 12) & (yy >= 30) & (yy < 150)
    res = measure.propose_anchors(cap, [60, 10, 140, 190], "capsule")
    assert res["radius_px"] == pytest.approx(12, abs=1.5)
    assert abs(res["base"][1] - 150) <= 3 and abs(res["tip"][1] - 30) <= 3
    assert measure.propose_anchors(cap, [60, 10, 140, 190], "blade")["width_px"] == pytest.approx(24, abs=3)
    disc = (xx + 0.5 - 90) ** 2 + (yy + 0.5 - 110) ** 2 <= 40 ** 2
    res = measure.propose_anchors(disc, [40, 60, 140, 160], "disc")
    assert np.abs(np.array(res["center"]) - [90, 110]).max() <= 2 and res["radius_px"] == pytest.approx(40, abs=2)
    with pytest.raises(BBRError):
        measure.propose_anchors(disc, [0, 0, 10, 10], "disc")


def test_anchors_cli_prints_a_snippet(tmp_path, capsys):
    mask = cone_mask(h=300, w=300)
    imageio.save_rgba(tmp_path / "front.png", paint(mask, (0.9, 0.9, 0.9)))
    spec = {"spec_version": "0.1", "name": "t", "views": {"front": {"path": "front.png"}}, "known_dimensions": {"height": 0.5}}
    (tmp_path / "view_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    overlay = tmp_path / "ov.png"
    assert cli.main(["anchors", "--view-spec", str(tmp_path / "view_spec.json"), "--view", "front", "--bbox", "100,30,200,190",
                     "--id", "horn", "--overlay", str(overlay)]) == 0
    out = json.loads(capsys.readouterr().out)
    front = out["horn"]["params"]["anchors_px"]["front"]
    assert set(front) == {"base", "mid", "tip", "radius_px"} and overlay.is_file()
    assert cli.main(["anchors", "--view-spec", str(tmp_path / "view_spec.json"), "--view", "top", "--bbox", "1,1,5,5"]) == 1


# --------------------------------------------------------------------------- spec
def scene(**params_by_id):
    comps = [{"id": k, "type": t, "params": p} for k, (t, p) in params_by_id.items()]
    return {"spec_version": "0.1", "name": "s", "components": comps}


def test_anchored_parts_need_no_size_and_refs_must_exist():
    ok = scene(a=("sphere", {"anchors_px": {"front": {"center": [1, 2], "radius_px": 3}}}), b=("sphere", {"mirror_of": "a"}))
    assert len(parse_scene_spec(ok).components) == 2
    with pytest.raises(Exception, match="size is required"):
        parse_scene_spec(scene(a=("sphere", {})))
    with pytest.raises(Exception, match="not a component id"):
        parse_scene_spec(scene(a=("sphere", {"mirror_of": "zz"})))
    with pytest.raises(Exception, match="cannot be the component itself"):
        parse_scene_spec(scene(a=("disc", {"attach_to": "a", "anchors_px": {}})))
    with pytest.raises(Exception, match="supported for types"):
        parse_scene_spec(scene(a=("box", {"mirror_of": "a"})))


def rod_mask(top=(100.0, 60.0), bottom=(140.0, 440.0), half=4.0, head=14.0, h=500, w=300):
    """Thin straight rod from ``top`` to ``bottom`` (pixels) with a wide twisted head over its upper 40 px."""
    yy, xx = np.mgrid[0:h, 0:w]
    px_, py_ = xx + 0.5, yy + 0.5
    d = np.array(bottom) - np.array(top)
    length = float(np.hypot(*d))
    u = d / length
    along = (px_ - top[0]) * u[0] + (py_ - top[1]) * u[1]
    across = np.abs(-(px_ - top[0]) * u[1] + (py_ - top[1]) * u[0])
    r = np.where(along < 40, head, half)
    return (along >= 0) & (along <= length) & (across <= r)


def test_rod_anchors_base_is_the_lower_end_with_sane_radius():
    m = rod_mask()
    res = measure.propose_anchors(m, [0, 0, 300, 500], "tube")
    assert np.abs(np.array(res["base"]) - [140, 440]).max() <= 3 and np.abs(np.array(res["tip"]) - [100, 60]).max() <= 3
    assert res["radius_px"][0] == pytest.approx(4, abs=1.5)               # the head does not make the base wide
    assert 2.5 <= res["radius_px"][1] <= 15 and res["confidence"] >= 0.8
    flipped = measure.propose_anchors(m[::-1], [0, 0, 300, 500], "tube")    # a view that is upside down: still base = lower
    assert flipped["base"][1] > flipped["tip"][1]
    cap = measure.propose_anchors(m, [0, 0, 300, 500], "capsule")
    assert cap["radius_px"] == pytest.approx(4, abs=1.5)
    hint = measure.propose_anchors(m, [0, 0, 300, 500], "tube", base_hint=[100, 60])
    assert hint["base"] == res["tip"]


def test_rod_anchors_consistent_across_views_and_checked(calibs):
    masks = {"front": rod_mask(), "side": rod_mask(top=(150.0, 60.0), bottom=(120.0, 440.0)),
             "back": rod_mask(top=(300.0 - 100.0, 60.0), bottom=(300.0 - 140.0, 440.0))[:, ::-1][:, ::-1]}
    found = {v: measure.propose_anchors(m, [0, 0, 300, 500], "tube") for v, m in masks.items()}
    assert all(a["base"][1] > a["tip"][1] for a in found.values())
    check = measure.check_anchor_views(found, calibs)
    assert check["confidence"] >= 0.8 and not check["warnings"]
    swapped = dict(found)
    swapped["back"] = measure.propose_anchors(masks["back"], [0, 0, 300, 500], "tube", base_hint=found["back"]["tip"])
    check = measure.check_anchor_views(swapped, calibs)
    assert any("swapped" in w for w in check["warnings"]) and check["confidence"] < 0.8


def test_anchors_cli_multi_view_rod(tmp_path, capsys):
    for name, mask in (("front", rod_mask()), ("side", rod_mask(top=(150.0, 60.0), bottom=(120.0, 440.0)))):
        imageio.save_rgba(tmp_path / f"{name}.png", paint(mask, (0.5, 0.3, 0.1)))
    spec = {"spec_version": "0.1", "name": "t", "views": {"front": {"path": "front.png"}, "side": {"path": "side.png", "camera_axis": "-X"}},
            "known_dimensions": {"height": 0.5}}
    (tmp_path / "view_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    labels = {"_regions": "regions", "front": {"staff": [{"polygon": [[0, 0], [300, 0], [300, 500], [0, 500]]}], "body": []},
              "side": {"staff": [{"polygon": [[0, 0], [300, 0], [300, 500], [0, 500]]}], "body": []}}
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    for name in ("front", "side"):
        (tmp_path / "regions" / name).mkdir(parents=True)
        np.save(tmp_path / "regions" / name / "id_map.npy", np.zeros((500, 300), np.uint8))
    code = cli.main(["anchors", "--view-spec", str(tmp_path / "view_spec.json"), "--view", "front", "--view", "side",
                     "--labels", str(tmp_path / "labels.json"), "--part", "staff"])
    cap = capsys.readouterr()
    assert code == 0 and "confidence" in cap.err
    anchors = json.loads(cap.out)["staff"]["params"]["anchors_px"]
    assert set(anchors) == {"front", "side"} and all(a["base"][1] > a["tip"][1] for a in anchors.values())
    assert cli.main(["anchors", "--view-spec", str(tmp_path / "view_spec.json"), "--view", "front", "--view", "side", "--bbox", "0,0,5,5"]) == 1


def test_gem_is_placed_like_a_capsule_from_base_tip_and_radius(calibs):
    """A crystal on a staff head: base / tip anchors in front + side, radius_px -> box [2r, 2r, length] along the axis."""
    base, tip = [0.05, 0.02, 0.30], [0.05, 0.02, 0.38]
    anchors = {v: {"base": px(calibs, v, base), "tip": px(calibs, v, tip), "radius_px": 15} for v in ("front", "side")}
    sol = T.solve_part("crystal", "gem", {"anchors_px": anchors}, calibs)
    assert sol.size == pytest.approx([0.03, 0.03, 0.08], abs=2 * MM)
    assert sol.location == pytest.approx([0.05, 0.02, 0.34], abs=2 * MM)
    spec = parse_scene_spec({"spec_version": "0.1", "name": "t", "components": [
        {"id": "crystal", "type": "gem", "params": {"facets": 6, "anchors_px": anchors}}]})
    assert spec.components[0].type == "gem"
    with pytest.raises(BBRError):
        T.solve_part("crystal", "gem", {"anchors_px": {"front": {"base": base[:2], "radius_px": 5}}}, calibs)
