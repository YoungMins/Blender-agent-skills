"""Defects of the blind twin-tail run: unmeasured-coordinate defaults, floating_part, strands, hanging-part base end, anchors CLI."""
import json
import sys

import numpy as np
import pytest

from bbr.core import gapcheck, imageio, measure, partgeom, summary, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.spec import parse_scene_spec
from bbr.core.zbuffer import depth_buffer
from conftest import SK, ellipse_mask, paint, rect_mask

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402

MM = 0.001


@pytest.fixture
def calibs():
    return {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
            "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0)),
            "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 450.0))}


def px(calibs, view, point):
    return [float(v) for v in np.round(calibs[view].world_to_px([point])[0])]


def anchors(calibs, views, base, tip, r=10):
    return {v: {"base": px(calibs, v, base), "tip": px(calibs, v, tip), "radius_px": r} for v in views}


# --------------------------------------------------------------------------- 1. defaults of an unmeasured coordinate
def test_unmeasured_y_is_mirrored_from_the_measured_partner(calibs):
    """tail_r is seen in the side view (y = 0.06 m), tail_l only in front + back: its y comes from the partner (x is mirrored)."""
    right = T.solve_part("tail_r", "capsule", {"anchors_px": anchors(calibs, ("front", "side"), [-0.09, 0.06, 0.3], [-0.12, 0.06, 0.1])}, calibs)
    left_anchors = anchors(calibs, ("front", "back"), [0.09, 0.0, 0.3], [0.12, 0.0, 0.1])
    left = T.solve_part("tail_l", "capsule", {"anchors_px": left_anchors}, calibs, auto_sibling=lambda: right)
    assert left.keypoints["base"][1] == pytest.approx(0.06, abs=2 * MM) and left.keypoints["tip"][1] == pytest.approx(0.06, abs=2 * MM)
    assert left.keypoints["base"][0] == pytest.approx(0.09, abs=2 * MM)                      # measured x untouched
    assert left.sources["base"]["y"] == "mirror:auto" and left.inferred
    assert any("mirrored from the partner" in w for w in left.warnings)


def test_a_partner_that_guessed_its_own_coordinate_is_not_a_measurement(calibs):
    guess = T.solve_part("tail_r", "capsule", {"anchors_px": anchors(calibs, ("front", "back"), [-0.09, 0, 0.3], [-0.12, 0, 0.1])}, calibs)
    left = T.solve_part("tail_l", "capsule", {"anchors_px": anchors(calibs, ("front", "back"), [0.09, 0, 0.3], [0.12, 0, 0.1])}, calibs,
                        auto_sibling=lambda: guess)
    assert left.sources["base"]["y"] == "default:body_centre"


def _box_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return c, np.array(tris)


def test_unmeasured_y_falls_back_on_contact_with_the_nearest_body_surface(calibs):
    """No partner, seen in the front only: the part sits on the body's front surface (the existing attach_to logic)."""
    body = _box_mesh([-0.1, -0.05, 0.0], [0.1, 0.05, 0.4])                       # front face at y = -0.05
    zbuf = {v: depth_buffer(*body, calibs[v]) for v in ("front", "back")}
    params = {"anchors_px": {"front": {"center": px(calibs, "front", [0.0, 0.0, 0.2]), "radius_px": 20}}}
    sol = T.solve_part("badge", "sphere", params, calibs, auto_depth_for=lambda v: zbuf.get(v))
    assert sol.keypoints["center"][1] == pytest.approx(-0.05 - 0.02, abs=2 * MM)         # surface height + its own half thickness, camera side
    assert sol.sources["center"]["y"] == "contact:auto:body@front"
    far = {"anchors_px": {"front": {"center": px(calibs, "front", [0.3, 0.0, 0.2]), "radius_px": 20}}}   # beside the body: no contact
    assert T.solve_part("badge", "sphere", far, calibs, auto_depth_for=lambda v: zbuf.get(v)).sources["center"]["y"] == "default:body_centre"


def test_explicit_sources_still_beat_the_automatic_ones(calibs):
    partner = T.solve_part("tail_r", "capsule", {"anchors_px": anchors(calibs, ("front", "side"), [-0.09, 0.06, 0.3], [-0.12, 0.06, 0.1])}, calibs)
    params = {"anchors_px": anchors(calibs, ("front", "back"), [0.09, 0, 0.3], [0.12, 0, 0.1]), "depth_m": 0.01}
    sol = T.solve_part("tail_l", "capsule", params, calibs, auto_sibling=lambda: partner)
    assert sol.sources["base"]["y"] == "depth_m" and sol.keypoints["base"][1] == pytest.approx(0.01)


# --------------------------------------------------------------------------- 1b. floating_part
def _tail_scene():
    head = _box_mesh([-0.1, -0.1, 0.4], [0.1, 0.1, 0.6])
    touching = _box_mesh([0.1, -0.02, 0.45], [0.16, 0.02, 0.55])               # shares the head's side face
    detached = _box_mesh([-0.16, 0.14, 0.45], [-0.12, 0.18, 0.55])             # 4 cm behind and 2 cm beside the head
    return {"head": head, "tail_l": touching, "tail_r": detached}


def test_floating_part_names_the_part_and_the_gap():
    found = gapcheck.floating_parts(_tail_scene(), 0.6)
    assert [g["part"] for g in found] == ["tail_r"]
    g = found[0]
    assert g["nearest"] == "head" and g["tol_m"] == pytest.approx(0.01)       # 25 % of its 4 cm minor dimension (1 % of the height is 6 mm)
    assert g["gap_m"] == pytest.approx(np.hypot(0.02, 0.04), abs=0.01)           # true gap 44.7 mm (sampling error a few mm)
    assert "tail_r floats" in gapcheck.format_floating(g) and "mm" in gapcheck.format_floating(g)


def test_floating_part_tolerance_skip_and_overlap():
    scene = _tail_scene()
    assert gapcheck.floating_parts(scene, 0.6, tols={"tail_r": 0.1}) == []                 # per-part tolerance (gap_tol_m)
    assert gapcheck.floating_parts(scene, 0.6, skip=["tail_r"]) == []                      # params.floating_ok
    scene["tail_r"] = _box_mesh([0.05, 0.05, 0.45], [0.2, 0.2, 0.55])                      # crosses the head: touching, not floating
    assert gapcheck.floating_parts(scene, 0.6) == []


def test_result_block_warns_about_floating_parts_and_tells_a_failed_run_what_to_do():
    report = {"name": "t", "status": "failed_validation", "passed": False,
              "floating_parts": [{"part": "tail_l", "gap_m": 0.038, "tol_m": 0.006, "nearest": "body"}],
              "validation": {"passed": False, "geometry": {"checks": {}}, "reference_match": {"passed": False, "total_error": 0.104, "threshold": 0.08, "per_view": {}}}}
    lines = summary.format_summary(report).splitlines()
    assert any(l.startswith("WARNING floating_part: tail_l floats 38 mm") for l in lines)
    assert lines[-1].startswith("=====") and lines[-2].startswith("next: FAILED")
    assert "re-run" in lines[-2] and "reference_match" in lines[-2]
    ok = summary.format_summary({"name": "t", "status": "passed", "passed": True})
    assert "next:" not in ok


# --------------------------------------------------------------------------- 2. strand
def test_strand_mesh_is_a_closed_tapered_tube_through_its_points():
    pts = [[0, 0, 0], [0.03, 0, -0.1], [-0.03, 0, -0.2], [0.02, 0, -0.3]]
    v, f = partgeom.strand_mesh(pts, [0.03, 0.025, 0.02, 0.01], flatten=0.5)
    assert partgeom.mesh_volume(v, f) > 0
    edges = {}
    for face in f:
        for a, b in zip(face, face[1:] + face[:1]):
            edges[(min(a, b), max(a, b))] = edges.get((min(a, b), max(a, b)), 0) + 1
    assert set(edges.values()) == {2}                                               # watertight
    sides, per_span = partgeom.DEFAULT_SIDES, partgeom.DEFAULT_STRAND_SPAN_SEGMENTS
    rings = v[:(3 * per_span + 1) * sides].reshape(-1, sides, 3)                    # ring k = the k-th cross-section along the curve
    for i, p in enumerate(pts):                                                     # the centre line passes through every anchor
        assert rings[i * per_span].mean(axis=0) == pytest.approx(p, abs=0.001)
    extent = lambda k, axis: float(np.ptp(rings[k][:, axis]))
    assert extent(0, 0) > extent(3 * per_span, 0) * 2.5                              # taper 0.03 -> 0.01
    assert extent(per_span, 1) < 0.65 * extent(per_span, 0)                          # flatten 0.5: much thinner in depth than wide
    with pytest.raises(Exception):
        partgeom.strand_mesh(pts[:2], [0.02, 0.02])


def _strand_anchors(calibs, views, pts, radii):
    out = {}
    for v in views:
        out[v] = {f"p{i}": px(calibs, v, p) for i, p in enumerate(pts)}
        out[v]["radius_px"] = radii
    return out


def test_strand_is_placed_from_ordered_anchors_with_a_radius_per_point(calibs):
    pts = [[-0.1, 0.02, 0.4], [-0.12, 0.03, 0.3], [-0.09, 0.04, 0.2], [-0.13, 0.05, 0.1]]
    sol = T.solve_part("tail", "strand", {"anchors_px": _strand_anchors(calibs, ("front", "side"), pts, [30, 25, 20, 10])}, calibs)
    for name, p in zip(("p0", "p1", "p2", "p3"), pts):
        assert sol.keypoints[name] == pytest.approx(p, abs=2 * MM)
    assert sol.location == pytest.approx(pts[0], abs=2 * MM)
    assert sol.params["radii"] == pytest.approx([0.03, 0.025, 0.02, 0.01], abs=1e-6)
    assert np.array(sol.params["points"])[0].tolist() == [0.0, 0.0, 0.0] and not sol.inferred
    two = _strand_anchors(calibs, ("front", "side"), pts[:2], [10, 10])
    with pytest.raises(Exception, match="at least 3"):
        T.solve_part("tail", "strand", {"anchors_px": two}, calibs)
    ends = _strand_anchors(calibs, ("front", "side"), pts, [30, 10])                       # base / tip radius only: interpolated
    assert T.solve_part("tail", "strand", {"anchors_px": ends}, calibs).params["radii"] == pytest.approx([0.03, 0.0233, 0.0167, 0.01], abs=1e-3)
    spec = parse_scene_spec({"spec_version": "0.1", "name": "t", "components": [
        {"id": "tail", "type": "strand", "params": {"anchors_px": ends, "flatten": 0.7}}]})
    assert spec.components[0].type == "strand"                                              # needs no size


def test_a_strand_hidden_in_the_side_view_takes_its_depth_from_the_partner(calibs):
    pts_r = [[-0.1, 0.05, 0.4], [-0.12, 0.05, 0.3], [-0.09, 0.05, 0.2]]
    pts_l = [[0.1, 0.0, 0.4], [0.12, 0.0, 0.3], [0.09, 0.0, 0.2]]
    right = T.solve_part("tail_r", "strand", {"anchors_px": _strand_anchors(calibs, ("front", "side"), pts_r, [20, 20, 10])}, calibs)
    left = T.solve_part("tail_l", "strand", {"anchors_px": _strand_anchors(calibs, ("front", "back"), pts_l, [20, 20, 10])}, calibs,
                        auto_sibling=lambda: right)
    assert [left.keypoints[f"p{i}"][1] for i in range(3)] == pytest.approx([0.05] * 3, abs=2 * MM)
    assert left.keypoints["p1"][0] == pytest.approx(0.12, abs=2 * MM)


# --------------------------------------------------------------------------- 2b/3. anchors on masks
def hanging_scene():
    """A head with a tail that hangs from its side: the tail's top end is 30 px from the head, its tip 53 px from the body
    (a shoulder block), so the distance rule alone cannot tell the ends apart. Returns (silhouette, tail, rest)."""
    head = ellipse_mask(400, 300, 80, 100, 55, 55)
    tail = rect_mask(400, 300, 185, 60, 207, 290)
    body = rect_mask(400, 300, 60, 150, 132, 300)
    sil = head | tail | body
    return sil, tail, sil & ~tail


def test_the_base_of_a_hanging_part_is_the_end_that_touches_the_most_other_foreground():
    sil, tail, rest = hanging_scene()
    top, bottom = np.array([196.0, 60.0]), np.array([196.0, 290.0])
    pts = measure.rest_points(rest)
    assert measure.attach_base(top, bottom, 230.0, pts) is None                       # the distance rule alone is undecided here
    assert measure.attach_base(top, bottom, 230.0, pts, measure.contact_reach(11.0, 230.0)) is False   # end_a (the top) is the base
    res = measure.propose_anchors(tail, [180, 55, 212, 295], "tube", rest=rest)
    assert res["base_by"] == "attach" and res["base"][1] < res["tip"][1]              # base at the top although the rod is vertical


def test_standing_parts_keep_the_old_rule_without_a_clear_contact():
    tail = rect_mask(300, 300, 140, 60, 160, 280)
    res = measure.propose_anchors(tail, [130, 50, 170, 290], "tube", rest=None)
    assert res["base_by"] == "lower" and res["base"][1] > res["tip"][1]


def wavy_mask(h=500, w=300):
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for y in range(40, 460, 2):
        cx, r = 150 + 40 * np.sin((y - 40) / 70.0), 22 - 14 * (y - 40) / 420.0
        m |= (yy - y) ** 2 + (xx - cx) ** 2 <= r * r
    return m


def test_strand_anchors_follow_the_medial_axis_from_the_attachment_end():
    tail = wavy_mask()
    head = ellipse_mask(500, 300, 20, 150, 70, 60)                                    # the head covers the top of the strand
    rest = (head & ~tail)
    res = measure.propose_anchors(tail, [60, 30, 240, 470], "strand", rest=rest, n_points=6)
    names = measure.strand_points(res)
    assert names == [f"p{i}" for i in range(6)] and res["suggested_type"] == "strand"
    ys = [res[n][1] for n in names]
    assert ys == sorted(ys) and ys[0] < 80 and ys[-1] > 420                            # ordered from the attachment end down
    assert res["base_by"] == "attach"
    for n in names:                                                                    # every point lies inside the mask
        assert tail[int(res[n][1]), int(res[n][0])]
    r = res["radius_px"]
    assert len(r) == 6 and r[0] > r[-1] * 1.4                                           # taper
    assert np.ptp([res[n][0] for n in names]) > 40                                      # it follows the waves, not a straight line
    with pytest.raises(Exception, match="at least 3"):
        measure.propose_anchors(tail, [60, 30, 240, 470], "strand", n_points=2)


# --------------------------------------------------------------------------- 4. anchors CLI
def _views(tmp_path, front_part=True, side_part=True):
    sil, tail, _ = hanging_scene()
    imageio.save_rgba(tmp_path / "front.png", paint(sil, (0.9, 0.5, 0.6)))
    imageio.save_rgba(tmp_path / "side.png", paint(sil, (0.9, 0.5, 0.6)))
    spec = {"spec_version": "0.1", "name": "t", "known_dimensions": {"height": 0.5},
            "views": {"front": {"path": "front.png"}, "side": {"path": "side.png", "camera_axis": "-X"}}}
    (tmp_path / "view_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    poly = [[180, 55], [212, 55], [212, 295], [180, 295]]
    labels = {"_regions": "regions", "front": {"tail": [{"polygon": poly}], "body": []},
              "side": {"tail": [{"polygon": poly}] if side_part else [], "body": []}}
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    for name in ("front", "side"):
        (tmp_path / "regions" / name).mkdir(parents=True)
        np.save(tmp_path / "regions" / name / "id_map.npy", np.zeros(sil.shape, np.uint8))


def run_anchors(tmp_path, capsys, *extra):
    code = cli.main(["anchors", "--view-spec", str(tmp_path / "view_spec.json"), "--labels", str(tmp_path / "labels.json"), "--part", "tail", *extra])
    cap = capsys.readouterr()
    return code, cap


def test_cli_base_hint_per_view_plain_or_named(tmp_path, capsys):
    _views(tmp_path)
    top, bottom = "196,62", "196,288"
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--view", "side", "--kind", "tube", "--base-hint", top, "--base-hint", top)
    assert code == 0
    a = json.loads(cap.out)["tail"]["params"]["anchors_px"]
    assert all(a[v]["base"][1] < a[v]["tip"][1] for v in ("front", "side"))
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--view", "side", "--kind", "tube",
                            "--base-hint", f"side:{bottom}", "--base-hint", f"front:{bottom}")
    a = json.loads(cap.out)["tail"]["params"]["anchors_px"]
    assert code == 0 and all(a[v]["base"][1] > a[v]["tip"][1] for v in ("front", "side"))      # the hints beat the contact rule
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--view", "side", "--base-hint", top)   # one plain hint for two views
    assert code == 1 and "one per view" in cap.err
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--base-hint", "top:1,2")
    assert code == 1 and "not one of --view" in cap.err


def test_cli_skips_a_view_where_the_part_is_not_labelled_and_prints_a_strand(tmp_path, capsys):
    _views(tmp_path, side_part=False)
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--view", "side", "--kind", "strand", "--points", "4", "--id", "tail",
                            "--overlay", str(tmp_path / "ov.png"))
    assert code == 0
    assert "view 'side' skipped" in cap.err and "no label pixels" in cap.err
    out = json.loads(cap.out)["tail"]
    assert out["suggested_type"] == "strand" and list(out["params"]["anchors_px"]) == ["front"]
    front = out["params"]["anchors_px"]["front"]
    assert [k for k in front if k.startswith("p")] == ["p0", "p1", "p2", "p3"] and len(front["radius_px"]) == 4
    assert front["p0"][1] < front["p3"][1] and (tmp_path / "ov.png").is_file()
    _views(tmp_path / "x") if False else None
    (tmp_path / "labels.json").write_text(json.dumps({"_regions": "regions", "front": {"body": []}, "side": {"body": []}}), encoding="utf-8")
    code, cap = run_anchors(tmp_path, capsys, "--view", "front", "--view", "side")
    assert code == 1 and "no label pixels in any" in cap.err
