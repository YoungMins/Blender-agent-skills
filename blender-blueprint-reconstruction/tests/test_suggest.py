"""suggest-parts: thin protrusions become part proposals (tip / base sides, matching across views, hair locks)."""
import json
import math

import numpy as np

from bbr.core import suggest
from bbr.core.calibration import ViewCalibration
from bbr.core.labels import polygon_mask
from conftest import ellipse_mask

SIZE = 420
CENTRE = (210.0, 230.0)
R = 80.0
M_PER_PX = 0.005


def _spike(angle_deg, length=55.0, half=11.0):
    """Triangle on the sphere surface at ``angle_deg`` (0 = image right, -90 = up): base inside the sphere, tip outside."""
    a = math.radians(angle_deg)
    u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array(CENTRE)
    base = c + u * (R - 4)
    tip = c + u * (R + length)
    return [base + n * half, tip, base - n * half], tip


def sphere_with_spikes(angles, **kw):
    mask = ellipse_mask(SIZE, SIZE, CENTRE[1], CENTRE[0], R, R)
    tips = []
    for ang in angles:
        poly, tip = _spike(ang, **kw)
        mask |= polygon_mask(mask.shape, poly)
        tips.append(tip)
    return mask, tips


def calib(view="front", axis="-Y", origin_x=CENTRE[0]):
    return ViewCalibration(view, axis, (SIZE, SIZE), M_PER_PX, (origin_x, CENTRE[1] + R))


def test_five_spikes_five_suggestions_with_tip_and_base_sides():
    angles = [-90, -50, -130, 15, 165]
    mask, tips = sphere_with_spikes(angles)
    res, overlay = suggest.suggest_parts({"front": mask}, {"front": calib()})
    sug = res["suggestions"]
    assert len(sug) == 5 and all(s["type"] == "cone" and s["views"] == ["front"] for s in sug)
    comps = {c["id"]: c for c in res["components"]}
    for tip in tips:
        best = min(comps.values(), key=lambda c: np.hypot(*(np.array(c["params"]["anchors_px"]["front"]["tip"]) - tip)))
        a = best["params"]["anchors_px"]["front"]
        assert np.hypot(*(np.array(a["tip"]) - tip)) < 6                               # the tip is the pointed end
        assert np.hypot(*(np.array(a["base"]) - CENTRE)) < np.hypot(*(np.array(a["tip"]) - CENTRE))   # the base is the body side
        assert a["radius_px"][0] > a["radius_px"][1]                                   # wide at the base, narrow at the tip
    assert len(overlay["front"]) == 5


def test_views_matched_by_z_and_x_and_mirror_pair():
    mask, _ = sphere_with_spikes([-60, -120, 20])
    back = mask[:, ::-1]                                                              # the back view is the mirror image
    side, _ = sphere_with_spikes([-60, -120, 20])                                      # a side view with spikes at the same heights
    res, _ = suggest.suggest_parts({"front": mask, "back": back, "side": side},
                                   {"front": calib(), "back": calib("back", "+Y", SIZE - CENTRE[0]), "side": calib("side", "-X")})
    pairs = [s for s in res["suggestions"] if {"front", "back"} <= set(s["views"])]
    assert len(pairs) == 3
    up = [s for s in res["suggestions"] if s["z_range_m"][1] > 0.7]
    assert len(up) == 2 and sum(bool(s["mirror_of"]) for s in up) == 1                # the two upper spikes mirror each other
    assert all(s["confidence"] > 0.5 for s in pairs)


def test_many_small_spikes_on_the_head_become_a_few_locks():
    angles = np.linspace(-125, -55, 10)                                                  # ten thin spikes over the crown
    mask, _ = sphere_with_spikes(angles, length=18.0, half=3.0)
    res, _ = suggest.suggest_parts({"front": mask}, {"front": calib()})
    locks = [s for s in res["suggestions"] if s["hair_lock"]]
    assert 3 <= len(locks) <= 8 and len(res["suggestions"]) <= 8
    comps = {c["id"]: c for c in res["components"]}
    assert all(comps[s["id"]]["type"] == "cone" and 0.4 <= comps[s["id"]]["params"]["flatten"] <= 0.6 for s in locks)


def test_label_polygon_covers_the_piece_and_json_is_pasteable():
    mask, _ = sphere_with_spikes([-90, 20, 160])
    res, overlay = suggest.suggest_parts({"front": mask}, {"front": calib()})
    for s in res["suggestions"]:
        (entry,) = res["labels"]["front"][s["id"]]
        piece = overlay["front"][s["number"]]
        covered = polygon_mask(mask.shape, entry["polygon"]) & mask
        assert (covered & piece).sum() / piece.sum() > 0.9 and covered.sum() < 1.3 * piece.sum()
    json.dumps(res)                                                                   # plain JSON, nothing numpy left


def test_already_labelled_parts_are_skipped_and_regions_replace_polygons():
    mask, _ = sphere_with_spikes([-90, 20])
    pieces, _, _ = suggest.find_pieces(mask)
    id_map = np.zeros(mask.shape, np.uint8)
    id_map[mask] = 1
    for i, p in enumerate(pieces, start=2):
        id_map[p] = i
    res, _ = suggest.suggest_parts({"front": mask}, {"front": calib()}, id_maps={"front": id_map})
    assert all(all(isinstance(e, int) for e in v) for v in res["labels"]["front"].values())
    labelled = {"body": mask & ~pieces[0], "horn": pieces[0]}
    res, _ = suggest.suggest_parts({"front": mask}, {"front": calib()}, label_masks={"front": labelled})
    assert len(res["suggestions"]) == 1 and res["already_labelled"][0]["part"] == "horn"


def test_fox_bench_finds_ears_hair_and_tail(fixtures_dir):
    from bbr.core import measure
    from bbr.core.calibration import calibrate_views
    from bbr.core.config import PipelineConfig
    from bbr.core.spec import load_view_spec
    case = fixtures_dir.parents[1] / "bench" / "cases" / "sd_fox_blind"
    spec = load_view_spec(case / "view_spec.json")
    _, masks = measure.load_view_data(spec, PipelineConfig())
    calibs, _ = calibrate_views(spec, masks)
    res, _ = suggest.suggest_parts(masks, calibs)
    ears = [s for s in res["suggestions"] if s["z_range_m"][1] >= 0.58 and not s["hair_lock"] and len(s["views"]) >= 2]
    assert ears and any(s["mirror_of"] for s in res["suggestions"])                   # an ear pair
    assert any(s["hair_lock"] for s in res["suggestions"])                            # hair spikes are grouped
    assert any(s["curved"] for s in res["suggestions"])                               # the tail
    assert len(res["suggestions"]) <= 25 and set(res["exclude_components"]) <= {s["id"] for s in res["suggestions"]}


def test_outline_and_simplify_round_trip():
    m = ellipse_mask(60, 60, 30, 30, 18, 12)
    from bbr.core.imageops import dilate
    poly = suggest.simplify([(x + 0.5, y + 0.5) for x, y in suggest.trace_outline(dilate(m, 1))])   # label_entries grows the piece by 1 px
    assert 6 <= len(poly) < 60
    filled = polygon_mask(m.shape, poly)
    assert (filled & m).sum() >= 0.99 * m.sum() and filled.sum() < 1.3 * m.sum()
