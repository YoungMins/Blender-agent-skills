"""Part-gap check: segment distance, label adjacency, which pairs are reported."""
import numpy as np
import pytest

from bbr.core import gapcheck as G
from conftest import rect_mask


def part(base, tip, r):
    return {"keypoints": {"base": np.array(base, float), "tip": np.array(tip, float)}, "clearance_radius": r}


def test_surface_gap_is_the_polyline_distance_minus_both_radii():
    a = part([0, 0, 0], [0, 0, 0.1], 0.01)
    assert G.surface_gap(a, part([0, 0, 0.124], [0, 0, 0.2], 0.01)) == pytest.approx(0.004)      # 24 mm - 20 mm
    assert G.surface_gap(a, part([0.1, 0, 0.05], [0.2, 0, 0.05], 0.01)) == pytest.approx(0.08)  # beside the middle
    assert G.surface_gap(a, part([0, 0, 0.1], [0, 0, 0.2], 0.01)) == 0.0                         # touching ends
    point = {"keypoints": {"center": np.array([0.0, 0.05, 0.05])}, "clearance_radius": 0.01}
    assert G.surface_gap(a, point) == pytest.approx(0.03)


def test_declared_pair_beyond_tolerance_is_reported_and_a_touching_one_is_not():
    placed = {"handle": part([0, 0, 0], [0, 0, 0.2], 0.0075), "head": part([0, 0, -0.03], [0, 0, -0.1], 0.004)}
    gaps = G.find_gaps(placed, [("handle", "head")])
    assert len(gaps) == 1 and gaps[0]["gap_m"] == pytest.approx(0.0185) and gaps[0]["why"] == "declared adjacent"
    assert "handle and head float" in G.format_gap(gaps[0]) and "params.joint" in G.format_gap(gaps[0])
    placed["head"] = part([0, 0, 0], [0, 0, -0.1], 0.004)                               # base on the handle base
    assert G.find_gaps(placed, [("handle", "head")]) == []
    far = {**placed, "head": part([0, 0, -0.03], [0, 0, -0.1], 0.004)}
    assert G.find_gaps(far, [("handle", "head")], tol={"head": 0.03}) == []              # per-part tolerance
    assert G.find_gaps({"handle": placed["handle"]}, [("handle", "head")]) == []        # unplaced partner: skipped


def labels(*views):
    return {view: {name: rect_mask(100, 100, *box) for name, box in per.items()} for view, per in views}


def test_label_adjacency_needs_contact_in_every_shared_view():
    touch = {"a": (10, 10, 30, 50), "b": (30, 10, 50, 50)}
    apart = {"a": (10, 10, 30, 50), "b": (60, 10, 80, 50)}
    both = labels(("front", touch), ("side", touch), ("back", touch))
    assert G.label_adjacent_pairs(both, ["a", "b"]) == [("a", "b")]
    one_apart = labels(("front", touch), ("side", apart), ("back", touch))
    assert G.label_adjacent_pairs(one_apart, ["a", "b"]) == []                          # hidden behind in one view: not adjacent
    assert G.label_adjacent_pairs(labels(("front", touch)), ["a", "b"]) == []           # one shared view is too little


def test_label_pairs_that_float_are_reported_too():
    placed = {"a": part([0, 0, 0], [0, 0, 0.1], 0.005), "b": part([0.1, 0, 0], [0.1, 0, 0.1], 0.005)}
    gaps = G.find_gaps(placed, [], [("a", "b")])
    assert gaps[0]["why"].startswith("labels touch") and gaps[0]["gap_m"] == pytest.approx(0.09)


def test_keypoints_typed_on_the_same_pixel_are_a_joint_that_must_stay_together():
    handle = {"front": {"base": [118, 543], "tip": [147, 443]}, "side": {"base": [280, 560], "tip": [236, 470]}, "back": {"base": [494, 516]}}
    head = {"front": {"base": [118, 543], "tip": [73, 612]}, "side": {"base": [280, 561], "tip": [335, 625]}, "back": {"base": [494, 516]}}
    other = {"front": {"base": [300, 300], "tip": [310, 400]}, "side": {"base": [90, 300]}}
    assert G.shared_anchors({"handle": handle, "head": head, "arm": other}) == [("handle", "head", "base", "base")]
    assert G.shared_anchors({"handle": handle, "head": {"front": head["front"]}}) == []                 # one view is not enough
    moved = {"front": {"base": [121, 543]}, "side": {"base": [290, 560]}}                                  # 10 px off in one view
    assert G.shared_anchors({"handle": handle, "head": moved}) == []
    placed = {"handle": part([0, 0, 0.106], [0.05, 0.07, 0.22], 0.0144), "head": part([0.009, 0.02, 0.127], [-0.02, -0.06, 0.06], 0.004)}
    gaps = G.find_gaps(placed, [], joints=[("handle", "head", "base", "base")])
    assert len(gaps) == 1 and gaps[0]["gap_m"] == pytest.approx(0.0305, abs=0.001) and gaps[0]["ka"] == "base"
    text = G.format_gap(gaps[0])
    assert "handle.base and head.base were anchored to the same pixel" in text and '"at": "base", "self": "base"' in text
    assert G.find_gaps({**placed, "head": part([0, 0.002, 0.106], [0, -0.06, 0.06], 0.004)}, [], joints=[("handle", "head", "base", "base")]) == []
