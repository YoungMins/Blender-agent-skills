"""labels.json -> per-part masks, polygon / split edits, z-range check, label gate of the texel projection."""
import numpy as np
import pytest

from bbr import BBRError
from bbr.core import labels
from bbr.core.calibration import ViewCalibration
from bbr.core.texelproj import project_texel_colors
from conftest import paint


def _grid():
    id_map = np.zeros((40, 60), np.uint8)
    id_map[:, :20] = 1
    id_map[:, 20:40] = 2
    id_map[:, 40:] = 3
    mask = np.ones((40, 60), bool)
    return id_map, mask


def test_regions_union_and_body_default():
    id_map, mask = _grid()
    m = labels.resolve_view({"hand": [1, "+3"], "horn": [2, "-2"]}, id_map, mask)
    assert (m["hand"] == ((id_map == 1) | (id_map == 3))).all()
    assert not m["horn"].any()                                              # "-2" removes region 2 again
    assert (m["body"] == (id_map == 2)).all()                               # unlabelled -> body
    assert sum(int(v.sum()) for v in m.values()) == mask.sum()              # exact partition


def test_polygon_split_and_within_edits():
    id_map, mask = _grid()
    m = labels.resolve_view({"a": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
                             "b": [{"split": 2, "line": [[30, 0], [30, 40]], "side": "left"}],
                             "c": [{"polygon": [[15, 0], [45, 0], [45, 40], [15, 40]], "within": 3}]}, id_map, mask)
    assert m["a"][:10, :10].all() and m["a"].sum() == 100
    assert m["b"][:, 30:40].all() and not m["b"][:, :30].any()              # heading down, "left" = image-right
    assert (m["c"] == ((id_map == 3) & (np.arange(60) < 45)[None])).all()


def test_split_side_convention_and_errors():
    s = labels.side_of_line((10, 10), [[5, 0], [5, 10]], "right")           # heading down the image: right hand = image-left
    assert s[:, :5].all() and not s[:, 5:].any()
    id_map, mask = _grid()
    with pytest.raises(BBRError):
        labels.resolve_view({"x": [9]}, id_map, mask)
    with pytest.raises(BBRError):
        labels.resolve_view({"x": ["oops"]}, id_map, mask)


def test_later_part_wins_a_shared_region_and_body_never_does():
    id_map, mask = _grid()
    m = labels.resolve_view({"a": [1], "b": [1], "body": [1]}, id_map, mask)
    assert (m["b"] == (id_map == 1)).all() and not m["a"].any() and not (m["body"] & (id_map == 1)).any()


def test_load_label_masks_checks_shape(tmp_path):
    id_map, mask = _grid()
    (tmp_path / "regions" / "front").mkdir(parents=True)
    np.save(tmp_path / "regions" / "front" / "id_map.npy", id_map)
    (tmp_path / "labels.json").write_text('{"front": {"horn": [1]}}', encoding="utf-8")
    out = labels.load_label_masks(tmp_path / "labels.json", {"front": mask, "side": mask})
    assert set(out) == {"front"} and out["front"]["horn"].sum() == 800
    with pytest.raises(BBRError):
        labels.load_label_masks(tmp_path / "labels.json", {"front": mask[:30]})


def test_z_range_disagreement_is_flagged():
    front = ViewCalibration("front", "-Y", (60, 100), 0.01, (30.0, 90.0))
    side = ViewCalibration("side", "-X", (60, 100), 0.01, (30.0, 90.0))

    def lm(y0, y1):
        m = np.zeros((100, 60), bool)
        m[y0:y1, 10:20] = True
        return {"hand": m, "body": ~m}

    calibs = {"front": front, "side": side}
    ok = labels.check_z_ranges(labels.part_z_ranges({"front": lm(20, 40), "side": lm(21, 40)}, calibs), 0.9)
    assert not ok["flagged"] and ok["parts"]["hand"]["views"] == 2
    bad = labels.check_z_ranges(labels.part_z_ranges({"front": lm(20, 40), "side": lm(50, 70)}, calibs), 0.9)
    assert bad["flagged"] == ["hand"] and bad["parts"]["hand"]["spread_m"] == pytest.approx(0.30, abs=1e-6)
    assert "DISAGREE" in labels.format_z_check(bad)


def test_texel_projection_rejects_pixels_of_another_label():
    h = w = 20
    mask = np.ones((h, w), bool)
    red, blue = paint(mask, (1.0, 0.0, 0.0)), paint(mask, (0.0, 0.0, 1.0))
    front = ViewCalibration("front", "-Y", (w, h), 0.1, (10.0, 10.0))
    back = ViewCalibration("side", "-X", (w, h), 0.1, (10.0, 10.0))
    pts = np.array([[0.0, 0.0, 0.0]])                                       # projects at pixel (10, 10) in both views
    nrm = np.array([[-0.6, -0.8, 0.0]])                                     # front and side see it, front better
    views = [(red, mask, front), (blue, mask, back)]
    own = np.ones((h, w), bool)
    foreign = np.ones((h, w), bool)
    foreign[2:18, 2:18] = False                                             # the front pixel carries ANOTHER label
    rgb, has, info = project_texel_colors(pts, nrm, views, label_masks=[foreign, own])
    np.testing.assert_allclose(rgb[0], [0, 0, 1], atol=1e-6)               # rejected in front -> takes the side view
    assert has[0] and info["label_rejected"][0]
    _, has2, info2 = project_texel_colors(pts, nrm, [views[0]], label_masks=[foreign])
    assert not has2[0] and info2["label_rejected"][0]                       # no other view: left for the inpainting
    rgb3, _, info3 = project_texel_colors(pts, nrm, views, label_masks=[own, None])
    np.testing.assert_allclose(rgb3[0], [1, 0, 0], atol=1e-6)
    assert not info3["label_rejected"][0]


def test_rect_edit_matches_polygon_and_within():
    id_map, mask = _grid()
    m = labels.resolve_view({"a": [{"rect": [10, 5, 0, 0]}],                          # corners in any order
                             "c": [{"rect": [15, 0, 45, 40], "within": 3}]}, id_map, mask)
    assert m["a"][:5, :10].all() and m["a"].sum() == 50
    assert (m["c"] == ((id_map == 3) & (np.arange(60) < 45)[None])).all()
    with pytest.raises(BBRError):
        labels.resolve_view({"x": [{"rect": [1, 2, 3]}]}, id_map, mask)
