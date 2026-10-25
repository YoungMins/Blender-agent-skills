"""The horizontal origin follows the body core, not the silhouette bbox that a held prop drags."""
import pytest

from bbr.core.calibration import body_core_centre, calibrate_views
from bbr.core.imageops import mask_bbox
from bbr.core.spec import parse_view_spec
from bbr.core.symmetry import mirror_iou
from conftest import ellipse_mask, rect_mask


def _penguin():
    """Body centred on column 100 (x 70..130, rows 40..200) plus a thick held prop at x 150..190, rows 120..190."""
    body = ellipse_mask(220, 220, 120, 100, 80, 30)
    prop = rect_mask(220, 220, 150, 120, 190, 190)
    return body | prop, prop


def _spec():
    return parse_view_spec({"spec_version": "0.1", "known_dimensions": {"height": 1.0},
                            "views": {"front": {"path": "f.png"}, "side": {"path": "s.png", "camera_axis": "-X"}}})


def test_unlabelled_core_ignores_thin_prop():
    mask = ellipse_mask(220, 220, 120, 100, 80, 30) | rect_mask(220, 220, 130, 70, 215, 76)   # thin spear to the right
    x0, y0, x1, y1 = mask_bbox(mask)
    assert (x0 + x1) / 2 > 110                                         # the bbox centre is dragged
    assert body_core_centre(mask, (x0, y0, x1, y1)) == pytest.approx(100.0, abs=1.0)


def test_labelled_core_uses_body_label_only():
    mask, prop = _penguin()
    bbox = mask_bbox(mask)
    labels = {"spatula": prop, "body": mask & ~prop}
    assert body_core_centre(mask, bbox, labels) == pytest.approx(100.0, abs=1.0)
    assert body_core_centre(mask, bbox, {"spatula": prop}) == pytest.approx(100.0, abs=1.0)   # no body label: mask minus parts
    assert body_core_centre(mask, bbox, {"_note": prop}) != pytest.approx(100.0, abs=1.0)    # "_" labels are not parts


def test_calibrate_views_origin_and_mirror_iou():
    mask, prop = _penguin()
    side = ellipse_mask(220, 160, 120, 80, 80, 40)
    calibs, _ = calibrate_views(_spec(), {"front": mask, "side": side}, {"front": {"spatula": prop, "body": mask & ~prop}})
    assert calibs["front"].origin_px[0] == pytest.approx(100.0, abs=1.0)
    assert calibs["front"].origin_px[1] == mask_bbox(mask)[3]
    bbox_calibs, _ = calibrate_views(_spec(), {"front": mask, "side": side})
    # the unlabelled core is still a body-core centre, but with a thick prop the labelled one is the exact axis
    assert mirror_iou(mask, calibs["front"], prop) >= 0.95
    assert mirror_iou(mask, calibs["front"], prop) > mirror_iou(mask, bbox_calibs["front"], prop)


def test_small_axis_offsets_keep_the_bbox_centre():
    body = ellipse_mask(220, 220, 120, 100, 80, 30)
    mask = body | rect_mask(220, 220, 130, 150, 133, 190)         # a 3 px sliver on one side: the axis moves by about 1 px
    calibs, _ = calibrate_views(_spec(), {"front": mask, "side": body})
    assert calibs["front"].origin_px[0] == (mask_bbox(mask)[0] + mask_bbox(mask)[2]) / 2.0


def test_front_and_back_switch_to_the_body_axis_together():
    from bbr.core.calibration import _body_axes
    front = ellipse_mask(220, 220, 120, 100, 80, 30) | rect_mask(220, 220, 150, 100, 156, 200)    # a thin sword drags the bbox
    back = ellipse_mask(220, 220, 120, 100, 80, 30) | rect_mask(220, 220, 130, 150, 133, 190)     # off by noise only
    bboxes = {"front": mask_bbox(front), "back": mask_bbox(back)}
    both = _body_axes({"front": front, "back": back}, bboxes, None)
    assert set(both) == {"front", "back"} and both["back"] == pytest.approx(100.0, abs=1.0)
    assert _body_axes({"back": back}, bboxes, None) == {}          # alone, the back view keeps its bbox centre
