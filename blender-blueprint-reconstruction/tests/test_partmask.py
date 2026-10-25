"""Hull exclusion: part silhouettes are removed from the carve masks without holes."""
import numpy as np

from bbr.core.calibration import ViewCalibration
from bbr.core.partmask import exclude_part_pixels, exclude_parts_from_masks, triangles_mask


def _body():
    m = np.zeros((60, 60), bool)
    m[10:50, 15:45] = True
    return m


def test_part_outside_the_body_is_removed():
    m = _body()
    m[25:30, 45:58] = True                                                 # a horn / sword sticking out
    part = np.zeros_like(m)
    part[24:31, 44:58] = True
    out = exclude_part_pixels(m, part, dilate_px=1)
    assert not out[:, 46:].any()
    assert out[10:50, 15:43].all()                                         # body untouched away from the part


def test_part_overlapping_the_interior_leaves_no_hole():
    m = np.zeros((60, 60), bool)
    m[5:55, 5:55] = True
    part = np.zeros_like(m)
    part[25:35, 25:35] = True                                              # lies entirely inside the body
    out = exclude_part_pixels(m, part, dilate_px=2)
    assert np.array_equal(out, m)


def test_part_touching_the_edge_removes_only_the_protruding_pixels():
    m = _body()
    part = np.zeros_like(m)
    part[20:30, 40:50] = True                                              # half in, half out
    out = exclude_part_pixels(m, part, dilate_px=0)
    assert not out[:, 45:].any()                                           # the protruding half is removed
    assert out[10:20, 15:45].all() and out[30:50, 15:45].all()             # the rest of the body is untouched


def test_existing_holes_stay_holes_and_slivers_are_dropped():
    m = _body()
    m[20:24, 20:24] = False                                                # a genuine hole of the reference
    m[2:4, 50:52] = True                                                   # tiny detached speck
    out = exclude_part_pixels(m, np.zeros_like(m), dilate_px=0)
    assert not out[20:24, 20:24].any() and not out[2:4, 50:52].any()


def test_removal_never_empties_the_mask():
    m = _body()
    assert np.array_equal(exclude_part_pixels(m, np.ones_like(m), 0), m)


def test_triangles_mask_uses_the_view_calibration():
    calib = ViewCalibration("front", "-Y", (40, 40), 0.01, (20.0, 40.0))   # origin at the bottom centre, 1 cm / px
    verts = np.array([[-0.1, 0, 0.1], [0.1, 0, 0.1], [0.1, 0, 0.2], [-0.1, 0, 0.2]])
    mask = triangles_mask(verts, np.array([[0, 1, 2], [0, 2, 3]]), calib)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (10, 29, 20, 29)    # x -0.1..0.1 -> px 10..30, z 0.1..0.2 -> px 30..20


def test_exclude_parts_from_masks_leaves_other_views_and_input_untouched():
    calib = ViewCalibration("front", "-Y", (60, 60), 0.01, (30.0, 55.0))
    m2 = _body()
    m2[25:31, 50:58] = True
    back = _body()
    original = m2.copy()
    tri = (np.array([[0.2, 0, 0.25], [0.28, 0, 0.25], [0.24, 0, 0.3]]), np.array([[0, 1, 2]]))
    out, stats = exclude_parts_from_masks({"front": m2, "back": back}, {"front": calib}, [tri], ["front", "back"], 1)
    assert stats["front"]["part_px"] > 0 and stats["front"]["mask_px_after"] < stats["front"]["mask_px_before"]
    assert "back" not in stats and out["back"] is back
    assert np.array_equal(original, m2)


def test_horn_on_a_dome_leaves_no_slot_in_the_body_outline():
    yy, xx = np.mgrid[0:200, 0:120]
    dome = ((yy - 150) ** 2 + (xx - 60) ** 2 < 90 ** 2) & (yy < 190)       # helmet-like body, outline top at y = 60
    horn = np.zeros_like(dome)
    horn[10:100, 52:66] = True                                              # a 14 px wide horn rising out of the dome
    m = dome | horn
    out = exclude_part_pixels(m, horn, dilate_px=1)
    assert not out[10:56, 50:68].any()                                      # the protruding half is gone
    overlap = dome & horn
    assert out[overlap].mean() > 0.9 and out[75:100, 52:66].all()           # the wedge inside the outline is kept: no slot
    assert (out & ~m).sum() == 0 and (dome & ~horn & ~out).sum() < 40       # at most a few rim pixels


def test_thick_part_inside_a_wide_notch_is_not_refilled():
    m = _body()
    m[25:40, 45:58] = True                                                  # a 15 px thick arm leaving the body
    part = np.zeros_like(m)
    part[25:40, 40:58] = True
    out = exclude_part_pixels(m, part, dilate_px=0)
    assert not out[:, 46:].any()                                            # the protruding part is removed


def test_thin_leftover_next_to_the_cut_is_dropped_but_thin_body_parts_far_away_stay():
    m = _body()
    m[25:35, 45:58] = True                                                  # the part
    m[22:24, 45:58] = True                                                  # a 2 px crescent the part model does not cover
    m[50:53, 15:45] = True                                                  # a thin body feature far from the cut
    part = np.zeros_like(m)
    part[25:35, 44:58] = True
    out = exclude_part_pixels(m, part, dilate_px=1)
    assert not out[20:25, 45:].any() and out[50:53, 15:45].all()
