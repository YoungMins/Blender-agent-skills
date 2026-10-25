"""Hull exclusion driven by part label masks."""
import numpy as np

from bbr.core.partmask import exclude_label_pixels


def test_label_exclusion_removes_protruding_part_but_leaves_no_interior_hole():
    m = np.zeros((60, 60), bool)
    m[5:55, 5:55] = True
    m[25:30, 55:60] = True                                                  # a sword sticking out
    hand = np.zeros_like(m)
    hand[25:35, 25:35] = True                                               # a hand labelled inside the body outline
    sword = np.zeros_like(m)
    sword[25:30, 50:60] = True
    out = exclude_label_pixels(m, hand | sword, 0)
    assert out[25:35, 25:35].all()                                          # no interior hole
    assert not out[25:30, 55:60].any()                                      # the protruding part is gone


def test_label_exclusion_keeps_detached_body_piece_and_dome_wedge():
    yy, xx = np.mgrid[0:200, 0:120]
    dome = ((yy - 150) ** 2 + (xx - 60) ** 2 < 90 ** 2) & (yy < 190)
    horn = np.zeros_like(dome)
    horn[10:100, 52:66] = True
    m = dome | horn
    m[190:196, 0:4] = True                                                  # a small real detached piece: not dropped
    out = exclude_label_pixels(m, horn, 0)
    assert not out[10:56, 50:68].any() and out[75:100, 52:66].all()         # horn gone above the dome, no slot in it
    assert out[190:196, 0:4].all()
