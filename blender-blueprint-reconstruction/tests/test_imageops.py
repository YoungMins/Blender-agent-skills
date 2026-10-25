"""Image operations: masks, flood fill, labelling, morphology, sheet splitting."""
import itertools
import time
from collections import deque

import numpy as np
import pytest

from bbr import ImageError
from bbr.core import imageops as ops
from conftest import ellipse_mask, paint, rect_mask


def _bfs_labels(mask, full):
    """Reference labelling (python BFS) for cross-checking label_components."""
    offs = [o for o in itertools.product((-1, 0, 1), repeat=mask.ndim) if any(o)]
    if not full:
        offs = [o for o in offs if sum(map(abs, o)) == 1]
    lab = np.zeros(mask.shape, int)
    n = 0
    for start in zip(*np.nonzero(mask)):
        if lab[start]:
            continue
        n += 1
        lab[start] = n
        q = deque([start])
        while q:
            p = q.popleft()
            for o in offs:
                t = tuple(a + b for a, b in zip(p, o))
                if all(0 <= t[i] < mask.shape[i] for i in range(mask.ndim)) and mask[t] and not lab[t]:
                    lab[t] = n
                    q.append(t)
    return lab, n


@pytest.mark.parametrize("shape,full", [((24, 31), True), ((24, 31), False), ((9, 10, 11), True), ((9, 10, 11), False)])
def test_label_components_matches_bfs(shape, full):
    rng = np.random.default_rng(7)
    for density in (0.3, 0.5, 0.65):
        m = rng.random(shape) < density
        got, n = ops.label_components(m, full)
        ref, rn = _bfs_labels(m, full)
        assert n == rn
        assert np.array_equal(got, ref)  # same labels: both follow raster order of the first voxel


def test_label_empty_and_full():
    assert ops.label_components(np.zeros((5, 5), bool))[1] == 0
    labels, n = ops.label_components(np.ones((5, 5), bool), False)
    assert n == 1 and labels.min() == 1


def test_mask_on_solid_background():
    m = ellipse_mask(100, 120, 50, 60, 30, 40)
    got = ops.extract_mask(paint(m, (0.9, 0.9, 0.9)))
    assert (got ^ m).sum() == 0
    assert ops.mask_bbox(got) == ops.mask_bbox(m)


def test_white_object_on_white_bg_with_dark_outline():
    h, w = 120, 160
    outer = rect_mask(h, w, 30, 20, 130, 100)
    inner = rect_mask(h, w, 34, 24, 126, 96)
    img = np.ones((h, w, 4), np.float32)
    img[outer & ~inner] = (0.1, 0.1, 0.1, 1.0)           # dark outline, white interior, white background
    got = ops.extract_mask(img)
    assert got[60, 80] and got[25, 35]                    # interior kept
    assert not got[5, 5]
    assert (got ^ outer).sum() == 0


def test_enclosed_background_colour_regions_are_filled_when_not_removed():
    """remove_enclosed=False: enclosed regions of background colour stay foreground (border flood fill only)."""
    m = ellipse_mask(100, 100, 50, 50, 40, 40) & ~ellipse_mask(100, 100, 50, 50, 5, 5)
    got = ops.extract_mask(paint(m, (0.9, 0.2, 0.2)), remove_enclosed=False)
    assert got[50, 50] and not got[2, 2]


def test_enclosed_chroma_hole_is_removed_by_default(caplog):
    """A chroma backdrop seen through a ring (staff / hair / hat gap): removed, warned about, ring kept."""
    m = ellipse_mask(200, 200, 100, 100, 80, 80) & ~ellipse_mask(200, 200, 100, 100, 25, 25)
    img = paint(m, (0.9, 0.2, 0.2))
    with caplog.at_level("WARNING"):
        got = ops.extract_mask(img)
    assert not got[100, 100] and got[100, 40] and not got[2, 2]
    assert (got ^ m).sum() == 0
    assert "1 enclosed background hole" in caplog.text
    assert ops.extract_mask(img, remove_enclosed=False)[100, 100]


def test_small_enclosed_hole_and_neutral_backdrop_are_kept():
    m = ellipse_mask(200, 200, 100, 100, 80, 80) & ~ellipse_mask(200, 200, 100, 100, 1.5, 1.5)   # ~7 px < 0.02 % of the image
    assert ops.extract_mask(paint(m, (0.9, 0.2, 0.2)))[100, 100]
    big = ellipse_mask(200, 200, 100, 100, 80, 80) & ~ellipse_mask(200, 200, 100, 100, 25, 25)
    assert ops.extract_mask(paint(big, (0.9, 0.2, 0.2), bg=(0.95, 0.95, 0.95)))[100, 100]   # white-ish backdrop: off by default
    assert not ops.extract_mask(paint(big, (0.9, 0.2, 0.2), bg=(0.95, 0.95, 0.95)), remove_enclosed=True)[100, 100]


def test_enclosed_look_alike_with_texture_or_off_colour_is_kept():
    """Shaded / textured regions near the backdrop colour and holes that differ from it stay foreground."""
    ring = ellipse_mask(200, 200, 100, 100, 80, 80) & ~ellipse_mask(200, 200, 100, 100, 25, 25)
    img = paint(ring, (0.9, 0.2, 0.2))
    hole = ellipse_mask(200, 200, 100, 100, 25, 25)
    noise = np.random.default_rng(0).uniform(-0.07, 0.07, (200, 200))
    textured = img.copy()
    textured[hole, 1] += noise[hole].astype(np.float32)               # checker-like jitter, still within tolerance
    assert ops.extract_mask(textured)[100, 100]
    darker = img.copy()
    darker[hole, :3] = (0.0, 0.8 - 0.1, 0.2)                             # 0.1 off the backdrop: inside tol, above half of it
    assert ops.extract_mask(darker)[100, 100]


def test_alpha_mode_holes_and_specks():
    h, w = 200, 200
    m = rect_mask(h, w, 40, 40, 160, 160)
    big_hole = rect_mask(h, w, 70, 70, 130, 130)          # 9 % of the image: kept as hole
    small_hole = rect_mask(h, w, 50, 50, 54, 54)          # 0.04 %: filled
    speck = rect_mask(h, w, 5, 5, 8, 8)
    alpha = (m & ~big_hole & ~small_hole) | speck
    img = np.zeros((h, w, 4), np.float32)
    img[alpha] = (0.5, 0.5, 0.5, 1.0)
    got = ops.extract_mask(img)  # auto -> alpha
    assert got[52, 52] and not got[100, 100] and not got[6, 6] and got[45, 45]


def test_color_mode_with_explicit_background():
    m = rect_mask(50, 50, 10, 10, 40, 40)
    img = paint(m, (1, 1, 1), bg=(1, 0, 1))
    assert (ops.extract_mask(img, "color", color=(1, 0, 1)) ^ m).sum() == 0
    with pytest.raises(ImageError):
        ops.extract_mask(img, "banana")


def test_flood_fill_speed_on_large_image():
    h, w = 887, 1774
    m = ellipse_mask(h, w, 440, 880, 380, 700) & ~ellipse_mask(h, w, 440, 880, 100, 200)
    img = paint(m, (0.95, 0.95, 0.95))
    img[::7, ::5, :3] += 0.01  # mild texture
    t = time.perf_counter()
    got = ops.extract_mask(img)
    elapsed = time.perf_counter() - t
    assert elapsed < 3.0, elapsed
    assert got.sum() > 0.5 * m.sum()


def test_bbox_crop_pad_resize():
    m = rect_mask(20, 30, 5, 3, 12, 9)
    assert ops.mask_bbox(m) == (5, 3, 12, 9)
    with pytest.raises(ImageError):
        ops.mask_bbox(np.zeros((4, 4), bool))
    assert ops.crop(m, (5, 3, 12, 9)).all()
    assert ops.pad_to(np.ones((2, 2)), (6, 4)).sum() == 4
    assert ops.pad_to(np.ones((2, 2)), (6, 4), align="topleft")[0, 0] == 1
    assert ops.resize_nearest(m, (60, 40)).sum() == m.sum() * 4
    ramp = np.tile(np.linspace(0, 1, 4, dtype=np.float32), (4, 1))
    up = ops.resize_bilinear(ramp, (8, 8))
    assert up.shape == (8, 8) and np.all(np.diff(up[0]) >= 0) and up[0, 0] == 0 and up[0, -1] == 1
    assert ops.resize_bilinear(np.ones((3, 3, 4), np.float32), (5, 7)).shape == (7, 5, 4)


def test_morphology():
    m = rect_mask(30, 30, 10, 10, 20, 20)
    assert ops.dilate(m, 2).sum() == 14 * 14
    assert ops.erode(m, 2).sum() == 6 * 6
    assert (ops.erode(ops.dilate(m, 3), 3) == m).all()
    assert ops.mask_edges(m).sum() == 36
    assert ops.erode(np.ones((5, 5), bool), 1).all()          # image border does not erode
    d = ops.distance_to(rect_mask(40, 40, 20, 20, 21, 21), 20)
    assert d[20, 20] == 0 and d[20, 26] == pytest.approx(6, abs=1) and d[20 + 8, 20 + 8] == pytest.approx(8 * 1.414, rel=0.15)


def test_split_sheet_synthetic_three_objects():
    h, w = 200, 600
    m = np.zeros((h, w), bool)
    m |= ellipse_mask(h, w, 100, 100, 70, 50)              # tall-ish
    m |= ellipse_mask(h, w, 100, 300, 80, 35)
    m |= rect_mask(h, w, 440, 30, 540, 180)
    m |= rect_mask(h, w, 380, 90, 436, 100)                # thin arm detached by a small gap from the third object
    img = paint(m, (0.8, 0.8, 0.8))
    boxes = ops.split_sheet(img, 3)
    assert len(boxes) == 3
    for x0, y0, x1, y1 in boxes:
        assert (y0, y1) == (0, h)                          # full height
    assert boxes[0][2] <= boxes[1][0] and boxes[1][2] <= boxes[2][0]
    assert boxes[0][0] <= 50 and boxes[0][2] >= 150
    assert boxes[1][0] <= 265 and boxes[1][2] >= 335
    assert boxes[2][0] <= 380 and boxes[2][2] >= 540       # the arm stays with the 3rd object
    with pytest.raises(ImageError):
        ops.split_sheet(img, 5)
