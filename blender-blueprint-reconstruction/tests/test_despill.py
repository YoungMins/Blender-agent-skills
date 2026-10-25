"""Chroma-backdrop decontamination, background-safe bilinear sampling and the background-leak statistic."""
import numpy as np

from bbr.core.calibration import ViewCalibration
from bbr.core.colorproj import ViewSampler
from bbr.core.despill import background_leak_ratio, chroma_channel, decontaminate_view

GREEN = (0.0, 0.8, 0.2)
RED = np.array([0.9, 0.1, 0.1])


def _edge_image(h=40, w=40):
    """Red square (x 10..30) with a 1 px column at each side blended 50% with the green backdrop."""
    img = np.empty((h, w, 4), np.float32)
    img[...] = (*GREEN, 1.0)
    img[10:30, 10:30, :3] = RED
    blend = 0.5 * RED + 0.5 * np.array(GREEN)
    img[10:30, 10, :3] = blend
    img[10:30, 29, :3] = blend
    mask = np.zeros((h, w), bool)
    mask[10:30, 10:30] = True          # the blended edge pixels count as foreground (as extract_mask does)
    return img, mask


def test_chroma_channel_only_for_saturated_backdrops():
    assert chroma_channel(GREEN) == 1
    assert chroma_channel((0.1, 0.2, 0.9)) == 2
    assert chroma_channel((1.0, 1.0, 1.0)) is None
    assert chroma_channel((0.5, 0.5, 0.5)) is None


def test_decontamination_recovers_foreground_on_a_half_blended_edge():
    img, mask = _edge_image()
    out, eff = decontaminate_view(img, mask, GREEN, edge_px=6)
    np.testing.assert_allclose(out[20, 10, :3], RED, atol=1e-5)
    np.testing.assert_allclose(out[20, 29, :3], RED, atol=1e-5)
    np.testing.assert_allclose(out[20, 20, :3], RED, atol=1e-6)    # interior untouched
    assert eff.sum() == mask.sum()
    assert img[20, 10, 1] > 0.4                                      # input not modified in place


def test_spill_clamp_removes_green_cast_when_blend_is_off_the_line():
    img, mask = _edge_image()
    img[20, 10, :3] = (0.5, 0.7, 0.3)    # not on the backdrop -> interior line: only the clamp applies
    out, _ = decontaminate_view(img, mask, GREEN, edge_px=6)
    assert out[20, 10, 1] <= max(out[20, 10, 0], out[20, 10, 2]) + 1e-6


def test_backdrop_pocket_inside_the_mask_is_dropped():
    img, mask = _edge_image()
    img[18:22, 18:22, :3] = GREEN        # enclosed backdrop gap, flood fill keeps it in the mask
    _, eff = decontaminate_view(img, mask, GREEN, edge_px=6)
    assert not eff[18:22, 18:22].any() and eff[15, 15]


def test_non_chroma_backdrop_is_untouched():
    img = np.ones((20, 20, 4), np.float32)
    img[5:15, 5:15, :3] = 0.2
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    out, eff = decontaminate_view(img, mask, (1.0, 1.0, 1.0))
    assert out is img and eff is mask


def test_sampling_never_returns_the_backdrop_colour():
    img, mask = _edge_image()
    img2, mask2 = decontaminate_view(img, mask, GREEN, edge_px=6)
    calib = ViewCalibration("front", "-Y", (40, 40), 0.1, (20.0, 20.0))
    sampler = ViewSampler(img2, mask2, calib, edge_erode_px=1, snap_px=4)
    # sweep points across and just outside the red square, incl. positions whose bilinear taps straddle the border
    xs = np.linspace(-2.0, 2.0, 41)
    pts = np.array([[x, 0.0, z] for x in xs for z in np.linspace(-2.0, 2.0, 41)])
    col, _, valid = sampler.sample(pts)
    assert valid.any()
    dist = np.linalg.norm(col[valid] - np.array(GREEN), axis=1)
    assert dist.min() > 0.5
    np.testing.assert_allclose(col[valid], np.tile(RED, (int(valid.sum()), 1)), atol=1e-4)


def test_bilinear_tap_outside_mask_uses_nearest_pixel():
    img = np.zeros((6, 6, 4), np.float32)
    img[...] = (*GREEN, 1.0)
    img[:, :3, :3] = RED
    mask = np.zeros((6, 6), bool)
    mask[:, :3] = True
    calib = ViewCalibration("front", "-Y", (6, 6), 1.0, (3.0, 3.0))
    s = ViewSampler(img, mask, calib, edge_erode_px=0, snap_px=0)
    col, _, _ = s.sample(np.array([[-0.05, 0.0, 0.0]]))             # px x=2.95: taps 2 and 3, tap 3 is backdrop
    np.testing.assert_allclose(col[0], RED, atol=1e-6)


def test_leak_ratio_counts_backdrop_and_chroma_dominant_texels():
    rgb = np.zeros((2, 4, 3))
    rgb[0, 0] = GREEN                    # backdrop colour
    rgb[0, 1] = (0.1, 0.6, 0.2)          # green-dominant, far from the backdrop colour
    rgb[0, 2] = RED
    rgb[0, 3] = (0.9, 0.9, 0.9)
    rgb[1, :] = (0.3, 0.3, 0.8)          # blue: not leaked (uncovered texels are ignored anyway)
    covered = np.zeros((2, 4), bool)
    covered[0] = True
    assert background_leak_ratio(rgb, covered, [GREEN]) == 0.5
    assert background_leak_ratio(rgb, covered, []) == 0.0
    assert background_leak_ratio(rgb, np.zeros((2, 4), bool), [GREEN]) == 0.0
