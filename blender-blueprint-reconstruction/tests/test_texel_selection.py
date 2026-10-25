"""Texel view selection (priority + seam band + edge distrust), inpainting and the source-map statistics."""
import numpy as np

from bbr.core.calibration import ViewCalibration
from bbr.core.texelproj import inpaint_colors, project_texel_colors, source_map_colors, texture_shares
from bbr.core.uvraster import uv_islands
from conftest import paint, rect_mask


def _front_side(w=20, h=20, colors=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
    mask = np.ones((h, w), bool)
    return [(paint(mask, colors[0]), mask, ViewCalibration("front", "-Y", (w, h), 0.1, (10.0, 10.0))),
            (paint(mask, colors[1]), mask, ViewCalibration("side", "-X", (w, h), 0.1, (10.0, 10.0)))]


def _normal(deg_from_front):
    """Unit normal in the XY plane tilted from -Y (front) toward -X (side)."""
    a = np.radians(deg_from_front)
    return np.array([[-np.sin(a), -np.cos(a), 0.0]])


def _first(rgb_info):
    return rgb_info[0][0]


def test_front_wins_over_side_until_side_faces_much_better():
    views = _front_side()
    pts = np.zeros((1, 3))
    kw = {"seam_blend_width": 0.0, "priority_margin": 0.12}
    np.testing.assert_allclose(_first(project_texel_colors(pts, _normal(45), views, **kw)), [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(_first(project_texel_colors(pts, _normal(50), views, **kw)), [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(_first(project_texel_colors(pts, _normal(80), views, **kw)), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(_first(project_texel_colors(pts, _normal(45), views, view_priority=("side", "front"), **kw)),
                               [0, 0, 1], atol=1e-6)                       # priority is configurable
    np.testing.assert_allclose(_first(project_texel_colors(pts, _normal(46), views, seam_blend_width=0.0, priority_margin=0.0)),
                               [0, 0, 1], atol=1e-6)                       # no bias: the best-facing view wins


def _switch_angle(views, threshold):
    pts = np.zeros((1, 3))
    angles = np.arange(40, 90, 0.25)
    red = [project_texel_colors(pts, _normal(a), views, seam_blend_width=0.0)[0][0, 0] for a in angles]
    return angles[int(np.argmax(np.array(red) < threshold))]


def test_seam_band_blends_only_near_the_switch():
    views = _front_side(colors=((1.0, 0.0, 0.0), (0.8, 0.0, 0.0)))        # similar colours: the conflict gate stays open
    pts = np.zeros((1, 3))
    switch = _switch_angle(views, 0.9)
    near, _, info = project_texel_colors(pts, _normal(switch - 0.5), views, seam_blend_width=0.1)
    assert 0.8 < near[0, 0] < 1.0 and (info["weights"][0] > 0).all()
    far, _, info = project_texel_colors(pts, _normal(switch - 15), views, seam_blend_width=0.1)
    np.testing.assert_allclose(far[0], [1, 0, 0], atol=1e-6)
    assert info["weights"][0, 1] == 0.0
    np.testing.assert_allclose(info["weights"].sum(axis=1), 1.0)


def test_conflicting_views_are_not_blended():
    views = _front_side()                                                   # red vs blue
    switch = _switch_angle(views, 0.5)
    rgb, _, info = project_texel_colors(np.zeros((1, 3)), _normal(switch - 0.5), views, seam_blend_width=0.1)
    np.testing.assert_allclose(rgb[0], [1, 0, 0], atol=1e-6)                # primary only
    assert info["conflict"][0]


def test_silhouette_edge_samples_lose_to_the_other_view():
    h = w = 40
    full = np.ones((h, w), bool)
    narrow = rect_mask(h, w, 12, 0, 40, 40)                                 # front mask edge at x = 12
    front = (paint(narrow, (1.0, 0.0, 0.0)), narrow, ViewCalibration("front", "-Y", (w, h), 0.1, (20.0, 20.0)))
    side = (paint(full, (0.0, 0.0, 1.0)), full, ViewCalibration("side", "-X", (w, h), 0.1, (20.0, 20.0)))
    pt = np.array([[-0.75, 0.0, 0.0]])                                      # front pixel x = 12.5: 0.5 px inside the mask
    n = _normal(52)                                                         # front 0.62; side 0.79 - 0.24 rank penalty
    kw = {"seam_blend_width": 0.0, "edge_erode_px": 0, "priority_margin": 0.12}
    on = project_texel_colors(pt, n, [front, side], **kw)[0][0]
    off = project_texel_colors(pt, n, [front, side], edge_distrust_px=0, **kw)[0][0]
    np.testing.assert_allclose(on, [0, 0, 1], atol=1e-6)                    # the edge penalty flips the choice
    np.testing.assert_allclose(off, [1, 0, 0], atol=1e-6)


def test_inpaint_uses_neighbours_never_background():
    size = 16
    covered = np.ones((size, size), bool)
    valid = np.zeros((size, size), bool)
    valid[:, :8] = True
    img = np.zeros((size, size, 3))
    img[...] = (1.0, 0.0, 0.0)                                              # value of the unseen texels (background-like)
    img[:, :8] = (0.1, 0.8, 0.2)
    pos = np.zeros((size, size, 3))
    pos[..., 0] = np.arange(size)[None, :]
    out, method = inpaint_colors(img, valid, covered, np.zeros((size, size), np.int64), pos, uv_px=16)
    np.testing.assert_allclose(out[:, 8:], np.broadcast_to((0.1, 0.8, 0.2), (size, 8, 3)))
    assert (method[:, 8:] == 1).all() and (method[:, :8] == 0).all()


def test_unseen_texel_from_projection_gets_neighbour_colour():
    # full pipeline on synthetic data: a texel behind a depth occluder is unseen by every view -> inpainted
    h = w = 20
    mask = np.ones((h, w), bool)
    front = ViewCalibration("front", "-Y", (w, h), 0.1, (10.0, 10.0))
    img = paint(mask, (0.2, 0.7, 0.3), bg=(1.0, 0.0, 1.0))
    pts = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    nrm = np.tile([0.0, -1.0, 0.0], (2, 1))
    zbuf = np.full((h, w), -np.inf)
    zbuf[10, 10] = 5.0                                                      # occludes the first texel only
    rgb, seen, _ = project_texel_colors(pts, nrm, [(img, mask, front)], [zbuf])
    assert seen.tolist() == [False, True]
    covered = np.ones((1, 2), bool)
    grid = np.zeros((1, 2, 3))
    grid[0, :, 0] = pts[:, 0]
    out, _ = inpaint_colors(rgb.reshape(1, 2, 3), seen.reshape(1, 2), covered, np.zeros((1, 2), np.int64), grid)
    np.testing.assert_allclose(out[0, 0], [0.2, 0.7, 0.3], atol=1e-6)


def test_inpaint_falls_back_to_3d_across_islands():
    size = 16
    covered = np.zeros((size, size), bool)
    covered[:, :4] = True
    covered[:, 10:14] = True
    valid = np.zeros((size, size), bool)
    valid[:, :4] = True                                                     # island 1 has no valid texel at all
    img = np.zeros((size, size, 3))
    img[:, :4] = (0.0, 0.0, 1.0)
    img[:, 10:14] = (1.0, 0.0, 0.0)
    island = np.full((size, size), -1, np.int64)
    island[:, :4], island[:, 10:14] = 0, 1
    pos = np.zeros((size, size, 3))
    pos[..., 0] = np.arange(size)[None, :] * 0.1
    out, method = inpaint_colors(img, valid, covered, island, pos, uv_px=16, cells=16)
    np.testing.assert_allclose(out[:, 10:14], np.broadcast_to((0.0, 0.0, 1.0), (size, 4, 3)))
    assert (method[:, 10:14] == 2).all()
    assert (out[:, 4:10] == 0).all()                                        # uncovered texels stay untouched


def test_uv_islands_split_disjoint_triangles():
    uvs = np.array([[[0, 0], [.1, 0], [0, .1]], [[.1, 0], [.1, .1], [0, .1]], [[.5, .5], [.6, .5], [.5, .6]]], float)
    isl = uv_islands(uvs)
    assert isl[0] == isl[1] != isl[2]


def test_source_map_and_shares_sum_to_one():
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.6, .4, 0], [0, 0, 0]], float)
    has = np.array([1, 1, 1, 1, 0], bool)
    conflict = np.array([0, 0, 0, 1, 1], bool)
    keys = ["front", "side", "back"]
    col = source_map_colors(w, has, keys)
    np.testing.assert_allclose(col[:3], np.eye(3))
    np.testing.assert_allclose(col[3], [.6, .4, 0])
    np.testing.assert_allclose(col[4], [.5, .5, .5])
    sh = texture_shares(w, has, conflict, keys)
    assert np.isclose(sum(sh["views"].values()) + sh["blend_share"] + sh["inpainted_share"], 1.0, atol=1e-3)
    assert sh["views"] == {"front": 0.2, "side": 0.2, "back": 0.2}
    assert sh["blend_share"] == 0.2 and sh["inpainted_share"] == 0.2 and sh["conflict_ratio"] == 0.4


def test_projection_reports_trust_that_drops_toward_the_silhouette_edge():
    h = w = 40
    mask = rect_mask(h, w, 10, 0, 40, 40)                                    # silhouette edge at x = 10
    view = (paint(mask, (0.6, 0.4, 0.2)), mask, ViewCalibration("front", "-Y", (w, h), 0.1, (20.0, 20.0)))
    pts = np.array([[(10.5 - 20.0) * 0.1, 0.0, 0.0], [(30.0 - 20.0) * 0.1, 0.0, 0.0]])    # 0.5 px and 20 px inside
    _, seen, info = project_texel_colors(pts, np.tile([0.0, -1.0, 0.0], (2, 1)), [view], edge_erode_px=0)
    assert seen.all() and info["trust"][0] < 0.2 and info["trust"][1] == 1.0
    _, none, info = project_texel_colors(pts, np.tile([0.0, 1.0, 0.0], (2, 1)), [view])   # facing away: unseen, no trust
    assert not none.any() and (info["trust"] == 0.0).all()


def _underside_scene():
    """Side wall (island 0): good colour with a near-black silhouette-edge row at the bottom; underside (island 1): unseen."""
    size = 16
    covered = np.zeros((size, size), bool)
    covered[:8], covered[10:] = True, True
    valid = np.zeros((size, size), bool)
    valid[:8] = True
    img = np.zeros((size, size, 3))
    img[:8] = (0.55, 0.38, 0.25)
    img[7] = (0.01, 0.01, 0.01)                                              # the wall's bottom edge row: backdrop-mixed, dark
    trust = np.where(np.arange(size)[:, None] == 7, 0.1, 1.0) * np.ones((size, size))
    island = np.full((size, size), -1, np.int64)
    island[:8], island[10:] = 0, 1
    pos = np.zeros((size, size, 3))
    pos[:8, :, 0] = np.arange(size)[None, :] * 0.1
    pos[:8, :, 2] = (7 - np.arange(8))[:, None] * 0.1 + 0.1                  # wall height; row 7 is the bottom edge
    pos[10:, :, 0] = np.arange(size)[None, :] * 0.1
    pos[10:, :, 2] = 0.0                                                     # underside at z = 0, right below the bottom edge row
    return img, valid, covered, island, pos, trust >= 0.5


def test_unseen_underside_is_never_black_when_the_nearest_texels_are_silhouette_edge_samples():
    img, valid, covered, island, pos, donor = _underside_scene()
    naive, _ = inpaint_colors(img, valid, covered, island, pos, uv_px=4, cells=16)
    assert naive[10:].max() < 0.1                                            # the failure: copied from the dark edge row
    out, method = inpaint_colors(img, valid, covered, island, pos, uv_px=4, cells=16, donor=donor)
    assert (method[10:] == 2).all()                                          # 3-D nearest valid donor (another island)
    assert out[10:].min() > 0.2                                              # no black
    np.testing.assert_allclose(out[10:].reshape(-1, 3).mean(axis=0), (0.55, 0.38, 0.25), atol=0.05)
    assert (out[:8, :, 0][:7] == 0.55).all() and (out[7] == 0.01).all()      # valid texels never change


def test_inpaint_without_any_trusted_donor_still_uses_the_valid_texels():
    img, valid, covered, island, pos, _ = _underside_scene()
    out, _ = inpaint_colors(img, valid, covered, island, pos, uv_px=4, cells=16, donor=np.zeros_like(valid))
    assert np.isfinite(out).all() and out[10:].max() > 0.0
