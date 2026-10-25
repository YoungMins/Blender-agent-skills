"""UV rasterizer, depth buffer, texel colour projection, island removal, colour metric (pure numpy)."""
import numpy as np

from bbr.core.calibration import ViewCalibration
from bbr.core.colormetric import color_error
from bbr.core.islands import small_island_vertices, vertex_components
from bbr.core.meshsmooth import taubin_smooth_edges, triangle_vertex_normals
from bbr.core.texelproj import dilate_colors, project_texel_colors
from bbr.core.uvraster import rasterize_triangles, rasterize_uv_mesh
from bbr.core.zbuffer import depth_buffer, visible
from conftest import ellipse_mask, paint, rect_mask


def test_right_triangle_coverage_and_exact_interpolation():
    # lower-left right triangle covering half of an 8x8 texture; world position = 2*uv on the z=0.5 plane
    verts = np.array([[0, 0, 0.5], [2, 0, 0.5], [0, 2, 0.5]], float)
    uvs = np.array([[[0, 0], [1, 0], [0, 1]]], float)
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))
    r = rasterize_uv_mesh(verts, np.array([[0, 1, 2]]), uvs, normals, 8)
    mask = r["mask"]
    assert mask.shape == (8, 8)
    assert int(mask.sum()) == 36                       # texels with x + y + 1 <= 8 (centres inside the diagonal)
    ys, xs = np.nonzero(mask)
    expected = np.stack([(xs + 0.5) / 8 * 2, (ys + 0.5) / 8 * 2, np.full(len(xs), 0.5)], axis=1)
    np.testing.assert_allclose(r["position"][ys, xs], expected, atol=1e-12)
    np.testing.assert_allclose(r["normal"][ys, xs], np.tile([0, 0, 1.0], (len(xs), 1)), atol=1e-12)
    assert not mask[7, 7] and mask[0, 0]


def test_quad_fills_square_and_interpolates_normals():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = verts[tris][..., :2]
    normals = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 1.0, 0], [0, 1.0, 0]])
    r = rasterize_uv_mesh(verts, tris, uvs, normals, (16, 8))
    assert r["mask"].all()
    np.testing.assert_allclose(np.linalg.norm(r["normal"], axis=2), 1.0, atol=1e-9)
    np.testing.assert_allclose(r["position"][..., 0][0], (np.arange(16) + 0.5) / 16, atol=1e-12)
    assert r["normal"][0, 0, 2] > r["normal"][-1, -1, 2]   # blends toward +y at the (1,1) corner


def test_adjacent_triangles_leave_no_gap_and_chunking_matches():
    n = 40
    g = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    full = rasterize_triangles(pts[tris] * 64, (64, 64))
    assert (full >= 0).all()
    import bbr.core.uvraster as uvr
    old = uvr.PAIR_CHUNK
    uvr.PAIR_CHUNK = 500
    try:
        assert (rasterize_triangles(pts[tris] * 64, (64, 64)) >= 0).all()
    finally:
        uvr.PAIR_CHUNK = old


def test_degenerate_triangle_covers_nothing():
    ids = rasterize_triangles(np.array([[[1, 1], [5, 5], [3, 3]]], float), (8, 8))
    assert (ids == -1).all()


def _front_calib(size=(40, 40), m_per_px=0.05):
    return ViewCalibration("front", "-Y", size, m_per_px, (20.0, 40.0))


def test_depth_buffer_and_visibility():
    # two parallel squares facing the front camera (-Y): near at y=-0.5, far at y=+0.5, x in [-.5,.5], z in [.2,1]
    def sq(y):
        return [[-0.5, y, 0.2], [0.5, y, 0.2], [0.5, y, 1.0], [-0.5, y, 1.0]]
    verts = np.array(sq(-0.5) + sq(0.5), float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    calib = _front_calib()
    zbuf = depth_buffer(verts, tris, calib)
    assert np.isclose(zbuf[30, 20], 0.5)                   # height along camera_axis -Y: nearer = larger (0.5)
    assert np.isinf(zbuf[0, 0]) and zbuf[0, 0] < 0
    pts = np.array([[0.0, -0.5, 0.6], [0.0, 0.5, 0.6]])
    assert visible(pts, calib, zbuf, 0.01).tolist() == [True, False]


def test_texel_projection_view_blend_and_occlusion():
    h = w = 20
    mask = np.ones((h, w), bool)
    red = paint(mask, (1.0, 0.0, 0.0))
    blue = paint(mask, (0.0, 0.0, 1.0))
    front = ViewCalibration("front", "-Y", (w, h), 0.1, (10.0, 10.0))
    back = ViewCalibration("back", "+Y", (w, h), 0.1, (10.0, 10.0))
    pts = np.array([[0.0, 0.0, 0.0]])
    # facing the front camera only: the back view (facing < 0) must not contribute
    rgb, seen, _ = project_texel_colors(pts, np.array([[0.0, -1.0, 0.0]]), [(red, mask, front), (blue, mask, back)])
    np.testing.assert_allclose(rgb[0], [1, 0, 0], atol=1e-6)
    assert seen[0]
    # facing the back camera only
    rgb2, _, _ = project_texel_colors(pts, np.array([[0.0, 1.0, 0.0]]), [(red, mask, front), (blue, mask, back)])
    np.testing.assert_allclose(rgb2[0], [0, 0, 1], atol=1e-6)
    # oblique towards both (normal +z with a small y tilt each way is impossible; use equal facing via +x/-y mix)
    both_front = np.array([[0.0, -0.6, 0.8]])
    rgb3, _, _ = project_texel_colors(pts, both_front, [(red, mask, front), (blue, mask, back)])
    np.testing.assert_allclose(rgb3[0], [1, 0, 0], atol=1e-6)
    # occluded in the only view: not seen (to be inpainted), never stamped with the occluded view's colour
    zbuf = np.full((h, w), 5.0)
    rgb4, seen4, _ = project_texel_colors(pts, np.array([[0.0, -1.0, 0.0]]), [(red, mask, front)], [zbuf])
    assert not seen4[0]
    assert not np.allclose(rgb4[0], [1, 0, 0])


def test_texel_projection_does_not_bleed_background():
    h = w = 40
    mask = ellipse_mask(h, w, 20, 20, 10, 10)
    img = paint(mask, (0.9, 0.2, 0.1), bg=(0.0, 1.0, 0.0))   # green background
    calib = ViewCalibration("front", "-Y", (w, h), 0.1, (20.0, 20.0))
    far = np.array([[1.9, 0.0, 0.0]])                          # projects at x=39: background, far from the mask
    rgb, _, _ = project_texel_colors(far, np.array([[0.0, -1.0, 0.0]]), [(img, mask, calib)])
    assert rgb[0, 1] <= 0.5                                    # never green


def test_dilate_colors_pads_and_limits_radius():
    img = np.zeros((10, 10, 3))
    img[5, 5] = (1, 0, 0)
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    out, filled = dilate_colors(img, mask, 2)
    assert filled[5, 7] and filled[3, 5] and not filled[5, 8] and not filled[0, 0]
    np.testing.assert_allclose(out[5, 7], (1, 0, 0))
    assert (out[0, 0] == 0).all()


def test_island_components_and_small_island_removal():
    # island A: chain 0-1-2-3, island B: 4-5, isolated vertex 6
    edges = np.array([[0, 1], [1, 2], [2, 3], [4, 5]])
    assert vertex_components(7, edges).tolist() == [0, 0, 0, 0, 4, 4, 6]
    weight = np.array([1, 1, 1, 1, 0.05, 0.05, 0.01], float)
    drop = small_island_vertices(7, edges, weight, 0.05)
    assert drop.tolist() == [False] * 4 + [True, True, True]
    assert not small_island_vertices(7, edges, weight, 0.0).any()
    assert not small_island_vertices(2, np.array([[0, 1]]), np.array([1.0, 1.0]), 5.0).any()  # largest always kept


def test_long_chain_components_converge():
    n = 5000
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    assert (vertex_components(n, edges) == 0).all()


def test_edge_taubin_smooths_noise():
    rng = np.random.default_rng(0)
    n = 30
    g = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(g, g)
    z = rng.normal(0, 0.01, xx.shape)
    verts = np.stack([xx.ravel(), yy.ravel(), z.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    edges = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                            np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])
    out = taubin_smooth_edges(verts, edges, 5)
    assert out[:, 2].std() < 0.6 * verts[:, 2].std()


def test_triangle_vertex_normals_point_outward_for_ccw():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    nrm = triangle_vertex_normals(verts, np.array([[0, 1, 2]]))
    np.testing.assert_allclose(nrm, np.tile([0, 0, 1.0], (3, 1)))


def test_color_error_masks_and_value():
    h = w = 30
    full = rect_mask(h, w, 0, 0, 30, 30)
    ref = paint(full, (0.5, 0.5, 0.5))
    out = paint(full, (0.6, 0.5, 0.3))
    m = rect_mask(h, w, 5, 5, 25, 25)
    res = color_error(ref, out, m, m)
    assert np.isclose(res["error"], (0.1 + 0.0 + 0.2) / 3)
    assert res["pixels"] == 16 * 16                        # 20x20 intersection eroded by 2 px
    far = rect_mask(h, w, 0, 0, 3, 3)
    assert color_error(ref, out, m, far)["error"] is None  # empty intersection
    assert color_error(ref, ref, m, m)["error"] == 0.0
