"""Distance transform, Surface Nets and the continuous superellipse hull field."""
import numpy as np

from bbr.core import carve
from bbr.core.distance import edt, signed_distance
from bbr.core.surface import surface_nets
from test_carve import PX, iou, sphere_views


def mesh_volume(verts, quads):
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def edge_counts(quads):
    e = np.sort(np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def test_edt_matches_brute_force():
    rng = np.random.default_rng(3)
    fg = rng.random((23, 31)) > 0.93
    fg[5, 7] = True
    yy, xx = np.nonzero(fg)
    gy, gx = np.mgrid[0:23, 0:31]
    brute = np.sqrt((gy[..., None] - yy) ** 2 + (gx[..., None] - xx) ** 2).min(axis=-1)
    dist, iy, ix = edt(fg, cap=64, return_index=True)
    assert np.allclose(dist, brute, atol=1e-4)
    assert np.allclose(np.hypot(gy - iy, gx - ix), brute, atol=1e-4)


def test_signed_distance_sign_and_boundary():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    sd = signed_distance(m, cap=30)
    assert (sd[m] < 0).all() and (sd[~m] > 0).all()
    assert np.isclose(sd[5, 10], -0.5) and np.isclose(sd[4, 10], 0.5)


def test_surface_nets_sphere_is_closed_and_accurate():
    n, cell, r = 48, 0.025, 0.4
    c = (np.arange(n) + 0.5) * cell
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    field = np.sqrt((x - n * cell / 2) ** 2 + (y - n * cell / 2) ** 2 + (z - n * cell / 2) ** 2) - r
    verts, quads = surface_nets(field, (cell / 2,) * 3, cell)
    vol = mesh_volume(verts, quads)
    exact = 4 / 3 * np.pi * r ** 3
    assert vol > 0 and abs(vol / exact - 1) < 0.03
    assert (edge_counts(quads) == 2).all()  # watertight: every edge shared by exactly two quads


def test_surface_nets_keeps_largest_component():
    f = np.ones((30, 12, 12), np.float32)
    f[2:20, 2:10, 2:10] = -1
    f[24:27, 5:8, 5:8] = -1
    verts, quads = surface_nets(f)
    assert verts[:, 0].max() < 22


def test_surface_nets_min_island_frac_keeps_disconnected_parts():
    f = np.ones((30, 12, 12), np.float32)
    f[2:20, 2:10, 2:10] = -1
    f[24:27, 5:8, 5:8] = -1
    verts, quads = surface_nets(f, min_island_frac=0.01)
    assert verts[:, 0].max() > 24 and (edge_counts(quads) == 2).all()


def test_superellipse_hull_of_sphere_masks():
    masks, calibs, dims = sphere_views()
    kw = dict(dims=dims, resolution=96, carve_views=["front", "side"])
    field, origin, cell = carve.build_hull_field(masks, calibs, **kw)
    verts, quads = surface_nets(field, origin + 0.5 * cell, cell)
    exact = 4 / 3 * np.pi * 0.4 ** 3
    assert abs(mesh_volume(verts, quads) / exact - 1) < 0.08
    rect, o, c = carve.build_hull_field(masks, calibs, cross_section="rectangle", **kw)
    v2, q2 = surface_nets(rect, o + 0.5 * c, c)
    assert mesh_volume(v2, q2) / exact > 1.25  # cylinder intersection ~1.33x


def test_hull_field_preserves_front_silhouette():
    masks, calibs, dims = sphere_views()
    field, origin, cell = carve.build_hull_field(masks, calibs, dims, 96, carve_views=["front", "side"])
    grid = carve.VoxelGrid(field < 0, origin, cell)
    proj = grid.occ.any(axis=1)  # [nx, nz]
    ref = masks["front"]
    x = origin[0] + (np.arange(proj.shape[0]) + 0.5) * cell
    z = origin[2] + (np.arange(proj.shape[1]) + 0.5) * cell
    u = np.floor(calibs["front"].origin_px[0] + x / PX).astype(int)
    v = np.floor(calibs["front"].origin_px[1] - z / PX).astype(int)
    ref_s = ref[v[None, :], u[:, None]]
    assert (proj & ref_s).sum() / (proj | ref_s).sum() > 0.97


def _rod_views(side_runs, front_runs, z1=1.0, size=300):
    """Front (-Y) / side (-X) masks from world-unit x / y intervals, z in [0, z1] (z=0 at row 250)."""
    from bbr.core.calibration import ViewCalibration
    from bbr.core.conventions import axis_index, view_frame
    from test_carve import hull_silhouette, iou  # noqa: F401

    c = size // 2
    calibs = {"front": ViewCalibration("front", "-Y", (size, size), PX, (c, 250)),
              "side": ViewCalibration("side", "-X", (size, size), PX, (c, 250))}
    rs = axis_index(view_frame("-X").right_axis)[1]

    def mask(runs, sign):
        m = np.zeros((size, size), bool)
        for lo, hi, zlo, zhi in runs:
            u0, u1 = sorted((c + sign * lo / PX, c + sign * hi / PX))
            m[int(250 - zhi / PX):int(250 - zlo / PX), int(round(u0)):int(round(u1))] = True
        return m

    masks = {"front": mask(front_runs, 1), "side": mask(side_runs, rs)}
    return masks, calibs, rs


def _grid_ycols(grid, lo, hi):
    y = grid.centers(1)
    return (y >= lo) & (y <= hi)


def test_pair_selection_unit():
    sel = carve.select_run_pairs([0.2, 0.04], [0.2, 0.04], 4.0)
    assert [(i, j) for i, j, _, _ in sel] == [(0, 0), (1, 1)]
    sel = carve.select_run_pairs([0.3], [0.3, 0.02], 4.0)
    assert [(i, j) for i, j, _, _ in sel] == [(0, 0), (0, 1)]
    assert sel[1][2] == 0.08 and sel[1][3] == 0.02  # long axis clamped to aspect x short
    assert carve.select_run_pairs([0.3], [0.02], 4.0) == [(0, 0, 0.3, 0.02)]


def test_hull_field_no_phantom_slab_between_thin_runs():
    # body (wide) + thin rod, rod offset in both views; ratio wide/thin = 5 > 4
    front = [(-0.3, 0.1, 0.0, 1.0), (0.30, 0.38, 0.0, 1.0)]
    side = [(-0.2, 0.2, 0.0, 1.0), (0.30, 0.38, 0.0, 1.0)]
    masks, calibs, rs = _rod_views(side, front)
    dims = {"width": 0.8, "depth": 0.8, "height": 1.0}
    field, origin, cell = carve.build_hull_field(masks, calibs, dims, 96, carve_views=["front", "side"])
    grid = carve.VoxelGrid(field < 0, origin, cell)
    x, y = grid.centers(0), grid.centers(1)
    slab = (x[:, None] >= -0.3 + cell) & (x[:, None] <= 0.1 - cell) & _grid_ycols(grid, 0.31, 0.37)[None, :]
    slab_t = (x[:, None] >= 0.31) & (x[:, None] <= 0.37) & (np.abs(y)[None, :] <= 0.2 - cell)
    assert not grid.occ[slab].any()
    assert not grid.occ[slab_t].any()
    from test_carve import hull_silhouette
    for name in ("front", "side"):
        assert iou(hull_silhouette(grid, calibs[name]), masks[name]) >= 0.97, name


def test_hull_field_needle_clamped_in_x():
    front = [(-0.3, 0.3, 0.0, 1.0)]
    side = [(-0.3, 0.3, 0.0, 1.0), (0.38, 0.42, 0.4, 0.5)]   # thin tip (half 0.02) at z 0.4..0.5
    masks, calibs, rs = _rod_views(side, front)
    dims = {"width": 0.8, "depth": 0.9, "height": 1.0}
    field, origin, cell = carve.build_hull_field(masks, calibs, dims, 96, carve_views=["front", "side"])
    grid = carve.VoxelGrid(field < 0, origin, cell)
    tip = grid.occ[:, _grid_ycols(grid, 0.385, 0.415), :]
    assert tip.any()
    xs = grid.centers(0)[tip.any(axis=(1, 2))]
    assert (xs.max() - xs.min()) / 2 <= 4.0 * 0.02 + cell
