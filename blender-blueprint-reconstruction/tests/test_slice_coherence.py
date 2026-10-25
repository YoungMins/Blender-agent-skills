"""Spiky silhouettes must not become stacks of horizontal rings: slice_coherence + the ring metric."""
import numpy as np
import pytest

from bbr.core import carve
from bbr.core.ringmetric import FLAG_PCT, ring_score
from conftest import rect_mask
from test_carve import PX, iou, sphere_views

RES = 160
HEIGHT = 0.8
COHERENCE = 0.03                 # 4 slices at RES


def spiky_sphere_views(period=14, length=14, thick=3, radius=0.4, size=240):
    """Sphere views whose outline carries triangular teeth (hair spikes) every ``period`` rows, left and right."""
    masks, calibs, dims = sphere_views(radius, size)
    disc = masks["front"].copy()
    c, r = size / 2, radius / PX
    for k, row in enumerate(range(int(c - r * 0.8), int(c + r * 0.8), period)):
        for sign in (-1, 1) if k % 2 == 0 else (1, -1):
            for dy in range(-thick // 2, thick // 2 + 1):
                reach = int(length * (1 - abs(dy) / (thick / 2 + 1)))
                x_edge = c + sign * np.sqrt(max(r * r - (row + dy - c) ** 2, 0.0))
                for dx in range(reach):
                    disc[row + dy, int(round(x_edge + sign * dx))] = True
    return {"front": disc, "side": disc.copy()}, {k: calibs[k] for k in ("front", "side")}, dims


def hull(masks, calibs, dims, **kw):
    return carve.build_hull_field(masks, calibs, dims, RES, carve_views=["front", "side"], **kw)


def score(field, cell):
    return 100.0 * ring_score(field, cell, HEIGHT)


def test_open_along_z_flattens_thin_bumps_keeps_plateaus_and_steps():
    vals = np.full((1, 40), 1.0)
    vals[0, 10:13] = 1.5          # a 3-slice bump
    vals[0, 25:] = 2.0            # a step that stays
    vals[0, 30:36] = 2.5          # a 6-slice plateau on top of it
    out = carve._open_along_z(vals, np.ones_like(vals, bool), 5)
    assert np.allclose(out[0, 10:13], 1.0)
    assert np.allclose(out[0, 25:30], 2.0) and np.allclose(out[0, 30:36], 2.5)
    assert (out <= vals + 1e-12).all()


def test_regularize_extents_flattens_spikes_keeps_tips_and_parts():
    nz = 40
    center = np.zeros((6, nz))
    half = np.zeros((6, nz))
    half[:4] = 1.0                       # four columns inside every slice
    half[4, 10:13] = 1.0                 # a spike tip: inside 3 slices only (must survive: it defines the bounding box)
    half[0, 20:22] = 1.2                 # a 2-slice spike (20 % of the half-width) of one column
    half[1, 30:32] = 1.8                 # a 2-slice bulge of 80 %: a part (brim), kept
    c2, h2 = carve._regularize_extents(center, half, 5)
    assert np.allclose(h2[4, 10:13], 1.0) and (h2[4, :10] == 0).all()
    assert (h2[:4] > 0).all() and h2[0, 20:22].max() < 1.1 and h2[1, 30:32].min() > 1.7
    assert carve._regularize_extents(center, half, 1)[1] is half


def test_spiky_sphere_has_no_rings_with_slice_coherence():
    masks, calibs, dims = spiky_sphere_views()
    f_off, _, cell = hull(masks, calibs, dims)
    f_on, _, cell = hull(masks, calibs, dims, slice_coherence=COHERENCE)
    assert score(f_off, cell) > FLAG_PCT, "the teeth must show up as rings without the regularisation"
    assert score(f_on, cell) < min(FLAG_PCT, 0.5 * score(f_off, cell))
    plain, pc, pd = sphere_views()
    f_ref, _, _ = hull({k: plain[k] for k in ("front", "side")}, {k: pc[k] for k in ("front", "side")}, pd)
    assert score(f_ref, cell) < FLAG_PCT / 4              # a smooth sphere is no ring


def test_slice_coherence_keeps_the_body_silhouette():
    masks, calibs, dims = spiky_sphere_views()
    field, origin, cell = hull(masks, calibs, dims, slice_coherence=COHERENCE)
    grid = carve.VoxelGrid(field < 0, origin, cell)
    from test_carve import hull_silhouette
    disc = sphere_views()[0]["front"]
    assert iou(hull_silhouette(grid, calibs["front"]), disc) > 0.96      # the teeth are a few % of the area
    assert iou(hull_silhouette(grid, calibs["front"]), masks["front"]) > 0.95


def test_thin_brim_is_a_part_not_a_spike():
    """A wide, thin disc (hat brim, 2.5 % of the height, thinner than the opening element) is far bigger than a spike."""
    size, brim = 240, 5                                   # px; 5 px = 0.025 m of a 1.0 m figure
    body = rect_mask(size, size, 100, 20, 140, 220)
    disc = body | rect_mask(size, size, 40, 100, 200, 100 + brim)
    calibs = sphere_views()[1]
    calibs = {k: calibs[k] for k in ("front", "side")}
    dims = {"width": 160 * PX, "depth": 160 * PX, "height": 200 * PX}
    for sc in (0.0, carve.DEFAULT_SLICE_COHERENCE):
        field, _, cell = carve.build_hull_field({"front": disc, "side": disc}, calibs, dims, 120,
                                                carve_views=["front", "side"], slice_coherence=sc)
        assert sc == 0.0 or int(sc * dims["height"] / cell) > brim * PX / cell, "the brim must be thinner than the element"
        inside = field < 0
        diag = np.hypot(*np.meshgrid(*[(np.arange(n) + 0.5) * cell - n * cell / 2 for n in inside.shape[:2]], indexing="ij"))
        brim_rows = inside[..., :].sum(axis=(0, 1)).argmax()
        assert diag[inside[..., brim_rows]].max() > 0.9 * 80 * PX, f"brim lost with slice_coherence={sc}"


def test_negative_slice_coherence_rejected():
    masks, calibs, dims = sphere_views()
    with pytest.raises(Exception):
        carve.build_hull_field(masks, calibs, dims, 32, carve_views=["front", "side"], slice_coherence=-0.1)
