"""Silhouette carving of the hull field (grid orientation, extents, top-view consistency)."""
import numpy as np
import pytest

from bbr.core import carve
from bbr.core.calibration import ViewCalibration, calibrate_views
from bbr.core.conventions import AXES
from bbr.core.spec import parse_view_spec
from conftest import ellipse_mask, rect_mask

PX = 0.005  # m per px of the synthetic sphere views


def sphere_views(radius=0.4, size=240):
    """Analytic sphere (centre z=radius) seen from front, side (-X) and top, all at PX m/px."""
    c = size / 2
    r = radius / PX
    disc = ellipse_mask(size, size, c, c, r, r)
    calibs = {
        "front": ViewCalibration("front", "-Y", (size, size), PX, (c, c + r)),
        "side": ViewCalibration("side", "-X", (size, size), PX, (c, c + r)),
        "top": ViewCalibration("top", "+Z", (size, size), PX, (c, c)),
    }
    return {k: disc for k in calibs}, calibs, {"width": 2 * radius, "depth": 2 * radius, "height": 2 * radius}


def hull_silhouette(grid, calib):
    """Rasterise the hull silhouette onto a view's pixel grid (independent of carve internals)."""
    w, h = calib.image_size
    vv, uu = np.mgrid[0:h, 0:w]
    plane = calib.px_to_plane(np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], 1))
    idx = np.floor((plane - grid.origin) / grid.cell).astype(int)
    axis = int(np.argmax(np.abs(AXES[calib.camera_axis])))
    proj = grid.occ.any(axis=axis)
    keep = [i for i in range(3) if i != axis]
    ok = np.ones(len(idx), bool)
    for i in keep:
        ok &= (idx[:, i] >= 0) & (idx[:, i] < grid.occ.shape[i])
    out = np.zeros(len(idx), bool)
    out[ok] = proj[idx[ok][:, keep[0]], idx[ok][:, keep[1]]]
    return out.reshape(h, w)


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def carve_grid(masks, calibs, dims, resolution, **kw):
    """Occupancy grid of ``build_hull_field`` (rectangle cross-section = pure silhouette carve)."""
    field, origin, cell = carve.build_hull_field(masks, calibs, dims, resolution, cross_section="rectangle", **kw)
    return carve.VoxelGrid(field < 0, origin, cell)


def test_carve_box_exact_and_top_consistency():
    size = 200
    box_f = rect_mask(size, size, 50, 40, 150, 190)    # 100 x 150 px
    box_s = rect_mask(size, size, 70, 40, 130, 190)    # 60 px deep
    spec = parse_view_spec({"spec_version": "0.1", "known_dimensions": {"height": 0.75},
                            "views": {"front": {"path": "f"}, "side": {"path": "s", "camera_axis": "-X"}}})
    calibs, info = calibrate_views(spec, {"front": box_f, "side": box_s})
    grid = carve_grid({"front": box_f, "side": box_s}, calibs, info["dimensions"], 64,
                     carve_views=["front", "side"])
    occ = grid.occ
    spans = []
    for i in range(3):
        idx = np.flatnonzero(occ.any(axis=tuple(j for j in range(3) if j != i)))
        spans.append((idx[-1] - idx[0] + 1) * grid.cell)
    for got, want in zip(spans, (0.5, 0.3, 0.75)):
        assert got == pytest.approx(want, abs=2 * grid.cell)
    # a top mask whose aspect contradicts width/depth is ignored (carve still works)
    bad_top = rect_mask(size, size, 20, 90, 180, 110)
    calibs["top"] = ViewCalibration("top", "+Z", (size, size), PX, (100.0, 100.0))
    grid2 = carve_grid({"front": box_f, "side": box_s, "top": bad_top}, calibs, info["dimensions"], 64,
                      carve_views=["front", "side", "top"])
    assert grid2.occ.sum() == occ.sum()


def nose_masks(axis):
    """Side mask: body u 100..200, nose protruding at image RIGHT (u 200..230) for rows 90..110."""
    side = np.zeros((200, 260), bool)
    side[20:180, 100:200] = True
    side[90:110, 200:230] = True
    front = rect_mask(200, 260, 120, 20, 180, 180)
    spec = parse_view_spec({"spec_version": "0.1", "known_dimensions": {"height": 0.8},
                            "views": {"front": {"path": "f"}, "side": {"path": "s", "camera_axis": axis}}})
    calibs, info = calibrate_views(spec, {"front": front, "side": side})
    return {"front": front, "side": side}, calibs, info["dimensions"]


@pytest.mark.parametrize("axis,sign", [("-X", -1), ("+X", +1)])
def test_side_camera_axis_places_nose(axis, sign):
    """Camera -X: image right = -Y, so the nose lands at negative Y; camera +X mirrors it to +Y."""
    masks, calibs, dims = nose_masks(axis)
    grid = carve_grid(masks, calibs, dims, 96, margin=0.0, carve_views=["front", "side"])
    m = calibs["side"].m_per_px
    ys, zs = grid.centers(1), grid.centers(2)
    # rows 90..110 are the nose; z measured up from the bbox bottom at row 180
    nose_z = (zs > (180 - 108) * m) & (zs < (180 - 92) * m)
    body_z = (zs > (180 - 85) * m) & (zs < (180 - 25) * m) | (zs > (180 - 175) * m) & (zs < (180 - 115) * m)
    nose_y = ys[grid.occ[:, :, nose_z].any(axis=(0, 2))]
    body_y = ys[grid.occ[:, :, body_z].any(axis=(0, 2))]
    tol = 2 * grid.cell
    if sign < 0:
        assert nose_y.min() < body_y.min() - 20 * m and nose_y.max() == pytest.approx(body_y.max(), abs=tol)
        assert nose_y.min() < -0.05          # nose at negative Y
    else:
        assert nose_y.max() > body_y.max() + 20 * m and nose_y.min() == pytest.approx(body_y.min(), abs=tol)
        assert nose_y.max() > 0.05           # nose at positive Y
    all_y = ys[grid.occ.any(axis=(0, 2))]
    assert (all_y.min() + all_y.max()) / 2 == pytest.approx(0.0, abs=tol)    # bbox-centre convention
