"""Layered parts (core/layers.py): a face recessed inside a helmet on a synthetic head."""
import numpy as np
import pytest

from bbr import CalibrationError
from bbr.core import carve, labels as labels_core, layers
from bbr.core.calibration import ViewCalibration
from bbr.core.surface import surface_nets
from conftest import rect_mask
from test_hull_field import edge_counts

PX = 0.005
SIZE = 200
ORIGIN = (100.0, 190.0)      # world x = 0 / y = 0 at column 100; z = 0 at row 190
BODY_RES = 96


def _stack(*rects):
    m = np.zeros((SIZE, SIZE), bool)
    for x0, y0, x1, y1 in rects:
        m |= rect_mask(SIZE, SIZE, x0, y0, x1, y1)
    return m


def head():
    """Helmet with a recessed face. Side view (camera -X, face to image right = -Y): the dome / rim (rows 30-85)
    reaches column 150 (y = -0.25), the face rows (85-140) only column 135 (y = -0.175), the neck (140-190) column 120.
    The face label is the front rectangle columns 75..125 x rows 85..135 and the side rectangle columns 110..135."""
    calibs = {"front": ViewCalibration("front", "-Y", (SIZE, SIZE), PX, ORIGIN),
              "side": ViewCalibration("side", "-X", (SIZE, SIZE), PX, ORIGIN)}
    masks = {"front": _stack((55, 30, 145, 190)),
             "side": _stack((50, 30, 150, 85), (50, 85, 135, 140), (50, 140, 120, 190))}
    face = {"front": _stack((75, 85, 125, 135)), "side": _stack((110, 85, 135, 135))}
    dims = {"width": 0.45, "depth": 0.5, "height": 0.8}
    return calibs, masks, face, dims


def body_field(res=BODY_RES):
    calibs, masks, face, dims = head()
    return carve.build_hull_field(masks, calibs, dims, res, carve_views=["front", "side"])


def layer_of(res=96, **kw):
    calibs, masks, face, dims = head()
    return layers.build_layer_field(face, calibs, "front", "side", resolution=res, **kw)


def front_y(field, origin, cell, x, z):
    """Smallest y of the inside voxels of the column nearest (x, z); None when the column is empty."""
    i = int((x - origin[0]) / cell)
    k = int((z - origin[2]) / cell)
    inside = np.flatnonzero(field[i, :, k] < 0)
    return None if inside.size == 0 else float(origin[1] + (inside[0] + 0.5) * cell)


def centre_column(layer, x=0.0, z=0.40):
    return int((z - layer.origin[2]) / layer.cell), int((x - layer.origin[0]) / layer.cell)


def test_layer_field_matches_its_masks():
    layer = layer_of()
    assert layer.cell < 0.003 and (layer.field < 0).any()
    k, i = centre_column(layer)                     # face centre: x = 0, z = (190 - 110) * PX = 0.40 m
    assert layer.y_front[k, i] == pytest.approx(-0.175, abs=2 * layer.cell)
    assert layer.y_back[k, i] == pytest.approx(-0.05, abs=2 * layer.cell)
    cols = np.isfinite(layer.y_front)               # footprint stays inside the front label (x +-0.125, z 0.275..0.525)
    xs, zs = layer.centers(0), layer.centers(2)
    assert xs[cols.any(axis=0)].min() > -0.125 - 2 * layer.cell and xs[cols.any(axis=0)].max() < 0.125 + 2 * layer.cell
    assert zs[cols.any(axis=1)].min() > 0.275 - 2 * layer.cell and zs[cols.any(axis=1)].max() < 0.525 + 2 * layer.cell
    assert (layer.footprint_sd[cols] < 0).all() and (layer.footprint_sd[~cols] > 0).all()


def test_layer_front_sits_behind_the_rim_front():
    field, origin, cell = body_field()
    layer = layer_of()
    new, stats = layers.recess_field(field, origin, cell, layer)
    rim_y = front_y(new, origin, cell, 0.0, 0.62)   # dome / rim rows 30-85: z = 0.525 .. 0.80
    assert rim_y == pytest.approx(-0.25, abs=2 * cell)
    k, i = centre_column(layer)
    expected = (150 - 135) * PX                     # rim front (column 150) minus face front (column 135) = 0.075 m
    assert layer.y_front[k, i] - rim_y == pytest.approx(expected, abs=3 * cell)
    assert stats["carved_cells"] > 0


def test_body_has_no_hole_where_the_layer_sits():
    field, origin, cell = body_field()
    layer = layer_of()
    new, _ = layers.recess_field(field, origin, cell, layer)
    for x in (-0.08, 0.0, 0.08):
        y = front_y(new, origin, cell, x, 0.40)     # body front moved back to the layer's back surface of that column
        k, i = centre_column(layer, x)
        assert layer.y_back[k, i] - 0.006 < y < layer.y_back[k, i] + cell, (x, y, layer.y_back[k, i])
        i, k = int((x - origin[0]) / cell), int((0.40 - origin[2]) / cell)
        col = new[i, :, k] < 0
        first = np.flatnonzero(col)[0]
        assert col[first:int((0.2 - origin[1]) / cell)].all()      # solid from the recess floor to the back
    verts, quads = surface_nets(new, origin + 0.5 * cell, cell)    # watertight-ish: one closed surface
    assert len(quads) > 0 and (edge_counts(quads) == 2).all()


def test_union_of_layer_and_body_keeps_the_silhouette():
    field, origin, cell = body_field()
    layer = layer_of()
    new, _ = layers.recess_field(field, origin, cell, layer)
    for x in (-0.08, 0.0, 0.08):                    # layer front = the face front at the face centre, no gap to the body
        k, i = centre_column(layer, x)
        body_front = front_y(new, origin, cell, x, 0.40)
        assert layer.y_back[k, i] > body_front - cell - 0.002    # layer back reaches the recessed body surface (one body voxel)
        assert layer.y_front[k, i] < body_front                    # and its front stands in front of it


def test_recess_is_confined_to_the_layer_region():
    field, origin, cell = body_field()
    layer = layer_of()
    new, stats = layers.recess_field(field, origin, cell, layer)
    changed = (field < 0) != (new < 0)
    assert changed.any() and not ((new < 0) & ~(field < 0)).any()  # only removal, never growth
    ci = np.nonzero(changed)
    cx = origin[0] + (ci[0] + 0.5) * cell
    cy = origin[1] + (ci[1] + 0.5) * cell
    cz = origin[2] + (ci[2] + 0.5) * cell
    assert (np.abs(cx) <= 0.125 + 2 * cell).all()                              # face width
    assert (cz >= 0.275 - 2 * cell).all() and (cz <= 0.525 + 2 * cell).all()   # face rows
    assert (cy < -0.03).all()                                                  # nothing behind the recess floor
    for x, z in ((0.0, 0.62), (0.0, 0.15), (0.20, 0.40)):                      # rim, neck, beside the face: untouched
        assert front_y(new, origin, cell, x, z) == front_y(field, origin, cell, x, z)
    assert stats["max_recess_m"] > 0.1


def test_layer_errors():
    calibs, masks, face, dims = head()
    with pytest.raises(CalibrationError, match="no label pixels"):
        layers.build_layer_field({"front": face["front"], "side": np.zeros_like(face["side"])}, calibs, "front", "side")
    bad = dict(calibs, front=ViewCalibration("front", "+Y", (SIZE, SIZE), PX, ORIGIN))
    with pytest.raises(CalibrationError, match="must look from -Y"):
        layers.build_layer_field(face, bad, "front", "side")
    far = {"front": _stack((75, 20, 125, 40)), "side": face["side"]}
    with pytest.raises(CalibrationError, match="do not overlap in z"):
        layers.build_layer_field(far, calibs, "front", "side")


def test_part_masks_of_a_label():
    face = np.zeros((4, 4), bool)
    face[1:3, 1:3] = True
    lm = {"front": {"body": ~face, "face": face}, "side": {"body": np.ones((4, 4), bool)},
          "back": {"face": np.zeros((4, 4), bool)}}
    got = labels_core.part_masks(lm, "face")
    assert list(got) == ["front"] and int(got["front"].sum()) == 4
