"""Contour plates (wings, fins): outline from a label mask, plane pose, mesh, fit (numpy only)."""
import numpy as np
import pytest

from bbr import BBRError
from bbr.core import partfit as P, partgeom, plate, triangulate as T
from bbr.core.calibration import ViewCalibration
from bbr.core.partmask import triangles_mask

MM = 0.001


@pytest.fixture
def calibs():
    return {"front": ViewCalibration("front", "-Y", (400, 500), MM, (200.0, 450.0)),
            "side": ViewCalibration("side", "-X", (300, 500), MM, (150.0, 450.0)),
            "back": ViewCalibration("back", "+Y", (400, 500), MM, (200.0, 450.0))}


def px(calibs, view, point):
    return [float(v) for v in np.round(calibs[view].world_to_px([point])[0])]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def wing_truth(base, axis, normal, thickness=0.004, bend=0.0):
    """A triangle wing: root edge 0.07 m wide at the base, tip 0.16 m along the axis (a bat-wing-like triangle)."""
    n = unit(normal)
    u = unit(np.asarray(axis, float) - (np.asarray(axis, float) @ n) * n)
    outline = np.array([[0.0, -0.035], [0.16, 0.0], [0.0, 0.035]])
    tris = plate.triangulate_polygon(outline)
    verts, faces = partgeom.plate_mesh(outline, tris, thickness, u, n, bend)
    return verts + base, partgeom.polygons_triangles(faces), u, n


def area_of(poly, tris):
    """Polygon area from its triangles (2D cross products written out)."""
    return 0.5 * sum(abs((poly[b][0] - poly[a][0]) * (poly[c][1] - poly[a][1]) - (poly[b][1] - poly[a][1]) * (poly[c][0] - poly[a][0]))
                     for a, b, c in tris)


def masks_of(verts, tris, calibs, views=("front", "side", "back")):
    return {v: triangles_mask(verts, tris, calibs[v]) for v in views}


def test_outline_of_a_triangle_label_is_a_triangle():
    m = np.zeros((80, 100), bool)
    yy, xx = np.mgrid[:80, :100]
    m[(yy > 10) & (yy < 70) & (xx > 10) & (xx < 10 + (yy - 10) * 1.4)] = True
    poly = plate.outline_from_mask(m)
    assert 3 <= len(poly) <= 6
    tris = plate.triangulate_polygon(poly)
    area = area_of(poly, tris)
    assert area == pytest.approx(float(m.sum()), rel=0.08)


def test_outline_keeps_the_largest_blob_and_fills_holes():
    m = np.zeros((60, 60), bool)
    m[10:40, 10:40] = True
    m[20:25, 20:25] = False                                   # a hole
    m[50:53, 50:53] = True                                    # a speck
    poly = plate.outline_from_mask(m)
    assert poly[:, 0].max() <= 40.5 and poly[:, 1].max() <= 40.5
    tris = plate.triangulate_polygon(poly)
    area = area_of(poly, tris)
    assert area == pytest.approx(900.0, rel=0.03)


def test_ear_clipping_handles_a_concave_polygon():
    poly = np.array([[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]], float)           # a notch on top
    tris = plate.triangulate_polygon(poly)
    assert len(tris) == 3
    area = area_of(poly, tris)
    assert area == pytest.approx(10.0)


@pytest.mark.parametrize("bend", [0.0, 35.0])
def test_plate_mesh_is_closed_with_outward_normals(bend):
    outline = np.array([[0.0, -0.03], [0.15, 0.0], [0.0, 0.03]])
    verts, faces = partgeom.plate_mesh(outline, plate.triangulate_polygon(outline), 0.004, (1, 0, 0), (0, 0, 1), bend)
    assert partgeom.mesh_volume(verts, faces) > 0
    edges: dict = {}
    for f in faces:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            edges[e] = edges.get(e, 0) + 1
    assert set(edges.values()) == {2}                         # watertight
    if bend:
        assert verts[:, 2].max() > 0.015                      # the tip curls toward +normal
    else:
        assert np.ptp(verts[:, 2]) == pytest.approx(0.004)
    flipped = outline * np.array([1.0, -1.0])                  # a mirrored outline is wound the other way: still outward
    v2, f2 = partgeom.plate_mesh(flipped, plate.triangulate_polygon(flipped), 0.004, (1, 0, 0), (0, 0, 1), bend)
    assert partgeom.mesh_volume(v2, f2) > 0


def test_normal_from_an_edge_on_line(calibs):
    # the plate is vertical and runs along X in the side view's plane: seen edge-on from the side (camera -X) it is a line
    n = plate.normal_from_lines({"side": [[150, 450], [100, 450]]}, calibs, np.array([0.0, -1.0, 0.0]))
    assert abs(n[1]) == pytest.approx(1.0) or abs(n[2]) == pytest.approx(1.0)
    n2 = plate.normal_from_lines({"side": [[150, 450], [150, 350]]}, calibs, np.array([0.0, -1.0, 0.0]))     # a vertical line
    assert abs(n2[1]) == pytest.approx(1.0)                  # the plate is the XZ plane: normal along Y
    assert n2 @ np.array([0.0, -1.0, 0.0]) > 0


def solve_wing(calibs, base, tip, mask_front, extra=None):
    params = {"thickness": 0.004, "face_view": "front", "anchors_px": {
        "front": {"base": px(calibs, "front", base), "tip": px(calibs, "front", tip)},
        "side": {"base": px(calibs, "side", base), "tip": px(calibs, "side", tip)}}, **(extra or {})}
    sol = T.solve_part("wing", "plate", params, calibs, label_for=lambda v: mask_front if v == "front" else None)
    return sol, params


def test_solve_places_the_plate_through_base_and_tip(calibs):
    base = np.array([0.03, 0.01, 0.30])
    axis = unit([0.9, 0.1, 0.25])
    verts, tris, u, n = wing_truth(base, axis, [0.15, -1.0, 0.1])
    masks = masks_of(verts, tris, calibs)
    tip = base + 0.16 * u
    sol, _ = solve_wing(calibs, base, tip, masks["front"])
    assert np.allclose(sol.keypoints["base"], base, atol=0.003)
    assert np.allclose(sol.keypoints["tip"], tip, atol=0.004)
    got_n = np.asarray(sol.params["normal"])
    assert abs(got_n @ (sol.keypoints["tip"] - sol.keypoints["base"])) < 1e-6          # the plane contains base -> tip
    assert got_n @ np.array([0.0, -1.0, 0.0]) > 0                                      # faces the face-on camera
    # the lifted outline seen from the face-on view is the label
    shape = P.shape_of("plate", P.pose_from_solution(sol, {}))
    assert P.iou(P.render(shape, calibs["front"]), masks["front"]) > 0.93


def test_fit_recovers_the_tilt_from_the_other_views(calibs):
    """A triangle wing tilted 25 degrees away from the face-on camera; the anchors give no depth for the tip, so the
    initial plane faces the camera: only the side / back silhouettes can tell the tilt."""
    base = np.array([0.03, 0.01, 0.30])
    verts, tris, u, n = wing_truth(base, unit([0.9, 0.1, 0.25]), [0.45, -1.0, 0.15])
    masks = masks_of(verts, tris, calibs)
    tip = base + 0.16 * u
    params = {"thickness": 0.004, "face_view": "front", "anchors_px": {
        "front": {"base": px(calibs, "front", base), "tip": px(calibs, "front", tip)}, "side": {"base": px(calibs, "side", base)}}}
    sol = T.solve_part("wing", "plate", params, calibs, label_for=lambda v: masks["front"] if v == "front" else None)
    assert abs(np.asarray(sol.params["normal"]) @ n) < 0.95                            # the start is visibly wrong
    new, report = P.fit_part("wing", "plate", sol, params, calibs, masks, [P.Target("wing", False, masks)])
    assert report["fitted"] and report["loss_fit"] < report["loss_init"] / 2
    assert abs(np.asarray(new.params["normal"]) @ n) > 0.98                            # within ~11 degrees
    iou = P.part_iou("plate", new, {**params, **new.params}, calibs, masks, masks)
    assert min(v["iou"] for v in iou.values()) > 0.9


def test_a_mirrored_plate_without_anchors_copies_the_outline_mirrored(calibs):
    base = np.array([0.03, 0.01, 0.30])
    verts, tris, u, n = wing_truth(base, unit([0.9, 0.1, 0.25]), [0.15, -1.0, 0.1])
    masks = masks_of(verts, tris, calibs)
    sol, _ = solve_wing(calibs, base, base + 0.16 * u, masks["front"])
    twin = T.solve_part("wing_l", "plate", {"mirror_of": "wing"}, calibs, sibling_for=lambda _id: sol, label_for=lambda v: None)
    assert np.allclose(twin.keypoints["base"], sol.keypoints["base"] * [-1, 1, 1])
    v1, f1 = partgeom.plate_mesh(sol.params["outline"], sol.params["outline_tris"], 0.004, sol.params["axis"], sol.params["normal"])
    v2, f2 = partgeom.plate_mesh(twin.params["outline"], twin.params["outline_tris"], 0.004, twin.params["axis"], twin.params["normal"])
    assert partgeom.mesh_volume(v2, f2) > 0
    assert np.allclose(np.sort(v2 + twin.keypoints["base"], axis=0), np.sort(v1 * [-1, 1, 1] + twin.keypoints["base"], axis=0), atol=1e-6)


def test_a_plate_without_a_label_is_an_error(calibs):
    params = {"thickness": 0.004, "anchors_px": {"front": {"base": [200, 300], "tip": [260, 280]},
                                                  "side": {"base": [150, 300], "tip": [140, 280]}}}
    with pytest.raises(BBRError, match="label"):
        T.solve_part("wing", "plate", params, calibs, label_for=lambda v: None)
