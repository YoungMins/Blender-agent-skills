"""Hull debris: detached fragments of the carved hull that belong to no part (numpy only)."""
import numpy as np
import pytest

from bbr.core import debris as D, gapcheck
from bbr.core.calibration import ViewCalibration
from bbr.core.surface import surface_nets

CELL = 0.01
N = 70                                    # voxels per side: x, y in -0.35 .. 0.35, z in 0 .. 0.7
ORIGIN = np.array([-0.35, -0.35, 0.0])
PX = 0.01


def calibs():
    return {"front": ViewCalibration("front", "-Y", (120, 90), PX, (60.0, 80.0)),
            "side": ViewCalibration("side", "-X", (120, 90), PX, (60.0, 80.0))}


def box(occ, lo, hi):
    i0 = np.round((np.asarray(lo) - ORIGIN) / CELL).astype(int)
    i1 = np.round((np.asarray(hi) - ORIGIN) / CELL).astype(int)
    occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = True


def masks_of(occ):
    out = {}
    idx = np.argwhere(occ)
    pts = ORIGIN + (idx + 0.5) * CELL
    for view, c in calibs().items():
        m = np.zeros((90, 120), bool)
        uv = np.floor(c.world_to_px(pts)).astype(int)
        m[uv[:, 1], uv[:, 0]] = True
        out[view] = m
    return out


def label_of(masks, view, lo, hi, other=None):
    """Label mask: the carve mask restricted to the pixel box of the world box lo..hi in ``view`` (x/y by view, z up)."""
    c = calibs()[view]
    px = np.floor(c.world_to_px(np.array([lo, hi], float))).astype(int)
    x0, x1 = sorted((px[0][0], px[1][0]))
    y0, y1 = sorted((px[0][1], px[1][1]))
    m = np.zeros_like(masks[view])
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m & masks[view]


@pytest.fixture
def scene():
    """A body with a leg, plus a one-view fragment (tail tip beside the foot: separate in front, inside the leg's
    silhouette in side), a speck, and a real detached part (a staff)."""
    occ = np.zeros((N, N, N), bool)
    box(occ, (-0.2, -0.2, 0.1), (0.2, 0.2, 0.6))           # body
    box(occ, (-0.03, -0.03, 0.0), (0.03, 0.03, 0.1))        # leg
    box(occ, (0.28, -0.03, 0.03), (0.34, 0.03, 0.07))       # fragment A: front blob of its own, side inside the leg
    box(occ, (0.30, 0.28, 0.5), (0.32, 0.30, 0.52))         # speck B: separate blob in both views
    box(occ, (-0.34, -0.34, 0.3), (-0.28, -0.28, 0.65))     # staff C: separate blob in both views, a real part
    return occ, masks_of(occ)


def field_of(occ):
    return np.where(occ, -1.0, 1.0).astype(np.float32)


def run(scene, labels):
    occ, masks = scene
    return D.find_debris(field_of(occ), ORIGIN, CELL, masks, calibs(), ["front", "side"], labels)


def by_box(report, x, y, z):
    for e in report:
        lo, hi = e["box_m"]
        if lo[0] - 0.02 <= x <= hi[0] + 0.02 and lo[1] - 0.02 <= y <= hi[1] + 0.02 and lo[2] - 0.02 <= z <= hi[2] + 0.02:
            return e
    raise AssertionError(f"no island at {(x, y, z)} in {[e['box_m'] for e in report]}")


def test_one_view_fragment_under_different_labels_is_dropped(scene):
    occ, masks = scene
    labels = {"front": {"tail": label_of(masks, "front", (0.26, 0, 0.0), (0.36, 0, 0.1))},     # tail label under the fragment
              "side": {"foot": label_of(masks, "side", (0, -0.04, 0.0), (0, 0.04, 0.1))}}      # a foot in the other view
    drop, report = run(scene, labels)
    a = by_box(report, 0.31, 0.0, 0.05)
    assert a["dropped"] and a["rule"] == "a"
    assert a["views"]["front"]["separate"] and not a["views"]["side"]["separate"]
    assert drop[tuple(np.round((np.array([0.31, 0.0, 0.05]) - ORIGIN) / CELL).astype(int))]


def test_the_same_labelled_part_under_both_views_is_a_real_detached_part(scene):
    occ, masks = scene
    labels = {"front": {"staff": label_of(masks, "front", (-0.36, 0, 0.28), (-0.26, 0, 0.66)),
                        "tail": label_of(masks, "front", (0.26, 0, 0.0), (0.36, 0, 0.1))},
              "side": {"staff": label_of(masks, "side", (0, -0.36, 0.28), (0, -0.26, 0.66)),
                       "foot": label_of(masks, "side", (0, -0.04, 0.0), (0, 0.04, 0.1))}}
    drop, report = run(scene, labels)
    staff = by_box(report, -0.31, -0.31, 0.47)
    assert not staff["dropped"]                                   # labelled staff in both views: kept (and large)
    assert not drop[tuple(np.round((np.array([-0.31, -0.31, 0.47]) - ORIGIN) / CELL).astype(int))]


def test_a_small_unlabelled_speck_is_dropped_when_the_scene_has_labels(scene):
    occ, masks = scene
    labels = {"front": {"staff": label_of(masks, "front", (-0.36, 0, 0.28), (-0.26, 0, 0.66))},
              "side": {"staff": label_of(masks, "side", (0, -0.36, 0.28), (0, -0.26, 0.66))}}
    drop, report = run(scene, labels)
    speck = by_box(report, 0.31, 0.29, 0.51)
    assert speck["dropped"] and speck["rule"] == "b"
    assert speck["volume_pct"] < 100 * D.VOLUME_FRAC


def test_without_part_labels_every_island_is_kept(scene):
    """Hull-only scenes (a hatted figure, a creature's legs) say nothing about what is a part: nothing is dropped."""
    drop, report = run(scene, None)
    assert not drop.any()
    assert len(report) >= 3 and not any(e["dropped"] for e in report)


def test_a_big_unlabelled_island_is_kept_even_in_a_labelled_scene(scene):
    occ, masks = scene
    labels = {"front": {"tail": label_of(masks, "front", (0.26, 0, 0.0), (0.36, 0, 0.1))},
              "side": {"tail": label_of(masks, "side", (0, -0.04, 0.0), (0, 0.04, 0.1))}}
    drop, report = run(scene, labels)
    staff = by_box(report, -0.31, -0.31, 0.47)
    assert not staff["dropped"] and staff["volume_pct"] > 100 * D.VOLUME_FRAC


def test_dropping_leaves_one_closed_surface_per_kept_island(scene):
    occ, masks = scene
    labels = {"front": {"tail": label_of(masks, "front", (0.26, 0, 0.0), (0.36, 0, 0.1))},
              "side": {"foot": label_of(masks, "side", (0, -0.04, 0.0), (0, 0.04, 0.1))}}
    drop, report = run(scene, labels)
    f = D.drop_debris(field_of(occ), drop)
    assert (f[drop] > 0).all() and (f[~drop] == field_of(occ)[~drop]).all()
    kept = sum(1 for e in report if not e["dropped"])
    verts, quads = surface_nets(f, ORIGIN + 0.5 * CELL, CELL, min_island_frac=0.0)
    from bbr.core.islands import vertex_components
    edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    n = len(np.unique(vertex_components(len(verts), edges)))
    assert n == 1 + kept


def test_the_report_names_what_was_dropped_and_where(scene):
    occ, masks = scene
    labels = {"front": {"tail": label_of(masks, "front", (0.26, 0, 0.0), (0.36, 0, 0.1))},
              "side": {"foot": label_of(masks, "side", (0, -0.04, 0.0), (0, 0.04, 0.1))}}
    _, report = run(scene, labels)
    lines = D.format_debris_report(report, "body")
    assert len(lines) == 1 and lines[0].startswith("body dropped ") and "rule a" in lines[0] and "drop_debris" in lines[0]
    assert D.format_debris_report([e for e in report if not e["dropped"]], "body") == []


def test_floating_tolerance_follows_the_size_of_the_part():
    """max(1 % of the height, 25 % of the part's minor dimension): a thick fin may sit 15 mm off, a thin plate may not."""
    def cube(lo, hi):
        lo, hi = np.array(lo, float), np.array(hi, float)
        v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
        return v, np.array(f)

    body = cube((-0.1, -0.1, 0.0), (0.1, 0.1, 1.0))
    thick = cube((0.115, -0.04, 0.4), (0.195, 0.04, 0.5))          # 80 mm minor dimension, 15 mm off the body
    thin = cube((0.115, -0.04, 0.6), (0.195, -0.036, 0.7))         # 4 mm plate, 15 mm off the body
    assert gapcheck.minor_dimension(thick[0]) == pytest.approx(0.08, abs=1e-6)
    out = gapcheck.floating_parts({"body": body, "fin": thick, "plate": thin}, 1.0)
    names = {g["part"]: g for g in out}
    assert "fin" not in names and "plate" in names
    assert names["plate"]["tol_m"] == pytest.approx(0.01, abs=1e-6)                 # still reported, with the tolerance it was held to
    out2 = gapcheck.floating_parts({"body": body, "fin": cube((0.135, -0.04, 0.4), (0.215, 0.04, 0.5))}, 1.0)
    assert out2 and out2[0]["part"] == "fin" and out2[0]["tol_m"] == pytest.approx(0.02, abs=1e-6)     # 35 mm off: floats even so


def test_mesh_components_split_a_hull_with_a_detached_island():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)
    t = np.array([[0, 1, 2], [0, 1, 3], [4, 5, 6]])
    parts = gapcheck.mesh_components(v, t)
    assert len(parts) == 2 and len(parts[0][1]) == 2 and len(parts[1][1]) == 1
