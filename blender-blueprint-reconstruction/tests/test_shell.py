"""Layer shell (glass visor): a closed thin sheet over the layer's front surface (numpy only)."""
import numpy as np
import pytest

from bbr import CalibrationError
from bbr.core import layers


def dome_layer(nx=12, nz=10, cell=0.01):
    """A layer whose front surface is a dome over an elliptical footprint (y_front < 0 in the middle)."""
    x = (np.arange(nx) + 0.5) / nx * 2 - 1
    z = (np.arange(nz) + 0.5) / nz * 2 - 1
    r2 = x[None, :] ** 2 + z[:, None] ** 2
    yf = np.where(r2 < 0.9, -0.03 * (1 - r2), np.nan).astype(np.float32)
    field = np.zeros((nx, 4, nz), np.float32)
    return layers.LayerField(field, np.array([-0.06, -0.05, 0.3]), cell, yf, yf + 0.02, np.zeros_like(yf))


def signed_volume(verts, quads):
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def test_shell_is_closed_and_outward():
    lf = dome_layer()
    verts, quads = layers.shell_mesh(lf, 0.002)
    edges = np.sort(np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()                                      # watertight
    area = np.isfinite(lf.y_front)
    vol = signed_volume(verts, quads)
    assert vol > 0                                                  # outward normals
    cells = ((area[:-1, :-1] & area[:-1, 1:] & area[1:, :-1] & area[1:, 1:]).sum())
    assert vol == pytest.approx(cells * lf.cell ** 2 * 0.002, rel=1e-6)


def test_front_faces_minus_y_and_follows_the_dome():
    lf = dome_layer()
    verts, quads = layers.shell_mesh(lf, 0.002)
    n_front = int((np.isfinite(lf.y_front)[:-1, :-1] & np.isfinite(lf.y_front)[:-1, 1:]
                   & np.isfinite(lf.y_front)[1:, :-1] & np.isfinite(lf.y_front)[1:, 1:]).sum())
    q = quads[:n_front]
    normal = np.cross(verts[q[:, 1]] - verts[q[:, 0]], verts[q[:, 2]] - verts[q[:, 0]])
    assert (normal[:, 1] < 0).all()
    assert verts[:, 1].min() == pytest.approx(np.nanmin(lf.y_front))
    assert verts[:, 1].max() == pytest.approx(np.nanmax(lf.y_front) + 0.002)


def test_rejects_bad_input():
    lf = dome_layer()
    with pytest.raises(ValueError):
        layers.shell_mesh(lf, 0.0)
    empty = layers.LayerField(lf.field, lf.origin, lf.cell, np.full_like(lf.y_front, np.nan), lf.y_back, lf.footprint_sd)
    with pytest.raises(CalibrationError):
        layers.shell_mesh(empty)


def test_spec_checks_cover():
    from bbr.core import spec as S
    base = {"spec_version": "0.1", "name": "a", "components": [
        {"id": "body", "type": "silhouette_hull", "params": {"layers": ["face"]}},
        {"id": "face", "type": "layer_hull", "params": {"label": "face"}},
        {"id": "visor", "type": "layer_hull", "params": {"shell": True, "cover": ["face"]}}]}
    assert S.validate_scene_spec(base) == []
    sc = S.parse_scene_spec(base)
    assert not S.is_textured_component(sc.get_component("visor"), character=True)       # glass: no texture by default
    assert S.is_textured_component(sc.get_component("face"), character=True)
    bad = {**base, "components": base["components"][:2] + [{"id": "visor", "type": "layer_hull", "params": {"shell": True, "cover": ["body"]}}]}
    assert any("cover" in e for e in S.validate_scene_spec(bad))
    bad = {**base, "components": base["components"][:2] + [{"id": "visor", "type": "layer_hull", "params": {"cover": ["face"]}}]}
    assert any("shell" in e for e in S.validate_scene_spec(bad))


# --------------------------------------------------------------------------- face detail: profile relief
from bbr.core.calibration import ViewCalibration

MM = 0.001


def side_face(nose_px=8, front_u=300):
    """Side label (camera -X, image right = -Y): a smooth front contour at ``front_u`` with a nose bump of
    ``nose_px`` around rows 140..160."""
    m = np.zeros((300, 400), bool)
    rows = np.arange(100, 220)
    for r in rows:
        bump = nose_px * np.exp(-0.5 * ((r - 150) / 4.0) ** 2)
        m[r, 200:int(round(front_u + bump)) + 1] = True
    return m


def test_profile_relief_finds_the_nose_and_flattens_the_mask():
    cal = ViewCalibration("side", "-X", (400, 300), MM, (200.0, 250.0))
    rel = layers.profile_relief(side_face(), cal, smooth=0.08)
    assert rel.amplitude == pytest.approx(8 * MM, abs=2.5 * MM)
    z_nose = float(cal.px_to_plane(np.array([[0.0, 150.5]]))[0, 2])
    k = int(np.argmin(np.abs(rel.z - z_nose)))
    assert rel.dy[k] < -4 * MM                                         # the nose sticks out toward -Y
    assert np.abs(rel.dy[np.abs(rel.z - z_nose) > 0.04]).max() < 1 * MM  # nothing far from the nose
    front_cols = [np.nonzero(rel.mask[r])[0].max() for r in range(100, 220)]
    assert max(front_cols) - min(front_cols) <= 3                        # the carved profile has no bump left
    flat = layers.profile_relief(side_face(nose_px=0), cal)
    assert flat.amplitude < 0.5 * MM


def test_profile_relief_needs_a_side_view():
    with pytest.raises(CalibrationError):
        layers.profile_relief(side_face(), ViewCalibration("front", "-Y", (400, 300), MM, (200.0, 250.0)))


def test_apply_relief_moves_the_midline_front_only():
    lf = dome_layer(nx=20, nz=20)
    rel = layers.ProfileRelief(np.zeros((1, 1), bool), np.array([0.3, 0.6]), np.array([-0.005, -0.005]))
    xs = lf.centers(0)
    zs = lf.centers(2)
    j = 10
    yf, yb = float(lf.y_front[j, 10]), float(lf.y_back[j, 10])
    pts = np.array([[xs[10], yf, zs[j]], [xs[10], yb, zs[j]], [xs[1], float(lf.y_front[j, 1]) if np.isfinite(lf.y_front[j, 1]) else 0.0, zs[j]],
                    [xs[10], yf, 0.9]])
    out = layers.apply_relief(pts, lf, rel, x0=float(xs[10]), width=0.01)
    assert out[0, 1] == pytest.approx(yf - 0.005, abs=1e-4)       # midline front: full feature
    assert out[1, 1] == pytest.approx(yb)                            # back: untouched
    assert abs(out[2, 1] - pts[2, 1]) < 1e-3                         # far from the midline: almost nothing
    assert out[3, 1] == pytest.approx(pts[3, 1])                     # outside the profile's z range


def test_feature_half_widths_from_the_front_view():
    cal = ViewCalibration("front", "-Y", (200, 200), MM, (100.0, 190.0))
    rgb = np.zeros((200, 200, 3)); rgb[...] = (0.95, 0.72, 0.55)      # skin
    mask = np.zeros((200, 200), bool); mask[20:180, 40:160] = True
    rgb[20:60, 40:160] = (0.35, 0.2, 0.1)                                 # hair fringe across the face
    rgb[140:150, 80:120] = (0.6, 0.1, 0.1)                                # a 40 px mouth on the midline
    rgb[150:160, 30:50] = (0.1, 0.1, 0.1)                                 # dark, but off the midline
    z, hw = layers.feature_half_widths(rgb, mask, cal, 0.0)
    assert np.all(np.diff(z) > 0)

    def at(row):
        return float(np.interp(cal.px_to_plane(np.array([[0.0, row + 0.5]]))[0, 2], z, hw))
    assert at(40) == pytest.approx(60 * MM)      # fringe: the whole 120 px label width
    assert at(145) == pytest.approx(20 * MM)     # mouth
    assert at(100) == 0.0 and at(170) == 0.0     # skin on the midline (a shaded nose; the dark blob at 150-160 is off-centre)
    assert 0.0 < at(153) < 20 * MM               # the mouth width tapers off below it


def test_apply_relief_keeps_a_wide_feature_flat_across():
    lf = dome_layer(nx=20, nz=20)
    xs, zs = lf.centers(0), lf.centers(2)
    j = 10
    hw = np.array([0.05, 0.05])
    rel = layers.ProfileRelief(np.zeros((1, 1), bool), np.array([0.3, 0.6]), np.array([-0.005, -0.005]), hw)
    p_mid = [xs[10], float(lf.y_front[j, 10]), zs[j]]
    p_side = [xs[10] + 0.03, float(lf.y_front[j, 13]), zs[j]]
    out = layers.apply_relief(np.array([p_mid, p_side]), lf, rel, x0=float(xs[10]), width=0.005)
    assert out[0, 1] == pytest.approx(p_mid[1] - 0.005, abs=1e-4)
    assert out[1, 1] == pytest.approx(p_side[1] - 0.005, abs=1e-3)   # inside the half-width: full depth too
