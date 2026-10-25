"""Hard-surface kit: arch / torus / per-axis rounded_box geometry, mirror pivot + bevel bool in the spec, texture policy
(cutters never textured), width-based sizing."""
import copy
import json
import math
from collections import Counter

import numpy as np
import pytest

from bbr import BBRError
from bbr.core import partgeom as g
from bbr.core import spec as S
from bbr.core.calibration import body_bbox, calibrate_views, derive_height
from bbr.core.imageops import mask_bbox
from conftest import paint, rect_mask


def _closed(faces):
    edges = Counter(frozenset((f[i], f[(i + 1) % len(f)])) for f in faces for i in range(len(f)))
    return set(edges.values()) == {2}


# --------------------------------------------------------------------------- geometry
@pytest.mark.parametrize("axis,bulge,size,half_w,height,length", [
    ("X", None, (0.8, 0.4, 0.2), 0.2, 0.2, 0.8),
    ("Y", "-Z", (0.4, 0.8, 0.1), 0.2, 0.1, 0.8),
    ("Z", None, (0.4, 0.3, 0.8), 0.15, 0.4, 0.8),
])
def test_arch_is_a_closed_half_ellipse_prism(axis, bulge, size, half_w, height, length):
    v, f = g.arch_mesh(size, axis, bulge)
    assert _closed(f)
    assert g.mesh_volume(v, f) == pytest.approx(math.pi * half_w * height / 2 * length, rel=0.01)   # positive: wound outward
    assert np.allclose(v.max(axis=0) - v.min(axis=0), size, atol=1e-9)                              # size = bounding extents
    assert np.allclose((v.max(axis=0) + v.min(axis=0)) / 2, 0, atol=1e-9)                           # centred


def test_arch_bulge_direction_and_flat_base():
    v, _ = g.arch_mesh((0.8, 0.4, 0.2), "X", "-Z")
    base = v[np.isclose(v[:, 2], 0.1)]                                                              # apex down, flat base up
    assert v[:, 2].min() == pytest.approx(-0.1) and len(base) >= 2
    assert abs(base[:, 1]).max() == pytest.approx(0.2)                                              # the base spans the full width


def test_arch_band_is_a_shell_segment():
    v, f = g.arch_mesh((0.8, 0.4, 0.2), "X", thickness=0.02)
    assert _closed(f) and g.mesh_volume(v, f) == pytest.approx(0.8 * math.pi / 2 * (0.2 * 0.2 - 0.18 * 0.18), rel=0.01)
    assert np.allclose(v.max(axis=0) - v.min(axis=0), (0.8, 0.4, 0.2))                              # outer size unchanged
    with pytest.raises(BBRError):
        g.arch_mesh((0.8, 0.4, 0.2), "X", thickness=0.2)
    with pytest.raises(BBRError):
        g.arch_mesh((0.8, 0.4, 0.2), "X", bulge="+X")                                               # along the axis
    with pytest.raises(BBRError):
        g.arch_mesh((0.8, 0.4, 0.2), "W")


def test_torus_volume_extents_and_axis():
    v, f = g.torus_mesh(0.1, 0.01, "X")
    assert _closed(f) and g.mesh_volume(v, f) == pytest.approx(2 * math.pi ** 2 * 0.1 * 0.01 ** 2, rel=0.03)
    assert np.allclose(v.max(axis=0), (0.01, 0.11, 0.11), atol=1e-3) and np.allclose(v.min(axis=0), -v.max(axis=0), atol=1e-3)
    e, fe = g.torus_mesh((0.07, 0.05), 0.01, "Z")                                                  # elliptic ring in XY
    assert g.mesh_volume(e, fe) > 0 and e[:, 0].max() == pytest.approx(0.08, abs=1e-3) and e[:, 1].max() == pytest.approx(0.06, abs=1e-3)
    with pytest.raises(BBRError):
        g.torus_mesh(0.05, 0.06)                                                                   # the tube swallows the hole
    with pytest.raises(BBRError):
        g.torus_mesh((0.2, 0.05), 0.04)                                                            # tighter than the ellipse's curvature


def test_rounded_box_per_axis_radius():
    size = (0.4, 0.2, 0.1)
    v, f = g.rounded_box_mesh(size, (0.0, 0.03, 0.01), 4)
    assert _closed(f) and np.allclose(v.max(axis=0) - v.min(axis=0), size)
    # edges parallel to X get an elliptic 0.03 x 0.01 profile; nothing is cut along X
    assert g.mesh_volume(v, f) == pytest.approx(0.4 * 0.2 * 0.1 - (1 - math.pi / 4) * 0.03 * 0.01 * 4 * 0.4, rel=0.01)
    assert not ((v[:, 1] > 0.0999) & (v[:, 2] > 0.0499)).any()                                      # no sharp edge vertex left
    uniform, uf = g.rounded_box_mesh(size, (0.02,) * 3, 3)
    assert _closed(uf) and _closed(g.rounded_box_mesh(size, (0.01, 0.02, 0.0))[1])
    assert g.mesh_volume(uniform, uf) < 0.4 * 0.2 * 0.1
    sharp, sf = g.rounded_box_mesh(size, (0, 0, 0))
    assert g.mesh_volume(sharp, sf) == pytest.approx(0.4 * 0.2 * 0.1)


# --------------------------------------------------------------------------- spec
@pytest.fixture
def data(fixtures_dir):
    return json.loads((fixtures_dir / "scene_spec_min.json").read_text())


def _with(data, **kw):
    d = copy.deepcopy(data)
    d["components"][0].update(kw)
    return d


def test_bevel_accepts_a_boolean_in_both_validators(data):
    for value in (False, True, None, 0.002, {"width": 0.002}):
        d = _with(data, bevel=value)
        assert S.validate_scene_spec(d) == [] and S.schema_errors_builtin(d, S.SCENE_SCHEMA_NAME) == [], value
    assert S.validate_scene_spec(_with(data, bevel="wide"))
    assert S.parse_scene_spec(_with(data, bevel=False)).get_component("top").bevel is False


def test_new_types_and_true_size_torus(data):
    data["components"] += [
        {"id": "lid", "type": "arch", "size": [0.8, 0.4, 0.2], "params": {"axis": "X", "thickness": 0.01, "bulge": "+Z"}},
        {"id": "lid2", "type": "half_cylinder", "size": [0.8, 0.4, 0.2]},
        {"id": "ring", "type": "torus", "params": {"major_radius": [0.06, 0.05], "minor_radius": 0.01, "axis": "X"}},
        {"id": "rb", "type": "rounded_box", "size": [1, 1, 1], "params": {"radius": [0.01, 0.02, 0.0]}}]
    assert S.validate_scene_spec(data) == [] and S.schema_errors_builtin(data, S.SCENE_SCHEMA_NAME) == []
    two = copy.deepcopy(data)
    two["components"][-1]["params"]["radius"] = [0.01, 0.02]
    assert S.validate_scene_spec(two)                                                           # a 3-vector or a number
    bad_bulge = copy.deepcopy(data)
    bad_bulge["components"][-4]["params"]["bulge"] = "Z"
    assert S.validate_scene_spec(bad_bulge)
    no_size = copy.deepcopy(data)
    no_size["components"][-4].pop("size")                                                       # an arch needs a size
    assert any("lid" in e for e in S.validate_scene_spec(no_size))


def test_mirror_center_is_parsed_validated_and_round_trips(data):
    sc = S.parse_scene_spec(_with(data, mirror_center=[0.0, 0.05, 0.0], mirror={"y": True}))
    assert sc.get_component("top").mirror_center == [0.0, 0.05, 0.0]
    dumped = sc.to_dict()
    assert "mirror_center" not in dumped["components"][1]                                       # unset stays out of the dump
    assert S.parse_scene_spec(json.loads(json.dumps(dumped))).get_component("top").mirror_center == [0.0, 0.05, 0.0]
    assert S.parse_scene_spec(data).get_component("top").mirror_center is None
    with pytest.raises(Exception, match="mirror_center needs a mirror axis"):
        S.parse_scene_spec(_with(data, mirror_center=[0, 0, 0]))
    assert S.validate_scene_spec(_with(data, mirror_center=[0, 0], mirror={"x": True}))         # not a 3-vector


# --------------------------------------------------------------------------- texture policy
class _View:
    def __init__(self, roles=("color",), synthetic=False):
        self._roles, self.reference_quality = roles, {"synthetic": synthetic}

    def enabled_views(self, role=None):
        return [object()] if role is None or role in self._roles else []


def test_project_textures_modes():
    assert S.project_textures({}, _View()) is True                                  # auto + real references
    assert S.project_textures({"texture": "auto"}, _View(synthetic=True)) is False  # renders of the spec itself stay flat
    assert S.project_textures({"texture": "project"}, _View(synthetic=True)) is True
    assert S.project_textures({"texture": "flat"}, _View()) is False
    assert S.project_textures({}, _View(roles=("validate",))) is False              # no colour view: nothing to project
    assert S.project_textures({"texture": "project"}, None) is False


def test_cutters_are_never_textured_and_opt_outs_win():
    def comp(**kw):
        return S.Component(id="c", type=kw.pop("type", "box"), **kw)
    assert S.is_textured_component(comp(), project=True) and not S.is_textured_component(comp())
    cutter = comp(boolean={"target": "t", "operation": "DIFFERENCE"}, params={"color_projection": True})
    assert not S.is_textured_component(cutter, project=True) and not S.is_textured_component(cutter, character=True)
    assert not S.is_textured_component(comp(params={"color_projection": False}), project=True)
    assert S.is_textured_component(comp(params={"color_projection": True}))                  # per-component opt in
    assert S.is_textured_component(comp(type="cone"))                                         # thin parts: always
    assert not S.is_textured_component(comp(type="silhouette_hull"), project=True)
    flat = S.parse_scene_spec({"spec_version": "0.1", "name": "n", "style": {"texture": "flat"},
                               "components": [{"id": "a", "type": "box", "size": [1, 1, 1]}]})
    assert flat.style["texture"] == "flat" and S.parse_scene_spec(
        {"spec_version": "0.1", "name": "n", "components": []}).style["texture"] == "auto"


# --------------------------------------------------------------------------- width-based sizing
def _chest_masks():
    """Front 160x100 px: body 100 wide + 8 px ring handles on both sides (silhouette 116 wide, 70 tall);
    side: body 60 wide + a 15 px lock (silhouette 75 wide, same 70 px height)."""
    front = rect_mask(100, 160, 30, 20, 130, 90) | rect_mask(100, 160, 22, 40, 30, 60) | rect_mask(100, 160, 130, 40, 138, 60)
    side = rect_mask(100, 160, 50, 20, 110, 90) | rect_mask(100, 160, 110, 50, 125, 62)
    return {"front": front, "side": side}


AXES = {"front": "-Y", "side": "-X"}


def test_body_bbox_opens_protrusions_away():
    front = _chest_masks()["front"]
    box = mask_bbox(front)
    assert box[2] - box[0] == 116
    opened = body_bbox(front, box)
    assert opened[2] - opened[0] == 100


def test_height_from_width_depends_on_what_the_width_measures():
    masks = _chest_masks()
    h_sil, note, alt = derive_height(masks, AXES, width=1.16)                       # silhouette width 116 px -> 10 mm/px
    assert h_sil == pytest.approx(0.70) and "silhouette" in note and "front" in note
    h_body, note_b, alt_b = derive_height(masks, AXES, width=1.0, size_of="body")   # the same scale, 100 px of body
    assert h_body == pytest.approx(0.70) and "body" in note_b and alt_b["silhouette"] == pytest.approx(0.6034, abs=1e-3)
    assert alt["body"] == pytest.approx(1.16 / 100 * 70)                            # 1.16 m read as the body width
    h_depth, note_d, _ = derive_height(masks, AXES, depth=0.75)                     # side silhouette: 75 px incl. the lock
    assert h_depth == pytest.approx(0.70) and "side" in note_d
    with pytest.raises(BBRError):
        derive_height(masks, AXES)
    with pytest.raises(BBRError, match="size_of"):
        derive_height(masks, AXES, width=1.0, size_of="hull")


def test_width_only_known_dimension_calibrates_like_the_derived_height():
    masks = _chest_masks()
    vs = S.parse_view_spec({"spec_version": "0.1", "known_dimensions": {"width": 1.16},
                            "views": {"front": {"path": "f.png"}, "side": {"path": "s.png", "camera_axis": "-X"}}})
    calibs, info = calibrate_views(vs, masks)
    assert info["dimensions"]["width"] == pytest.approx(1.16) and info["provenance"]["height"] == "derived:front"
    assert info["dimensions"]["height"] == pytest.approx(0.70) and calibs["front"].m_per_px == pytest.approx(0.01)
    assert calibs["side"].m_per_px == pytest.approx(0.01)                           # the side view shares the scale via the height
    assert info["dimensions"]["depth"] == pytest.approx(0.75)
    vs.dimension_provenance = {"width": "typed by hand"}
    assert calibrate_views(vs, masks)[1]["provenance"]["width"] == "known (typed by hand)"


def test_draft_view_spec_cli_width_and_provenance(tmp_path, capsys):
    import prepare_references as cli
    from bbr.core import imageio
    folder = tmp_path / "chest" / "references"
    folder.mkdir(parents=True)
    masks = _chest_masks()
    for key, m in (("front", masks["front"]), ("side", masks["side"]), ("back", masks["front"])):
        imageio.save_rgba(folder / f"{key}.png", paint(m, (0.5, 0.3, 0.1), bg=(0.5, 0.5, 0.5)))
    assert cli.main(["draft-view-spec", "--dir", str(folder)]) == 1                           # no size at all
    assert not (folder / "view_spec.json").exists()                                             # and nothing is written
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--width", "1.16"]) == 0
    spec = json.loads((folder / "view_spec.json").read_text())
    assert spec["known_dimensions"] == {"width": 1.16, "depth": None, "height": pytest.approx(0.70, abs=0.01)}
    assert "silhouette" in spec["dimension_provenance"]["height"] and "--size-of body" in capsys.readouterr().out
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--width", "1.0", "--size-of", "body"]) == 0
    spec = json.loads((folder / "view_spec.json").read_text())
    assert spec["known_dimensions"]["width"] is None and "body" in spec["dimension_provenance"]["height"]   # not recorded as known
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "0.7"]) == 0
    assert "dimension_provenance" not in json.loads((folder / "view_spec.json").read_text())


def test_texture_size_follows_surface_over_reference_resolution():
    from bbr.core.texelproj import suggest_texture_size as size
    assert size(0.0004, 0.00126) == 128                       # a 4 cm bracket: the minimum
    assert size(1.72, 0.00126) == 1024                        # a chest body (1.08 M reference px of surface): the maximum
    assert size(0.05, 0.00126) == 256                         # 31 k px of surface -> 177 px side -> 256
    assert size(0, 0.001) == 128 and size(1.0, 0) == 128


def test_per_corner_normals_keep_a_flat_face_flat():
    from bbr.core.uvraster import rasterize_uv_mesh
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    uvs = np.array([[[0.0, 0], [1, 0], [0, 1]]])
    tilted = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])                       # smooth vertex normals averaged with a neighbour
    smooth = rasterize_uv_mesh(verts, np.array([[0, 1, 2]]), uvs, tilted, 16)
    flat = rasterize_uv_mesh(verts, np.array([[0, 1, 2]]), uvs, np.tile([0.0, 0, 1], (1, 3, 1)), 16)
    inside = flat["mask"]
    assert np.allclose(flat["normal"][inside], [0, 0, 1])
    assert not np.allclose(smooth["normal"][inside], [0, 0, 1], atol=0.05)
