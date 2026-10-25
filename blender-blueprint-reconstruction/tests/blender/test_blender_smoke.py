"""Blender smoke test. Run:  blender -b --factory-startup -P tests/blender/test_blender_smoke.py
Builds the simple table into an empty scene twice, checks for duplicates, renders one mask, calls the API.
Exits 0 on success, 1 on failure."""
import sys
import traceback
from pathlib import Path

SK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SK / "scripts"))

import bpy  # noqa: E402


def agent_id_counts() -> dict:
    counts: dict = {}
    for obj in bpy.data.objects:
        if "agent_id" in obj.keys():
            counts[obj["agent_id"]] = counts.get(obj["agent_id"], 0) + 1
    return counts


def check_hull(api, tmp: Path) -> None:
    """Silhouette hull (small settings): cleanup + retopo + UV + texture; second build is idempotent; vertex mode works."""
    import json
    import bmesh
    ex = SK / "examples" / "robot_character"
    spec = json.loads((ex / "scene_spec.json").read_text(encoding="utf-8"))
    params = spec["components"][0]["params"]
    params.update(resolution=96, texture_size=512, target_faces=3000)
    path = tmp / "hull_scene_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    vs = str(ex / "view_spec.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for _ in range(2):
        res = api.build_from_scene_spec(str(path), vs)
        assert res["ok"], res
    obj = bpy.data.objects["robot_character:body"] if "robot_character:body" in bpy.data.objects else next(
        o for o in bpy.data.objects if o.get("agent_id") == "robot_character:body")
    me = obj.data
    assert sum(1 for o in bpy.data.objects if o.get("agent_id") == "robot_character:body") == 1
    assert len(me.uv_layers) == 1 and len(me.polygons) <= 3300, (len(me.uv_layers), len(me.polygons))
    assert [i.name for i in bpy.data.images if i.name.endswith("_basecolor")] == ["robot_character_body_basecolor"]
    assert me.materials[0].name == "M_ref_texture" and me.materials[0].node_tree.nodes["bbr_basecolor"].image.size[0] == 512
    bm = bmesh.new()
    bm.from_mesh(me)
    assert not any(not e.is_manifold for e in bm.edges), "hull is not watertight"
    bm.free()
    assert not any(o.get("agent_id", "").startswith("bbr:preview_light_") for o in bpy.data.objects)
    print("HULL OK", len(obj.data.polygons), "faces")


def check_parts(api, tmp: Path) -> None:
    """Part primitives: closed meshes, idempotent; hull exclusion + textured parts (small settings); measure helper."""
    import json
    import bmesh
    parts = {"spec_version": "0.1", "name": "parts", "components": [
        {"id": "horn", "type": "cone", "location": [0, 0, 0.1], "params": {"base_radius": 0.02, "tip_radius": 0.003,
         "length": 0.15, "axis": [0.5, 0, 1], "bend": 80, "bend_dir": [-1, 0, 0]}},
        {"id": "sword", "type": "blade", "location": [0.1, 0, 0.3], "params": {"length": 0.3, "width": 0.04,
         "thickness": 0.01, "tip_length": 0.05, "axis": [0, 0, -1]}},
        {"id": "shield", "type": "disc", "location": [0, 0.1, 0.3], "params": {"radius": 0.1, "thickness": 0.02,
         "dome_height": 0.03, "dome_radius": 0.04, "axis": [0, 1, 0]}},
        {"id": "brim", "type": "disc", "location": [0, 0, 0.5], "params": {"radius": 0.12, "thickness": 0.006, "radius_ratio": 0.7}},
        {"id": "crystal", "type": "gem", "location": [0.2, 0, 0.4], "size": [0.04, 0.04, 0.09], "params": {"facets": 6}},
        {"id": "lock", "type": "strand", "location": [-0.2, 0, 0.5], "params": {
            "points": [[0, 0, 0], [0.03, 0, -0.1], [-0.02, 0, -0.2], [0.02, 0, -0.3]], "radii": [0.03, 0.025, 0.02, 0.008], "flatten": 0.7}}]}
    path = tmp / "parts_spec.json"
    path.write_text(json.dumps(parts), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for _ in range(2):
        res = api.build_from_scene_spec(str(path))
        assert res["ok"], res
    for cid in ("horn", "sword", "shield", "brim", "crystal", "lock"):
        objs = [o for o in bpy.data.objects if o.get("agent_id") == f"parts:{cid}"]
        assert len(objs) == 1, (cid, len(objs))
        bm = bmesh.new()
        bm.from_mesh(objs[0].data)
        assert len(bm.faces) >= 9 and not any(not e.is_manifold for e in bm.edges), f"{cid} is not closed"
        bm.free()
        assert tuple(objs[0].scale) == (1.0, 1.0, 1.0)
    lock = next(o for o in bpy.data.objects if o.get("agent_id") == "parts:lock")           # partfit's strand rasterizer = the Blender mesh
    import numpy as np
    from bbr.core import partfit
    from bbr.core.calibration import ViewCalibration
    cal = ViewCalibration("front", "-Y", (400, 400), 0.002, (200.0, 380.0))
    bpy.context.view_layer.update()
    world = np.array([tuple(lock.matrix_world @ v.co) for v in lock.data.vertices])
    tris = np.array([[p.vertices[0], p.vertices[i], p.vertices[i + 1]] for p in lock.data.polygons for i in range(1, len(p.vertices) - 1)])
    spec_pose = partfit.Pose([-0.2, 0, 0.5], [0, 0, 0], None, {"points": [[0, 0, 0], [0.03, 0, -0.1], [-0.02, 0, -0.2], [0.02, 0, -0.3]],
                                                               "radii": [0.03, 0.025, 0.02, 0.008], "flatten": 0.7})
    a, b = partfit.render(partfit.shape_of("strand", spec_pose), cal), partfit.render(("mesh", world, tris), cal)
    assert partfit.iou(a, b) > 0.995
    brim = next(o for o in bpy.data.objects if o.get("agent_id") == "parts:brim")
    assert abs(brim.dimensions.y / brim.dimensions.x - 0.7) < 0.01, tuple(brim.dimensions)          # elliptical brim
    gem = next(o for o in bpy.data.objects if o.get("agent_id") == "parts:crystal")
    assert len(gem.data.polygons) == 12 and abs(gem.dimensions.z - 0.09) < 1e-4, (len(gem.data.polygons), tuple(gem.dimensions))
    assert not any(poly.use_smooth for poly in gem.data.polygons) and not gem.modifiers                # flat shaded, no auto bevel
    bad = dict(parts, components=[{"id": "horn", "type": "cone", "params": {"length": 0.1}}])
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert api.build_from_scene_spec(str(path))["ok"] is False                       # missing base_radius: reported, not raised

    ex = SK / "examples" / "robot_character"
    spec = json.loads((ex / "scene_spec.json").read_text(encoding="utf-8"))
    spec["components"][0]["params"].update(resolution=96, texture_size=512, target_faces=3000, exclude_components=["horn"])
    spec["components"].append({"id": "horn", "type": "cone", "stage": "secondary", "location": [0.0, 0.0, 0.6],
                               "params": {"base_radius": 0.03, "tip_radius": 0.0, "length": 0.2}})
    path = tmp / "hull_parts_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    vs = str(ex / "view_spec.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for _ in range(2):
        res = api.build_from_scene_spec(str(path), vs)
        assert res["ok"], res
    names = sorted(i.name for i in bpy.data.images if i.name.endswith("_basecolor"))
    assert names == ["robot_character_body_basecolor", "robot_character_horn_basecolor"], names
    horn = next(o for o in bpy.data.objects if o.get("agent_id") == "robot_character:horn")
    body = next(o for o in bpy.data.objects if o.get("agent_id") == "robot_character:body")
    assert horn.data.materials[0].name == "M_ref_texture:horn" and body.data.materials[0].name == "M_ref_texture"
    assert horn.data.materials[0].node_tree.nodes["bbr_basecolor"].image.name == "robot_character_horn_basecolor"
    assert len(horn.data.uv_layers) == 1 and len(body.data.uv_layers) == 1
    spec["components"][0]["params"]["exclude_components"] = ["nope"]
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert api.build_from_scene_spec(str(path), vs)["ok"] is False
    m = api.measure_reference(vs, "front", points=[[10, 20]], bbox=[10, 20, 50, 60])
    assert m["ok"] and m["measurement"]["plane"] == ["x", "z"] and m["measurement"]["bbox"]["size"]["z"] > 0, m
    assert api.measure_reference(vs, "nope", points=[[1, 1]])["ok"] is False
    sug = api.suggest_parts(vs, out_dir=str(tmp / "suggest"))
    assert sug["ok"] and isinstance(sug["suggestions"], list) and (tmp / "suggest" / "suggestions.json").is_file(), sug
    assert api.suggest_parts(str(tmp / "no_such_view_spec.json"))["ok"] is False
    print("PARTS OK")


def check_measurements(api, tmp: Path) -> None:
    """Object-ID render, misattribution / occlusion on an embedded vs a flush box, held-out pose search."""
    import json
    import numpy as np
    from bbr.blender import attribution, projection, reconstruction as rc, render, scene, synthetic
    from bbr.core import imageio
    from bbr.core.spec import load_view_spec

    def box(cid, loc, size, mat):
        return {"id": cid, "type": "box", "location": loc, "size": size, "material": mat}
    mats = {k: {"base_color": c} for k, c in (("a", [0.9, 0.2, 0.2]), ("b", [0.2, 0.3, 0.9]), ("c", [0.1, 0.7, 0.2]))}
    spec = {"spec_version": "0.1", "name": "meas", "materials": mats, "components": [
        box("body", [0, 0, 0.5], [1.0, 0.5, 1.0], "a"), box("embedded", [0, 0, 0.5], [0.3, 0.2, 0.3], "b"),
        box("flush", [0, 0.3, 0.5], [0.3, 0.1, 0.3], "c")]}
    (tmp / "meas_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert api.build_from_scene_spec(str(tmp / "meas_spec.json"))["ok"]
    views = {"front": {"path": "mf.png"}, "side": {"path": "ms.png", "camera_axis": "-X"}, "back": {"path": "mb.png"},
             "h": {"path": "h55.png", "camera_axis": {"azimuth": 45, "elevation": 0}}}
    (tmp / "meas_views.json").write_text(json.dumps({"spec_version": "0.1", "views": views}), encoding="utf-8")
    synthetic.make_synthetic_references(load_view_spec(tmp / "meas_views.json", check_files=False))
    centre = np.array([0.0, 0.0, 0.5])
    imageio.save_rgba(tmp / "h55.png", render.render_color_dir(55.0, 0.0, (400, 460), 0.004, centre, transparent=True))
    inp = rc.load_inputs(str(tmp / "meas_views.json"), str(tmp / "meas_spec.json"))
    objs = scene.model_meshes()
    names = [attribution.component_name(inp.scene, o) for o in objs]
    ids_front = render.render_object_ids("front", inp.calibs["front"], objs)
    ids_back = render.render_object_ids("back", inp.calibs["back"], objs)
    assert set(np.unique(ids_front)) == {0, names.index("body") + 1}, np.unique(ids_front)
    assert set(np.unique(ids_back)) == {0, names.index("body") + 1, names.index("flush") + 1}, np.unique(ids_back)
    assert all(tuple(o.color) == (1.0, 1.0, 1.0, 1.0) for o in objs)                 # object colours restored
    res = attribution.analyze(inp, rc.resolve_config(None, inp.scene.workflow))
    assert res["occluded_objects"] == ["embedded"], res["occluded_objects"]
    assert res["objects"]["embedded"]["occlusion_share"] > 0.9 and res["objects"]["flush"]["occlusion_share"] < 0.1
    assert res["objects"]["body"]["occlusion_share"] < 0.1 and res["objects"]["flush"]["best_view"] in ("side", "back")
    hold = projection.compare_holdout(inp.view_spec)
    h = hold["views"]["h"]
    assert abs(h["best_azimuth"] - 55.0) <= 5.0 and h["iou"] > 0.9 and h["iou"] > h["iou_nominal"] + 0.02, h
    assert h["color_error"] is not None and h["azimuth_searched"] and set(inp.masks) == {"front", "side", "back"}
    audit = api.audit_reconstruction(str(tmp / "meas_views.json"), str(tmp / "meas_spec.json"), str(tmp / "audit"))
    assert audit["ok"] and audit["attribution"]["occluded_objects"] == ["embedded"], audit
    assert audit["warnings"] == ["occluded_part:embedded"] and "h" in audit["holdout"]["views"], audit["warnings"]
    sheet = render.render_turntable(tmp / "turntable.png", n=4, tile=(96, 96))
    assert sheet.shape == (96, 4 * 96, 3) and (tmp / "turntable.png").is_file()
    print("MEASUREMENTS OK", {k: round(v["occlusion_share"], 3) for k, v in res["objects"].items()}, round(h["iou"], 3))


def check_textured_defaults() -> None:
    """Character scenes (silhouette_hull, or parts_labels with style.hard_surface false) texture every part by default; others
    (hard-surface, with or without labels) only cone / blade / disc unless the scene projects."""
    from bbr.blender import partstex
    from bbr.core.spec import parse_scene_spec

    def scene(extra=None, hull=True, labels=None, style=None):
        comps = [{"id": "body", "type": "silhouette_hull"}] if hull else []
        comps += [{"id": "arm", "type": "capsule", "size": [0.02, 0.02, 0.1]}, {"id": "ball", "type": "sphere", "size": [0.02] * 3},
                  {"id": "can", "type": "cylinder", "size": [0.02, 0.02, 0.05], "params": {"color_projection": False}},
                  {"id": "ear", "type": "cone", "params": {"anchors_px": {"front": {"base": [1, 2], "tip": [3, 4], "radius_px": [3, 1]}}}},
                  *(extra or [])]
        return parse_scene_spec({"spec_version": "0.1", "name": "s", "components": comps, **({"parts_labels": labels} if labels else {}), **({"style": style} if style else {})})

    ids = lambda sc: [c.id for c in partstex.textured_parts(sc)]
    assert ids(scene()) == ["arm", "ball", "ear"], ids(scene())                     # capsule, sphere: textured; cylinder opted out
    assert ids(scene(hull=False)) == ["ear"], ids(scene(hull=False))                # hard-surface scene: cone / blade / disc only
    assert ids(scene(hull=False, labels="labels.json", style={"hard_surface": False})) == ["arm", "ball", "ear"]   # labels + a non-hard-surface style: character
    assert ids(scene(hull=False, labels="labels.json")) == ["ear"]                  # labels in a hard-surface scene only gate the sampling
    assert not partstex.is_character_scene(scene(hull=False, labels="labels.json")) and partstex.is_character_scene(scene())
    print("TEXTURED DEFAULTS OK")


def check_stop_secondary(api, tmp) -> None:
    """``stop_after="secondary"`` builds the parts stage, writes its own preview, runs the part-gap check and stops."""
    from bbr.blender import reconstruction as rc
    assert rc.STOP_AFTER_CHOICES == ("blockout", "primary", "secondary", "final")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = api.run_reconstruction(str(SK / "examples" / "simple_table" / "view_spec.json"),
                                 str(SK / "examples" / "simple_table" / "scene_spec.json"), str(tmp / "secondary"), stop_after="secondary")
    assert out["ok"] and out["status"] == "stopped_after_secondary", out
    assert out["report"]["part_gaps"] == [] and Path(out["report"]["outputs"]["stage_secondary_preview"]).is_file(), out["report"]["outputs"]
    dim = out["report"]["dimension_check"]
    assert dim["passed"] and dim["tolerance"] == 0.03 and "relative_error" in dim, dim      # the parts-only check ran
    assert "Dimensions :" in __import__("bbr.core.summary", fromlist=["x"]).format_summary(out["report"])
    print("STOP SECONDARY OK")


def check_hard_kit(api, tmp: Path) -> None:
    """Hard-surface kit: arch / torus / per-axis rounded_box build closed, mirror_center pivots, the Mirror modifier runs before
    the Boolean (a cutter must be mirrored too), and a projecting scene textures every part except hidden cutters."""
    import json
    import bmesh
    import numpy as np
    from bbr.blender import hullpost, partstex, scene
    from bbr.core.spec import parse_scene_spec

    def evaluated(cid):
        obj = next(o for o in bpy.data.objects if o.get("agent_id") == f"kit:{cid}")
        return obj, hullpost.world_triangles([obj])[0]

    def box(cid, loc, size, **kw):
        return {"id": cid, "type": "box", "location": loc, "size": size, **kw}

    ex = SK / "examples" / "simple_table"
    comps = [
        {"id": "lid", "type": "arch", "location": [0, 0, 0.5], "size": [0.8, 0.4, 0.2], "params": {"axis": "X"}},
        {"id": "band", "type": "half_cylinder", "location": [0.2, 0, 0.5], "size": [0.1, 0.42, 0.22], "params": {"thickness": 0.01}},
        {"id": "ring", "type": "torus", "location": [0.5, 0, 0.3], "params": {"major_radius": [0.05, 0.04], "minor_radius": 0.008, "axis": "X"}},
        {"id": "rb", "type": "rounded_box", "location": [0, 0, 0.1], "size": [0.2, 0.2, 0.1], "params": {"radius": [0.0, 0.03, 0.01]}},
        box("off", [0.5, 0.3, 0.1], [0.05, 0.05, 0.05], mirror={"y": True}, mirror_center=[0, 0.2, 0]),
        box("slab", [0.3, 0.8, 0.5], [0.2, 0.2, 0.2], mirror={"x": True}),
        box("slot", [0.3, 0.8, 0.5], [0.1, 0.3, 0.1], boolean={"target": "slab", "operation": "DIFFERENCE"}),
        box("slab2", [0.3, 1.2, 0.5], [0.2, 0.2, 0.2], mirror={"x": True}),
        box("slot2", [0.3, 1.2, 0.5], [0.1, 0.3, 0.1], mirror={"x": True}, boolean={"target": "slab2", "operation": "DIFFERENCE"}),
    ]
    spec = {"spec_version": "0.1", "name": "kit", "style": {"texture": "project"}, "components": comps}
    path = tmp / "kit_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    vs = str(ex / "view_spec.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for _ in range(2):
        res = api.build_from_scene_spec(str(path), vs)
        assert res["ok"], res
    for cid in ("lid", "band", "ring", "rb"):
        bm = bmesh.new()
        bm.from_mesh(next(o for o in bpy.data.objects if o.get("agent_id") == f"kit:{cid}").data)
        assert not any(not e.is_manifold for e in bm.edges), f"{cid} is not closed"
        bm.free()
    obj, v = evaluated("lid")
    assert np.allclose(v.max(axis=0) - v.min(axis=0), (0.8, 0.4, 0.2), atol=5e-3), v.max(axis=0) - v.min(axis=0)   # the auto bevel trims the base corners
    _, ring = evaluated("ring")
    assert np.allclose(ring.max(axis=0) - ring.min(axis=0), (0.016, 0.1 + 0.016, 0.08 + 0.016), atol=2e-3)
    _, rb = evaluated("rb")
    assert np.allclose(rb.max(axis=0) - rb.min(axis=0), (0.2, 0.2, 0.1), atol=1e-6)
    _, off = evaluated("off")                                   # mirrored about y = 0.2: instances at y 0.3 and 0.1, not -0.3
    assert abs(off[:, 1].min() - 0.075) < 1e-4 and abs(off[:, 1].max() - 0.325) < 1e-4, (off[:, 1].min(), off[:, 1].max())
    pivot = scene.find_by_agent_id("bbr:pivot:kit:off")
    assert pivot is not None and abs(pivot.location.y - 0.2) < 1e-6 and scene.find_by_agent_id("bbr:pivot:kit:lid") is None
    # Mirror runs before Boolean: an unmirrored cutter only cuts the instance it overlaps; a mirrored one cuts both
    half = lambda v: ((v[:, 0] > 0).sum(), (v[:, 0] < 0).sum())
    pos, neg = half(evaluated("slab")[1])
    assert pos != neg, (pos, neg)
    pos2, neg2 = half(evaluated("slab2")[1])
    assert pos2 == neg2, (pos2, neg2)
    # texture projection: every part but the cutters, and the mirrored ones too
    sc = parse_scene_spec(spec)
    assert [c.id for c in partstex.textured_parts(sc, True)] == [c["id"] for c in comps if "boolean" not in c]
    names = sorted(i.name[len("kit_"):-len("_basecolor")] for i in bpy.data.images if i.name.endswith("_basecolor"))
    assert names == sorted(c["id"] for c in comps if "boolean" not in c), names
    assert not any(o.get("bbr_cutter") for o in scene.model_meshes()) and len(scene.model_meshes(True)) - len(scene.model_meshes()) == 2
    assert next(o for o in bpy.data.objects if o.get("agent_id") == "kit:slab2").data.materials[0].name == "M_ref_texture:slab2"
    spec["style"]["texture"] = "flat"
    path.write_text(json.dumps(spec), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert api.build_from_scene_spec(str(path), vs)["ok"] and not [i for i in bpy.data.images if i.name.endswith("_basecolor")]
    print("HARD KIT OK")


def check_lathe_attach(api, tmp: Path) -> None:
    """Lathe (explicit profile and measured profile_from), surface attach on a lathe, and smooth-by-angle normals without bevel."""
    import json
    import numpy as np
    from bbr.blender import hullpost
    ex = SK / "examples" / "simple_table"
    comps = [
        {"id": "col", "type": "lathe", "location": [0.1, 0.2, 0.0], "params": {"profile": [[0, 0.2], [0.1, 0.2], [0.8, 0.15]], "depth_ratio": 0.8, "segments": 48}, "bevel": False},
        {"id": "plaque", "type": "box", "size": [0.08, 0.02, 0.1], "stage": "secondary", "bevel": False,
         "params": {"attach_to": "col", "attach_at": {"azimuth_deg": 90, "height": 0.4}, "standoff": 0.002}},
        {"id": "whole", "type": "lathe", "params": {"profile_from": {"view": "front", "bbox_px": [0, 0, 100000, 100000]}}}]
    spec = {"spec_version": "0.1", "name": "lath", "components": comps}
    path = tmp / "lathe_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    vs = str(ex / "view_spec.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for _ in range(2):
        res = api.build_from_scene_spec(str(path), vs)
        assert res["ok"], res
    obj = next(o for o in bpy.data.objects if o.get("agent_id") == "lath:col")
    v = hullpost.world_triangles([obj])[0]
    assert np.allclose(v.min(axis=0), (0.1 - 0.2, 0.2 - 0.16, 0.0), atol=1e-3) and abs(v[:, 2].max() - 0.8) < 1e-6, (v.min(axis=0), v.max(axis=0))
    sharp = obj.data.attributes.get("sharp_edge")
    assert sharp is not None and sum(d.value for d in sharp.data) >= 2 * 48, "cap rims must be sharp edges (no cap normal over the wall)"
    plaque = next(o for o in bpy.data.objects if o.get("agent_id") == "lath:plaque")
    import math
    tilt = math.atan2(0.05, 0.7)                    # the column narrows: its normal leans up by this angle
    off = 0.002 + 0.01                              # standoff + half the 2 cm thickness
    r = 0.2 - 0.05 * (0.4 - 0.1) / 0.7
    assert abs(plaque.location.x - (0.1 + r + off * math.cos(tilt))) < 1e-3 and abs(plaque.location.y - 0.2) < 1e-3, tuple(plaque.location)
    assert abs(plaque.location.z - (0.4 + off * math.sin(tilt))) < 1e-3, tuple(plaque.location)
    assert abs(np.degrees(plaque.rotation_euler.z) - 90.0) < 1.0 and tuple(plaque.scale) == (1.0, 1.0, 1.0), tuple(plaque.rotation_euler)
    whole = next(o for o in bpy.data.objects if o.get("agent_id") == "lath:whole")
    assert abs(hullpost.world_triangles([whole])[0][:, 2].max() - 0.72) < 0.01, "profile_from must follow the reference silhouette height"
    print("LATHE ATTACH OK")


def check_plate(api, tmp: Path) -> None:
    """A contour plate (wing): built from its solved outline; a plate with a size and no anchors is still the hard-surface box."""
    import json
    from bbr.blender import hullpost
    spec = {"spec_version": "0.1", "name": "pl", "category": "other", "dimensions": {"width": 1.0, "depth": 1.0, "height": 1.0},
            "components": [
                {"id": "wing", "type": "plate", "stage": "primary", "params": {
                    "outline": [[0.0, -0.03], [0.15, 0.0], [0.0, 0.03]], "outline_tris": [[0, 1, 2]], "thickness": 0.004,
                    "axis": [1, 0, 0], "normal": [0, 0, 1], "bend": 30}, "location": [0.1, 0.0, 0.5]},
                {"id": "slab", "type": "plate", "stage": "primary", "size": [0.2, 0.1, 0.02], "location": [0.0, 0.0, 0.1]}]}
    path = tmp / "plate_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = api.build_from_scene_spec(str(path))
    assert res["ok"], res
    wing = next(o for o in bpy.data.objects if o.get("agent_id") == "pl:wing")
    v = hullpost.world_triangles([wing])[0]
    assert abs(v[:, 0].min() - 0.1) < 1e-3 and v[:, 2].max() > 0.5 + 0.01, (v.min(axis=0), v.max(axis=0))   # the bend lifts the tip
    slab = next(o for o in bpy.data.objects if o.get("agent_id") == "pl:slab")
    assert len(slab.data.vertices) == 8, "a plate with a size and no anchors is the box"
    print("PLATE OK")


def check_materials(api, tmp: Path) -> None:
    """Material classes (WI-1): a class reaches the Principled values of a projected hull (stylized column) and a flat part
    (physical column), the projected material keeps its name and active node, the colour metric render ignores the class,
    and the lit preview restores the scene's world without leaving the temporary one behind."""
    import json
    import numpy as np
    from bbr.blender import material, render, synthetic
    from bbr.core import pbr

    ex = SK / "examples" / "robot_character"
    vs = str(ex / "view_spec.json")
    spec = json.loads((ex / "scene_spec.json").read_text(encoding="utf-8"))
    spec["components"][0]["params"].update(resolution=96, texture_size=512, target_faces=3000)
    spec.setdefault("materials", {})["gilded"] = {"class": "gold", "roughness": 0.2}
    spec["components"].append({"id": "plinth", "type": "box", "location": [0.0, 0.0, -0.02], "size": [0.05, 0.05, 0.02],
                               "material": "gilded", "params": {"color_projection": False}})

    def build(hull_class):
        if hull_class:
            spec["components"][0]["material"] = hull_class
        else:
            spec["components"][0].pop("material", None)
        path = tmp / "materials_spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        res = api.build_from_scene_spec(str(path), vs)
        assert res["ok"], res
        dims, _ = synthetic.model_dimensions()
        return render.render_color("front", synthetic.synthetic_calibration("front", "-Y", dims))

    def principled(obj):
        bsdf = next(n for n in obj.data.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        return {k: bsdf.inputs[s].default_value for k, s in material.PBR_SOCKETS.items()}

    plain = build(None)
    classed = build("metal_steel")
    assert np.array_equal(plain, classed), "the colour metric render must not depend on the material class"
    body = next(o for o in bpy.data.objects if o.get("agent_id") == "robot_character:body")
    mat = body.data.materials[0]
    assert mat.name.startswith(material.REF_TEXTURE_MATERIAL) and mat.node_tree.nodes.active.name == material.REF_TEXTURE_NODE
    for key, val in pbr.resolve("metal_steel", "stylized").items():
        assert abs(principled(body)[key] - val) < 1e-6, (key, principled(body)[key], val)
    plinth = next(o for o in bpy.data.objects if o.get("agent_id") == "robot_character:plinth")
    want = pbr.resolve({"class": "gold", "roughness": 0.2}, "physical")
    assert want == {"metallic": 1.0, "roughness": 0.2}
    for key, val in want.items():
        assert abs(principled(plinth)[key] - val) < 1e-6, (key, principled(plinth)[key], val)

    sc = bpy.context.scene
    old_world, worlds = sc.world, {w.name for w in bpy.data.worlds}
    render.render_preview(tmp / "materials_preview.png", size=(96, 96))
    assert sc.world == old_world and {w.name for w in bpy.data.worlds} == worlds, ({w.name for w in bpy.data.worlds}, worlds)
    assert render.PREVIEW_WORLD not in bpy.data.worlds
    print("MATERIALS OK")


def check_glass_and_relief(api, tmp: Path) -> None:
    """Detail pass on sd_astronaut (small settings): the visor is a closed see-through shell, left out of the colour render
    and the opaque mesh list; the face layer carries the profile relief and a 2048 texture by default."""
    import json
    import bmesh
    import numpy as np
    from bbr.blender import hull, render, scene, synthetic

    ex = SK / "examples" / "sd_astronaut"
    spec = json.loads((ex / "scene_spec.json").read_text(encoding="utf-8"))
    spec["components"][0]["params"].update(resolution=96, texture_size=512, target_faces=3000)
    spec["parts_labels"] = str(ex / "labels.json")
    path = tmp / "astronaut_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = api.build_from_scene_spec(str(path), str(ex / "view_spec.json"))
    assert res["ok"], res
    visor = scene.find_by_agent_id("sd_astronaut:visor")
    face = scene.find_by_agent_id("sd_astronaut:face")
    assert visor.get(scene.PROP_TRANSPARENT) == 1 and not face.get(scene.PROP_TRANSPARENT)
    assert visor not in scene.model_meshes(opaque=True) and visor in scene.model_meshes()
    bsdf = next(n for n in visor.data.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    assert bsdf.inputs["Transmission Weight"].default_value == 1.0
    bm = bmesh.new()
    bm.from_mesh(visor.data)
    assert not any(not e.is_manifold for e in bm.edges), "the visor shell is not closed"
    bm.free()
    assert face.data.materials[0].node_tree.nodes["bbr_basecolor"].image.size[0] == 2048
    hand = scene.find_by_agent_id("sd_astronaut:hand_r")
    assert hand is not None and len(hand.data.polygons) > 400, "the glove (capsule with fingers) was not built"
    dims, _ = synthetic.model_dimensions()
    cal = synthetic.synthetic_calibration("front", "-Y", dims)
    with_glass = render.render_color("front", cal)
    visor.hide_render = True
    assert np.array_equal(with_glass, render.render_color("front", cal)), "the glass must not reach the colour render"
    visor.hide_render = False
    print("GLASS RELIEF OK")


def check_update_component(api, tmp: Path) -> None:
    """update_component: merge + rebuild of one object; a failing rebuild restores the spec file."""
    import json
    from bbr.blender import reconstruction as rc
    spec = {"spec_version": "0.1", "name": "upd", "components": [
        {"id": "a", "type": "box", "location": [0, 0, 0.5], "size": [1.0, 1.0, 1.0]},
        {"id": "b", "type": "box", "location": [2, 0, 0.5], "size": [1.0, 1.0, 1.0]}]}
    path = tmp / "upd_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert api.build_from_scene_spec(str(path))["ok"]
    res = api.update_component(str(path), "a", {"size": [2.0, 1.0, 1.0]})
    assert res["ok"], res
    a = next(o for o in bpy.data.objects if o.get("agent_id") == "upd:a")
    b = next(o for o in bpy.data.objects if o.get("agent_id") == "upd:b")
    assert abs(a.dimensions.x - 2.0) < 1e-4 and abs(b.dimensions.x - 1.0) < 1e-4, (tuple(a.dimensions), tuple(b.dimensions))
    assert sum(1 for o in bpy.data.objects if o.get("agent_id") == "upd:a") == 1
    assert json.loads(path.read_text(encoding="utf-8"))["components"][0]["size"] == [2.0, 1.0, 1.0]
    before = path.read_bytes()
    orig = rc.texture_parts

    def boom(*args, **kwargs):
        raise RuntimeError("forced rebuild failure")
    rc.texture_parts = boom
    try:
        failed = api.update_component(str(path), "a", {"size": [3.0, 1.0, 1.0]})
    finally:
        rc.texture_parts = orig
    assert failed["ok"] is False, failed
    assert path.read_bytes() == before, "spec file not restored after a failed rebuild"
    assert api.update_component(str(path), "a", {"size": [-1, 1, 1]})["ok"] is False and path.read_bytes() == before
    print("UPDATE OK")


def main() -> None:
    from bbr import api
    from bbr.blender import render, scene, synthetic

    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = str(SK / "examples" / "simple_table" / "scene_spec.json")

    first = api.build_from_scene_spec(spec)
    assert first["ok"], first
    counts1 = agent_id_counts()
    second = api.build_from_scene_spec(spec)
    assert second["ok"], second
    counts2 = agent_id_counts()
    assert counts1 == counts2, "second build changed the object set"
    assert all(v == 1 for v in counts2.values()), f"duplicate agent_ids: {counts2}"
    assert "simple_table:top" in counts2 and "simple_table:leg_fl" in counts2

    dims, _ = synthetic.model_dimensions()
    calib = synthetic.synthetic_calibration("front", "-Y", dims)
    mask = render.render_mask("front", calib)
    assert mask.shape == (calib.image_size[1], calib.image_size[0]) and mask.any(), "empty mask"
    assert 0.02 < mask.mean() < 0.5, f"implausible mask coverage {mask.mean():.3f}"

    info = api.inspect_scene()
    assert info["ok"] and len(info["scene"]["objects"]) >= 3, info
    assert not scene.duplicate_agent_ids()
    bad = api.build_from_scene_spec(str(SK / "does_not_exist.json"))
    assert bad["ok"] is False and "error" in bad
    vs = str(SK / "examples" / "simple_table" / "view_spec.json")
    tmp = Path(__import__("tempfile").mkdtemp(prefix="bbr_smoke_"))
    cur = api.render_reference_views(vs, str(tmp))  # scene already built: rendered as is
    assert cur["ok"] and cur["source"] == "current_scene" and cur["renders"], cur
    val = api.validate_reconstruction(vs, spec, str(tmp))
    assert val["ok"] and "diagnosis" in val and "warnings" in val and "recommendation" in val, val
    view = val["report"]["reference_match"]["per_view"]["front"]
    assert {"bbox_delta_m", "largest_mismatch_region", "components_in_region"} <= set(view), view
    bpy.ops.wm.read_factory_settings(use_empty=True)
    empty = api.render_reference_views(vs, str(tmp), None)
    assert empty["ok"] and empty["source"] == "sibling_scene_spec", empty
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import json
    lone = json.loads(Path(vs).read_text(encoding="utf-8"))  # same views, but no scene_spec.json beside it
    for entry in lone["views"].values():
        entry["path"] = str((Path(vs).parent / entry["path"]).resolve())
    (tmp / "view_spec.json").write_text(json.dumps(lone), encoding="utf-8")
    none = api.render_reference_views(str(tmp / "view_spec.json"), str(tmp))
    assert none["ok"] is False and "nothing to render" in none["error"], none
    check_textured_defaults()
    check_hull(api, tmp)
    check_parts(api, tmp)
    check_measurements(api, tmp)
    check_stop_secondary(api, tmp)
    check_hard_kit(api, tmp)
    check_lathe_attach(api, tmp)
    check_plate(api, tmp)
    check_materials(api, tmp)
    check_glass_and_relief(api, tmp)
    check_update_component(api, tmp)
    print("SMOKE OK", {k: v for k, v in counts2.items()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
