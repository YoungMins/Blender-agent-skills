"""Build the synthetic ground-truth benchmark cases.   Run inside Blender:

    blender -b --factory-startup -P bench/make_synthetic.py -- [--out bench/cases] [--cases chibi,creature,hatted]
                                                              [--px-per-m 500] [--no-variants]

For every case this writes ``<out>/<case>/``:
  gt_scene_spec.json   the ground-truth SceneSpec (primitives only, recentred: bbox centre x = y = 0, soles at z = 0)
  gt.obj, gt.blend     the ground-truth mesh (world space, evaluated) and scene
  references/          orthographic front / side / back renders (the training views)
  holdout/az_NNN.png   transparent renders at 8 azimuths with ONE shared camera (held out: never given to the pipeline,
                       except az_045 / az_225 which the view spec lists with the ``holdout`` role)
  view_spec.json       front / side / back + two holdout views, known height
  recon_scene_spec.json  default reconstruction spec: a silhouette hull only (the hull-only baseline)
  case.json            what ``score.py`` / ``run_bench.py`` need (GT files, held-out camera, perturbation)
and two perturbed variants ``<case>__scale2`` (side view 2 % too short, so the derived depth drifts 2 %) and
``<case>__noacc`` (the accessory is missing from the side view), which reuse the base case's ground truth.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SK / "scripts"))

import bpy  # noqa: E402

from bbr.blender import checkpoint, reconstruction as rc, synthetic  # noqa: E402
from bbr.core import imageio, imageops  # noqa: E402
from bbr.core.spec import load_view_spec, parse_scene_spec  # noqa: E402

HOLDOUT_IN_VIEW_SPEC = (45, 225)
RECON_RESOLUTION = 224
RECON_TEXTURE_SIZE = 1024
SCALE_DRIFT = 0.02


def comp(cid, ctype, loc, size=None, material=None, rot=None, stage="primary", **params):
    c = {"id": cid, "type": ctype, "stage": stage, "confidence": "observed", "location": list(loc)}
    if size is not None:
        c["size"] = list(size)
    if rot is not None:
        c["rotation_deg"] = list(rot)
    if material:
        c["material"] = material
    if params:
        c["params"] = params
    return c


def mats(**colors):
    return {k: {"base_color": list(v), "roughness": 0.6} for k, v in colors.items()}


def chibi():
    """Big sphere head with two horns, capsule torso / arms / legs, boxy feet."""
    cs = [comp("head", "sphere", (0, 0, 0.36), (0.26, 0.24, 0.26), "skin"),
          comp("torso", "capsule", (0, 0, 0.17), (0.17, 0.13, 0.22), "cloth"),
          comp("hair", "sphere", (0, 0.03, 0.40), (0.24, 0.20, 0.20), "dark")]
    for s, x in (("l", -1), ("r", 1)):
        cs += [comp(f"horn_{s}", "cone", (x * 0.075, 0.0, 0.46), None, "horn", stage="secondary", base_radius=0.024,
                    tip_radius=0.004, length=0.11, axis=[x * 0.45, 0, 1]),
               comp(f"arm_{s}", "capsule", (x * 0.125, 0, 0.19), (0.05, 0.05, 0.16), "skin", rot=(0, -x * 15, 0)),
               comp(f"leg_{s}", "capsule", (x * 0.045, 0, 0.065), (0.065, 0.075, 0.13), "dark"),
               comp(f"foot_{s}", "rounded_box", (x * 0.045, -0.02, 0.018), (0.075, 0.12, 0.036), "red", radius=0.012)]
    return cs, mats(skin=(0.95, 0.75, 0.6), cloth=(0.2, 0.4, 0.85), dark=(0.3, 0.2, 0.18), horn=(0.95, 0.8, 0.1),
                    red=(0.85, 0.2, 0.2)), ["horn_l", "horn_r"]


def creature():
    """Quadruped with a long curled tail, ears and a snout."""
    cs = [comp("body", "capsule", (0, 0, 0.17), (0.14, 0.34, 0.14), "fur", axis="Y"),
          comp("head", "sphere", (0, -0.2, 0.24), (0.13, 0.13, 0.13), "fur"),
          comp("snout", "rounded_box", (0, -0.275, 0.225), (0.06, 0.05, 0.05), "nose", radius=0.01),
          comp("tail_1", "capsule", (0, 0.25, 0.20), (0.05, 0.05, 0.14), "fur", rot=(-60, 0, 0)),
          comp("tail_2", "capsule", (0, 0.33, 0.30), (0.04, 0.04, 0.12), "stripe", rot=(-30, 0, 0)),
          comp("tail_3", "capsule", (0, 0.37, 0.40), (0.03, 0.03, 0.10), "fur", rot=(-10, 0, 0))]
    for s, x in (("l", -1), ("r", 1)):
        cs.append(comp(f"ear_{s}", "cone", (x * 0.04, -0.19, 0.29), None, "stripe", stage="secondary", base_radius=0.022,
                       tip_radius=0.003, length=0.07, axis=[x * 0.25, 0, 1]))
        for e, y in (("f", -0.11), ("b", 0.11)):
            cs.append(comp(f"leg_{e}{s}", "cylinder", (x * 0.05, y, 0.05), (0.045, 0.045, 0.1), "fur"))
    return cs, mats(fur=(0.75, 0.5, 0.25), stripe=(0.35, 0.22, 0.1), nose=(0.15, 0.1, 0.1)), ["tail_1", "tail_2", "tail_3"]


def hatted():
    """Robed figure under a wide-brim pointed hat, holding a tall staff with an orb."""
    cs = [comp("robe", "capsule", (0, 0, 0.2), (0.2, 0.16, 0.3), "robe"),
          comp("head", "sphere", (0, 0, 0.42), (0.15, 0.14, 0.15), "skin"),
          comp("brim", "cylinder", (0, 0, 0.47), (0.44, 0.44, 0.015), "hat"),
          comp("hat", "cone", (0, 0, 0.4775), None, "hat", stage="secondary", base_radius=0.1, tip_radius=0.01,
               length=0.2, axis=[0, 0, 1]),
          comp("staff", "cylinder", (0.24, -0.02, 0.30), (0.022, 0.022, 0.60), "wood", stage="secondary"),
          comp("orb", "sphere", (0.24, -0.02, 0.62), (0.07, 0.07, 0.07), "glow", stage="secondary")]
    for s, x in (("l", -1), ("r", 1)):
        cs += [comp(f"arm_{s}", "capsule", (x * 0.13, 0, 0.24), (0.05, 0.05, 0.2), "robe", rot=(0, -x * 10, 0)),
               comp(f"foot_{s}", "rounded_box", (x * 0.045, -0.015, 0.012), (0.07, 0.11, 0.024), "wood", radius=0.01)]
    return cs, mats(robe=(0.4, 0.2, 0.6), skin=(0.8, 0.55, 0.35), hat=(0.25, 0.15, 0.45), wood=(0.5, 0.33, 0.18),
                    glow=(0.3, 0.9, 0.9)), ["staff", "orb"]


CASES = {"chibi": chibi, "creature": creature, "hatted": hatted}


def scene_spec(name, components, materials):
    return {"spec_version": "0.1", "name": name, "category": "character", "object_type": "character", "units": "meters",
            "dimensions": {"width": None, "depth": None, "height": None}, "symmetry": {"x": False, "y": False, "z": False},
            "style": {"hard_surface": False, "bevel": False, "weighted_normals": False}, "materials": materials,
            "components": components}


def build(spec_dict):
    """Fresh scene with the SceneSpec built (primitives only)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = parse_scene_spec(spec_dict)
    inp = rc.Inputs(spec, spec.to_dict())
    rc.rebuild_all(inp, rc.resolve_config(None, spec.workflow))
    return inp


def recentre(spec_dict):
    """Shift the components so the bbox centre is at x = y = 0 and the soles at z = 0 (the pipeline's calibration does the same)."""
    build(spec_dict)
    dims, centre = synthetic.model_dimensions()
    box_min_z = centre[2] - dims["height"] / 2.0
    shift = np.array([-centre[0], -centre[1], -box_min_z])
    out = copy.deepcopy(spec_dict)
    for c in out["components"]:
        c["location"] = [round(float(v), 5) for v in np.asarray(c["location"]) + shift]
    return out


def recon_spec(name, height):
    return {"spec_version": "0.1", "name": name, "category": "character", "object_type": "character", "units": "meters",
            "dimensions": {"width": None, "depth": None, "height": round(height, 4)},
            "symmetry": {"x": False, "y": False, "z": False},
            "style": {"hard_surface": False, "bevel": False, "weighted_normals": False}, "materials": {},
            "workflow": {"max_iterations": 3, "error_threshold": 0.08, "blockout_threshold": 0.12, "auto_correct": True},
            "components": [{"id": "body", "type": "silhouette_hull", "stage": "primary", "confidence": "observed",
                            "location": [0, 0, 0],
                            "params": {"resolution": RECON_RESOLUTION, "carve_views": ["front", "side"],
                                       "cross_section": "superellipse", "exponent": 2.0,
                                       "color_views": ["front", "side", "back"], "smooth_iterations": 14,
                                       "color_projection": True, "texture_size": RECON_TEXTURE_SIZE}}]}


def view_spec(height, holdout_prefix="holdout"):
    views = {"front": {"path": "references/front.png"}, "side": {"path": "references/side.png", "camera_axis": "-X"},
             "back": {"path": "references/back.png"}}
    for az in HOLDOUT_IN_VIEW_SPEC:
        views[f"h{az}"] = {"path": f"{holdout_prefix}/az_{az:03d}.png", "camera_axis": {"azimuth": az, "elevation": 0},
                           "roles": ["holdout"]}
    return {"spec_version": "0.1", "views": views, "known_dimensions": {"height": round(height, 4)},
            "symmetry": {"x": False, "y": False, "z": False},
            "reference_quality": {"orthographic": True, "notes": "synthetic, rendered from the ground-truth scene"}}


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def squeeze_height(path, factor):
    """Scale the picture vertically by ``factor`` about the sole line (bottom of the figure); background fills the rest."""
    img = imageio.load_rgba(path)
    mask = imageops.extract_mask(img, "auto", None, 0.12)
    bottom = imageops.mask_bbox(mask)[3]
    bg = img[0, 0]
    out = np.tile(bg, (img.shape[0], img.shape[1], 1))
    ys = np.arange(img.shape[0])
    src = np.floor(bottom - (bottom - ys - 0.5) / factor).astype(int)
    ok = (src >= 0) & (src < img.shape[0])
    out[ys[ok]] = img[src[ok]]
    imageio.save_rgba(path, out)


def check_masks(d, dims, px):
    """The pipeline reads the silhouette from the picture: every part must contrast with the background, else the
    calibration (silhouette bbox vs the known height) silently drifts. Fail loudly instead."""
    for view, (w_dim, h_dim) in (("front", ("width", "height")), ("side", ("depth", "height")), ("back", ("width", "height"))):
        img = imageio.load_rgba(d / "references" / f"{view}.png")
        x0, y0, x1, y1 = imageops.mask_bbox(imageops.extract_mask(img, "auto", None, 0.12))
        got, want = (x1 - x0, y1 - y0), (dims[w_dim] * px, dims[h_dim] * px)
        if abs(got[0] - want[0]) > 2 or abs(got[1] - want[1]) > 2:
            raise SystemExit(f"{d.name}/{view}: silhouette bbox {got} px != expected {tuple(round(v) for v in want)}: "
                             "a material is too close to the background colour")


def check_area(d, px):
    """The front silhouette read from the picture must match the ground-truth alpha silhouette (same camera, scale ``px``):
    a part whose colour is close to the background is flood-filled away (a skin-coloured head between hat and shoulders)
    and would make every score judge the reference instead of the pipeline."""
    front = imageops.extract_mask(imageio.load_rgba(d / "references" / "front.png"), "auto", None, 0.12)
    gt = imageio.load_rgba(d / "holdout" / "az_000.png")[..., 3] >= 0.5
    got, want = front.sum() / px ** 2, gt.sum() / px ** 2
    if abs(got / want - 1.0) > 0.02:
        raise SystemExit(f"{d.name}: front reference silhouette area {got:.4f} m2 != ground truth {want:.4f} m2: "
                         "a material is too close to the background colour (it is lost from the mask)")


def make_case(name, builder, root, px, variants):
    components, materials, accessory = builder()
    gt = recentre(scene_spec(f"{name}_gt", components, materials))
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "gt_scene_spec.json", gt)
    build(gt)
    dims, centre = synthetic.model_dimensions()
    height = dims["height"]
    assert abs(centre[0]) < 1e-3 and abs(centre[1]) < 1e-3, (name, centre)
    write_json(d / "view_spec.json", view_spec(height))
    vs = load_view_spec(d / "view_spec.json", check_files=False)
    synthetic.make_synthetic_references(vs, px, dims)
    check_masks(d, dims, px)
    cam = synthetic.holdout_camera(px)
    synthetic.render_azimuths(cam, d / "holdout")
    check_area(d, px)
    mesh_info = synthetic.export_obj(d / "gt.obj")
    checkpoint.save_blend(d / "gt.blend")
    write_json(d / "recon_scene_spec.json", recon_spec(name, height))
    case = {"name": name, "kind": "synthetic", "view_spec": "view_spec.json", "scene_spec": "recon_scene_spec.json",
            "gt_obj": "gt.obj", "holdout_dir": "holdout", "holdout_camera": cam, "height_m": round(height, 5),
            "dimensions_m": {k: round(v, 5) for k, v in dims.items()}, "perturbation": None, "mesh": mesh_info,
            "accessory": accessory}
    write_json(d / "case.json", case)
    print(f"case {name}: height {height:.3f} m, {mesh_info['triangles']} triangles")
    if variants:
        make_variants(name, d, root, case, gt, components, materials, accessory, dims, px)


def make_variants(name, base_dir, root, case, gt, components, materials, accessory, dims, px):
    holdout_rel = f"../{name}/holdout"
    for suffix, label in (("scale2", f"side view {SCALE_DRIFT * 100:g} % too short (derived depth drifts)"),
                          ("noacc", f"{', '.join(accessory)} missing from the side view")):
        v = root / f"{name}__{suffix}"
        (v / "references").mkdir(parents=True, exist_ok=True)
        for view in ("front", "back", "side"):
            shutil.copy(base_dir / "references" / f"{view}.png", v / "references" / f"{view}.png")
        if suffix == "scale2":
            squeeze_height(v / "references" / "side.png", 1.0 - SCALE_DRIFT)
        else:
            stripped = copy.deepcopy(gt)
            stripped["components"] = [c for c in stripped["components"] if c["id"] not in accessory]
            build(stripped)
            side_only = {"spec_version": "0.1", "views": {"side": {"path": str(v / "references" / "side.png"),
                                                                  "camera_axis": "-X"}}}
            write_json(v / "_side_only.json", side_only)
            synthetic.make_synthetic_references(load_view_spec(v / "_side_only.json", check_files=False), px, dims)
            (v / "_side_only.json").unlink()
        vspec = view_spec(case["height_m"], holdout_rel)
        write_json(v / "view_spec.json", vspec)
        write_json(v / "recon_scene_spec.json", recon_spec(f"{name}_{suffix}", case["height_m"]))
        vcase = {**case, "name": f"{name}__{suffix}", "gt_obj": f"../{name}/gt.obj", "holdout_dir": holdout_rel,
                 "perturbation": label}
        write_json(v / "case.json", vcase)
        print(f"variant {name}__{suffix}: {label}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SK / "bench" / "cases"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--px-per-m", type=float, default=synthetic.SYNTHETIC_PX_PER_M)
    ap.add_argument("--no-variants", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.out)
    for name in args.cases.split(","):
        make_case(name, CASES[name], root, args.px_per_m, not args.no_variants)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
