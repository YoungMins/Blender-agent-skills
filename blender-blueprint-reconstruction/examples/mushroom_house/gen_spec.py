# Mushroom house, rebuilt with the new skill features: lathe (cap/body from labels, slab from a pixel box), surface attach
# (door, windows), labels (_chimney ignored) and color_views. Every number comes from `measure --points-file` (metres.json).
import json, sys
from pathlib import Path
here = Path(__file__).parent
M = json.loads((here / "metres.json").read_text())["views"]
F, S, B = M["front"], M["side"], M["back"]
C = []


def add(id, type, stage, size=None, mat="stone", params=None, loc=None, **kw):
    d = {"id": id, "type": type, "stage": stage, "confidence": kw.pop("confidence", "observed"), "material": mat}
    if loc: d["location"] = [round(v, 5) for v in loc]
    if size: d["size"] = [round(v, 5) for v in size]
    if params: d["params"] = params
    d.update(kw)
    C.append(d)


# --- three lathes: slab from a pixel box, body and cap from their labels (the chimney is labelled _chimney and left out)
add("base_slab", "lathe", "blockout", mat="stone", params={"profile_from": {"view": "front", "bbox_px": [60, 568, 625, 616], "bottom_z": 0}, "segments": 64})
add("body", "lathe", "blockout", mat="cream", params={"profile_from": {"part": "body"}, "segments": 64})
add("cap", "lathe", "blockout", mat="red", params={"profile_from": {"part": "cap"}, "segments": 64})


# --- chimney: x / width from the front, y / depth / z from the side
def chimney(name, mat="stone", **params):
    f, s = F[name], S[name]
    add(name, "box", "blockout", [f["size"]["x"], s["size"]["y"], s["size"]["z"]], mat,
        params or None, [(f["min"]["x"] + f["max"]["x"]) / 2, (s["min"]["y"] + s["max"]["y"]) / 2, (s["min"]["z"] + s["max"]["z"]) / 2])


chimney("chim_cap")
chimney("chim_shaft", color_views=["front", "back"])        # the side view draws the chimney on the near cap, contradicting front/back
chimney("chim_base", color_views=["front", "back"])

# --- door: parts seated on the body along its surface normal at their own front-view pixel; the front protrudes ~3 cm
PTS = json.loads((here / "pts.json").read_text())
fw, ww = F["door_frame"]["size"]["x"], F["door_wood"]["size"]["x"]
T_FRAME, T_WOOD, PROTRUDE = 0.14, 0.147, 0.03


def centre(key):
    x0, y0, x1, y1 = PTS["front"][key]
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def door_part(id, type, mat, key, thick, proud, size, **params):
    add(id, type, "primary", size, mat, {**params, "attach_to": "body", "attach_at": {"view": "front", "px": centre(key)},
                                         "standoff": PROTRUDE + proud - thick})


lo, arch = F["door_lo_frame"], F["door_arch_frame"]
door_part("door_frame_lo", "box", "stone", "door_lo_frame", T_FRAME, 0.0, [fw, T_FRAME, lo["size"]["z"]])
door_part("door_frame_arch", "arch", "stone", "door_arch_frame", T_FRAME, 0.0, [fw, T_FRAME, arch["size"]["z"]], axis="Y", bulge="+Z", segments=32)
door_part("door_lo", "box", "wood", "door_lo", T_WOOD, 0.005, [ww, T_WOOD, F["door_lo"]["size"]["z"]])
door_part("door_arch", "arch", "wood", "door_arch", T_WOOD, 0.005, [ww, T_WOOD, F["door_arch"]["size"]["z"]], axis="Y", bulge="+Z", segments=32)
# the knob sits on the (flat) door itself: the target is door_lo, half sunk
add("door_knob", "sphere", "secondary", [0.06] * 3, "brass", {"segments": 16, "attach_to": "door_lo", "attach_at": {"view": "front", "px": PTS["front"]["knob"]},
                                                              "standoff": -0.012}, bevel=False)

# --- windows: ring, glass and two bars share one attach point per window (view pixel), centred on the surface
R, r = 0.108, 0.017


def window(tag, view, px, **kw):
    at = {"view": view, "px": px}
    for pid, typ, size, mat, params in (
            ("ring", "torus", None, "wood", {"axis": "Y", "major_radius": R, "minor_radius": r}),
            ("glass", "cylinder", [0.2, 0.012, 0.2], "glass", {"axis": "Y", "vertices": 32}),
            ("barh", "box", [0.2, 0.02, 0.014], "wood", {}),
            ("barv", "box", [0.014, 0.02, 0.2], "wood", {})):
        half = {"ring": r, "glass": 0.006, "barh": 0.01, "barv": 0.01}[pid]
        add(f"win_{tag}_{pid}", typ, "secondary", size, mat, {**params, "attach_to": "body", "attach_at": at, "standoff": -half},
            bevel=False, **kw)


window("f", "front", PTS["front"]["win_f"], mirror={"x": True})
window("s", "side", PTS["side"]["win_s"])
window("b", "back", PTS["back"]["win_b"])

spec = {"spec_version": "0.1", "name": "mushroom_house", "category": "prop", "object_type": "mushroom_house", "units": "meters",
        "dimensions": {"width": 1.3024, "depth": 1.2087, "height": 1.2}, "symmetry": {"x": False, "y": False, "z": False},
        "style": {"hard_surface": True, "bevel": True, "bevel_width": 0.003, "bevel_segments": 2, "weighted_normals": True, "texture": "project"},
        "parts_labels": "labels.json",
        "materials": {"stone": {"base_color": [0.5, 0.47, 0.42, 1], "roughness": 0.8}, "cream": {"base_color": [0.93, 0.87, 0.72, 1], "roughness": 0.7},
                      "red": {"base_color": [0.8, 0.1, 0.08, 1], "roughness": 0.5}, "wood": {"base_color": [0.4, 0.22, 0.1, 1], "roughness": 0.7},
                      "brass": {"base_color": [0.7, 0.5, 0.15, 1], "roughness": 0.35, "metallic": 0.9}, "glass": {"base_color": [0.55, 0.72, 0.85, 1], "roughness": 0.2}},
        "components": C,
        "workflow": {"max_iterations": 3, "error_threshold": 0.08, "blockout_threshold": 0.12, "auto_correct": False}}
Path(sys.argv[1]).write_text(json.dumps(spec, indent=1))
print(len(C), "components")
