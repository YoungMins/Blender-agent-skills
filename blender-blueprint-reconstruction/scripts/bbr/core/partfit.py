"""Automatic per-part fitting (numpy only): refine an anchor-triangulated part against its label masks.

The triangulation of :mod:`triangulate` is the initialisation; a CMA-ES (deterministic seed) then moves the part's
parameters so that its silhouette, rendered at the reference calibration of every view where the part has a label
mask, covers its label and stays inside the reference silhouette. No Euler angles are optimised: a part is described by

* cone / tube: base, mid, tip (a 3-point centre line), base / tip radius, flatten;
* capsule: base, tip (the two ends), radius at each end (tapered when they differ); cylinder: base, tip, radius;
* blade: base, tip, width, thickness, roll about its axis;
* plate (contour plate, :mod:`bbr.core.plate`): plane origin, plane normal, uniform scale of the outline, thickness; the
  outline polygon is fixed (the label mask in the face-on view), so the face-on silhouette is the label whatever the plane
  and the OTHER views fit its pose;
* disc: centre, normal, radius, thickness, dome height;  sphere / ellipsoid: centre, three semi-axes;
* strand (Catmull-Rom tube): its N points p0 ... pN-1, a radius per point, flatten.

Loss per (part, view), every term normalised by the label area so that small and large parts weigh the same
(``R`` = rendered part, ``L`` = its label, ``S`` = the full reference silhouette, areas in pixels):

* (a) ``|R and not S| / |L|``: the part leaves the silhouette;
* (b) ``|L and not R| / |L|``: labelled pixels the part does not cover;
* (c) ``w_inside * |R and S and not L| / |L|``: pixels of the part inside the body but not labelled. They MAY be
  occluded, so the weight is a small tie-breaker (it stops a part from growing into the body for free), not a penalty;
* (d) contact, for ``attach_to`` parts: the part's lowest point must rest ``standoff`` above the target surface
  (depth buffer in the attach view); this is what pins the coordinate no label measures;
* (e) mirror consistency, for a ``mirror_of`` partner with its own anchors, or a partner named by a
  ``symmetric_pairs`` hint: a pull toward the mirrored sibling. The pull has a dead zone ``mirror_tol`` (parameter
  units, meters for positions and radii) inside which the partner is free to follow its own references, so two
  references that disagree by a little are not forced to an exact mirror (``PAIR_WEIGHT`` for hints).
  A partner WITHOUT anchors is fitted jointly instead: its geometry is the exact mirror of the primary's and its label
  masks are extra loss terms;
* (f) a small regulariser toward the initial values (unobservable parameters stay put);
* (g) strand only: the attachment end p0 is held near its triangulated place (``ANCHOR_WEIGHT``, tighter bounds), so the
  strand stays attached to its parent, and the second differences of the points are penalised (``SMOOTH_WEIGHT``) so it
  does not kink; an inner point sliding along the strand is not seen by any silhouette, so it is held near its
  initial place (``TANGENT_WEIGHT``) while it may move freely across the strand.

A coordinate no loss view measures (a part labelled in one view only) is locked at its triangulated / contact value.
Coarse-to-fine: a crop of at most ``COARSE_PX`` around the label bounding box, then the same crop at full resolution.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .. import BBRError, get_logger
from . import partgeom, plate, triangulate
from .calibration import ViewCalibration
from .uvraster import pair_chunks
from .triangulate import PartSolution
from .zbuffer import view_height

log = get_logger(__name__)

FIT_TYPES = ("cone", "capsule", "cylinder", "blade", "plate", "disc", "sphere", "strand")
COARSE_PX = 128              # coarse stage: the crop around the label bbox is downscaled to at most this
WINDOW_PAD_FRAC = 0.35       # crop margin around the label / initial-part bbox (of its larger side)
WINDOW_PAD_PX = 6
INSIDE_WEIGHT = 0.1          # term (c)
ANCHOR_WEIGHT = 0.15         # term (g): p0 per coordinate and squared scale unit (``REG_WEIGHTS["p0"]``) ...
ANCHOR_BOUND = 2.0           # ... and p0 may move at most this many scale units (the other points 6)
TANGENT_WEIGHT = 0.05        # term (g): per squared slide (scale units) of an inner point along the strand: silhouettes do not see it
SMOOTH_WEIGHT = 0.02         # term (g): per squared second difference of the points, in units of the mean point spacing
REG_WEIGHT = 0.0005          # term (f): per parameter and unit of the normalised squared deviation ...
REG_WEIGHTS = {"flatten": 0.05, "thickness": 0.02, "dome": 0.02, "roll": 0.02, "normal": 0.05, "scale": 0.05, "p0": ANCHOR_WEIGHT}   # ... stiffer where
# the silhouettes barely observe the parameter (an elliptic cross-section mimics a larger round one in two views)
CONTACT_WEIGHT = 0.1         # term (d)
MIRROR_WEIGHT = 0.02         # term (e)
PAIR_WEIGHT = 0.1            # term (e) for an explicit symmetric_pairs hint
JOINT_SLACK_M = 0.009       # a joint keypoint may move this far from its pin to follow the labels (gap tolerance is 10 mm)
JOINT_WEIGHT = 5.0           # penalty for leaving that ball, per squared radius
FAIL_LOSS = 1e3
POLISH = (1, 0.1, 800)                                    # (supersampling, first step, max evals) of the final pattern search
STAGES = ((True, 0.5, 1000, 2), (False, 0.25, 500, 1))   # (coarse?, initial sigma in scale units, max evals, supersampling)
STRAND_STAGES = ((True, 0.3, 3000, 2), (True, 0.15, 2000, 2), (False, 0.1, 400, 1))   # ~4 x the parameters, a cheap coarse render
STRAND_POLISH = (1, 0.05, 150)
STRAND_BUDGET_S = 60.0       # wall-clock cap per strand; each stage (and the polish) ends at its share of it ...
STRAND_STAGE_ENDS = (0.4, 0.65, 0.9, 1.0)   # ... cumulative: coarse, coarse, fine, polish
COARSE_SIDES, COARSE_SEGMENTS = 16, 10
MIN_LABEL_PX = 12            # a label smaller than this (in a view) is ignored


# --------------------------------------------------------------------------- state (named parameter blocks)
State = dict[str, np.ndarray]


POINT_NAME = re.compile(r"p\d+")


def is_point(name: str) -> bool:
    """A parameter block that is a 3D position: a strand point ``p<i>`` (or ``base`` / ``tip`` / ``center``)."""
    return name in ("base", "tip", "center") or POINT_NAME.fullmatch(name) is not None


def strand_points(st: Mapping[str, np.ndarray]) -> np.ndarray:
    """``(N, 3)`` points p0 ... pN-1 of a strand state, in order."""
    return np.array([st[f"p{i}"] for i in range(sum(1 for n in st if POINT_NAME.fullmatch(n)))], dtype=float)


def strand_kink(pts: np.ndarray) -> float:
    """Mean squared second difference of the strand points in units of the mean point spacing (0 for a straight,
    evenly spaced line; about 0.1 for a gentle wave; > 1 for a kink or two bunched points)."""
    d2 = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
    spacing = max(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()), 1e-9)
    return float(np.mean(np.sum(d2 ** 2, axis=1))) / spacing ** 2


def layout(ptype: str, taper: bool = True, n_points: int = 0) -> list[tuple[str, int]]:
    """Ordered ``(name, size)`` parameter blocks of a part type (``n_points``: the points of a strand)."""
    if ptype == "strand":
        return [(f"p{i}", 3) for i in range(n_points)] + [("radii", n_points), ("flatten", 1)]
    if ptype == "cone":
        return [("base", 3), ("tip", 3), ("bend", 2), ("rb", 1), ("rt", 1), ("flatten", 1)]
    if ptype == "capsule":
        return [("base", 3), ("tip", 3), ("r0", 1)] + ([("r1", 1)] if taper else [])
    if ptype == "cylinder":
        return [("base", 3), ("tip", 3), ("r0", 1)]
    if ptype == "blade":
        return [("base", 3), ("tip", 3), ("width", 1), ("thickness", 1), ("roll", 1)]
    if ptype == "disc":
        return [("center", 3), ("normal", 3), ("radius", 1), ("thickness", 1), ("dome", 1)]
    if ptype == "plate":
        return [("center", 3), ("normal", 3), ("scale", 1), ("thickness", 1)]
    if ptype == "sphere":
        return [("center", 3), ("semi", 3)]
    raise BBRError(f"no part fitting for type {ptype!r}; supported: {list(FIT_TYPES)}")


def _pack(st: Mapping[str, np.ndarray], lay: Sequence[tuple[str, int]]) -> np.ndarray:
    return np.concatenate([np.asarray(st[n], dtype=float).reshape(k) for n, k in lay])


def _unpack(x: np.ndarray, lay: Sequence[tuple[str, int]]) -> State:
    out, i = {}, 0
    for n, k in lay:
        out[n] = x[i:i + k].copy()
        i += k
    return out


def cone_frame(base: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Two unit vectors perpendicular to the chord base -> tip: the frame a cone's bend offset is expressed in (fixed
    during a fit, taken from the initial chord)."""
    return partgeom.orthonormal_frame(np.asarray(tip, float) - np.asarray(base, float))[:2]


def cone_mid(st: State, frame: np.ndarray) -> np.ndarray:
    """The centre line's middle point: the chord middle plus the bend offset (2 numbers in ``frame``). Parameterising
    the middle point this way removes the freedom to slide it along the arc (the same arc, a different point)."""
    return (st["base"] + st["tip"]) / 2.0 + st["bend"][0] * frame[0] + st["bend"][1] * frame[1]


def _bend_offset(base: np.ndarray, mid: np.ndarray, tip: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Bend offset of the arc through base, mid, tip: where the arc's own middle lies relative to the chord middle."""
    length, axis, bend, bend_dir = partgeom.arc_through_points(base, mid, tip)
    if bend_dir is None:
        return np.zeros(2)
    pts, *_ = partgeom.cone_path(length, axis, bend, bend_dir, 2)
    off = base + pts[1] - (base + tip) / 2.0
    return np.array([off @ frame[0], off @ frame[1]])


def state_from_solution(ptype: str, sol: PartSolution, params: Mapping[str, Any], taper: bool = True,
                        frame: Optional[np.ndarray] = None) -> State:
    """Initial state of a part from its triangulated solution (``params`` = the component's params; ``frame`` = the
    cone bend frame, default from the part's own chord)."""
    kp, sc = sol.keypoints, sol.scalars
    if ptype == "strand":
        names = sorted((n for n in kp if POINT_NAME.fullmatch(n)), key=lambda n: int(n[1:]))
        st = {f"p{i}": np.asarray(kp[n], float).copy() for i, n in enumerate(names)}
        st["radii"] = np.asarray(sc["radii"], float).copy()
        st["flatten"] = np.array([float(sol.params.get("flatten", params.get("flatten", 1.0)))])
        return st
    if ptype == "cone":
        rb, rt = sc["radius"]
        frame = cone_frame(kp["base"], kp["tip"]) if frame is None else frame
        mid = kp["mid"] if "mid" in kp else (kp["base"] + kp["tip"]) / 2.0
        return {"base": kp["base"].copy(), "tip": kp["tip"].copy(), "bend": _bend_offset(kp["base"], mid, kp["tip"], frame),
                "rb": np.array([rb]), "rt": np.array([rt]), "flatten": np.array([float(sol.params.get("flatten", params.get("flatten", 1.0)))])}
    if ptype in ("capsule", "cylinder"):
        r = float(sc["radius"])
        st = {"base": kp["base"].copy(), "tip": kp["tip"].copy(), "r0": np.array([r])}
        if ptype == "capsule" and taper:
            st["r1"] = np.array([float(sc.get("radius_tip", r))])
        return st
    if ptype == "blade":
        return {"base": kp["base"].copy(), "tip": kp["tip"].copy(), "width": np.array([float(sc["width"])]),
                "thickness": np.array([float(sc["thickness"])]), "roll": np.array([float(params.get("roll", 0.0))])}
    if ptype == "plate":
        return {"center": kp["base"].copy(), "normal": np.asarray(sc["normal"], float).copy(),
                "scale": np.array([float(sc.get("scale", 1.0))]), "thickness": np.array([float(sc["thickness"])])}
    if ptype == "disc":
        return {"center": kp["center"].copy(), "normal": np.asarray(sc["normal"], float).copy(),
                "radius": np.array([float(sc["radius"])]), "thickness": np.array([float(sc["thickness"])]),
                "dome": np.array([float(sol.params.get("dome_height", params.get("dome_height", 0.0)))])}
    if ptype == "sphere":
        return {"center": kp["center"].copy(), "semi": np.asarray(sc["semi"], float).copy()}
    raise BBRError(f"no part fitting for type {ptype!r}")


def mirror_state(st: State, frame: Optional[np.ndarray] = None, dst_frame: Optional[np.ndarray] = None) -> State:
    """The state reflected across X = 0 (a cone's bend offset is re-expressed in ``dst_frame``; ``frame`` = its own)."""
    flip = np.array([-1.0, 1.0, 1.0])
    out = {k: (v * flip if v.size == 3 and (is_point(k) or k == "normal") else v.copy()) for k, v in st.items()}
    if "bend" in st:
        mid = cone_mid(st, frame if frame is not None else cone_frame(st["base"], st["tip"])) * flip
        dst = dst_frame if dst_frame is not None else cone_frame(out["base"], out["tip"])
        out["bend"] = _bend_offset(out["base"], mid, out["tip"], dst)
    if "roll" in out:
        out["roll"] = -out["roll"]
    return out


# --------------------------------------------------------------------------- state -> pose (what the builders receive)
@dataclass
class Pose:
    """Builder inputs of a part: object ``location`` / ``rotation_deg`` / ``size`` and the primitive ``params``."""

    location: list[float]
    rotation_deg: list[float]
    size: Optional[list[float]]
    params: dict[str, Any]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise BBRError("degenerate direction")
    return v / n


def pose_from_state(ptype: str, st: State, params: Mapping[str, Any], face_calib: Optional[ViewCalibration] = None,
                    frame: Optional[np.ndarray] = None) -> Pose:
    """Pose of a state. ``params`` are the component's params (tessellation, tip_width, ...) the geometry is built with;
    ``face_calib`` (blade) is the calibration of the view that measured the width; ``frame`` the cone bend frame."""
    p = dict(params)
    if ptype == "sphere":
        return Pose(list(st["center"]), [0.0] * 3, [2.0 * float(s) for s in st["semi"]], {})
    if ptype == "disc":
        n = _unit(st["normal"])
        radius, dome = float(st["radius"][0]), max(float(st["dome"][0]), 0.0)
        geo = {"radius": radius, "axis": list(n), "thickness": float(st["thickness"][0]), "dome_height": dome}
        if p.get("dome_radius") is not None:
            geo["dome_radius"] = min(float(p["dome_radius"]), radius)
        return Pose(list(st["center"]), [0.0] * 3, None, geo)
    if ptype == "plate":
        if face_calib is None or p.get("outline_px") is None:
            raise BBRError("a plate needs its outline polygon and the view it is seen face-on in")
        n = _unit(st["normal"])
        outline, u = plate.local_outline(np.asarray(p["outline_px"], float), face_calib, st["center"], n,
                                         np.asarray(p["axis"], float), float(st["scale"][0]))
        geo = {"outline": outline.tolist(), "outline_tris": p["outline_tris"], "outline_px": p["outline_px"],
               "thickness": float(st["thickness"][0]), "axis": list(u), "normal": list(n), "bend": float(p.get("bend", 0.0)),
               "face_view": p.get("face_view")}
        return Pose(list(st["center"]), [0.0] * 3, None, geo)
    if ptype == "strand":
        pts = strand_points(st)
        return Pose(list(pts[0]), [0.0] * 3, None, {"points": (pts - pts[0]).tolist(), "radii": [max(float(r), 0.0) for r in st["radii"]],
                                                    "flatten": float(st["flatten"][0])})
    base, tip = st["base"], st["tip"]
    length = float(np.linalg.norm(tip - base))
    axis = _unit(tip - base)
    if ptype == "cone":
        rb, rt = float(st["rb"][0]), max(float(st["rt"][0]), 0.0)
        length, ax, bend, bend_dir = partgeom.arc_through_points(base, cone_mid(st, frame), tip)
        geo = {"base_radius": rb, "tip_radius": rt, "length": length, "axis": list(ax), "bend": bend,
               "flatten": float(st["flatten"][0])}
        if bend_dir is not None:
            geo["bend_dir"] = list(bend_dir)
        return Pose(list(base), [0.0] * 3, None, geo)
    if ptype == "blade":
        if face_calib is None:
            raise BBRError("a blade needs the view that measured its width (width_px)")
        wdir = triangulate.blade_width_dir(axis, face_calib, float(st["roll"][0]))
        return Pose(list(base), [0.0] * 3, None, {"length": length, "width": float(st["width"][0]),
                                                  "thickness": float(st["thickness"][0]), "axis": list(axis),
                                                  "width_dir": list(wdir)})
    r0 = min(float(st["r0"][0]), length / 2.0)                        # capsule / cylinder
    geo = {"axis": "Z"}
    if "r1" in st and abs(float(st["r1"][0]) - r0) > 1e-9:
        geo["tip_radius"] = min(float(st["r1"][0]), length / 2.0)
    rot = partgeom.euler_xyz_deg(partgeom.rotation_z_to(axis))
    return Pose(list((base + tip) / 2.0), rot, [2.0 * r0, 2.0 * r0, length], geo)


def pose_from_solution(sol: PartSolution, params: Mapping[str, Any]) -> Pose:
    """What the Blender builders get for a solution (the component's params merged with the solution's)."""
    return Pose(list(sol.location), list(sol.rotation_deg), None if sol.size is None else list(sol.size),
                {**params, **sol.params})


def euler_matrix(deg: Sequence[float]) -> np.ndarray:
    """Rotation matrix ``Rz @ Ry @ Rx`` of Blender XYZ-Euler degrees."""
    x, y, z = (math.radians(a) for a in deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


# --------------------------------------------------------------------------- silhouettes
Shape = tuple  # ("mesh", verts, tris) | ("capsule", c0, r0, c1, r1) | ("ellipsoid", centre, semi)


def _cylinder_mesh(c0: np.ndarray, c1: np.ndarray, r: float, sides: int = 16) -> tuple[np.ndarray, np.ndarray]:
    u, v, _ = partgeom.orthonormal_frame(c1 - c0)
    phi = np.arange(sides) * (2.0 * np.pi / sides)
    ring = r * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
    verts = np.concatenate([c0 + ring, c1 + ring, c0[None], c1[None]])
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris += [(i, j, sides + j), (i, sides + j, sides + i), (2 * sides, j, i), (2 * sides + 1, sides + i, sides + j)]
    return verts, np.array(tris, dtype=np.int64)


def shape_of(ptype: str, pose: Pose, coarse: bool = False) -> Shape:
    """Renderable shape of a pose, in world coordinates. ``coarse`` uses a lighter tessellation."""
    loc, p = np.asarray(pose.location, float), pose.params
    if ptype == "sphere":
        return ("ellipsoid", loc, np.asarray(pose.size, float) / 2.0)
    if ptype in ("capsule", "cylinder"):
        rot = euler_matrix(pose.rotation_deg)
        axis, size = rot[:, 2], pose.size
        length = float(size[2])
        if ptype == "cylinder":
            verts, tris = _cylinder_mesh(loc - axis * length / 2.0, loc + axis * length / 2.0, size[0] / 2.0,
                                         8 if coarse else int(p.get("vertices", 16)))
            return ("mesh", verts, tris)
        r0 = size[0] / 2.0
        r1 = float(p.get("tip_radius", r0))
        return ("capsule", loc - axis * (length / 2.0 - r0), r0, loc + axis * (length / 2.0 - r1), r1)
    if ptype == "cone":
        sides = int(p.get("sides", partgeom.DEFAULT_SIDES))
        segs = int(p.get("segments", partgeom.DEFAULT_SEGMENTS))
        if coarse:
            sides, segs = min(sides, COARSE_SIDES), min(segs, COARSE_SEGMENTS)
        verts, faces = partgeom.cone_mesh(
            float(p["base_radius"]), float(p.get("tip_radius", 0.0)), float(p["length"]), p.get("axis", (0, 0, 1)),
            float(p.get("bend", 0.0)), p.get("bend_dir"), segs, sides, float(p.get("flatten", 1.0)), weld=False)
    elif ptype == "blade":
        verts, faces = partgeom.blade_mesh(
            float(p["length"]), float(p["width"]), float(p["thickness"]), p.get("tip_width"), p.get("tip_thickness"),
            float(p.get("tip_length", 0.0)), p.get("axis", (0, 0, 1)), p.get("width_dir"))
    elif ptype == "plate":
        verts, faces = partgeom.plate_mesh(p["outline"], p["outline_tris"], float(p["thickness"]), p["axis"], p["normal"],
                                           float(p.get("bend", 0.0)), 1 if coarse and abs(float(p.get("bend", 0.0))) > 0 else 0)
    elif ptype == "strand":
        verts, faces = partgeom.strand_mesh(
            p["points"], p["radii"], float(p.get("flatten", 1.0)), 3 if coarse else int(p.get("segments", partgeom.DEFAULT_STRAND_SPAN_SEGMENTS)),
            8 if coarse else int(p.get("sides", partgeom.DEFAULT_SIDES)), weld=False)
    elif ptype == "disc":
        verts, faces = partgeom.disc_mesh(
            float(p["radius"]), float(p["thickness"]), float(p.get("dome_height", 0.0)), p.get("dome_radius"),
            24 if coarse else int(p.get("segments", partgeom.DEFAULT_DISC_SEGMENTS)),
            3 if coarse else int(p.get("dome_rings", partgeom.DEFAULT_DOME_RINGS)), p.get("axis", (0, 0, 1)))
    else:
        raise BBRError(f"no silhouette for type {ptype!r}")
    return ("mesh", np.asarray(verts, float) + loc, partgeom.polygons_triangles(faces))


def mirror_shape(shape: Shape) -> Shape:
    """The shape reflected across X = 0 (triangle winding does not matter for silhouettes)."""
    flip = np.array([-1.0, 1.0, 1.0])
    if shape[0] == "mesh":
        return ("mesh", shape[1] * flip, shape[2])
    if shape[0] == "capsule":
        return ("capsule", shape[1] * flip, shape[2], shape[3] * flip, shape[4])
    return ("ellipsoid", shape[1] * flip, shape[2])


def _mesh_mask(verts: np.ndarray, tris: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """Like :func:`partmask.triangles_mask` with edge functions instead of barycentric coordinates (about twice as fast;
    the fit renders thousands of times)."""
    w, h = calib.image_size
    out = np.zeros((h, w), dtype=bool)
    xy = calib.world_to_px(verts)[tris]
    for tri, px, py in pair_chunks(xy, (w, h)):
        t = xy[tri]
        x, y = px + 0.5, py + 0.5
        e = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e.append((t[:, b, 0] - t[:, a, 0]) * (y - t[:, a, 1]) - (t[:, b, 1] - t[:, a, 1]) * (x - t[:, a, 0]))
        inside = ((e[0] >= 0) & (e[1] >= 0) & (e[2] >= 0)) | ((e[0] <= 0) & (e[1] <= 0) & (e[2] <= 0))
        out[py[inside], px[inside]] = True
    return out


def render(shape: Shape, calib: ViewCalibration) -> np.ndarray:
    """Orthographic silhouette of a shape on the view's pixel grid, bool ``(H, W)``."""
    if shape[0] == "mesh":
        return _mesh_mask(shape[1], shape[2], calib)
    w, h = calib.image_size
    out = np.zeros((h, w), dtype=bool)
    mpp = calib.m_per_px
    if shape[0] == "capsule":
        _, c0, r0, c1, r1 = shape
        p = calib.world_to_px(np.array([c0, c1]))
        rp = np.array([r0, r1]) / mpp
        lo = np.floor((p - rp[:, None]).min(axis=0)).astype(int)
        hi = np.ceil((p + rp[:, None]).max(axis=0)).astype(int) + 1
    else:
        _, c, semi = shape
        frame = triangulate.view_frame(calib.camera_axis)
        sh = abs(float(semi @ frame.right)) / mpp                                # semi-axes in px (axis-aligned ellipsoid)
        sv = abs(float(semi @ frame.up)) / mpp
        pc = calib.world_to_px(c[None])[0]
        lo = np.floor(pc - [sh, sv]).astype(int)
        hi = np.ceil(pc + [sh, sv]).astype(int) + 1
    x0, y0, x1, y1 = max(lo[0], 0), max(lo[1], 0), min(hi[0], w), min(hi[1], h)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1]
    q = np.stack([xx + 0.5, yy + 0.5], axis=-1)
    if shape[0] == "ellipsoid":
        out[y0:y1, x0:x1] = ((q[..., 0] - pc[0]) / max(sh, 1e-9)) ** 2 + ((q[..., 1] - pc[1]) / max(sv, 1e-9)) ** 2 <= 1.0
        return out
    d, dr = p[1] - p[0], rp[1] - rp[0]
    w0 = q - p[0]
    a = float(d @ d - dr * dr)
    b = w0 @ d + rp[0] * dr
    c = (w0 ** 2).sum(-1) - rp[0] ** 2
    best = np.minimum(c, a - 2.0 * b + c)
    if a > 1e-9:
        t = b / a
        inner = (t > 0) & (t < 1)
        best = np.where(inner, np.minimum(best, c - b * b / a), best)
    out[y0:y1, x0:x1] = best <= 0
    return out


# --------------------------------------------------------------------------- loss
def view_loss(rendered: np.ndarray, label: np.ndarray, silhouette: np.ndarray, w_inside: float = INSIDE_WEIGHT
              ) -> dict[str, float]:
    """Terms (a), (b), (c) of one view, normalised by the label area. Masks may be soft (coverage 0..1 after
    downscaling) or boolean; ``total`` = a + b + w_inside * c."""
    r = rendered.astype(float)
    lab, sil = label.astype(float), silhouette.astype(float)
    area = max(float(lab.sum()), 4.0)
    a = float((r * (1.0 - sil)).sum()) / area
    b = float((lab * (1.0 - r)).sum()) / area
    c = float((r * sil * (1.0 - lab)).sum()) / area
    return {"outside": a, "uncovered": b, "inside": c, "total": a + b + w_inside * c}


def iou(rendered: np.ndarray, label: np.ndarray) -> float:
    r, lab = np.asarray(rendered, bool), np.asarray(label, bool)
    union = int((r | lab).sum())
    return float((r & lab).sum()) / union if union else 1.0


# --------------------------------------------------------------------------- per-view data at a resolution level
@dataclass
class ViewData:
    """One (target, view): label, silhouette and the crop window; downscaled copies are made per level."""

    view: str
    target: int                 # index into the target list (its mirrored flag)
    calib: ViewCalibration
    label: np.ndarray
    sil: np.ndarray
    window: tuple[int, int, int, int]
    _levels: dict[int, tuple[ViewCalibration, np.ndarray, np.ndarray]] = None  # type: ignore[assignment]

    def level(self, k: int) -> tuple[ViewCalibration, np.ndarray, np.ndarray]:
        """``(calibration of the crop, label, silhouette)`` downscaled by the integer factor ``k`` (block means)."""
        if self._levels is None:
            self._levels = {}
        if k not in self._levels:
            x0, y0, x1, y1 = self.window
            x0, y0 = x0 // k * k, y0 // k * k
            cw, ch = -(-(x1 - x0) // k), -(-(y1 - y0) // k)
            h, w = self.label.shape

            def crop(m: np.ndarray) -> np.ndarray:
                out = np.zeros((ch * k, cw * k))
                sub = m[y0:min(y0 + ch * k, h), x0:min(x0 + cw * k, w)]
                out[:sub.shape[0], :sub.shape[1]] = sub
                return out.reshape(ch, k, cw, k).mean(axis=(1, 3))

            c = self.calib
            cal = ViewCalibration(c.view, c.camera_axis, (cw, ch), c.m_per_px * k,
                                  ((c.origin_px[0] - x0) / k, (c.origin_px[1] - y0) / k))
            self._levels[k] = (cal, crop(self.label), crop(self.sil))
        return self._levels[k]


def _window(label: np.ndarray, init_render: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    both = label | init_render
    ys, xs = np.nonzero(both)
    if len(ys) == 0:
        return None
    h, w = label.shape
    pad = int(WINDOW_PAD_FRAC * max(xs.max() - xs.min(), ys.max() - ys.min())) + WINDOW_PAD_PX
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0), min(int(xs.max()) + 1 + pad, w),
            min(int(ys.max()) + 1 + pad, h))


# --------------------------------------------------------------------------- optimiser
def cma_es(f: Callable[[np.ndarray], float], x0: np.ndarray, sigma0: float, lo: np.ndarray, hi: np.ndarray, *,
           max_evals: int, seed: int = 0, tol: float = 1e-3, stall_gens: int = 20, ftol: float = 2e-4,
           trace: Optional[list] = None, deadline: Optional[float] = None) -> tuple[np.ndarray, float, int]:
    """Minimal CMA-ES with box repair (clipping) and a deterministic seed -> ``(best x, best f, evaluations)``.

    ``f`` should be scaled so that one unit of every coordinate is a plausible deviation (the caller normalises)."""
    n = len(x0)
    rng = np.random.default_rng(seed)
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / float(np.sum(w ** 2))
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m, sigma = x0.astype(float).copy(), float(sigma0)
    cov, pc, ps = np.eye(n), np.zeros(n), np.zeros(n)
    bmat, dvec = np.eye(n), np.ones(n)
    best_x, best_f = m.copy(), float(f(m))
    evals, gen, hist = 1, 0, []
    while evals < max_evals and (deadline is None or time.perf_counter() < deadline):
        z = rng.standard_normal((lam, n))
        xs = np.clip(m + sigma * ((z * dvec) @ bmat.T), lo, hi)
        fs = np.array([f(x) for x in xs])
        evals += lam
        y = (xs - m) / sigma
        order = np.argsort(fs)
        if fs[order[0]] < best_f:
            best_f, best_x = float(fs[order[0]]), xs[order[0]].copy()
        ys = y[order[:mu]]
        ymean = w @ ys
        m = np.clip(m + sigma * ymean, lo, hi)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (bmat @ ((bmat.T @ ymean) / dvec))
        hsig = float(np.linalg.norm(ps)) / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + (math.sqrt(cc * (2 - cc) * mueff) * ymean if hsig else 0.0)
        cov = ((1 - c1 - cmu) * cov + c1 * (np.outer(pc, pc) + (0.0 if hsig else cc * (2 - cc)) * cov)
               + cmu * (ys.T * w) @ ys)
        sigma = min(sigma * math.exp((cs / damps) * (float(np.linalg.norm(ps)) / chi - 1)), 3.0)
        cov = (cov + cov.T) / 2.0
        evals_c, bmat = np.linalg.eigh(cov)
        dvec = np.sqrt(np.maximum(evals_c, 1e-20))
        gen += 1
        hist.append(best_f)
        if trace is not None:
            trace.append((evals, best_f, sigma))
        if sigma * dvec.max() < tol or (len(hist) > stall_gens and hist[-stall_gens - 1] - best_f < ftol):
            break
    return best_x, best_f, evals


def compass_search(f: Callable[[np.ndarray], float], x0: np.ndarray, step0: float, lo: np.ndarray, hi: np.ndarray, *,
                   max_evals: int, step_min: float = 0.01, deadline: Optional[float] = None) -> tuple[np.ndarray, float, int]:
    """Deterministic coordinate pattern search (polish after CMA-ES): try +-step on every coordinate, keep improvements,
    halve the step when nothing improves."""
    x, fx, evals, step = x0.copy(), float(f(x0)), 1, step0
    while step >= step_min and evals < max_evals and (deadline is None or time.perf_counter() < deadline):
        improved = False
        for i in range(len(x)):
            for sign in (1.0, -1.0):
                y = x.copy()
                y[i] = np.clip(y[i] + sign * step, lo[i], hi[i])
                if y[i] == x[i]:
                    continue
                fy = float(f(y))
                evals += 1
                if fy < fx - 1e-9:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            step *= 0.5
    return x, fx, evals


# --------------------------------------------------------------------------- fitting
class MissingFaceView(BBRError):
    """A blade / disc / plate part has no face-on view to measure from (a ``mirror_of`` partner may take it from its partner)."""


@dataclass
class Target:
    """Label masks of one part to explain. ``mirrored``: its geometry is the primary's reflected across X = 0."""

    part_id: str
    mirrored: bool
    labels: Mapping[str, np.ndarray]


def _scales(ptype: str, st: State, mpp: float, ref_m: Optional[float] = None) -> tuple[State, State, State]:
    """Per-parameter ``(scale, lower, upper)``: one scale unit is a plausible deviation of the triangulation (``ref_m``:
    the size of a plate's outline)."""
    if ptype == "plate":
        ref = float(ref_m or 0.1)
    elif ptype in ("cone", "capsule", "cylinder", "blade"):
        ref = float(np.linalg.norm(st["tip"] - st["base"]))
    elif ptype == "strand":
        ref = float(np.linalg.norm(np.diff(strand_points(st), axis=0), axis=1).sum())
    elif ptype == "disc":
        ref = 2.0 * float(st["radius"][0])
    else:
        ref = 2.0 * float(st["semi"].max())
    scale, lo, hi = {}, {}, {}
    for name, v in st.items():
        v = np.asarray(v, float)
        if is_point(name):
            s = np.full(3, max(0.06 * ref, 3 * mpp))
            reach = ANCHOR_BOUND if name == "p0" else 6
            lo[name], hi[name] = v - reach * s, v + reach * s
        elif name == "bend":
            s = np.full(2, max(0.06 * ref, 3 * mpp))
            lo[name], hi[name] = v - 6 * s, v + 6 * s
        elif name == "normal":
            s = np.full(3, 0.3 if ptype == "plate" else 0.15)   # a plate's tilt is only seen in the oblique views: search wider
            lo[name], hi[name] = v - 1.2, v + 1.2
        elif name in ("rb", "rt", "r0", "r1", "radius", "width", "semi", "radii"):
            s = np.maximum(0.12 * np.abs(v), 1.5 * mpp)
            lo[name], hi[name] = np.where(name == "rt", 0.0, 0.3 * v), np.maximum(3.0 * v, 0.5 * v.max())
        elif name == "thickness":
            s = np.maximum(0.25 * v, 0.5 * mpp)
            lo[name], hi[name] = 0.3 * v, 3.0 * v
        elif name == "dome":
            s = np.maximum(0.5 * v, 0.004 * ref)
            lo[name], hi[name] = np.zeros(1), 3.0 * v + 0.02 * ref
        elif name == "flatten":
            s = np.full(1, 0.25)
            lo[name], hi[name] = np.full(1, 0.2), np.full(1, 1.5)
        elif name == "scale":
            s = np.full(1, 0.1)
            lo[name], hi[name] = np.full(1, 0.6), np.full(1, 1.5)
        else:                                                    # roll
            s = np.full(1, 15.0)
            lo[name], hi[name] = np.full(1, -180.0), np.full(1, 180.0)
        scale[name] = s
        lo.setdefault(name, v - 6 * s)
    return scale, lo, hi


def _lock_mask(ptype: str, lay: Sequence[tuple[str, int]], st: State, locked: Sequence[str],
               measured_axes: set[int], free_axes: set[int]) -> np.ndarray:
    """Bool per vector entry: True = frozen. ``locked`` names are ``name`` or ``name.x|y|z``; a point's coordinate on
    an axis no loss view measures (and no contact term frees) stays at its initial value; a disc whose dome is 0 keeps 0."""
    frozen = []
    names = {n for n, _ in lay}
    for item in locked:
        if str(item).split(".")[0] not in names:
            raise BBRError(f"fit_locked: {item!r} is not a parameter of a {ptype}; parameters: {sorted(names)}")
    for name, k in lay:
        fz = np.zeros(k, dtype=bool)
        if is_point(name):
            for ax in range(3):
                fz[ax] = ax not in (measured_axes | free_axes)
        if name == "dome" and float(st["dome"][0]) <= 0.0:
            fz[:] = True
        for item in locked:
            base, _, comp = str(item).partition(".")
            if base == name:
                if comp:
                    fz["xyz".index(comp)] = True
                else:
                    fz[:] = True
        frozen.append(fz)
    return np.concatenate(frozen)


def contact_points(ptype: str, pose: Pose, shape: Shape) -> np.ndarray:
    """World points that must rest on the attach target: a disc's back face (sampled on polar rings), else the mesh
    vertices / the capsule end spheres' centres offset by their radii / the ellipsoid centre."""
    if ptype == "disc":
        n = _unit(np.asarray(pose.params["axis"], float))
        u, v, _ = partgeom.orthonormal_frame(n)
        phi = np.arange(12) * (2 * np.pi / 12)
        rho = np.array([0.0, 0.33, 0.66, 1.0])
        ring = np.concatenate([np.stack([np.cos(phi), np.sin(phi)], 1) * r for r in rho[1:]] + [np.zeros((1, 2))])
        pts = float(pose.params["radius"]) * (ring[:, :1] * u + ring[:, 1:] * v)
        return np.asarray(pose.location, float) + pts - n * float(pose.params["thickness"]) / 2.0
    if shape[0] == "mesh":
        return shape[1]
    if shape[0] == "capsule":
        return np.array([shape[1], shape[3]])
    return shape[1][None]


def contact_gap(points: np.ndarray, zbuf: np.ndarray, calib: ViewCalibration) -> Optional[float]:
    """Smallest height of ``points`` above the target surface (depth buffer ``zbuf`` of ``calib``'s view) among the
    points that lie over a surface; None when none does."""
    h, w = zbuf.shape
    uv = calib.world_to_px(points)
    px, py = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not ok.any():
        return None
    surf = zbuf[py[ok], px[ok]]
    ok2 = np.isfinite(surf)
    if not ok2.any():
        return None
    return float((view_height(points[ok][ok2], calib) - surf[ok2]).min())


@dataclass
class FitSetup:
    """Everything the objective needs (built once per fit)."""

    ptype: str
    params: Mapping[str, Any]
    face_calib: Optional[ViewCalibration]
    frame: Optional[np.ndarray]
    lay: list[tuple[str, int]]
    items: list[ViewData]
    mirrored: list[bool]
    w_inside: float
    contact: Optional[tuple[np.ndarray, ViewCalibration, float, float]]   # zbuf, calib, standoff, tolerance (m)


def render_soft(shape: Shape, cal: ViewCalibration, ss: int) -> np.ndarray:
    """Coverage in 0..1 of the shape on ``cal``'s grid, from an ``ss`` x ``ss`` supersampled render (``ss`` 1 = binary)."""
    if ss <= 1:
        return render(shape, cal)
    w, h = cal.image_size
    fine = ViewCalibration(cal.view, cal.camera_axis, (w * ss, h * ss), cal.m_per_px / ss, (cal.origin_px[0] * ss, cal.origin_px[1] * ss))
    return render(shape, fine).reshape(h, ss, w, ss).mean(axis=(1, 3))


def evaluate(setup: FitSetup, st: State, k: int, coarse: bool, ss: int = 1) -> tuple[float, dict[str, float]]:
    """Mean per-view loss (a + b + c) plus the contact term (d) of a state at downscale ``k``; ``FAIL_LOSS`` when the
    state cannot be built."""
    try:
        pose = pose_from_state(setup.ptype, st, setup.params, setup.face_calib, setup.frame)
        shape = shape_of(setup.ptype, pose, coarse)
    except (BBRError, ValueError, KeyError, FloatingPointError):
        return FAIL_LOSS, {}
    mirrored = mirror_shape(shape)
    total = 0.0
    for item in setup.items:
        cal, lab, sil = item.level(k)
        terms = view_loss(render_soft(mirrored if setup.mirrored[item.target] else shape, cal, ss), lab, sil, setup.w_inside)
        total += terms["total"]
    total /= max(len(setup.items), 1)
    extra = {}
    if setup.contact is not None:
        zbuf, calib, standoff, tol = setup.contact
        gap = contact_gap(contact_points(setup.ptype, pose, shape), zbuf, calib)
        extra["contact"] = CONTACT_WEIGHT * min(((gap - standoff) / tol) ** 2, 100.0) if gap is not None else CONTACT_WEIGHT * 100.0
        total += extra["contact"]
    return total, extra


def fit_part(part_id: str, ptype: str, sol: PartSolution, params: Mapping[str, Any],
             calibs: Mapping[str, ViewCalibration], silhouettes: Mapping[str, np.ndarray], targets: Sequence[Target], *,
             depth_for: Optional[Callable[[str], np.ndarray]] = None, locked: Sequence[str] = (), taper: bool = True,
             seed: int = 0, mirror_sibling: Optional[PartSolution] = None, w_inside: float = INSIDE_WEIGHT,
             mirror_tol: float = 0.0, mirror_weight: float = MIRROR_WEIGHT, joint_point: Optional[str] = None,
             joint_slack: float = JOINT_SLACK_M, time_budget: float = STRAND_BUDGET_S) -> tuple[PartSolution, dict[str, Any]]:
    """Fit one anchored part (and its exact mirror partners, ``targets`` with ``mirrored=True``) to its label masks.

    ``sol`` / ``params`` are the triangulated solution and the component's params; ``silhouettes`` the full reference
    silhouettes; ``depth_for(view)`` the attach target's depth buffer (``params.attach_to``). Returns the fitted
    :class:`PartSolution` (keypoints, scalars, pose, params: what the builders and ``mirror_of`` partners read) and a
    JSON-able report (loss before / after, per-view IoU before / after, free parameters, evaluations, seconds).

    ``mirror_sibling``: the solution whose mirror image softly attracts this part (term (e)), with dead zone
    ``mirror_tol`` and weight ``mirror_weight``; the report gets ``mirror_dev_max`` (largest deviation from it, in
    parameter units).

    ``joint_point``: the keypoint pinned by ``params.joint``; it starts on the pin but stays free within ``joint_slack``
    (m) of it, so the rest of the part keeps following its labels instead of the pin dragging the whole part.

    ``time_budget`` (s): wall-clock cap of a strand fit (the other types are cheap and run their fixed evaluation budgets)."""
    t0 = time.perf_counter()
    if ptype not in FIT_TYPES:
        raise BBRError(f"{part_id}: no fitting for type {ptype!r}")
    face_view = sol.scalars.get("face_view")
    if ptype in ("blade", "disc", "plate") and face_view not in calibs:
        raise MissingFaceView(f"{part_id}: fitting a {ptype} needs radius_px / width_px in a face-on view")
    if ptype == "plate":
        if sol.params.get("outline_px") is None:
            raise MissingFaceView(f"{part_id}: a plate that copies its mirror partner's outline has nothing to fit on its own")
        params = {**params, **sol.params}
        if joint_point == "base":
            joint_point = "center"
    face_calib = calibs[face_view] if ptype in ("blade", "plate") else None
    frame = cone_frame(sol.keypoints["base"], sol.keypoints["tip"]) if ptype == "cone" else None
    st0 = state_from_solution(ptype, sol, params, taper, frame)
    lay = layout(ptype, taper, sum(1 for n in st0 if POINT_NAME.fullmatch(n)))
    pose0 = pose_from_state(ptype, st0, params, face_calib, frame)
    shape0 = shape_of(ptype, pose0)
    items: list[ViewData] = []
    for ti, tgt in enumerate(targets):
        for view, label in tgt.labels.items():
            if view not in calibs or view not in silhouettes or int(label.sum()) < MIN_LABEL_PX:
                continue
            init_r = render(mirror_shape(shape0) if tgt.mirrored else shape0, calibs[view])
            win = _window(np.asarray(label, bool), init_r)
            if win is not None:
                items.append(ViewData(view, ti, calibs[view], np.asarray(label, bool), np.asarray(silhouettes[view], bool), win))
    if not items:
        raise BBRError(f"{part_id}: no label mask in any calibrated view; nothing to fit")
    measured = {i for it in items for i in range(3) if i != triangulate.depth_axis(it.calib)}
    attach = params.get("attach_to")
    contact = None
    free_axes: set[int] = set()
    if attach and depth_for is not None:
        aview = triangulate._attach_view(part_id, params, calibs, sorted({it.view for it in items}))
        ac = calibs[aview]
        contact = (depth_for(aview), ac, float(params.get("standoff", 0.0)), 1.5 * ac.m_per_px)
        free_axes.add(triangulate.depth_axis(ac))
    mpp = float(np.mean([it.calib.m_per_px for it in items]))
    ref_m = float(np.ptp(np.asarray(pose0.params["outline"], float), axis=0).max()) if ptype == "plate" else None
    scale, lo_s, hi_s = _scales(ptype, st0, mpp, ref_m)
    frozen = _lock_mask(ptype, lay, st0, locked, measured, free_axes)
    x0 = _pack(st0, lay)
    s_vec, lo_v, hi_v = _pack(scale, lay), _pack(lo_s, lay), _pack(hi_s, lay)
    j_off = _offset(lay, joint_point) if joint_point in dict(lay) else None    # the pin bounds the keypoint, it does not freeze it
    if j_off is not None:
        lo_v[j_off:j_off + 3], hi_v[j_off:j_off + 3] = x0[j_off:j_off + 3] - joint_slack, x0[j_off:j_off + 3] + joint_slack
    free = ~frozen
    prior_x = None
    if mirror_sibling is not None:                          # (e): pull toward the sibling's mirror image
        sib_frame = cone_frame(mirror_sibling.keypoints["base"], mirror_sibling.keypoints["tip"]) if ptype == "cone" else None
        sib_st = state_from_solution(ptype, mirror_sibling, params, taper, sib_frame)
        if {n: np.shape(v) for n, v in sib_st.items()} == {n: np.shape(v) for n, v in st0.items()}:
            prior_x = _pack(mirror_state(sib_st, sib_frame, frame), lay)
        else:
            log.warning("%s: the symmetric partner has a different parameter layout (strand points); no symmetry term", part_id)
    setup = FitSetup(ptype, params, face_calib, frame, lay, items, [t.mirrored for t in targets], w_inside, contact)

    if ptype == "strand":
        pts0 = strand_points(st0)
        tang0 = np.gradient(pts0, axis=0)
        tang0 /= np.maximum(np.linalg.norm(tang0, axis=1, keepdims=True), 1e-9)
        slide_scale = float(scale["p1"][0])

    reg_w = np.concatenate([np.full(k, REG_WEIGHTS.get(n, REG_WEIGHT)) for n, k in lay])[free]

    def make_objective(k: int, coarse: bool, ss: int = 1) -> Callable[[np.ndarray], float]:
        def f(zf: np.ndarray) -> float:
            x = x0.copy()
            x[free] = x0[free] + zf * s_vec[free]
            loss, _ = evaluate(setup, _unpack(x, lay), k, coarse, ss)
            dz = (x[free] - x0[free]) / s_vec[free]
            loss += float(reg_w @ dz ** 2)
            if j_off is not None:
                loss += JOINT_WEIGHT * (max(float(np.linalg.norm(x[j_off:j_off + 3] - x0[j_off:j_off + 3])) - joint_slack, 0.0) / joint_slack) ** 2
            if ptype == "strand":
                pts = strand_points(_unpack(x, lay))
                slide = np.einsum("ij,ij->i", (pts - pts0)[1:-1], tang0[1:-1]) / slide_scale
                loss += SMOOTH_WEIGHT * strand_kink(pts) + TANGENT_WEIGHT * float(slide @ slide)
            if prior_x is not None:
                over = np.maximum(np.abs(x[free] - prior_x[free]) - mirror_tol, 0.0) / s_vec[free]
                loss += mirror_weight * float(np.mean(over ** 2))
            return loss
        return f

    lo_z, hi_z = (lo_v - x0)[free] / s_vec[free], (hi_v - x0)[free] / s_vec[free]
    wmax = max(max(it.window[2] - it.window[0], it.window[3] - it.window[1]) for it in items)
    k_coarse = max(1, -(-wmax // COARSE_PX))
    z = np.zeros(int(free.sum()))
    evals = 0
    if free.any():
        stages, polish = (STRAND_STAGES, STRAND_POLISH) if ptype == "strand" else (STAGES, POLISH)
        ends = [t0 + f * time_budget for f in STRAND_STAGE_ENDS] if ptype == "strand" else [None] * (len(stages) + 1)
        for (coarse, sigma, budget, ss), end in zip(stages, ends):
            z, _, n_ev = cma_es(make_objective(k_coarse if coarse else 1, coarse, ss), z, sigma, lo_z, hi_z,
                                max_evals=budget, seed=seed, deadline=end)
            evals += n_ev
        if polish is not None:
            z, _, n_ev = compass_search(make_objective(1, False, polish[0]), z, polish[1], lo_z, hi_z, max_evals=polish[2],
                                        deadline=ends[-1])
            evals += n_ev
    x_fit = x0.copy()
    x_fit[free] = x0[free] + z * s_vec[free]
    st_fit = _unpack(x_fit, lay)
    f_fine = make_objective(1, False)
    loss_init, loss_fit = f_fine(np.zeros(int(free.sum()))), f_fine(z)
    if loss_fit > loss_init:                                    # never return something worse than the triangulation
        st_fit, x_fit, loss_fit = st0, x0, loss_init
    new = solution_from_state(ptype, st_fit, sol, params, face_calib, [it.view for it in items], x_fit != x0, lay, frame)
    names_free = [(f"{n}.{'xyz'[i]}" if k == 3 else f"{n}.{i}" if k > 1 else n) for n, k in lay for i in range(k)
                  if free[_offset(lay, n) + i]]
    report = {"fitted": True, "loss_init": round(loss_init, 5), "loss_fit": round(loss_fit, 5), "evals": evals,
              "seconds": round(time.perf_counter() - t0, 2), "free": names_free,
              "locked": [n for n in dict.fromkeys(n for n, _ in lay) if not any(free[_offset(lay, n):_offset(lay, n) + dict(lay)[n]])],
              "joint_with": [t.part_id for t in targets if t.mirrored], "coarse_k": k_coarse}
    if prior_x is not None:
        report["mirror_dev_max"] = round(float(np.abs(x_fit - prior_x)[free].max(initial=0.0)), 5)
        report["mirror_tol"] = mirror_tol
    return new, report


def _offset(lay: Sequence[tuple[str, int]], name: str) -> int:
    off = 0
    for n, k in lay:
        if n == name:
            return off
        off += k
    raise KeyError(name)


def solution_from_state(ptype: str, st: State, old: PartSolution, params: Mapping[str, Any],
                        face_calib: Optional[ViewCalibration], views: Sequence[str], changed: Optional[np.ndarray] = None,
                        lay: Optional[Sequence[tuple[str, int]]] = None, frame: Optional[np.ndarray] = None) -> PartSolution:
    """A :class:`PartSolution` for a fitted state (what ``solve_part`` would return for these measurements): keypoints,
    scalars (read by ``mirror_of`` partners) and the pose / primitive parameters."""
    pose = pose_from_state(ptype, st, params, face_calib, frame)
    kp = {n: np.asarray(st[n], float).copy() for n in st if is_point(n)}
    if ptype == "cone":
        kp["mid"] = cone_mid(st, frame)
    sc = dict(old.scalars)
    if ptype == "cone":
        sc["radius"] = (float(st["rb"][0]), float(st["rt"][0]))
    elif ptype in ("capsule", "cylinder"):
        sc["radius"] = float(pose.size[0] / 2.0)
        if "tip_radius" in pose.params:
            sc["radius_tip"] = pose.params["tip_radius"]
        else:
            sc.pop("radius_tip", None)
    elif ptype == "blade":
        sc.update(width=float(st["width"][0]), thickness=float(st["thickness"][0]), width_dir=np.asarray(pose.params["width_dir"]))
    elif ptype == "plate":
        sc.update(thickness=float(st["thickness"][0]), normal=_unit(st["normal"]), scale=float(st["scale"][0]),
                  half_width=plate.outline_half_width(pose.params["outline"]))
        reach = float(np.linalg.norm(np.asarray(old.keypoints["tip"], float) - np.asarray(old.keypoints["base"], float))) / max(float(old.scalars.get("scale", 1.0)), 1e-9)
        kp = {"base": np.asarray(st["center"], float).copy(),
              "tip": np.asarray(st["center"], float) + reach * float(st["scale"][0]) * np.asarray(pose.params["axis"], float)}
    elif ptype == "disc":
        sc.update(radius=float(st["radius"][0]), thickness=float(st["thickness"][0]), normal=_unit(st["normal"]))
    elif ptype == "sphere":
        sc["semi"] = [float(s) for s in st["semi"]]
    elif ptype == "strand":
        sc["radii"] = [float(r) for r in pose.params["radii"]]
    sources = {n: dict(per) for n, per in old.sources.items()}
    if lay is not None and changed is not None:
        for name, k in lay:
            if name in kp:
                off = _offset(lay, name)
                for ax in range(3):
                    if changed[off + ax] and any(v for v in views):
                        sources.setdefault(name, {})["xyz"[ax]] = "fit:" + "+".join(views)
    geo = dict(pose.params)
    keep = {k: v for k, v in old.params.items() if k not in geo}
    return PartSolution(ptype, kp, sc, [float(v) for v in pose.location], [float(v) for v in pose.rotation_deg],
                        None if pose.size is None else [float(v) for v in pose.size], {**keep, **triangulate._py(geo)},
                        sources, old.spread_m, list(old.warnings))


# --------------------------------------------------------------------------- report
def part_iou(ptype: str, sol: PartSolution, params: Mapping[str, Any], calibs: Mapping[str, ViewCalibration],
             silhouettes: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray],
             others: Optional[Mapping[str, np.ndarray]] = None) -> dict[str, dict[str, float]]:
    """Per view where ``labels`` has the part: IoU of the part as the builders make it against its label (``iou``), the
    same IoU ignoring the part's pixels over the label of OTHER parts (``iou_vis``: neighbouring primitives overlap at
    joints and a part may sit behind another, so spill there is not an error; ``others`` = union of their labels), the
    share of the label it covers and the share of the part that leaves the reference silhouette (full resolution)."""
    shape = shape_of(ptype, pose_from_solution(sol, params))
    out: dict[str, dict[str, float]] = {}
    for view, label in labels.items():
        if view not in calibs or int(label.sum()) < MIN_LABEL_PX:
            continue
        r = render(shape, calibs[view])
        lab = np.asarray(label, bool)
        sil = np.asarray(silhouettes[view], bool) if view in silhouettes else np.ones_like(lab)
        skip = np.asarray(others[view], bool) & ~lab if others is not None and view in others else np.zeros_like(lab)
        out[view] = {"iou": round(iou(r, lab), 4), "iou_vis": round(iou(r & ~skip, lab), 4), "covered": round(float((r & lab).sum()) / max(int(lab.sum()), 1), 4),
                     "spill": round(float((r & ~sil).sum()) / max(int(r.sum()), 1), 4), "label_px": int(lab.sum())}
    return out


def format_line(part_id: str, report: Mapping[str, Any]) -> str:
    """One terminal line per part (per view: IoU ignoring other parts' labels / strict IoU)."""
    views = "  ".join(f"{v} {d['iou_vis']:.2f}/{d['iou']:.2f}" + (f" (was {report['iou_init'][v]['iou']:.2f})" if v in report.get("iou_init", {}) else "")
                      for v, d in report["iou"].items())
    how = (f"fitted in {report['seconds']:.1f}s, {report['evals']} evals" if report.get("fitted")
           else report.get("note", "not fitted"))
    return f"partfit {part_id:<14} IoU {views or 'no label'}   [{how}]"
