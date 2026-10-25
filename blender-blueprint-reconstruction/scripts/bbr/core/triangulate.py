"""Anchor-based part placement (numpy only): image anchors -> world pose, no hand-typed 3D values.

A part is described by pixel anchors per view (``params.anchors_px = {view: {name: [x, y], ..., radius_px: ...}}``).
Each orthographic view fixes the two world axes of its image plane (:func:`pixel_to_plane`); a keypoint seen in two views
is triangulated per axis (mean of the estimates, spread reported: front/back give x and z, side/right give y and z, so
z is measured twice). A coordinate no view measures (a part visible in one view) comes from, in this order:

1. ``params.depth_m``: explicit override (a number, or ``{keypoint: value}``), marked inferred;
2. ``params.mirror_of``: the same keypoint of a sibling part mirrored across X = 0;
3. ``params.attach_to`` (+ ``attach_side``, ``standoff``): contact with a target surface. The target's depth buffer in
   the view where the part is seen gives the surface height under the part's footprint; the part sits on its
   maximum (so it never sinks in) plus ``standoff`` plus its own half thickness, on the camera side of that view.
4. the part's OTHER keypoints: a point hidden in a view is simply omitted there (``anchors_px.side`` without ``tip``);
   an axis that no view measures for it is taken from the part's own measured keypoints, with a warning, and the part
   is marked inferred.

``params.joint = {"part": id, "at": "tip"|"base", "self": "base"|"tip"}`` pins this part's ``self`` keypoint to the
partner's ``at`` keypoint (a handle's base on the head's base), whatever the pixels say; the keypoint needs no anchor.

Then the pose and the primitive's parameters follow from the 3D keypoints (:func:`solve_part`); the Blender side only
passes them to the existing builders. Units: meters.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .. import BBRError
from . import partgeom, plate
from .calibration import ViewCalibration
from .conventions import AXES, DEFAULT_CAMERA_AXIS, normalize_axis, view_frame

AXIS_LETTERS = "xyz"
AGREE_WARN_M = 0.005          # views disagreeing by more than this on a coordinate are reported
MIN_CONTACT_COVERAGE = 0.5    # share of the footprint that must lie on the target surface
POINTS: dict[str, tuple[str, ...]] = {
    "cone": ("base", "mid", "tip"), "capsule": ("base", "tip"), "cylinder": ("base", "tip"),
    "blade": ("base", "tip"), "plate": ("base", "tip"), "disc": ("center",), "sphere": ("center",), "gem": ("base", "tip"),
    "strand": ()}      # strand: p0 (base) ... pN-1 (tip), N >= 3, named in the anchors; plate: root edge (base) and tip
REQUIRED: dict[str, tuple[str, ...]] = {
    "cone": ("base", "tip"), "capsule": ("base", "tip"), "cylinder": ("base", "tip"), "blade": ("base", "tip"),
    "plate": ("base", "tip"), "disc": ("center",), "sphere": ("center",), "gem": ("base", "tip"), "strand": ()}
SCALAR_KEYS: dict[str, str] = {"cone": "radius_px", "capsule": "radius_px", "cylinder": "radius_px",
                               "blade": "width_px", "plate": "", "disc": "radius_px", "sphere": "radius_px",
                               "gem": "radius_px", "strand": "radius_px"}
STRAND_POINT = re.compile(r"p\d+")
MIN_STRAND_POINTS = 3
ANCHOR_TYPES = tuple(POINTS)
JOINT_KEYS = ("base", "tip", "center")
JOINT_WARN_M = 0.01           # own anchors this far from the joint point are reported (the joint wins)
_ORDER = ("base", "tip", "center", "mid")      # mid last: it may be interpolated from base and tip
CENTRE_DEFAULT_AXES = (0, 1)   # x, y: the model origin is the body axis / bbox centre, so 0 is the body's centre line


def _strand_order(names) -> list[str]:
    return sorted(names, key=lambda k: int(k[1:]))


@dataclass
class PartSolution:
    """World pose / primitive parameters of an anchored part plus how every coordinate was obtained."""

    type: str
    keypoints: dict[str, np.ndarray]
    scalars: dict[str, Any]
    location: list[float]
    rotation_deg: list[float]
    size: Optional[list[float]]
    params: dict[str, Any]
    sources: dict[str, dict[str, str]] = field(default_factory=dict)
    spread_m: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def inferred(self) -> bool:
        """True when any coordinate did not come from a view measurement."""
        return any(not s.startswith(("views:", "fit:", "joint:")) for per in self.sources.values() for s in per.values())

    def report(self) -> dict[str, Any]:
        """JSON-able summary (keypoints, per-axis source, view disagreement in mm, warnings)."""
        return {"type": self.type, "inferred": self.inferred, "warnings": list(self.warnings),
                "clearance_radius": round(clearance_radius(self.type, self.scalars), 5),
                "keypoints": {k: [round(float(c), 5) for c in v] for k, v in self.keypoints.items()},
                "sources": self.sources,
                "view_spread_mm": {k: {a: round(v * 1000.0, 2) for a, v in d.items()} for k, d in self.spread_m.items()},
                "location": [round(float(c), 5) for c in self.location]}


def clearance_radius(ptype: str, sc: Mapping[str, Any]) -> float:
    """Meters from the keypoint polyline to the part's surface, for the gap check: the radius of a tube, half the
    thickness of a flat blade, the smallest semi-axis of a sphere, the radius of a disc."""
    if ptype == "cone":
        return float(max(sc["radius"]))
    if ptype in ("capsule", "cylinder", "gem"):
        return float(max(sc["radius"], sc.get("radius_tip", 0.0)))
    if ptype == "blade":
        return float(sc["thickness"]) / 2.0
    if ptype == "plate":                       # the polyline is the root -> tip axis: the plate reaches half its width to the side
        return float(max(sc["thickness"] / 2.0, sc.get("half_width", 0.0)))
    if ptype == "disc":
        return float(sc["radius"])
    if ptype == "strand":
        return float(max(sc["radii"]))
    return float(min(sc["semi"]))


# --------------------------------------------------------------------------- pixel -> world
def depth_axis(calib: ViewCalibration) -> int:
    """Index (0..2) of the world axis a view cannot measure (its viewing axis)."""
    return AXIS_LETTERS.index(normalize_axis(calib.camera_axis)[1].lower())


def pixel_to_plane(calib: ViewCalibration, px: Sequence[float]) -> np.ndarray:
    """Pixel ``[x, y]`` -> world ``(3,)``: the two in-plane coordinates, ``nan`` for the depth axis."""
    out = calib.px_to_plane(np.asarray(px, dtype=float).reshape(1, 2))[0].copy()
    out[depth_axis(calib)] = np.nan
    return out


def triangulate_point(calibs: Mapping[str, ViewCalibration], obs: Mapping[str, Sequence[float]]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
    """World point from one pixel per view: ``(point (3,) with nan where no view measures, spread (3,) = max - min of
    the estimates in meters, count (3,), views per axis)``. x from front/back, y from side/right, z from both."""
    est: list[list[float]] = [[], [], []]
    used: list[list[str]] = [[], [], []]
    for view, px in obs.items():
        w = pixel_to_plane(calibs[view], px)
        for i in range(3):
            if not np.isnan(w[i]):
                est[i].append(float(w[i]))
                used[i].append(view)
    point = np.array([np.mean(e) if e else np.nan for e in est])
    spread = np.array([max(e) - min(e) if len(e) > 1 else 0.0 for e in est])
    return point, spread, np.array([len(e) for e in est]), used


def surface_height(zbuf: np.ndarray, px: Sequence[float], footprint_px: float) -> tuple[Optional[float], float]:
    """Highest surface height (along the view's camera axis) in the disc of ``footprint_px`` around ``px`` of a depth
    buffer (``-inf`` = no surface) and the share of the footprint that has a surface. ``(None, 0)`` when none does."""
    h, w = zbuf.shape
    cx, cy, r = float(px[0]), float(px[1]), max(float(footprint_px), 1.0)
    x0, x1 = max(int(math.floor(cx - r)), 0), min(int(math.ceil(cx + r)) + 1, w)
    y0, y1 = max(int(math.floor(cy - r)), 0), min(int(math.ceil(cy + r)) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= r * r
    vals = zbuf[y0:y1, x0:x1][inside]
    finite = np.isfinite(vals)
    if not finite.any():
        return None, 0.0
    return float(vals[finite].max()), float(finite.mean())


def surface_plane(zbuf: np.ndarray, calib: ViewCalibration, px: Sequence[float], footprint_px: float
                  ) -> Optional[tuple[float, float, float, float]]:
    """Plane ``height = h0 + a * right + b * up`` (meters along the view's camera axis; ``right`` / ``up`` = meters from
    the footprint centre along the image axes) least-squares fitted to the surface under the footprint, lifted by the
    largest residual so that it lies on top of every surface sample: ``(h0, a, b, coverage)``; None without surface."""
    h, w = zbuf.shape
    cx, cy, r = float(px[0]), float(px[1]), max(float(footprint_px), 1.0)
    x0, x1 = max(int(math.floor(cx - r)), 0), min(int(math.ceil(cx + r)) + 1, w)
    y0, y1 = max(int(math.floor(cy - r)), 0), min(int(math.ceil(cy + r)) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = ((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= r * r) & np.isfinite(zbuf[y0:y1, x0:x1])
    if sel.sum() < 6:
        return None
    u = (xx[sel] + 0.5 - cx) * calib.m_per_px           # right
    v = -(yy[sel] + 0.5 - cy) * calib.m_per_px          # up
    z = zbuf[y0:y1, x0:x1][sel]
    design = np.stack([np.ones_like(u), u, v], axis=1)
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    coef[0] += float((z - design @ coef).max())
    return float(coef[0]), float(coef[1]), float(coef[2]), float(sel.sum() / (math.pi * r * r))


# --------------------------------------------------------------------------- anchors
def _parse_anchors(part_id: str, ptype: str, anchors: Mapping[str, Any], calibs: Mapping[str, ViewCalibration]
                   ) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    """``(keypoint -> {view: px}, view -> scalar anchor value)`` with names and shapes validated."""
    obs: dict[str, dict[str, list[float]]] = {}
    scalars: dict[str, Any] = {}
    scalar_key = SCALAR_KEYS[ptype]
    for view, entry in anchors.items():
        if view not in calibs:
            raise BBRError(f"{part_id}: anchors_px view {view!r} is not calibrated; views: {sorted(calibs)}")
        for key, val in entry.items():
            if key in POINTS[ptype] or (ptype == "strand" and STRAND_POINT.fullmatch(key)):
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise BBRError(f"{part_id}: anchors_px.{view}.{key} must be [x, y] pixels, got {val!r}")
                obs.setdefault(key, {})[view] = [float(val[0]), float(val[1])]
            elif scalar_key and key == scalar_key:
                scalars[view] = val
            else:
                raise BBRError(f"{part_id}: a {ptype} takes anchors "
                               f"{['p0', 'p1', '...'] if ptype == 'strand' else list(POINTS[ptype])}"
                               f"{' + ' + scalar_key if scalar_key else ''}, got {key!r} (view {view})")
    return obs, scalars


def _view_value(calib: ViewCalibration, v: Any) -> Any:
    """px value (number or list) -> meters."""
    if isinstance(v, (list, tuple)):
        return [float(x) * calib.m_per_px for x in v]
    return float(v) * calib.m_per_px


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals))


def _attach_view(part_id: str, params: Mapping[str, Any], calibs: Mapping[str, ViewCalibration],
                 anchor_views: Sequence[str]) -> str:
    """View used for the contact solve: ``attach_side`` (a side name such as back / a view key); else the only anchor view."""
    side = params.get("attach_side")
    if side is None:
        if len(anchor_views) == 1:
            return anchor_views[0]
        raise BBRError(f"{part_id}: attach_to needs attach_side (back, front, left, right, top, bottom or a view key)")
    if side in calibs:
        return side
    axis = DEFAULT_CAMERA_AXIS.get(str(side))
    matches = [v for v, c in calibs.items() if axis and normalize_axis(c.camera_axis) == axis]
    matches = [v for v in matches if v in anchor_views] or matches
    if not matches:
        raise BBRError(f"{part_id}: attach_side {side!r} matches no calibrated view; views: {sorted(calibs)}")
    return matches[0]


def _resample(vals: Sequence[float], n: int) -> np.ndarray:
    """``vals`` (one per point, or just base / tip) as ``n`` values: linear interpolation along the strand."""
    v = np.asarray(vals, dtype=float).ravel()
    return v if len(v) == n else np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(v)), v)


def _strand_radii(part_id: str, views: Mapping[str, Any], calibs: Mapping[str, ViewCalibration], n: int) -> list[float]:
    """Radius (m) at each of the ``n`` points of a strand: ``radius_px`` per view as a number, a list (one per point, or
    base / tip), or ``{"base": r, "tip": r}``; the views are averaged."""
    per = []
    for v, val in views.items():
        if isinstance(val, Mapping):
            if not {"base", "tip"} <= set(val):
                raise BBRError(f"{part_id}: radius_px of a strand is a number, a list or {{\"base\": r, \"tip\": r}}, got {val!r}")
            val = [val["base"], val["tip"]]
        per.append(_resample(np.atleast_1d(_view_value(calibs[v], val)), n))
    return [float(x) for x in np.mean(per, axis=0)]


def _scalars(part_id: str, ptype: str, views: Mapping[str, Any], params: Mapping[str, Any],
             calibs: Mapping[str, ViewCalibration], sib: Optional[PartSolution], face_pref: Optional[str],
             npts: int = 0) -> dict[str, Any]:
    """Sizes in meters from the per-view scalar anchors (``radius_px`` / ``width_px``), else the mirrored sibling's."""
    out: dict[str, Any] = {}
    if ptype == "plate":                        # the outline comes from the label mask (see _plate_geometry), no scalar anchor
        thick = params.get("thickness", sib.scalars.get("thickness") if sib else None)
        if thick is None:
            raise BBRError(f"{part_id}: a plate needs params.thickness (meters)")
        out["thickness"] = float(thick)
        face = params.get("face_view") or (face_pref if face_pref in views else None) or (sib.scalars.get("face_view") if sib else None)
        if face:
            out["face_view"] = str(face)
        return out
    if ptype in ("disc", "blade"):
        thick = params.get("thickness", sib.scalars.get("thickness") if sib else None)
        if thick is None:
            raise BBRError(f"{part_id}: a {ptype} needs params.thickness (meters)")
        out["thickness"] = float(thick)
    if views and ptype == "strand":
        out["radii"] = _strand_radii(part_id, views, calibs, npts)
    elif views:
        if ptype in ("disc", "blade"):          # measured face-on in ONE view (an edge-on view would give the thickness)
            face = face_pref if face_pref in views else next(iter(views))
            out["face_view"] = face
            out["radius" if ptype == "disc" else "width"] = _view_value(calibs[face], views[face])
        elif ptype == "cone":
            pairs = []
            for v, val in views.items():
                m = _view_value(calibs[v], val)
                pairs.append(m if isinstance(m, list) else [m, m])
            out["radius"] = (_mean([p[0] for p in pairs]), _mean([p[1] for p in pairs]))
        elif ptype == "sphere":
            per: list[list[float]] = [[], [], []]
            for v, val in views.items():
                m = _view_value(calibs[v], val)
                rh, rv = (m if isinstance(m, list) else [m, m])[:2]
                h_i, v_i = [i for i in range(3) if i != depth_axis(calibs[v])]   # horizontal, vertical world axes
                per[h_i].append(rh)
                per[v_i].append(rv)
            known = [_mean(p) for p in per if p]
            out["semi"] = [_mean(p) if p else _mean(known) for p in per]     # unmeasured axis = mean of the others
        else:
            out["radius"] = _mean([_view_value(calibs[v], val) for v, val in views.items()])
    elif sib is not None:
        for key in ("radius", "radius_tip", "width", "semi", "radii"):
            if key in sib.scalars:
                out[key] = sib.scalars[key]
    need = {"disc": "radius", "blade": "width", "sphere": "semi", "strand": "radii"}.get(ptype, "radius")
    if need not in out:
        raise BBRError(f"{part_id}: anchors_px needs {SCALAR_KEYS[ptype]} in at least one view")
    return out


def _extent(ptype: str, sc: Mapping[str, Any], name: str, d: int) -> tuple[float, float]:
    """``(half thickness, footprint radius)`` in meters of a keypoint along depth axis ``d`` (for the contact solve)."""
    if ptype == "cone":
        rb, rt = sc["radius"]
        r = {"base": rb, "tip": rt}.get(name, (rb + rt) / 2.0)
        return r, r
    if ptype in ("capsule", "cylinder", "gem"):
        return sc["radius"], sc["radius"]
    if ptype == "strand":
        r = float(sc["radii"][0])
        return r, r
    if ptype == "blade":
        return sc["thickness"] / 2.0, sc["width"] / 2.0
    if ptype == "plate":
        return sc["thickness"] / 2.0, max(sc.get("half_width", 0.0), sc["thickness"] / 2.0)
    if ptype == "disc":
        return sc["thickness"] / 2.0, sc["radius"]
    return sc["semi"][d], _mean(sc["semi"])


# --------------------------------------------------------------------------- solve
def solve_part(part_id: str, ptype: str, params: Mapping[str, Any], calibs: Mapping[str, ViewCalibration],
               depth_for: Optional[Callable[[str], np.ndarray]] = None,
               sibling_for: Optional[Callable[[str], PartSolution]] = None,
               joint_for: Optional[Callable[[str], PartSolution]] = None,
               auto_sibling: Optional[Callable[[], Optional[PartSolution]]] = None,
               auto_depth_for: Optional[Callable[[str], Optional[np.ndarray]]] = None,
               auto_target: str = "body",
               label_for: Optional[Callable[[str], Optional[np.ndarray]]] = None) -> PartSolution:
    """Pose and primitive parameters of one anchored part.

    ``params`` is the component's ``params`` (``anchors_px``, ``mirror_of``, ``attach_to``, ``attach_side``,
    ``standoff``, ``depth_m``, ``thickness``, ``roll``, ``joint``). ``depth_for(view)`` returns the attach target's depth
    buffer in that view (see :mod:`zbuffer`); ``sibling_for(id)`` the solution of the ``mirror_of`` part, ``joint_for(id)``
    that of the ``joint`` partner.

    A coordinate no view measures and no explicit ``depth_m`` / ``mirror_of`` / ``attach_to`` gives is never left at a raw
    default; the fallbacks, in order: ``auto_sibling()`` = the ``_l`` / ``_r`` partner's solution (same keypoint mirrored
    across X = 0; None when there is no solved partner), ``auto_depth_for(view)`` = depth buffer of the nearest body / hull
    (``auto_target``) in a view that sees the keypoint: the part sits on its surface when at least half of the footprint lies
    over it, the part's own other measured keypoints, finally the body centre line (x = y = 0, the model origin). Each
    one is a warning and marks the part inferred.

    ``label_for(view)`` returns the part's label mask in that view (a ``plate``'s outline comes from it), None without one.
    """
    if ptype not in POINTS:
        raise BBRError(f"{part_id}: type {ptype!r} has no anchor placement; supported: {list(POINTS)}")
    obs, view_scalars = _parse_anchors(part_id, ptype, params.get("anchors_px") or {}, calibs)
    mirror_id = params.get("mirror_of")
    sib: Optional[PartSolution] = None
    if mirror_id:
        if sibling_for is None:
            raise BBRError(f"{part_id}: mirror_of {mirror_id!r} needs the sibling's solution")
        sib = sibling_for(str(mirror_id))
        if sib.type != ptype:
            raise BBRError(f"{part_id}: mirror_of {mirror_id!r} is a {sib.type}, not a {ptype}")
    if not obs and sib is None:
        raise BBRError(f"{part_id}: needs anchors_px or mirror_of")
    joint = _parse_joint(part_id, ptype, params)
    joint_sol = None
    if joint:
        if joint_for is None:
            raise BBRError(f"{part_id}: joint {joint[0]!r} needs the partner's solution")
        joint_sol = joint_for(joint[0])
        if joint[1] not in joint_sol.keypoints:
            raise BBRError(f"{part_id}: joint.at {joint[1]!r} is not a keypoint of {joint[0]} ({sorted(joint_sol.keypoints)})")
    if ptype == "strand":
        names = _strand_order(set(obs) | (set(sib.keypoints) if sib is not None and not obs else set()))
        if len(names) < MIN_STRAND_POINTS:
            raise BBRError(f"{part_id}: a strand needs at least {MIN_STRAND_POINTS} anchor points p0, p1, p2, ... (got {len(names)})")
    else:
        names = sorted((n for n in POINTS[ptype] if n in obs or (not obs and n in sib.keypoints)        # type: ignore[union-attr]
                        or (joint and n == joint[2])), key=_ORDER.index)
    for n in REQUIRED[ptype]:
        if n not in names:
            raise BBRError(f"{part_id}: a {ptype} needs the anchor {n!r}")
    attach = params.get("attach_to")
    attach_view = _attach_view(part_id, params, calibs, sorted({v for d in obs.values() for v in d})) if attach else None
    sc = _scalars(part_id, ptype, view_scalars, params, calibs, sib, attach_view, len(names))
    standoff = float(params.get("standoff", 0.0))
    depth_m = params.get("depth_m")
    pts: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, str]] = {}
    spreads: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    tol = float(params.get("agree_tol_m", AGREE_WARN_M))
    for name in names:
        if joint_sol is not None and name == joint[2]:                            # type: ignore[index]
            pt = joint_sol.keypoints[joint[1]].copy()                              # type: ignore[index]
            if name in obs:
                est, _, cnt, _ = triangulate_point(calibs, obs[name])
                off = float(np.abs(est - pt)[cnt > 0].max(initial=0.0))
                if off > JOINT_WARN_M:
                    warnings.append(f"{part_id}.{name}: its anchors are {off * 1000:.0f} mm from the joint with "
                                    f"{joint[0]}.{joint[1]}; the joint wins")     # type: ignore[index]
            pts[name], spreads[name] = pt, {}
            sources[name] = {a: f"joint:{joint[0]}" for a in AXIS_LETTERS}        # type: ignore[index]
            continue
        if name in obs:
            pt, spread, count, used = triangulate_point(calibs, obs[name])
        else:
            pt, spread, count, used = np.full(3, np.nan), np.zeros(3), np.zeros(3, int), [[], [], []]
        src = {AXIS_LETTERS[i]: "views:" + "+".join(used[i]) for i in range(3) if count[i]}
        spreads[name] = {AXIS_LETTERS[i]: float(spread[i]) for i in range(3) if count[i] > 1}
        for ax, s in spreads[name].items():
            if s > tol:
                warnings.append(f"{part_id}.{name}: views disagree on {ax} by {s * 1000:.1f} mm ({src[ax]}); mean used")
        missing = [i for i in range(3) if count[i] == 0]
        for i in missing:
            ax = AXIS_LETTERS[i]
            dm = depth_m.get(name) if isinstance(depth_m, Mapping) else depth_m
            if dm is not None and len(missing) == 1:
                pt[i], src[ax] = float(dm), "depth_m"
            elif sib is not None and name in sib.keypoints:
                m = sib.keypoints[name]
                pt[i], src[ax] = (-m[0] if i == 0 else m[i]), f"mirror:{mirror_id}"
            elif attach and i == depth_axis(calibs[attach_view]):                 # type: ignore[index]
                if name not in obs or attach_view not in obs[name]:
                    raise BBRError(f"{part_id}: attach_to needs anchors_px.{attach_view}.{name} (the pixel to drop on the surface)")
                if depth_for is None:
                    raise BBRError(f"{part_id}: attach_to {attach!r} needs the target's depth buffer")
                calib = calibs[attach_view]                                       # type: ignore[index]
                half, foot = _extent(ptype, sc, name, i)
                zbuf, tilt = depth_for(attach_view), 1.0                          # type: ignore[arg-type]
                height, cover = surface_height(zbuf, obs[name][attach_view], foot / calib.m_per_px)
                plane = surface_plane(zbuf, calib, obs[name][attach_view], foot / calib.m_per_px)                     if ptype == "disc" and params.get("tilt", True) else None
                if plane is not None:       # a disc lies flat on a sloped surface: tilt its normal to the fitted plane
                    height, (ga, gb) = plane[0], plane[1:3]
                    frame = view_frame(calib.camera_axis)
                    normal = AXES[normalize_axis(calib.camera_axis)] - ga * frame.right - gb * frame.up
                    sc["normal"] = normal / np.linalg.norm(normal)
                    tilt = float(sc["normal"] @ AXES[normalize_axis(calib.camera_axis)])
                if height is None:
                    raise BBRError(f"{part_id}: {attach!r} has no surface under {name} in view {attach_view}: "
                                   f"check the anchor pixel / attach_side")
                if cover < MIN_CONTACT_COVERAGE:
                    warnings.append(f"{part_id}.{name}: only {cover * 100:.0f}% of its footprint lies over {attach!r} in view {attach_view}")
                sign = float(AXES[normalize_axis(calib.camera_axis)][i])
                pt[i], src[ax] = sign * (height + (standoff + half) * tilt), f"contact:{attach}@{attach_view}"
            else:
                _auto_axis(part_id, ptype, sc, name, i, pt, src, obs, calibs, warnings, standoff, auto_sibling, auto_depth_for,
                           auto_target)
        pts[name] = pt
        sources[name] = src
    for name, pt in pts.items():            # an axis hidden in every view of this keypoint: the part's other measured points
        if name == "mid":
            continue
        for i in np.flatnonzero(np.isnan(pt)):
            ax = AXIS_LETTERS[i]
            donors = [n for n, q in pts.items() if n not in (name, "mid") and not np.isnan(q[i])
                      and sources[n].get(ax, "").startswith(("views:", "joint:"))]
            if donors:
                pt[i], sources[name][ax] = _mean([pts[n][i] for n in donors]), "inferred:" + "+".join(donors)
                warnings.append(f"{part_id}.{name}: no view measures {ax} (hidden): taken from {', '.join(donors)}; "
                                f"give depth_m, mirror_of or attach_to to measure it")
    for name, pt in pts.items():            # last resort: the body centre line (x, y); z is always measured by an upright view
        for i in CENTRE_DEFAULT_AXES:
            if np.isnan(pt[i]) and name != "mid":
                pt[i], sources[name][AXIS_LETTERS[i]] = 0.0, f"default:{auto_target}_centre"
                warnings.append(f"{part_id}.{name}: no view measures {AXIS_LETTERS[i]} and no partner or surface gives it: "
                                f"set to the {auto_target} centre line (0); give depth_m, mirror_of or attach_to to measure it")
    if "mid" in pts and np.isnan(pts["mid"]).any():                               # between base and tip
        b, t, m = pts["base"], pts["tip"], pts["mid"]
        known = ~np.isnan(m)
        d = (t - b)[known]
        frac = float(np.clip(((m - b)[known] @ d) / max(float(d @ d), 1e-12), 0.0, 1.0))
        for i in np.flatnonzero(~known):
            m[i] = b[i] + frac * (t[i] - b[i])
            sources["mid"][AXIS_LETTERS[i]] = "interpolated"
    for name, pt in pts.items():
        gaps = [AXIS_LETTERS[i] for i in range(3) if np.isnan(pt[i])]
        if gaps:
            seen = sorted(obs.get(name, {}))
            raise BBRError(f"{part_id}.{name}: no {'/'.join(gaps)} coordinate (seen only in {seen}); add the anchor in a second "
                           f"view, or attach_to (+ attach_side), mirror_of, or depth_m (a point hidden in a view is omitted there, never guessed)")
    loc, rot, size, geo = _geometry(part_id, ptype, pts, sc, params, calibs, sib, label_for)
    return PartSolution(ptype, pts, sc, _py(loc), _py(rot), _py(size), _py(geo), sources, spreads, warnings)


def _auto_axis(part_id: str, ptype: str, sc: Mapping[str, Any], name: str, i: int, pt: np.ndarray, src: dict[str, str],
               obs: Mapping[str, Mapping[str, Sequence[float]]], calibs: Mapping[str, ViewCalibration], warnings: list[str],
               standoff: float, auto_sibling: Optional[Callable[[], Optional[PartSolution]]],
               auto_depth_for: Optional[Callable[[str], Optional[np.ndarray]]], auto_target: str) -> None:
    """Fill the unmeasured coordinate ``i`` of keypoint ``name`` of ``pt`` from the automatic partner, else from contact with
    the nearest body surface (:func:`solve_part`); leaves ``nan`` when neither applies."""
    ax = AXIS_LETTERS[i]
    partner = auto_sibling() if auto_sibling is not None else None
    if (partner is not None and name in partner.keypoints
            and not partner.sources.get(name, {}).get(ax, "").startswith(("default:", "mirror:auto"))):   # a partner's own guess is no measurement
        m = partner.keypoints[name]
        pt[i], src[ax] = (-m[0] if i == 0 else m[i]), "mirror:auto"
        warnings.append(f"{part_id}.{name}: {ax} is not measured in any view: mirrored from the partner part (automatic _l/_r pairing)")
        return
    if auto_depth_for is None:
        return
    for view in obs.get(name, {}):
        calib = calibs[view]
        if depth_axis(calib) != i:
            continue
        zbuf = auto_depth_for(view)
        if zbuf is None:
            continue
        half, foot = _extent(ptype, sc, name, i)
        height, cover = surface_height(zbuf, obs[name][view], foot / calib.m_per_px)
        if height is None or cover < MIN_CONTACT_COVERAGE:
            continue
        sign = float(AXES[normalize_axis(calib.camera_axis)][i])
        pt[i], src[ax] = sign * (height + standoff + half), f"contact:auto:{auto_target}@{view}"
        warnings.append(f"{part_id}.{name}: {ax} is not measured in any view: placed in contact with the {auto_target} surface seen in {view}")
        return


def _parse_joint(part_id: str, ptype: str, params: Mapping[str, Any]) -> Optional[tuple[str, str, str]]:
    """``(partner id, partner keypoint, own keypoint)`` of ``params.joint`` (defaults: the partner's tip, own base)."""
    j = params.get("joint")
    if not j:
        return None
    if not isinstance(j, Mapping) or not j.get("part"):
        raise BBRError(f'{part_id}: joint must be {{"part": id, "at": "tip"|"base", "self": "base"|"tip"}}, got {j!r}')
    at, own = str(j.get("at", "tip")), str(j.get("self", "base"))
    if at not in JOINT_KEYS or own not in POINTS[ptype] or own == "mid":
        raise BBRError(f"{part_id}: joint.at must be one of {list(JOINT_KEYS)} and joint.self one of "
                       f"{[n for n in POINTS[ptype] if n != 'mid']} for a {ptype}, got at={at!r} self={own!r}")
    return str(j["part"]), at, own


def _py(v: Any) -> Any:
    """numpy scalars / arrays -> plain python (lists, floats) for the Blender side and JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    return float(v) if isinstance(v, np.floating) else v


def _rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about the unit ``axis``."""
    a = math.radians(deg)
    return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * float(axis @ v) * (1 - math.cos(a))


def blade_width_dir(axis: np.ndarray, face_calib: ViewCalibration, roll_deg: float = 0.0) -> np.ndarray:
    """Width direction of a blade: its axis rotated 90 degrees in the image plane of the view that measured the width
    (perpendicular to the axis), then rolled by ``roll_deg`` about the axis."""
    ia, ib = [i for i in range(3) if i != depth_axis(face_calib)]
    wdir = np.zeros(3)
    wdir[ia], wdir[ib] = -axis[ib], axis[ia]
    wdir = wdir - (wdir @ axis) * axis
    wdir = wdir / np.linalg.norm(wdir) if np.linalg.norm(wdir) > 1e-6 else partgeom.orthonormal_frame(axis)[0]
    return _rotate(wdir, axis, roll_deg)


def _plate_geometry(part_id: str, pts: Mapping[str, np.ndarray], sc: dict[str, Any], params: Mapping[str, Any],
                    calibs: Mapping[str, ViewCalibration], sib: Optional[PartSolution],
                    label_for: Optional[Callable[[str], Optional[np.ndarray]]]
                    ) -> tuple[list[float], list[float], Optional[list[float]], dict[str, Any]]:
    """Pose and params of a contour plate (see :mod:`bbr.core.plate`): outline from the label mask in ``face_view``, plane
    through ``base`` containing base -> tip, facing the face view (or the plane of ``edge_on_views`` lines)."""
    base, tip = pts["base"], pts["tip"]
    length = float(np.linalg.norm(tip - base))
    if length < 1e-9:
        raise BBRError(f"{part_id}: base and tip coincide")
    axis = (tip - base) / length
    face = sc.get("face_view")
    mask = None
    if label_for is not None:
        if face is None:
            areas = {v: int(m.sum()) for v in calibs for m in [label_for(v)] if m is not None}
            face = max(areas, key=areas.get) if areas else None
        mask = label_for(face) if face else None
    if mask is None or not np.asarray(mask).any():
        if sib is not None and "outline" in sib.params:                          # mirror of a solved plate: its outline, mirrored
            flip = np.array([-1.0, 1.0, 1.0])
            sp = sib.params
            normal = np.asarray(sp["normal"], float) * flip
            sc.update(face_view=sib.scalars.get("face_view"), normal=normal, half_width=sib.scalars.get("half_width", 0.0))
            outline = np.asarray(sp["outline"], float) * np.array([1.0, -1.0])
            return list(base), [0.0, 0.0, 0.0], None, {
                "outline": outline.tolist(), "outline_tris": sp["outline_tris"], "outline_px": None, "thickness": sc["thickness"],
                "axis": list(np.asarray(sp["axis"], float) * flip), "normal": list(normal), "bend": float(sp.get("bend", 0.0)),
                "face_view": sib.scalars.get("face_view")}
        raise BBRError(f"{part_id}: a plate takes its outline from its label mask in the view where it is seen face-on "
                       f"(face_view {face!r}): label the part there (parts_labels)")
    if face not in calibs:
        raise BBRError(f"{part_id}: face_view {face!r} is not calibrated; views: {sorted(calibs)}")
    fc = calibs[face]
    sc["face_view"] = face
    c = plate.camera_dir(fc)
    lines = params.get("edge_on_views")
    normal = plate.normal_from_lines(lines, calibs, c) if lines else plate.plate_normal(axis, fc)
    outline_px = plate.outline_from_mask(np.asarray(mask, bool), float(params.get("outline_eps_px", plate.DEFAULT_SIMPLIFY_PX)))
    tris = plate.triangulate_polygon(outline_px)
    outline, u = plate.local_outline(outline_px, fc, base, normal, axis, 1.0)
    sc.update(normal=normal, half_width=plate.outline_half_width(outline))
    return list(base), [0.0, 0.0, 0.0], None, {
        "outline": outline.tolist(), "outline_tris": tris.tolist(), "outline_px": outline_px.tolist(), "thickness": sc["thickness"],
        "axis": list(u), "normal": list(normal), "bend": float(params.get("bend", 0.0)), "face_view": face}


def _geometry(part_id: str, ptype: str, pts: Mapping[str, np.ndarray], sc: dict[str, Any], params: Mapping[str, Any],
              calibs: Mapping[str, ViewCalibration], sib: Optional[PartSolution],
              label_for: Optional[Callable[[str], Optional[np.ndarray]]] = None
              ) -> tuple[list[float], list[float], Optional[list[float]], dict[str, Any]]:
    """``(location, rotation_deg, size, params)`` of the primitive builders from 3D keypoints and sizes."""
    if ptype == "plate":
        return _plate_geometry(part_id, pts, sc, params, calibs, sib, label_for)
    if ptype == "sphere":
        c = pts["center"]
        return list(c), [0.0, 0.0, 0.0], [2.0 * s for s in sc["semi"]], {}
    if ptype == "disc":
        face = sc.get("face_view")
        if "normal" in sc and face in calibs:                                   # tilted onto the contact surface
            normal = sc["normal"]
        elif face in calibs:
            normal = AXES[normalize_axis(calibs[face].camera_axis)].copy()      # face-on view: normal toward its camera
        elif sib is not None and "normal" in sib.scalars:
            normal = sib.scalars["normal"] * np.array([-1.0, 1.0, 1.0])
        else:
            raise BBRError(f"{part_id}: cannot tell the disc's facing direction (give radius_px in the face-on view)")
        sc["normal"] = normal
        return list(pts["center"]), [0.0, 0.0, 0.0], None, {"radius": sc["radius"], "axis": list(normal)}
    if ptype == "strand":
        poly = np.array([pts[n] for n in pts])
        if float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) < 1e-9:
            raise BBRError(f"{part_id}: the strand points coincide")
        return list(poly[0]), [0.0, 0.0, 0.0], None, {"points": (poly - poly[0]).tolist(), "radii": list(sc["radii"])}
    base, tip = pts["base"], pts["tip"]
    length = float(np.linalg.norm(tip - base))
    if length < 1e-9:
        raise BBRError(f"{part_id}: base and tip coincide")
    axis = (tip - base) / length
    if ptype == "cone":
        rb, rt = sc["radius"]
        out: dict[str, Any] = {"base_radius": rb, "tip_radius": rt, "length": length, "axis": list(axis), "bend": 0.0}
        if "mid" in pts:
            length, ax, bend, bend_dir = partgeom.arc_through_points(base, pts["mid"], tip)
            out.update(length=length, axis=list(ax), bend=bend)
            if bend_dir is not None:
                out["bend_dir"] = list(bend_dir)
        return list(base), [0.0, 0.0, 0.0], None, out
    if ptype == "blade":
        face = sc.get("face_view")
        if face is not None:        # width direction: in the image plane of the view that measured the width, normal to the axis
            wdir = blade_width_dir(axis, calibs[face], float(params.get("roll", 0.0)))
        elif sib is not None and "width_dir" in sib.scalars:
            wdir = sib.scalars["width_dir"] * np.array([-1.0, 1.0, 1.0])
        else:
            raise BBRError(f"{part_id}: give width_px in a view where the blade is seen flat")
        sc["width_dir"] = wdir
        return list(base), [0.0, 0.0, 0.0], None, {"length": length, "width": sc["width"], "thickness": sc["thickness"],
                                                   "axis": list(axis), "width_dir": list(wdir)}
    r = sc["radius"]                                                                # capsule / cylinder
    rot = partgeom.euler_xyz_deg(partgeom.rotation_z_to(axis))
    geo: dict[str, Any] = {"axis": "Z"}
    if ptype == "capsule" and abs(sc.get("radius_tip", r) - r) > 1e-9:               # tapered capsule (fitted)
        geo["tip_radius"] = sc["radius_tip"]
    return list((base + tip) / 2.0), rot, [2.0 * r, 2.0 * r, length], geo     # gem: the same box, built as a bipyramid
