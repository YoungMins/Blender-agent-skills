"""Gaps between parts that should touch (numpy only).

Every anchored part is a polyline of 3D keypoints (``base`` [``mid``] ``tip``, or ``center``) with a clearance radius
(:func:`~bbr.core.triangulate.clearance_radius`). The surface gap of two parts is the distance of their polylines minus
both radii. Two parts are expected to touch when

* one lists the other in ``params.adjacent_to``, or
* their label masks touch (within ``LABEL_TOUCH_PX``) in EVERY view where both are labelled, in at least
  ``MIN_TOUCH_VIEWS`` views;

a pair that ends up farther apart than ``tol`` (default ``GAP_TOL_M``) floats. A third kind of contract is the shared
anchor: two parts whose ``anchors_px`` put a keypoint on the same pixel (within ``SAME_PIXEL_PX``) in every view where
both give it (at least ``MIN_TOUCH_VIEWS``) mean one physical point, a joint; the fit may move either part, so the two
keypoints must still be within ``tol`` in 3D (the usual way a handle and its head drift apart). Both are reported as a
``part_gap`` (pin the keypoint with ``params.joint`` to fix it).
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from . import geomscore
from .imageops import dilate

GAP_TOL_M = 0.01
LABEL_TOUCH_PX = 3         # labels grown by this many pixels must meet to count as touching in a view
MIN_LABEL_PX = 12          # a label smaller than this in a view is ignored
MIN_TOUCH_VIEWS = 2        # views where both parts are labelled and touch before the pair counts as adjacent
SAME_PIXEL_PX = 4.0        # anchors this close in a view are the same physical point
FLOAT_TOL_FRAC = 0.01      # floating_part: a part this far (share of the height) from every other mesh floats ...
FLOAT_MINOR_FRAC = 0.25    # ... or this share of its own minor dimension (the smallest extent of its oriented box), if that is more
FLOAT_MAX_PART_PTS = 3000  # surface samples per part ...
FLOAT_MAX_OTHER_PTS = 60000   # ... and of all the other meshes together


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Closest distance between segments ``p0-p1`` and ``q0-q1`` (either may be a point)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-18 and e < 1e-18:
        return float(np.linalg.norm(r))
    if a < 1e-18:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e < 1e-18:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def _polyline(keypoints: Mapping[str, Any]) -> np.ndarray:
    strand = sorted((n for n in keypoints if re.fullmatch(r"p\d+", n)), key=lambda n: int(n[1:]))
    names = strand or [n for n in ("base", "mid", "tip") if n in keypoints] or ["center"]
    return np.array([keypoints[n] for n in names], dtype=float).reshape(-1, 3)


def surface_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Gap in meters between two placed parts (``{"keypoints": {name: xyz}, "clearance_radius": r}``); 0 when they overlap."""
    pa, pb = _polyline(a["keypoints"]), _polyline(b["keypoints"])
    sa = [(pa[i], pa[i + 1]) for i in range(len(pa) - 1)] or [(pa[0], pa[0])]
    sb = [(pb[i], pb[i + 1]) for i in range(len(pb) - 1)] or [(pb[0], pb[0])]
    dist = min(_segment_distance(x0, x1, y0, y1) for x0, x1 in sa for y0, y1 in sb)
    return max(dist - float(a.get("clearance_radius", 0.0)) - float(b.get("clearance_radius", 0.0)), 0.0)


def label_adjacent_pairs(labels: Mapping[str, Mapping[str, np.ndarray]], parts: Iterable[str],
                         touch_px: int = LABEL_TOUCH_PX) -> list[tuple[str, str]]:
    """Pairs of ``parts`` whose labels touch in every view where both are labelled (``labels`` = ``{view: {part: mask}}``),
    in at least ``MIN_TOUCH_VIEWS`` views."""
    ids = sorted(parts)
    grown: dict[tuple[str, str], np.ndarray] = {}
    out: list[tuple[str, str]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            touching = shared = 0
            for view, per in labels.items():
                ma, mb = per.get(a), per.get(b)
                if ma is None or mb is None or int(ma.sum()) < MIN_LABEL_PX or int(mb.sum()) < MIN_LABEL_PX:
                    continue
                shared += 1
                if (view, a) not in grown:
                    grown[(view, a)] = dilate(np.asarray(ma, bool), touch_px)
                touching += int((grown[(view, a)] & np.asarray(mb, bool)).any())
            if shared >= MIN_TOUCH_VIEWS and touching == shared:
                out.append((a, b))
    return out


def shared_anchors(anchors: Mapping[str, Mapping[str, Mapping[str, Any]]], px: float = SAME_PIXEL_PX
                   ) -> list[tuple[str, str, str, str]]:
    """``(part a, part b, keypoint of a, keypoint of b)`` for keypoints typed on the same pixel in every view where both
    parts give them (at least ``MIN_TOUCH_VIEWS``). ``anchors`` = ``{part: params.anchors_px}``."""
    pts = ("base", "mid", "tip", "center")
    ids = sorted(anchors)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for ka in pts:
                for kb in pts:
                    d = [np.hypot(*(np.asarray(va[ka], float) - np.asarray(vb[kb], float)))
                         for view, va in anchors[a].items() if ka in va and view in anchors[b] and kb in anchors[b][view]
                         for vb in [anchors[b][view]]]
                    if len(d) >= MIN_TOUCH_VIEWS and max(d) <= px:
                        out.append((a, b, ka, kb))
    return out


def find_gaps(placed: Mapping[str, Mapping[str, Any]], declared: Iterable[tuple[str, str]],
              labelled: Iterable[tuple[str, str]] = (), tol: Optional[Mapping[str, float]] = None,
              default_tol: float = GAP_TOL_M, joints: Iterable[tuple[str, str, str, str]] = ()) -> list[dict[str, Any]]:
    """Pairs expected to touch (``declared`` ids, ``labelled`` = touching labels) whose surface gap exceeds the tolerance
    (the larger of the two parts' ``tol`` entries, else ``default_tol``), and ``joints`` (:func:`shared_anchors`) whose
    two keypoints ended up farther apart than that. Pairs with an unplaced part are skipped.
    Each entry: ``{a, b, gap_m, tol_m, why}`` (+ ``ka``, ``kb`` for a joint), largest gap first."""
    pairs: dict[tuple[str, str], str] = {}
    for why, src in (("labels touch in every shared view", labelled), ("declared adjacent", declared)):
        for a, b in src:
            pairs[tuple(sorted((a, b)))] = why                      # type: ignore[assignment]
    out = []
    for (a, b), why in pairs.items():
        if a not in placed or b not in placed:
            continue
        limit = max((tol or {}).get(a, default_tol), (tol or {}).get(b, default_tol))
        gap = surface_gap(placed[a], placed[b])
        if gap > limit:
            out.append({"a": a, "b": b, "gap_m": round(gap, 5), "tol_m": limit, "why": why})
    for a, b, ka, kb in joints:
        if a not in placed or b not in placed or ka not in placed[a]["keypoints"] or kb not in placed[b]["keypoints"]:
            continue
        limit = max((tol or {}).get(a, default_tol), (tol or {}).get(b, default_tol))
        gap = float(np.linalg.norm(np.asarray(placed[a]["keypoints"][ka], float) - np.asarray(placed[b]["keypoints"][kb], float)))
        if gap > limit:
            out.append({"a": a, "b": b, "ka": ka, "kb": kb, "gap_m": round(gap, 5), "tol_m": limit,
                        "why": "anchored to the same pixel in the references"})
    return sorted(out, key=lambda g: -g["gap_m"])


def format_gap(g: Mapping[str, Any]) -> str:
    """One warning line for a :func:`find_gaps` entry."""
    if "ka" in g:
        return (f"{g['a']}.{g['ka']} and {g['b']}.{g['kb']} were {g['why']} but end {g['gap_m'] * 1000:.0f} mm apart "
                f"(tolerance {g['tol_m'] * 1000:.0f} mm): pin one with params.joint, e.g. on {g['a']}: "
                f'{{"part": "{g["b"]}", "at": "{g["kb"]}", "self": "{g["ka"]}"}}')
    return (f"{g['a']} and {g['b']} float {g['gap_m'] * 1000:.0f} mm apart (tolerance {g['tol_m'] * 1000:.0f} mm; {g['why']}): "
            f"add params.joint to one of them, or fix its anchors")


def minor_dimension(verts: np.ndarray) -> float:
    """Smallest extent (m) of the oriented (principal-axes) bounding box of a point set: the thickness of a plate, the width of
    a rod, the diameter of a ball."""
    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    if len(v) < 4:
        return 0.0
    c = v - v.mean(axis=0)
    _, axes = np.linalg.eigh(c.T @ c)
    proj = c @ axes
    return float(np.ptp(proj, axis=0).min())


def mesh_components(verts: np.ndarray, tris: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected pieces of a triangle mesh as ``(verts, tris)``, largest surface area first."""
    from .islands import vertex_components
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(t) == 0:
        return []
    label = vertex_components(len(verts), np.concatenate([t[:, [0, 1]], t[:, [1, 2]]]))
    out = []
    for lab in np.unique(label[t[:, 0]]):
        sel = label[t[:, 0]] == lab
        used, inv = np.unique(t[sel], return_inverse=True)
        sub_v, sub_t = np.asarray(verts)[used], inv.reshape(-1, 3)
        out.append((float(geomscore.triangle_areas(sub_v, sub_t).sum()), sub_v, sub_t))
    return [(v, t) for _, v, t in sorted(out, key=lambda o: -o[0])]


def floating_parts(meshes: Mapping[str, tuple[np.ndarray, np.ndarray]], height_m: float, tol_frac: float = FLOAT_TOL_FRAC,
                   tols: Optional[Mapping[str, float]] = None, skip: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Parts whose minimum distance to the union of the OTHER meshes exceeds the tolerance (``tol_frac`` x ``height_m``, or
    ``tols[part]`` in meters, or :data:`FLOAT_MINOR_FRAC` of the part's own minor dimension when that is larger: a thick fin may
    sit further off than a thin plate): ``{part, gap_m, tol_m, nearest}`` largest gap first. ``meshes`` = ``{part: (verts, tris)}`` in
    world space. Distances are between surface samples (sampling density follows the tolerance), less their spacing, so
    parts that touch or overlap give 0. ``skip``: parts that may float on purpose (``params.floating_ok``)."""
    default_tol = tol_frac * height_m
    skipped = set(skip)
    pts: dict[str, np.ndarray] = {}
    spacing: dict[str, float] = {}
    minor: dict[str, float] = {part: minor_dimension(v) for part, (v, t) in meshes.items()}
    for part, (v, t) in meshes.items():
        area = float(geomscore.triangle_areas(v, t).sum()) if len(t) else 0.0
        if area <= 0:
            continue
        n = int(np.clip(area / max(default_tol / 4.0, 1e-6) ** 2, 200, FLOAT_MAX_OTHER_PTS // max(len(meshes), 1)))
        pts[part] = geomscore.sample_surface(v, t, n, orient_outward=False)[0]
        spacing[part] = float(np.sqrt(area / n))
    out = []
    for part, mine in pts.items():
        if part in skipped or len(pts) < 2:
            continue
        others = [o for o in pts if o != part]
        cloud = np.concatenate([pts[o] for o in others])
        owner = np.concatenate([np.full(len(pts[o]), i) for i, o in enumerate(others)])
        step = max(1, len(mine) // FLOAT_MAX_PART_PTS)
        mine = mine[::step]
        d = geomscore.nearest_distances(mine, cloud)
        k = int(d.argmin())
        near = others[int(owner[_nearest_index(mine[k], cloud)])]
        gap = max(float(d[k]) - 0.5 * (spacing[part] + spacing[near]), 0.0)
        limit = max((tols or {}).get(part, 0.0), default_tol, FLOAT_MINOR_FRAC * minor.get(part, 0.0))
        if gap > limit:
            out.append({"part": part, "gap_m": round(gap, 5), "tol_m": round(limit, 5), "nearest": near})
    return sorted(out, key=lambda g: -g["gap_m"])


def _nearest_index(p: np.ndarray, cloud: np.ndarray) -> int:
    return int(np.argmin(((cloud - p) ** 2).sum(axis=1)))


def format_floating(g: Mapping[str, Any]) -> str:
    """One warning line for a :func:`floating_parts` entry."""
    return (f"{g['part']} floats {g['gap_m'] * 1000:.0f} mm from the nearest other part ({g['nearest']}; tolerance "
            f"{g['tol_m'] * 1000:.0f} mm): attach it (params.attach_to, params.joint, or anchors in a second view), "
            f"or set params.floating_ok if it is meant to hover")
