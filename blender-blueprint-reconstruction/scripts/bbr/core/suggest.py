"""Automatic part proposals from thin protrusions (numpy only): ears, horns, tails, hair locks, limbs.

Silhouette hulls turn spiky hair into a drum and an agent finds thin parts by eye. This module does the search:

1. **Pieces.** Per view, ``mask - opening(mask, OPEN_FRAC * height)`` (2-4 % of the height), crumbs dropped, each connected piece
   above a minimum area, grown back a little into the body so it reaches its base.
2. **Fit.** Each piece goes through the anchor / shape scoring of ``--kind auto`` (:func:`~bbr.core.measure.propose_anchors`
   and :mod:`~bbr.core.partshape`): cone (spike, ear, horn), curved cone (tail), capsule (limb), one type for all views.
3. **Hair.** Many small pieces on the head region are merged into 3-8 cone "locks" (:data:`LOCK_FLATTEN`), not dozens of parts.
4. **Match.** Pieces of different views are one part when their z-ranges overlap (front / back also share x, the side view
   shares z only). A part seen in one view only is kept, as ``inferred``.
5. **Output.** Ready-to-paste ``labels.json`` entries (region ids when a regions dir exists, else polygons), ``scene_spec``
   components with ``anchors_px``, an ``exclude_components`` list, a confidence per part and a numbered overlay.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import BBRError
from . import imageio, labels as labels_core, measure, partshape, proposals
from .calibration import ViewCalibration, calibrate_views
from .config import PipelineConfig
from .conventions import view_frame
from .distance import edt
from .imageops import dilate, erode, label_components, mask_bbox
from .spec import load_view_spec

BODY = labels_core.BODY

OPEN_FRAC = 0.03             # opening radius, share of the silhouette height (2-4 % is the useful range)
MIN_AREA_FRAC = 0.0006       # smallest piece, share of the silhouette area ...
MIN_AREA_PX = 30             # ... and in pixels
REGROW_PX = 8                # a sharp tip lost with the crumbs grows back by this much (inside the thin part only)
CRUMB_PX = 1                 # slivers thinner than 2 x this + 1 px (along curved outlines) are dropped
WIDE_SCALE = 1.6            # the second, wider opening (x OPEN_FRAC) that finds ears and other thick protrusions
ABSORB = 0.5                 # a fine piece with this share of its area inside a wide piece is part of it
SPLIT_FRAC = 0.012           # a wide piece is cut where an erosion by this share of the height breaks it into parts
SPLIT_MIN_SEED = 0.15        # a part that survives the erosion with less than this share of the biggest one is no seed
GROW_FRAC = 0.5              # a piece is grown into the body by this share of the opening radius, to reach its base
MATCH_MIN_IOU = 0.35         # z-range (and x-range) overlap at which pieces of two views are one part
MIRROR_MIN_IOU = 0.7         # two parts with this z overlap at mirrored x are a pair (``mirror_of``)
MIN_EXCLUDE_CONF = 0.5       # parts below this, spheres (a cut-off stump) and parts seen in one view are listed but not in ``exclude_components``
HAIR_MIN_PIECES = 5          # at least this many small head pieces in a view become locks
HAIR_MAX_LEN_FRAC = 0.12     # a hair candidate is shorter than this share of the height ...
HAIR_BAND = 0.30             # ... and sits in the top share of the silhouette (the head)
LOCKS = (3, 8)               # number of locks
HAIR_CLOSE_FRAC = 0.04      # the gaps between the spikes of a lock are closed by this share of the height
TUFT_MIN_SPIKES = 3          # a wide piece on the head that holds this many spikes is a hair lock by itself
HAIR_MAX_AREA_FRAC = 0.01    # a hair candidate is also smaller than this share of the silhouette (an ear is not)
HAIR_LINK_FRAC = 0.12       # hair pieces whose centres are this close (share of the height) belong to one lock
LOCK_FLATTEN = 0.5           # flatten of a lock (hair is a flat fan, 0.4-0.6)
CURVE_MIN_BEND = 0.2         # a cone is curved (mid anchor on the centre line) when its centre line is this far off the chord, share of its length
CROSS_MIN_FACTOR = 0.6       # cross-view disagreements (warnings stay) lower a part's confidence by at most this factor
CROSS_DROP = 0.5             # a view whose base / tip height is off the others' median by this share of the part length is dropped from the part
SNAP_REACH = 0.5            # the tip is moved on at most this share of the axis length
POLYGON_TOL_PX = 1.5         # Douglas-Peucker tolerance of the label polygons
REGION_COVER = 0.7           # region ids stand in for a piece when they cover this much of it and lie this much inside it
VIEW_ORDER = ("front", "back", "side", "right", "left")
_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))   # clockwise on screen (y down)
_ANCHOR_KEYS = ("base", "mid", "tip", "center", "radius_px", "width_px")


# --------------------------------------------------------------------------- pieces
def _pieces_at(mask: np.ndarray, r: int, min_area: float) -> tuple[list[np.ndarray], np.ndarray]:
    core = dilate(erode(mask, r), r) & mask
    raw = mask & ~core
    thin = dilate(erode(raw, CRUMB_PX), CRUMB_PX)                                 # crumbs gone, but so are the sharp tips ...
    for _ in range(REGROW_PX):                                                     # ... which grow back along the piece
        thin = dilate(thin, 1) & raw
    comps, n = label_components(thin, full=True)
    areas = np.bincount(comps.ravel(), minlength=n + 1)
    grow = max(1, int(round(GROW_FRAC * r)))
    return [dilate(comps == i, grow) & mask for i in np.argsort(-areas[1:]) + 1 if areas[i] >= min_area], core


def _split(piece: np.ndarray, k: int) -> list[np.ndarray]:
    """Cut a piece where thin bridges join its parts (an ear and the hair beside it, joined along the head outline): the
    parts that survive an erosion by ``k`` px are the seeds, every piece pixel goes to its nearest seed."""
    lab, n = label_components(erode(piece, k), full=False)
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    seeds = [i for i in range(1, n + 1) if areas[i] >= SPLIT_MIN_SEED * areas[1:].max()]
    if len(seeds) < 2:
        return [piece]
    seed_map = np.where(np.isin(lab, seeds), lab, 0)
    _, iy, ix = edt(seed_map > 0, cap=min(max(piece.shape), 200), return_index=True)
    owner = np.where(iy >= 0, seed_map[np.clip(iy, 0, piece.shape[0] - 1), np.clip(ix, 0, piece.shape[1] - 1)], 0)
    return [piece & (owner == i) for i in seeds]


def find_pieces(mask: np.ndarray, open_frac: float = OPEN_FRAC, min_frac: float = MIN_AREA_FRAC
                ) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """``(pieces, core, spikes)``: the thin protrusions of ``mask`` as bool masks (biggest first), the opened body they stick
    out of, and how many separate spikes (pieces of the fine opening) each one holds.

    Two scales: the opening at ``OPEN_FRAC * height`` finds spikes, the one at ``WIDE_SCALE`` times that finds the
    wide ones (an ear is thicker than a lock, so the small opening cuts only its tip). A fine piece that lies in a wide one
    (``ABSORB`` of its area) is part of it: a wide piece with several fine ones is a tuft of spikes (hair), one with a single
    fine piece is an ear, horn or limb."""
    mask = np.asarray(mask, dtype=bool)
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) == 0:
        return [], mask.copy(), []
    r = max(2, int(round(open_frac * (ys[-1] - ys[0] + 1))))
    min_area = max(MIN_AREA_PX, min_frac * mask.sum())
    wide, core = _pieces_at(mask, int(round(WIDE_SCALE * r)), min_area)
    wide = [part for w in wide for part in _split(w, max(2, int(round(SPLIT_FRAC * (ys[-1] - ys[0] + 1)))))]
    wide = [w for w in wide if w.sum() >= min_area]
    fine, _ = _pieces_at(mask, r, min_area)
    spikes = [sum(int((f & w).sum() >= ABSORB * f.sum()) for f in fine) for w in wide]
    covered = np.zeros_like(mask)
    for w in wide:
        covered |= w
    extra = [f for f in fine if (f & covered).sum() < ABSORB * f.sum()]
    order = sorted(range(len(wide) + len(extra)), key=lambda i: -int((wide + extra)[i].sum()))
    return [(wide + extra)[i] for i in order], core, [(spikes + [1] * len(extra))[i] for i in order]


def _range(calib: ViewCalibration, mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((z0, z1), (h0, h1))`` of a mask in metres: height and the horizontal extent (world X for front / back, Y for side)."""
    x0, y0, x1, y1 = mask_bbox(mask)
    pts = calib.px_to_plane(np.array([[x0, y1], [x1, y0]], dtype=float))
    col = "XY".index(view_frame(calib.camera_axis).right_axis[1])
    return (float(pts[0, 2]), float(pts[1, 2])), (float(pts[:, col].min()), float(pts[:, col].max()))


def _iou1d(a: Sequence[float], b: Sequence[float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0.0) / union if union > 0 else 0.0


# --------------------------------------------------------------------------- fitting
def _clean(fit: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fit.items() if k in _ANCHOR_KEYS}


def fit_piece(piece: np.ndarray, rest: np.ndarray, force_type: Optional[str] = None) -> dict[str, Any]:
    """Anchors and type of one piece (``--kind auto``): ``{anchors, type, curved, confidence, length_px, base, tip}``."""
    x0, y0, x1, y1 = mask_bbox(piece)
    fit = measure.propose_anchors(piece, [x0 - 4, y0 - 4, x1 + 4, y1 + 4], "auto", None, rest, force_type=force_type)
    a = _clean(fit)
    if "tip" in a and "base" in a and not fit.get("curled"):
        a["tip"] = _snap_tip(piece, a)
    base = a.get("base") or a.get("center")
    tip = a.get("tip") or a.get("center")
    bend = max([int(m) / 100.0 for w in fit.get("warnings", []) for m in re.findall(r"(\d+)% of its length", w)] or [0.0])
    curved = bool(fit.get("curled")) and bend >= CURVE_MIN_BEND
    if "mid" in a and not curved:                      # a slight bend (an ear's outline) is no tail: keep the cone straight
        a["mid"] = _xy((np.array(a["base"]) + np.array(a["tip"])) / 2.0)
    return {"anchors": a, "type": fit["suggested_type"], "curved": curved, "confidence": float(fit["confidence"]),
            "length_px": float(fit.get("length_px") or 0.0), "base": list(base), "tip": list(tip),
            "warnings": list(fit.get("warnings") or []) if curved else []}


def _snap_tip(piece: np.ndarray, a: Mapping[str, Any]) -> list[float]:
    """The shape fit opens thin spikes away, which shortens a pointed end: move the tip on along the axis to the farthest
    piece pixel that still lies within the tip's width of it."""
    base, tip = np.array(a["base"], dtype=float), np.array(a["tip"], dtype=float)
    u = tip - base
    length = float(np.hypot(*u))
    if length < 1.0:
        return list(a["tip"])
    u /= length
    rr, cc = np.nonzero(piece)
    pts = np.stack([cc + 0.5, rr + 0.5], axis=1) - tip
    ahead, lateral = pts @ u, np.abs(pts[:, 0] * u[1] - pts[:, 1] * u[0])
    r_tip = a.get("radius_px", [2.0, 2.0])[-1] if isinstance(a.get("radius_px"), (list, tuple)) else 2.0
    near = (lateral <= r_tip + 3.0) & (ahead > 0) & (ahead <= SNAP_REACH * length)
    return _xy(tip + u * float(ahead[near].max())) if near.any() else list(a["tip"])


def _lock(members: Sequence[dict[str, Any]], shape: tuple[int, int], gap_px: int) -> dict[str, Any]:
    """One cone over several small spikes: their union with the gaps between them closed (``gap_px``, inside the silhouette),
    fitted like any piece; when that fails, a cone from their mean base to their mean tip, wide enough for the fan."""
    union = np.zeros(shape, dtype=bool)
    for m in members:
        union |= m["mask"]
    silhouette = members[0]["silhouette"]
    mask = erode(dilate(union, gap_px), gap_px) & silhouette | union
    comps, n = label_components(mask, full=True)
    if n > 1:
        mask = comps == (1 + int(np.argmax(np.bincount(comps.ravel())[1:])))
    try:
        fit = fit_piece(mask, members[0]["rest"], force_type=partshape.CONE)
        fit["confidence"] = min(fit["confidence"], 0.9)
        return {"mask": mask, "area": int(mask.sum()), "fit": fit, "rest": members[0]["rest"], "silhouette": silhouette,
                "members": len(members), "lock": True}
    except BBRError:
        pass
    w = np.array([m["area"] for m in members], dtype=float)
    base = np.average([m["fit"]["base"] for m in members], axis=0, weights=w)
    tip = np.average([m["fit"]["tip"] for m in members], axis=0, weights=w)
    rb = max(float(np.hypot(*(np.array(m["fit"]["base"]) - base))) for m in members) + float(np.mean([_radius(m["fit"], 0) for m in members]))
    rt = float(np.mean([_radius(m["fit"], -1) for m in members])) + 0.5 * max(float(np.hypot(*(np.array(m["fit"]["tip"]) - tip))) for m in members)
    fit = {"anchors": {"base": _xy(base), "mid": _xy((base + tip) / 2), "tip": _xy(tip), "radius_px": [round(rb, 1), round(rt, 1)]},
           "type": partshape.CONE, "curved": False, "confidence": 0.3, "length_px": float(np.hypot(*(tip - base))),
           "base": _xy(base), "tip": _xy(tip), "warnings": ["lock fitted from the mean of its spikes (low confidence)"]}
    return {"mask": union, "area": int(union.sum()), "fit": fit, "rest": members[0]["rest"], "silhouette": silhouette,
            "members": len(members), "lock": True}


def _radius(fit: Mapping[str, Any], end: int) -> float:
    r = fit["anchors"].get("radius_px", 2.0)
    return float(r[end]) if isinstance(r, (list, tuple)) else float(r)


def _xy(p: Sequence[float]) -> list[float]:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def _clusters(points: np.ndarray, link: float, max_groups: int) -> list[list[int]]:
    """Single-linkage groups of ``points`` closer than ``link``; ``link`` grows until at most ``max_groups`` remain."""
    n = len(points)
    dist = np.hypot(*(points[:, None, :] - points[None, :, :]).transpose(2, 0, 1))
    while True:
        parent = list(range(n))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= link:
                    parent[root(i)] = root(j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(root(i), []).append(i)
        if len(groups) <= max_groups or link > dist.max():
            return list(groups.values())
        link *= 1.3


def group_hair(pieces: list[dict[str, Any]], bbox: tuple, shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Merge the small head pieces into cone locks when there are ``HAIR_MIN_PIECES`` or more; others stay as they are.

    ``pieces``: dicts with ``mask``, ``area``, ``fit``, ``rest``. A hair candidate is shorter than ``HAIR_MAX_LEN_FRAC`` of the
    height and its centre lies in the top ``HAIR_BAND`` of the silhouette. Candidates close to each other (single linkage on
    the centres, ``HAIR_LINK_FRAC`` of the height, widened until at most ``LOCKS[1]`` groups remain) form one lock; a group
    that is a single piece stays a piece. A lone group of many is split in two until ``LOCKS[0]`` locks exist."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    hair = [p for p in pieces if p["fit"]["length_px"] <= HAIR_MAX_LEN_FRAC * height and not p["lock"]
            and p["area"] <= HAIR_MAX_AREA_FRAC * float(p["silhouette"].sum())
            and np.nonzero(p["mask"])[0].mean() <= y0 + HAIR_BAND * height]
    if len(hair) < HAIR_MIN_PIECES:
        return pieces
    centres = np.array([[np.nonzero(p["mask"])[1].mean(), np.nonzero(p["mask"])[0].mean()] for p in hair])
    groups = _clusters(centres, HAIR_LINK_FRAC * height, LOCKS[1])
    while len(groups) < LOCKS[0]:
        big = max(groups, key=len)
        if len(big) < 2:
            break
        centre = proposals.kmeans(centres[big], 2)
        side = np.abs(centres[big][:, None, :] - centre[None]).sum(axis=2).argmin(axis=1)
        groups.remove(big)
        groups += [[g for g, s_ in zip(big, side) if s_ == k] for k in (0, 1) if (side == k).any()]
    merged = {i for g in groups if len(g) > 1 for i in g}
    locks = [_lock([hair[i] for i in g], shape, int(round(HAIR_CLOSE_FRAC * height))) for g in groups if len(g) > 1]
    return [p for i, p in enumerate(pieces) if not any(p is hair[m] for m in merged)] + locks


# --------------------------------------------------------------------------- contours, labels
def trace_outline(mask: np.ndarray) -> list[tuple[int, int]]:
    """Moore-neighbour outline of the first connected piece of ``mask`` as pixel ``(x, y)`` coordinates."""
    p = np.pad(np.asarray(mask, dtype=bool), 1)
    ys, xs = np.nonzero(p)
    start = (int(xs[0]), int(ys[0]))
    cur, back = start, 4                        # entered from the left (background)
    out = [start]
    for _ in range(4 * int(p.sum()) + 8):
        for k in range(1, 9):
            d = (back + k) % 8
            nxt = (cur[0] + _DIRS[d][0], cur[1] + _DIRS[d][1])
            if p[nxt[1], nxt[0]]:
                prev = (cur[0] + _DIRS[(d - 1) % 8][0], cur[1] + _DIRS[(d - 1) % 8][1])
                back = _DIRS.index((prev[0] - nxt[0], prev[1] - nxt[1])) if (prev[0] - nxt[0], prev[1] - nxt[1]) in _DIRS else (d + 4) % 8
                cur = nxt
                break
        else:
            break
        if cur == start:
            break
        out.append(cur)
    return [(x - 1, y - 1) for x, y in out]


def simplify(points: Sequence[tuple[float, float]], tol: float = POLYGON_TOL_PX) -> list[tuple[float, float]]:
    """Douglas-Peucker on a closed outline (split at the point farthest from the first one)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 3:
        return [(float(x), float(y)) for x, y in pts]
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    ring = np.vstack([pts, pts[:1]])
    keep = np.zeros(len(ring), dtype=bool)
    keep[[0, far, len(ring) - 1]] = True

    def rec(lo: int, hi: int) -> None:
        if hi <= lo + 1:
            return
        a, ab = ring[lo], ring[hi] - ring[lo]
        norm = max(float(np.hypot(*ab)), 1e-9)
        d = np.abs(ab[0] * (ring[lo + 1:hi, 1] - a[1]) - ab[1] * (ring[lo + 1:hi, 0] - a[0])) / norm
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[lo + 1 + i] = True
            rec(lo, lo + 1 + i)
            rec(lo + 1 + i, hi)

    rec(0, far)
    rec(far, len(ring) - 1)
    return [(float(x), float(y)) for x, y in ring[:-1][keep[:-1]]]


def label_entries(mask: np.ndarray, id_map: Optional[np.ndarray] = None) -> list[Any]:
    """``labels.json`` items for a piece: region ids when ``id_map`` has regions that match it, else one polygon per component."""
    if id_map is not None:
        ids = [int(i) for i in np.unique(id_map[mask]) if i > 0]
        good = [i for i in ids if (mask & (id_map == i)).sum() >= REGION_COVER * (id_map == i).sum()]
        if good and (mask & np.isin(id_map, good)).sum() >= REGION_COVER * mask.sum():
            return good
    comps, n = label_components(mask, full=True)
    entries = []
    for i in range(1, n + 1):
        poly = simplify([(x + 0.5, y + 0.5) for x, y in trace_outline(dilate(comps == i, 1))])
        if len(poly) >= 3:
            entries.append({"polygon": [[round(x, 1), round(y, 1)] for x, y in poly]})
    return entries


# --------------------------------------------------------------------------- the whole search
def suggest_parts(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration],
                  label_masks: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
                  id_maps: Optional[Mapping[str, np.ndarray]] = None, open_frac: float = OPEN_FRAC,
                  ) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    """Part suggestions from the protrusions of the upright views. Returns ``(result, masks)``: ``result`` is JSON-able
    (``suggestions``, ``labels``, ``components``, ``exclude_components``, ``notes``), ``masks`` = ``{view: {number: mask}}``
    for the overlay. ``label_masks`` (``{view: {part: mask}}``): the body label is what pieces attach to, and pieces that lie
    in an already labelled part are skipped (listed under ``already_labelled``)."""
    views = [v for v in VIEW_ORDER if v in masks and v in calibs and calibs[v].camera_axis in ("-Y", "+Y", "-X", "+X")]
    if not views:
        raise BBRError("suggest-parts needs an upright front / back / side view with a mask")
    per_view: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    for v in views:
        mask = np.asarray(masks[v], dtype=bool)
        pieces, core, spikes = find_pieces(mask, open_frac)
        labelled = {p: np.asarray(m, dtype=bool) for p, m in (label_masks or {}).get(v, {}).items() if p != BODY and not p.startswith("_")}
        body = (label_masks or {}).get(v, {}).get(BODY)
        rest = np.asarray(body, dtype=bool) if body is not None and np.asarray(body).any() else core
        items = []
        bbox = mask_bbox(mask)
        top, height = bbox[1], bbox[3] - bbox[1]
        for piece, n_spikes in zip(pieces, spikes):
            owner = next((p for p, m in labelled.items() if (piece & m).sum() >= 0.5 * piece.sum()), None)
            if owner:
                skipped.append({"view": v, "part": owner, "bbox": list(mask_bbox(piece))})
                continue
            try:
                fit = fit_piece(piece, rest)
            except BBRError:
                continue
            tuft = n_spikes >= TUFT_MIN_SPIKES and np.nonzero(piece)[0].mean() <= top + HAIR_BAND * height
            items.append({"mask": piece, "area": int(piece.sum()), "fit": fit, "lock": bool(tuft), "rest": rest, "silhouette": mask})
        per_view[v] = group_hair(items, bbox, mask.shape)
        for it in per_view[v]:
            it["z"], it["h"] = _range(calibs[v], it["mask"])
    suggestions = _match(views, per_view)
    _unify_types(suggestions)
    _cross_check(suggestions, calibs)
    suggestions.sort(key=lambda s: (-len(s["pieces"]), -sum(p["area"] for p in s["pieces"].values())))   # several views first, then biggest
    for n, s in enumerate(suggestions, start=1):
        s["id"] = f"prot_{n}"
        s["number"] = n
    _mirror_pairs(suggestions)
    return _report(suggestions, views, masks, id_maps, skipped)


def _match(views: Sequence[str], per_view: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Group pieces of different views into parts by z-range overlap (front / back also by x-range, one-to-one; the side
    view may serve two mirrored parts)."""
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for i, v in enumerate(views):
        for p in per_view[v]:
            if id(p) in used:
                continue
            used.add(id(p))
            s = {"pieces": {v: p}, "match": {}}
            for w in views[i + 1:]:
                best, score = None, MATCH_MIN_IOU
                for q in per_view[w]:
                    if id(q) in used and w != "side":
                        continue
                    both_x = {v, w} <= {"front", "back"}
                    iou = _iou1d(p["z"], q["z"]) * (_iou1d(p["h"], q["h"]) if both_x else 1.0)
                    if id(q) in used:
                        iou *= 0.99 if iou >= 0.5 else 0.0          # a side piece shared by mirrored parts only on a clear match
                    if iou > score:
                        best, score = q, iou
                if best is not None:
                    used.add(id(best))
                    s["pieces"][w], s["match"][w] = best, round(score, 2)
            out.append(s)
    return out


def _unify_types(suggestions: Sequence[dict[str, Any]]) -> None:
    """One primitive for all views of a part (a confidence-weighted vote; a curled view decides alone)."""
    for s in suggestions:
        fits = {v: p["fit"] for v, p in s["pieces"].items()}
        types = {f["type"] for f in fits.values()}
        if len(types) == 1:
            continue
        score: dict[str, float] = {}
        for f in fits.values():
            score[f["type"]] = score.get(f["type"], 0.0) + f["confidence"]
        best = partshape.CONE if any(f["curved"] for f in fits.values()) else max(score, key=lambda t: (score[t], partshape.TYPE_RANK[t]))
        for v, p in s["pieces"].items():
            if p["fit"]["type"] != best and not p.get("lock"):
                try:
                    p["fit"] = fit_piece(p["mask"], p["rest"], force_type=best)
                except BBRError:
                    pass
                p["fit"]["warnings"].append(f"the views disagree on the type ({sorted(types)}); {best} used in every view")


def _outlier_views(found: Mapping[str, Mapping[str, Any]], calibs: Mapping[str, ViewCalibration]) -> list[str]:
    """Views (of at least three) whose base or tip height is off the median of the others by more than ``CROSS_DROP`` of the
    part length: the match picked another part's piece there (side-view hair spikes at similar heights are easily confused)."""
    zs = {}
    for v, a in found.items():
        c = calibs.get(v)
        if c is not None and "base" in a and "tip" in a and view_frame(c.camera_axis).up_axis[1] == "Z":
            zs[v] = tuple(float(c.px_to_plane(np.array([a[k]]))[0, 2]) for k in ("base", "tip"))
    if len(zs) < 3:
        return []
    length = float(np.mean([abs(z[0] - z[1]) for z in zs.values()]))
    out = []
    for v, z in zs.items():
        others = [w for u, w in zs.items() if u != v]
        if length > 0 and any(abs(z[i] - float(np.median([w[i] for w in others]))) > CROSS_DROP * length for i in (0, 1)):
            out.append(v)
    return out if len(out) < len(zs) / 2 else []


def _cross_check(suggestions: Sequence[dict[str, Any]], calibs: Mapping[str, ViewCalibration]) -> None:
    """Base / tip must run the same way in every view (the same end is the higher one): the minority view swaps them, as
    ``anchors`` does; remaining height disagreements between views are warnings and lower the confidence."""
    for s in suggestions:
        if len(s["pieces"]) < 2:
            continue
        found = {v: {**p["fit"]["anchors"], "confidence": p["fit"]["confidence"]} for v, p in s["pieces"].items()}
        flipped = measure.harmonize_base_tip(found, dict(calibs))
        for v in flipped:
            fit = s["pieces"][v]["fit"]
            fit["anchors"] = {k: found[v][k] for k in fit["anchors"]}
            fit["base"], fit["tip"] = fit["tip"], fit["base"]
        dropped = _outlier_views(found, calibs)
        for v in dropped:
            del found[v], s["pieces"][v]
        s["dropped_views"] = dropped
        check = measure.check_anchor_views(found, dict(calibs))
        low = min(f["confidence"] for f in (p["fit"] for p in s["pieces"].values()))
        s["cross_factor"] = max(CROSS_MIN_FACTOR, check["confidence"] / low) if low > 0 else 1.0
        s["cross_warnings"] = ([f"base / tip swapped in {', '.join(flipped)} to run the same way as in the other views"]
                                  if flipped and measure._elongation_of(found) >= measure.SYMMETRIC_ELONGATION else []) + [
            f"{v} view dropped: its base / tip heights disagree with the other views by more than {CROSS_DROP:.0%} of the part length "
            f"(probably another part's piece); the part has no {v} anchors" for v in dropped] + [
            w for w in check["warnings"] if "height disagrees" in w or "swapped between" in w]


def _mirror_pairs(suggestions: list[dict[str, Any]]) -> None:
    """A part at mirrored x (same z, same size) of a bigger one is its ``mirror_of``: the side view shows both overlapped."""
    for i, a in enumerate(suggestions):
        for b in suggestions[i + 1:]:
            if b.get("mirror_of") or a.get("mirror_of"):
                continue
            va = next((v for v in ("front", "back") if v in a["pieces"] and v in b["pieces"]), None)
            if va is None:
                continue
            pa, pb = a["pieces"][va], b["pieces"][va]
            same_type = pa["fit"]["type"] == pb["fit"]["type"]
            sizes = sorted([pa["area"], pb["area"]])
            if (same_type and _iou1d(pa["z"], pb["z"]) >= MIRROR_MIN_IOU and sizes[0] >= 0.6 * sizes[1]
                    and abs(sum(pa["h"]) + sum(pb["h"])) <= 0.3 * max(abs(sum(pa["h"])), abs(sum(pb["h"])), 1e-9)
                    and (sum(pa["h"]) * sum(pb["h"]) < 0)):
                b["mirror_of"] = a["id"]
                if "side" in b["pieces"] and "side" in a["pieces"] and b["pieces"]["side"] is a["pieces"]["side"]:
                    del b["pieces"]["side"]


def _report(suggestions: list[dict[str, Any]], views: Sequence[str], masks: Mapping[str, np.ndarray],
            id_maps: Optional[Mapping[str, np.ndarray]], skipped: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    labels: dict[str, dict[str, list[Any]]] = {v: {} for v in views}
    overlay: dict[str, dict[int, np.ndarray]] = {v: {} for v in views}
    components, out = [], []
    for s in suggestions:
        fits = {v: p["fit"] for v, p in s["pieces"].items()}
        typ = max((f["type"] for f in fits.values()), key=lambda t: partshape.TYPE_RANK[t])
        curved = any(f["curved"] for f in fits.values())
        lock = any(p.get("lock") for p in s["pieces"].values())
        conf = float(np.mean([f["confidence"] for f in fits.values()])) * min(float(s.get("cross_factor", 1.0)), 1.0)
        if len(fits) == 1:
            conf *= 0.7                                   # one view: depth and placement are guesses
        warnings = list(dict.fromkeys(re.sub(r"\d+% of its length", "n% of its length", w) for f in fits.values() for w in f["warnings"]))
        warnings += s.get("cross_warnings", [])
        if len(fits) == 1:
            warnings.append(f"seen in {next(iter(fits))} only: the other views hide it or it did not match; the part is inferred")
        params: dict[str, Any] = {"anchors_px": {v: dict(f["anchors"]) for v, f in fits.items()}}
        if lock:
            params["flatten"] = LOCK_FLATTEN
        if s.get("mirror_of"):
            params["mirror_of"] = s["mirror_of"]
        comp = {"id": s["id"], "type": typ, "stage": "secondary",
                "confidence": "observed" if len(fits) > 1 else "inferred", "params": params}
        if curved:
            comp["notes"] = "curved cone (tail-like): base / mid / tip follow the centre line"
        components.append(comp)
        for v, p in s["pieces"].items():
            labels[v][s["id"]] = label_entries(p["mask"], (id_maps or {}).get(v))
            overlay[v][s["number"]] = p["mask"]
        out.append({"id": s["id"], "number": s["number"], "type": typ, "curved": curved, "hair_lock": lock,
                    "views": sorted(fits), "confidence": round(conf, 2), "mirror_of": s.get("mirror_of"),
                    "z_range_m": [round(min(p["z"][0] for p in s["pieces"].values()), 4), round(max(p["z"][1] for p in s["pieces"].values()), 4)],
                    "match_iou": s["match"], "warnings": warnings})
    exclude = [s["id"] for s in out if s["confidence"] >= MIN_EXCLUDE_CONF and len(s["views"]) > 1 and s["type"] != partshape.SPHERE]
    result = {"suggestions": out, "labels": {v: l for v, l in labels.items() if l}, "components": components,
              "exclude_components": exclude, "already_labelled": skipped, "views": list(views),
              "notes": ["labels / components / exclude_components are ready to paste: rename the prot_N ids, check every overlay number "
                        "against the image, drop what is not a part. `labels` for `_l`/`_r` need the character's sides."]}
    return result, overlay


def render_overlays(rgba: Mapping[str, np.ndarray], overlay: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, np.ndarray]:
    """``{view: RGBA float image}``: the view with every suggestion tinted, outlined and numbered, rulers in VIEW pixels."""
    out = {}
    for v, parts in overlay.items():
        if v not in rgba or not parts:
            continue
        id_map = np.zeros(next(iter(parts.values())).shape, dtype=np.uint8)
        regions = []
        for n, m in sorted(parts.items(), reverse=True):
            id_map[m] = min(n, 255)
        for n, m in parts.items():
            if (id_map == n).any():
                region = id_map == n
                regions.append({"id": n, "label_px": list(proposals.inner_point(region, mask_bbox(region)))})
        out[v] = proposals.render_overlay(rgba[v], id_map, regions, 1)
    return out


# --------------------------------------------------------------------------- from files
def suggest_from_files(view_spec_path: Path | str, labels_path: Optional[Path | str] = None,
                       regions_dir: Optional[Path | str] = None, open_frac: float = OPEN_FRAC,
                       out_dir: Optional[Path | str] = None) -> dict[str, Any]:
    """Run :func:`suggest_parts` on the views of a ViewSpec (calibrated with the labels' body axis, like the pipeline).

    Region ids replace polygons in ``labels`` when a regions dir exists: ``regions_dir``, else the one named by the labels
    file, else ``regions/`` beside the ViewSpec or beside its ``references/`` folder. With ``out_dir`` it writes
    ``suggestions.json`` and one numbered overlay ``suggest_<view>.png`` (view pixels, rulers) per view; the result then
    lists them under ``files``."""
    spec = load_view_spec(view_spec_path)
    rgba, masks = measure.load_view_data(spec, PipelineConfig())
    label_masks = labels_core.load_label_masks(labels_path, masks, regions_dir) if labels_path else None
    calibs, _ = calibrate_views(spec, masks, label_masks)
    base = Path(spec.source_path).resolve().parent if spec.source_path else Path(".")
    if regions_dir:
        folders = [Path(regions_dir)]
    elif labels_path:
        folders = [labels_core.regions_dir(labels_path, labels_core.load_labels(labels_path))]
    else:
        folders = [base / "regions", base.parent / "regions"]          # case layout, then $W/references/view_spec.json beside $W/regions
    id_maps: dict[str, np.ndarray] = {}
    for folder in folders:
        for v in masks:
            f = folder / v / "id_map.npy"
            if v not in id_maps and f.is_file() and np.load(f).shape == masks[v].shape:
                id_maps[v] = np.load(f)
    result, overlay = suggest_parts(masks, calibs, label_masks, id_maps, open_frac)
    result["label_style"] = "region ids and polygons" if id_maps else "polygons"
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for v, img in render_overlays(rgba, overlay).items():
            imageio.save_rgba(out / f"suggest_{v}.png", img)
            files.append(str(out / f"suggest_{v}.png"))
        result["files"] = [str(out / "suggestions.json")] + files
        (out / "suggestions.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result
