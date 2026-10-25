"""Which primitive is this part, and where are its anchors? (numpy only; the engine of ``anchors --kind auto``.)

A part's label mask is explained by up to five candidate shapes, each rendered back to pixels and scored by its IoU with
the mask (the honest ``confidence`` of the proposal):

* ``sphere``: the area-equivalent disc (a compact blob: a fist, a ball, a pom-pom);
* ``cylinder``: a straight tube of constant half-width with flat caps (a lantern, a can);
* ``capsule``: pole to pole along the principal axis, radius from the area (a mitten, a limb, a staff);
* ``cone``, straight: a tube along the principal axis whose half-width is a line fit over the axis (an ear, a horn, a
  tapered sleeve), flat caps like the ``cone`` part type;
* ``cone``, curved: the same tube along a centre-favouring geodesic between the two geodesically farthest mask pixels
  (a J-curled tail), with a ``mid`` anchor on the curve.

The simplest shape wins unless a more complex one beats it by a margin (:data:`GAIN`, :data:`CURVE_GAIN`; a centre line
bending :data:`CURL_BEND` of its length is a cone whatever the IoUs), so ``suggested_type`` is a measured choice, not a label guess. base = ``base_hint``, else the end
nearer the rest of the silhouette (``rest_pts``, normally the body label: an ear sits on the head), else the wider end of
a cone, else the lower end.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Optional, Sequence

import numpy as np

from .. import BBRError
from .distance import edt
from .imageops import dilate, erode, label_components
from .labels import polygon_mask
from .measure import MIN_RADIUS_PX, TAPER_CLEAR, _arclength, _smooth, attach_base

SPHERE, CAPSULE, CYLINDER, CONE = "sphere", "capsule", "cylinder", "cone"
TYPE_RANK = {SPHERE: 0, CAPSULE: 1, CYLINDER: 2, CONE: 3}   # more specific = higher: one type per part takes the highest any view needs
GAIN = 0.03                # a more complex shape must beat the simpler one's IoU by this ...
CONE_OVER_SPHERE_GAIN = 0.0   # ... except a cone over a sphere (a tapered fan cannot be a ball: any tie goes to the cone)
CURVE_GAIN = 0.05          # a curved cone must beat the straight one by this ...
CURL_BEND = 0.12           # ... or, bending this much of its length (a J-curled tail), win over any simpler shape
CURL_MIN_IOU = 0.55
OPEN_PX = 2                # morphological opening radius that removes spikes before the shapes are fitted
OPEN_MAX_LOSS = 0.25       # ... unless it removes more than this share of the mask (a thin part)
CENTRE_PULL = 3.0          # geodesic step cost grows by this factor at the boundary (the path hugs the medial axis)
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


# --------------------------------------------------------------------------- geodesics
def _geodesic(mask: np.ndarray, start: tuple[int, int], weight: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra over the 8-connected pixels of ``mask`` from ``start`` (row, col): ``(dist, prev)`` as flat arrays of the
    mask padded by one pixel (``dist`` -1 where unreachable, ``prev`` -1 at the start). ``weight`` scales each step by
    the weight of the pixel entered."""
    pad = np.pad(mask, 1)
    h, w = pad.shape
    flat = pad.ravel().tolist()
    wt = None if weight is None else np.pad(weight, 1, constant_values=1.0).ravel().tolist()
    offsets = [(dy * w + dx, math.hypot(dy, dx)) for dy, dx in _STEPS]
    inf = math.inf
    dist = [inf] * (h * w)
    prev = [-1] * (h * w)
    s = (start[0] + 1) * w + start[1] + 1
    dist[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for off, step in offsets:
            nxt = cur + off
            if flat[nxt]:
                nd = d + step * (1.0 if wt is None else wt[nxt])
                if nd < dist[nxt]:
                    dist[nxt], prev[nxt] = nd, cur
                    heapq.heappush(heap, (nd, nxt))
    out = np.array(dist)
    out[~np.isfinite(out)] = -1.0
    return out, np.array(prev)


def _farthest(dist: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    """(row, col) in the unpadded mask of the largest geodesic distance."""
    i = int(np.argmax(dist))
    return i // (shape[1] + 2) - 1, i % (shape[1] + 2) - 1


def _path(prev: np.ndarray, shape: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    """Pixel ``(row, col)`` path from the geodesic's start to ``end``."""
    w = shape[1] + 2
    i = (end[0] + 1) * w + end[1] + 1
    out = []
    while i >= 0:
        out.append((i // w - 1, i % w - 1))
        i = int(prev[i])
    return np.array(out[::-1], dtype=float)


# --------------------------------------------------------------------------- rendering (to score a candidate)
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def _render_tube(shape: tuple[int, int], origin: tuple[int, int], path: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Flat-capped tube of half-width ``radii`` along the polyline ``path`` (``[x, y]`` image pixels): union of the quads
    between consecutive samples, so a tight curve never leaves an even-odd hole."""
    out = np.zeros(shape, dtype=bool)
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left, right = path + nrm * radii[:, None], path - nrm * radii[:, None]
    ox = np.array(origin, dtype=float)
    for i in range(len(path) - 1):
        out |= polygon_mask(shape, np.array([left[i], left[i + 1], right[i + 1], right[i]]) - ox)
    return out


def _render_capsule(shape: tuple[int, int], origin: tuple[int, int], a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Pixels within ``r`` of the segment a-b (the capsule's core line)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    p = np.stack([xx + 0.5 + origin[0], yy + 0.5 + origin[1]], axis=-1).astype(float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.hypot(*(p - (a + t[..., None] * ab)).transpose(2, 0, 1)) <= r


def _open(crop: np.ndarray) -> np.ndarray:
    """``crop`` without spikes thinner than ``2 * OPEN_PX + 1`` (fur, an outline stroke along a sleeve) that would move the
    axis ends; the largest piece, or ``crop`` itself when opening would eat more than ``OPEN_MAX_LOSS`` of it."""
    opened = dilate(erode(crop, OPEN_PX), OPEN_PX) & crop
    labels, n = label_components(opened)
    if n > 1:
        opened = labels == (1 + int(np.argmax(np.bincount(labels.ravel())[1:])))
    return opened if opened.sum() >= (1.0 - OPEN_MAX_LOSS) * crop.sum() else crop


def _xy(p: np.ndarray) -> list[float]:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


# --------------------------------------------------------------------------- candidates
def _line_fit(frac: np.ndarray, value: np.ndarray, sel: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Value at frac 0 and 1 of a line fitted on ``sel``, clipped to ``[lo, hi]``."""
    if sel.sum() >= 3:
        slope, icpt = np.polyfit(frac[sel], value[sel], 1)
        a, b = float(icpt), float(icpt + slope)
    else:
        a = b = float(np.median(value[value > 0])) if (value > 0).any() else lo
    return min(max(a, lo), hi), min(max(b, lo), hi)


def _axis_model(pts: np.ndarray, dt_px: np.ndarray, lo_pt: np.ndarray, hi_pt: np.ndarray) -> dict[str, Any]:
    """Cone / capsule dimensions of the mask pixels ``pts`` (``[x, y]``, ``dt_px`` = their distance to the boundary) for a
    straight axis from ``lo_pt`` to ``hi_pt``: the half-width line along the axis and the capsule radius from the area."""
    span = hi_pt - lo_pt
    length = max(float(np.hypot(*span)), 1.0)
    u = span / length
    t = (pts - lo_pt) @ u
    n_bins = int(length) + 1
    bins = np.clip((t / length * n_bins).astype(int), 0, n_bins - 1)
    count = np.bincount(bins, minlength=n_bins).astype(float)
    slice_r = np.zeros(n_bins)
    np.maximum.at(slice_r, bins, dt_px)
    half = np.minimum(count / 2.0, 1.5 * slice_r)
    frac = (np.arange(n_bins) + 0.5) / n_bins
    thick = float(dt_px.max())
    r_lo, r_hi = _line_fit(frac, half, (frac >= 0.1) & (frac <= 0.9) & (count > 0), MIN_RADIUS_PX, max(1.5 * thick, MIN_RADIUS_PX))
    disc = length * length - (4.0 - math.pi) * float(len(pts))      # capsule: area = pi r^2 + 2 r (length - 2 r)
    r_cap = (length - math.sqrt(max(disc, 0.0))) / (4.0 - math.pi)
    r_cap = min(max(r_cap, MIN_RADIUS_PX), 1.1 * thick, length / 2.0)
    return {"lo": lo_pt, "hi": hi_pt, "length": length, "r_lo": r_lo, "r_hi": r_hi, "r_cap": r_cap}


def _apex_end(pts: np.ndarray, apex: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Far end of the line from ``apex`` through ``centre``: the mask pixel on it (within 2 px) farthest from ``apex``."""
    u = centre - apex
    u = u / max(float(np.hypot(*u)), 1e-9)
    rel = pts - apex
    along, across = rel @ u, np.abs(rel @ np.array([-u[1], u[0]]))
    near = across <= 2.0
    return apex + u * float(along[near].max() if near.any() else along.max())


def _straight(crop: np.ndarray, origin: tuple[int, int], dt: np.ndarray) -> list[dict[str, Any]]:
    """Straight-axis models of ``crop`` (a bool mask cut at ``origin``): along the principal axis (``pca``; its extreme
    pixels are the ends) and along the line from each of two extreme pixels through the centroid (``apex``: an ear or a horn
    whose axis is not its principal axis, because it is a fan)."""
    rr, cc = np.nonzero(crop)
    pts = np.stack([cc + 0.5 + origin[0], rr + 0.5 + origin[1]], axis=1).astype(float)
    dt_px = dt[rr, cc]
    centre = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    t = (pts - centre) @ vecs[:, 1]
    ends = [(pts[t <= t.min() + 1.0].mean(axis=0), pts[t >= t.max() - 1.0].mean(axis=0))]
    e1 = pts[int(np.argmax(np.hypot(*(pts - centre).T)))]
    e2 = pts[int(np.argmax(np.hypot(*(pts - e1).T)))]
    ends += [(e, _apex_end(pts, e, centre)) for e in (e1, e2)]
    return [_axis_model(pts, dt_px, lo, hi) for lo, hi in ends]


def _curved(crop: np.ndarray, origin: tuple[int, int], dt: np.ndarray, contact: Optional[np.ndarray] = None
            ) -> Optional[dict[str, Any]]:
    """Curved-cone candidate: centre-favouring geodesic between two geodesically farthest pixels. With a ``contact`` mask
    (part pixels touching the body) the base end is the contact (its pixel nearest the contact centroid) and the tip is
    the pixel geodesically farthest from it; else the two farthest-apart pixels (the ends of a J)."""
    thick = float(dt[crop].max())
    if contact is not None and 0 < contact.sum() < 0.5 * crop.sum():
        cr, cc = np.nonzero(contact)
        k = int(np.argmin((cr - cr.mean()) ** 2 + (cc - cc.mean()) ** 2))
        p1 = (int(cr[k]), int(cc[k]))
    else:
        heart = np.unravel_index(int(np.argmax(np.where(crop, dt, -1.0))), crop.shape)
        d0, _ = _geodesic(crop, (int(heart[0]), int(heart[1])))
        p1 = _farthest(d0, crop.shape)
    d1, _ = _geodesic(crop, p1)
    p2 = _farthest(d1, crop.shape)
    if p1 == p2:
        return None
    pull = 1.0 + CENTRE_PULL * (1.0 - np.clip(dt / max(thick, 1e-9), 0.0, 1.0)) ** 2
    _, prev = _geodesic(crop, p1, pull)
    raw = _path(prev, crop.shape, p2)
    if len(raw) < 4:
        return None
    radii_raw = dt[raw[:, 0].astype(int), raw[:, 1].astype(int)]
    xy = _smooth(raw[:, ::-1] + 0.5 + np.array(origin, dtype=float), k=max(5, (len(raw) // 10) | 1))
    s = _arclength(xy)
    length = float(s[-1])
    dense = np.arange(0.0, length, 1.0)
    dense = np.append(dense, length) if dense[-1] < length else dense
    path = np.stack([np.interp(dense, s, xy[:, 0]), np.interp(dense, s, xy[:, 1])], axis=1)
    frac = dense / max(length, 1e-9)
    rad = np.interp(dense, s, _smooth(np.stack([radii_raw, radii_raw], axis=1), k=5)[:, 0])
    r_a, r_b = _line_fit(frac, rad, (frac >= 0.12) & (frac <= 0.88), MIN_RADIUS_PX, max(1.2 * thick, MIN_RADIUS_PX))
    chord = path[-1] - path[0]
    off = np.abs((path - path[0]) @ np.array([-chord[1], chord[0]])) / max(float(np.hypot(*chord)), 1e-9)
    return {"path": path, "length": length, "r_a": r_a, "r_b": r_b, "bend": float(off.max()) / max(length, 1e-9),
            "mid": path[len(path) // 2]}


def suggest(crop: np.ndarray, origin: tuple[int, int], rest: Optional[np.ndarray] = None,
            base_hint: Optional[Sequence[float]] = None, force_type: Optional[str] = None,
            contact: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Best of the candidate shapes for the bool mask ``crop`` whose pixel (0, 0) is image pixel ``origin`` ``(x, y)``.

    Returns the anchors of :func:`~bbr.core.measure.propose_anchors` plus ``suggested_type`` (``sphere`` / ``capsule`` /
    ``cone``), ``confidence`` (IoU of the chosen shape with the mask) and ``candidates`` (IoU per type). ``rest`` =
    ``[x, y]`` points of what the part is attached to (orients base / tip), ``contact`` = the part pixels touching it (the
    base end of a curved part); ``force_type`` restricts the choice to one type
    (the same type in every view of a part)."""
    if force_type is not None and force_type not in TYPE_RANK:
        raise BBRError(f"type must be one of {sorted(TYPE_RANK)}, got {force_type!r}")
    crop = _open(crop)
    shape = crop.shape
    rr, cc = np.nonzero(crop)
    area = len(rr)
    if area < 12:
        raise BBRError("the part is too small to find an axis in")
    dt = edt(~np.pad(crop, 2), cap=int(min(max(shape) + 2, 160)))[2:-2, 2:-2] - 0.5
    centre = np.array([cc.mean() + 0.5 + origin[0], rr.mean() + 0.5 + origin[1]])
    r_sphere = math.sqrt(area / math.pi)
    cands: dict[str, dict[str, Any]] = {SPHERE: {"iou": _iou(crop, _render_capsule(shape, origin, centre, centre, r_sphere))}}
    models = _straight(crop, origin, dt)
    pca = models[0]
    u = (pca["hi"] - pca["lo"]) / max(float(np.hypot(*(pca["hi"] - pca["lo"]))), 1e-9)
    inset = min(pca["r_cap"], pca["length"] / 2.0)
    cands[CAPSULE] = {"iou": _iou(crop, _render_capsule(shape, origin, pca["lo"] + u * inset, pca["hi"] - u * inset, pca["r_cap"]))}
    cone_s, straight = -1.0, pca
    for m in models:
        n = int(m["length"]) + 2
        chord = np.stack([np.linspace(m["lo"][0], m["hi"][0], n), np.linspace(m["lo"][1], m["hi"][1], n)], axis=1)
        iou = _iou(crop, _render_tube(shape, origin, chord, np.linspace(m["r_lo"], m["r_hi"], n)))
        if iou > cone_s:
            cone_s, straight = iou, m
    lo, hi = straight["lo"], straight["hi"]
    cone_c, curved = -1.0, None
    for seed in ((None, contact) if contact is not None else (None,)):   # the J's farthest ends, then the end that touches the body
        c = _curved(crop, origin, dt, seed) if area >= 40 else None
        if c is not None:
            iou = _iou(crop, _render_tube(shape, origin, c["path"], np.linspace(c["r_a"], c["r_b"], len(c["path"]))))
            if iou > cone_c:
                cone_c, curved = iou, c
    curl = bool(curved and curved["bend"] >= CURL_BEND and cone_c >= CURL_MIN_IOU and cone_c >= cone_s - CURVE_GAIN)
    use_curve = bool(curved and (cone_c >= cone_s + CURVE_GAIN or curl))
    cands[CONE] = {"iou": cone_c if use_curve else cone_s, "curved": use_curve}
    cyl = _render_tube(shape, origin, np.stack([np.linspace(pca["lo"][0], pca["hi"][0], int(pca["length"]) + 2),
                                                np.linspace(pca["lo"][1], pca["hi"][1], int(pca["length"]) + 2)], axis=1),
                       np.full(int(pca["length"]) + 2, (pca["r_lo"] + pca["r_hi"]) / 2.0))
    cands[CYLINDER] = {"iou": _iou(crop, cyl)}
    if force_type:
        chosen = force_type
    else:
        chosen = SPHERE
        for kind in (CAPSULE, CYLINDER, CONE):
            gain = CONE_OVER_SPHERE_GAIN if (kind == CONE and chosen == SPHERE) else GAIN
            if cands[kind]["iou"] >= cands[chosen]["iou"] + gain:
                chosen = kind
        if use_curve and curl:
            chosen = CONE
    candidates = {k: round(v["iou"], 3) for k, v in cands.items()}
    out: dict[str, Any]
    if chosen == SPHERE:
        out = {"kind": "sphere", "center": _xy(centre), "radius_px": round(r_sphere, 1)}
    else:
        if chosen in (CAPSULE, CYLINDER):
            a, b, mid = pca["lo"], pca["hi"], None
            ra = rb = pca["r_cap"] if chosen == CAPSULE else (pca["r_lo"] + pca["r_hi"]) / 2.0
            cone_curved = False
        else:
            cone_curved = cands[CONE]["curved"]
            if cone_curved:
                a, b, ra, rb, mid = curved["path"][0], curved["path"][-1], curved["r_a"], curved["r_b"], curved["mid"]
            else:
                a, b, ra, rb, mid = lo, hi, straight["r_lo"], straight["r_hi"], (lo + hi) / 2.0
        span = float(np.hypot(*(b - a)))
        wide = chosen == CONE and max(ra, rb) >= TAPER_CLEAR * min(ra, rb)
        verdict = attach_base(a, b, span, rest)
        if base_hint is not None:
            hint = np.asarray(base_hint, dtype=float)
            b_is_base, base_by = bool(np.linalg.norm(hint - b) < np.linalg.norm(hint - a)), "hint"
        elif verdict is not None:
            b_is_base, base_by = verdict, "attach"
        elif wide:
            b_is_base, base_by = rb > ra, "width"
        else:
            b_is_base, base_by = bool(b[1] > a[1]), "lower"
        if b_is_base:
            a, b, ra, rb = b, a, rb, ra
        out = {"kind": "tube" if chosen == CONE else "capsule", "base_by": base_by, "base": _xy(a)}
        if chosen == CONE:
            out["mid"] = _xy(mid)
        out["tip"] = _xy(b)
        out["radius_px"] = [round(ra, 1), round(rb, 1)] if chosen == CONE else round(ra, 1)
        out["length_px"] = round(span, 1)
        if chosen == CONE and cone_curved:
            out["warnings"] = [f"curved part ({100 * curved['bend']:.0f}% of its length off the chord): base / mid / tip follow its centre line"]
    out["suggested_type"] = chosen
    out["curled"] = bool(chosen == CONE and use_curve and curl)
    out["confidence"] = round(max(0.0, cands[chosen]["iou"]), 2)
    out["candidates"] = candidates
    return out
