"""Measuring reference images (numpy + PIL only): pixel -> world coordinates with the pipeline's own calibration.

Agents use this to place parts (horns, a sword, a shield) from the images instead of guessing. A pixel of an
orthographic view fixes the two world coordinates in the image plane (e.g. front: x and z); the third (depth) is
unknown from that view: read it from the perpendicular view.
"""
from __future__ import annotations

import heapq
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .. import BBRError
from . import imageio, imageops, labels as labels_core
from .distance import edt
from .calibration import ViewCalibration, calibrate_views
from .config import PipelineConfig
from .conventions import view_frame
from .spec import ViewSpec, load_view_spec

AXIS_NAMES = "xyz"


def load_view_data(view_spec: ViewSpec, config: PipelineConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load (crop + flip applied) images and foreground masks of every enabled, usable view (preview-only views skipped)."""
    rgba: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for view in view_spec.enabled_views():
        if not view.is_orthographic_usable:
            continue
        img = imageio.load_view_image(view)
        bg = view.background
        rgba[view.key] = img
        masks[view.key] = imageops.extract_mask(img, bg.mode, bg.color, bg.tolerance, config.max_hole_frac,
                                                config.min_component_frac, bg.remove_enclosed)
    return rgba, masks


def load_calibrations(view_spec_path: Path | str, config: Optional[PipelineConfig] = None,
                      labels_path: Optional[Path | str] = None) -> tuple[ViewSpec, dict[str, ViewCalibration], dict[str, Any]]:
    """``(view_spec, calibrations, info)`` exactly as the pipeline computes them (``info`` = dimensions, provenance, bboxes_px).
    ``labels_path``: the scene's ``parts_labels`` file, whose ``body`` label can move the body axis (the pipeline passes it)."""
    spec = load_view_spec(view_spec_path)
    _, masks = load_view_data(spec, config or PipelineConfig())
    label_masks = labels_core.load_label_masks(labels_path, masks) if labels_path else None
    calibs, info = calibrate_views(spec, masks, label_masks)
    return spec, calibs, info


def plane_axes(calib: ViewCalibration) -> tuple[str, str, str]:
    """World axis letters ``(horizontal, vertical, depth)`` of a view, e.g. front: ``('x', 'z', 'y')``."""
    f = view_frame(calib.camera_axis)
    return f.right_axis[1].lower(), f.up_axis[1].lower(), f.camera_axis[1].lower()


def pixel_to_world(calib: ViewCalibration, points_px: Sequence[Sequence[float]]) -> np.ndarray:
    """Pixel coordinates ``(N,2)`` (u right, v down, pixel centres at +0.5) -> world points ``(N,3)`` on the plane through
    the origin facing the camera; the depth coordinate is 0 (unknown)."""
    return calib.px_to_plane(np.asarray(points_px, dtype=float).reshape(-1, 2))


def _in_plane(calib: ViewCalibration, world: np.ndarray) -> list[dict[str, float]]:
    h, v, _ = plane_axes(calib)
    return [{h: round(float(p[AXIS_NAMES.index(h)]), 5), v: round(float(p[AXIS_NAMES.index(v)]), 5)} for p in world]


def measure_view(calib: ViewCalibration, points: Sequence[Sequence[float]] = (),
                 bbox: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """World coordinates of image pixels ``points`` ``[[x, y], ...]`` and/or of a pixel box ``bbox = [x0, y0, x1, y1]``.

    Result: ``view``, ``plane`` (the two measurable world axes), ``depth_axis`` (not measurable here), ``m_per_px``,
    ``points`` (``{px, world}`` with world keyed by axis letter) and ``bbox`` (``min``/``max``/``size`` per plane axis).
    """
    h, v, d = plane_axes(calib)
    out: dict[str, Any] = {"view": calib.view, "camera_axis": calib.camera_axis, "plane": [h, v], "depth_axis": d,
                           "m_per_px": calib.m_per_px, "origin_px": list(calib.origin_px), "image_size": list(calib.image_size), "points": [], "bbox": None}
    pts = [list(map(float, p)) for p in points]
    if pts:
        for px, world in zip(pts, _in_plane(calib, pixel_to_world(calib, pts))):
            out["points"].append({"px": px, "world": world})
    if bbox is not None:
        if len(bbox) != 4:
            raise BBRError(f"bbox needs x0,y0,x1,y1, got {list(bbox)}")
        x0, y0, x1, y1 = (float(b) for b in bbox)
        corners = _in_plane(calib, pixel_to_world(calib, [[x0, y0], [x1, y1]]))
        out["bbox"] = {"px": [x0, y0, x1, y1],
                       "min": {a: round(min(c[a] for c in corners), 5) for a in (h, v)},
                       "max": {a: round(max(c[a] for c in corners), 5) for a in (h, v)},
                       "size": {a: round(abs(corners[1][a] - corners[0][a]), 5) for a in (h, v)}}
    if not pts and bbox is None:
        raise BBRError("nothing to measure: pass points and/or a bbox")
    return out


def measure_reference(view_spec_path: Path | str, view: str, points: Sequence[Sequence[float]] = (),
                      bbox: Optional[Sequence[float]] = None, config: Optional[PipelineConfig] = None,
                      labels_path: Optional[Path | str] = None) -> dict[str, Any]:
    """:func:`measure_view` for one view of a ViewSpec file, calibrated like the pipeline (silhouette bbox + known height)."""
    spec, calibs, info = load_calibrations(view_spec_path, config, labels_path)
    if view not in calibs:
        raise BBRError(f"view {view!r} is not calibrated; calibrated views: {sorted(calibs)}")
    res = measure_view(calibs[view], points, bbox)
    res["dimensions"] = {k: round(v, 5) for k, v in info["dimensions"].items()}
    return res


def pixel_formula(calib: ViewCalibration) -> dict[str, str]:
    """The pixel -> metre formula of a view as text per world axis (``u`` right, ``v`` down): what ``px_to_plane`` computes."""
    f = view_frame(calib.camera_axis)
    ox, oy = calib.origin_px
    m = f"{calib.m_per_px:.7f}"

    def expr(sign: str, a: str, b: str) -> str:      # sign * (a - b) * m
        return f"({a} - {b}) * {m}" if sign == "+" else f"({b} - {a}) * {m}"

    return {f.right_axis[1].lower(): expr(f.right_axis[0], "u", f"{ox:g}"), f.up_axis[1].lower(): expr(f.up_axis[0], f"{oy:g}", "v")}


def calibration_summary(calibs: dict[str, ViewCalibration]) -> dict[str, Any]:
    """``{view: {camera_axis, m_per_px, origin_px, formula}}``: the numbers a batch pixel -> metre conversion needs."""
    return {k: {"camera_axis": c.camera_axis, "m_per_px": c.m_per_px, "origin_px": list(c.origin_px), "formula": pixel_formula(c)}
            for k, c in calibs.items()}


def measure_points(view_spec_path: Path | str, points: dict[str, dict[str, Sequence[float]]],
                   config: Optional[PipelineConfig] = None, labels_path: Optional[Path | str] = None) -> dict[str, Any]:
    """Batch :func:`measure_view`: ``points`` = ``{view: {name: [x, y] | [x0, y0, x1, y1]}}`` (pixels; two numbers a point, four
    a box), the same calibration as the pipeline. Returns ``{"units": "m", "views": {view: {name: {axis: metres, ...}}},
    "calibration", "dimensions"}``; a point gives its two plane axes, a box ``{min, max, size}`` per plane axis."""
    spec, calibs, info = load_calibrations(view_spec_path, config, labels_path)
    out: dict[str, Any] = {}
    for view, named in points.items():
        if view not in calibs:
            raise BBRError(f"view {view!r} is not calibrated; calibrated views: {sorted(calibs)}")
        if not isinstance(named, dict):
            raise BBRError(f"view {view!r}: expected {{name: [x, y] | [x0, y0, x1, y1]}}, got {type(named).__name__}")
        h, v, _ = plane_axes(calibs[view])
        res: dict[str, Any] = {}
        for name, px in named.items():
            if not isinstance(px, (list, tuple)) or len(px) not in (2, 4) or not all(isinstance(c, (int, float)) for c in px):
                raise BBRError(f"{view}.{name}: need [x, y] or [x0, y0, x1, y1] pixels, got {px!r}")
            m = measure_view(calibs[view], [px] if len(px) == 2 else (), px if len(px) == 4 else None)
            res[name] = m["points"][0]["world"] if len(px) == 2 else {k: m["bbox"][k] for k in ("min", "max", "size")}
        out[view] = res
    return {"units": "m", "views": out, "calibration": calibration_summary({k: calibs[k] for k in points}),
            "dimensions": {k: round(v, 5) for k, v in info["dimensions"].items()}}


def format_measurement(res: dict[str, Any]) -> str:
    """Human-readable lines of a :func:`measure_view` result."""
    h, v = res["plane"]
    lines = [f"view {res['view']} (camera {res['camera_axis']}): image x -> world {h}, image y -> world {v}; "
             f"{res['depth_axis']} is depth (not measurable here); {res['m_per_px'] * 1000:.3f} mm/px"]
    for item in res["points"]:
        w = item["world"]
        lines.append(f"  px ({item['px'][0]:g}, {item['px'][1]:g}) -> {h}={w[h]:+.4f} {v}={w[v]:+.4f}")
    if res["bbox"]:
        b = res["bbox"]
        lines.append(f"  bbox px {b['px']} -> {h} [{b['min'][h]:+.4f}, {b['max'][h]:+.4f}] size {b['size'][h]:.4f}; "
                     f"{v} [{b['min'][v]:+.4f}, {b['max'][v]:+.4f}] size {b['size'][v]:.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- anchor proposals
ANCHOR_KINDS = ("auto", "tube", "capsule", "blade", "plate", "disc", "sphere", "strand")   # plate = root edge + tip of a wing / fin (its outline is the label mask); strand = N points along a curved tube (hair lock, braid, cable); tube = cone / horn / limb with a taper; auto = pick one (see partshape)
_NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
END_FIT_TRIM = 0.12      # the radius fit ignores this share of the path at each end
MIN_RADIUS_PX = 0.5
END_TRIM = 0.25          # the end stretch of a skeleton path (share of its length) is not trusted: thinning bends it to a corner
END_WINDOW = 0.6         # the end is extrapolated from the path between END_TRIM and END_WINDOW
POINT_RATIO = 0.35       # an end narrower than this share of the other end is a point (horn tip): snapped to its extreme pixel
POINT_REACH = 0.25       # ... searched within this share of the length (min POINT_REACH_MIN_PX) of the extrapolated end
POINT_REACH_MIN_PX = 8.0
POINT_MAX_DT_PX = 3.5    # ... among pixels this close to the boundary
ROD_RATIO = 4.0          # sqrt(eigenvalue ratio) of the pixel scatter from which a part is a rod: ends from the principal axis (PCA)
ROD_VERTICAL = 0.35      # a rod whose axis rises at least this share of its length has base = its lower end (else: the wider end)
ROD_END_BAND = (0.05, 0.35)   # radius = median of the distance transform along the medial axis inside this share of the length at each end
ROD_FULL_CONFIDENCE_RATIO = 6.0
ROD_CONE_RATIO = 1.6          # a rod whose end radii differ by this much is a cone (tapered), else a capsule
ROD_ENDS_RATIO_WARN = 2.5     # base / tip radius ratio above which the label probably includes a head or twist
TAPER_CLEAR = 1.25       # end half-widths ratio from which a tapered part's base is its wider end
ATTACH_CLEAR = 0.15      # one end must be this much nearer (share of the length) to the rest of the silhouette than the other
CONTACT_REACH_FACTOR = 1.5   # an end's neighbourhood reaches this many part half-widths, or CONTACT_REACH_LENGTH of the length ...
CONTACT_REACH_LENGTH = 0.25  # (the larger; a label often reaches into the head, so the ends sit a little inside the part)
CONTACT_REACH_MIN_PX = 6.0
CONTACT_CLEAR = 2.0          # ... and the attachment end touches this many times more other foreground than the free end
CONTACT_MIN_POINTS = 20      # ... counted in at least this many points (rest_points samples)
STRAND_POINTS = 5            # default number of anchor points of a strand
MIN_STRAND_POINTS = 3
STRAND_VERTICAL = 0.5        # a strand whose height span is this share of its length is sampled at equal height steps ...
STRAND_MONOTONE_TOL = 0.03   # ... when it never backtracks by more than this share of its length (the views then agree on z)
MAX_REST_POINTS = 20000
CONTACT_PX = 3           # part pixels within this of the body count as touching it
END_OVERSHOOT = 0.25     # ... at most this share of the length beyond the trusted part


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: 1-px wide, 8-connected skeleton of a bool mask (numpy only)."""
    img = np.asarray(mask, dtype=np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            n = [p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx] for dy, dx in _NEIGHBOURS]   # P2..P9
            b = sum(x.astype(np.int16) for x in n)
            a = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = ((n[0] * n[2] * n[4]) == 0) & ((n[2] * n[4] * n[6]) == 0)
            else:
                c = ((n[0] * n[2] * n[6]) == 0) & ((n[0] * n[4] * n[6]) == 0)
            rm = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
    return img.astype(bool)


def _dijkstra(pixels: set, start: tuple) -> tuple[dict, dict]:
    dist = {start: 0.0}
    prev: dict = {}
    heap = [(0.0, start)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for dy, dx in _NEIGHBOURS:
            nxt = (cur[0] + dy, cur[1] + dx)
            if nxt in pixels:
                nd = d + math.hypot(dy, dx)
                if nd < dist.get(nxt, math.inf):
                    dist[nxt], prev[nxt] = nd, cur
                    heapq.heappush(heap, (nd, nxt))
    return dist, prev


def _longest_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Longest geodesic path ``[(row, col), ...]`` of a skeleton (double sweep per connected piece)."""
    pixels = {(int(r), int(c)) for r, c in zip(*np.nonzero(skeleton))}
    best: list[tuple[int, int]] = []
    seen: set = set()
    for seed in pixels:
        if seed in seen:
            continue
        dist, _ = _dijkstra(pixels, seed)
        seen.update(dist)
        a = max(dist, key=dist.get)
        dist_a, prev = _dijkstra(pixels, a)
        b = max(dist_a, key=dist_a.get)
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        if len(path) > len(best):
            best = path
    return best


def _smooth(path: np.ndarray, k: int = 5) -> np.ndarray:
    """Moving average of a polyline, end points kept."""
    if len(path) < k + 2:
        return path
    pad = k // 2
    ext = np.concatenate([np.repeat(path[:1], pad, 0), path, np.repeat(path[-1:], pad, 0)])
    out = np.stack([np.convolve(ext[:, i], np.ones(k) / k, mode="valid") for i in range(2)], axis=1)
    out[0], out[-1] = path[0], path[-1]
    return out


def _arclength(path: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def _at(path: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def _extend_end(path: np.ndarray, work: np.ndarray, lo: np.ndarray, hi: np.ndarray, at_start: bool
                ) -> tuple[np.ndarray, int]:
    """New end point of a skeleton path, and the path index from which the original polyline is kept.

    Thinning bends the last stretch of a flat or pointed end toward a corner of the outline, so the end region is
    discarded: a quadratic fitted to the path between ``END_TRIM`` and ``END_WINDOW`` of its length is extrapolated
    outward until it leaves the mask (or the box edge)."""
    s = _arclength(path)
    total = float(s[-1])
    so = s if at_start else total - s                       # distance from this end
    sel = (so >= END_TRIM * total) & (so <= END_WINDOW * total)
    if sel.sum() < 4:
        sel = so >= 0
    t = so[sel]
    deg = 2 if sel.sum() >= 6 else 1
    cr, cc = np.polyfit(t, path[sel][:, 0], deg), np.polyfit(t, path[sel][:, 1], deg)
    t0 = float(t.min())
    end = np.array([np.polyval(cr, t0), np.polyval(cc, t0)])
    for tt in np.arange(t0, -END_OVERSHOOT * total, -0.5):
        cur = np.array([np.polyval(cr, tt), np.polyval(cc, tt)])
        r, c = int(round(cur[0])), int(round(cur[1]))
        if not (lo[0] <= cur[0] <= hi[0] and lo[1] <= cur[1] <= hi[1]) or not work[r, c]:
            break
        end = cur
    keep = int(np.flatnonzero(so >= t0)[0] if at_start else np.flatnonzero(so >= t0)[-1])
    return end, keep


def rest_points(rest: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """``(N, 2)`` pixel ``[x, y]`` points of a bool ``rest`` mask (thinned to at most ``MAX_REST_POINTS``), None when empty."""
    if rest is None or not np.asarray(rest).any():
        return None
    rr, cc = np.nonzero(rest)
    step = max(1, len(rr) // MAX_REST_POINTS)
    return np.stack([cc[::step] + 0.5, rr[::step] + 0.5], axis=1).astype(float)


def contact_reach(half_width: float, length: float) -> float:
    """Neighbourhood radius (px) of an end of a part for the attachment test of :func:`attach_base`."""
    return max(CONTACT_REACH_MIN_PX, CONTACT_REACH_FACTOR * half_width, CONTACT_REACH_LENGTH * length)


def attach_base(end_a: np.ndarray, end_b: np.ndarray, length: float, rest: Optional[np.ndarray],
                reach: Optional[float] = None) -> Optional[bool]:
    """True when ``end_b`` is the base, False for ``end_a``, None when undecided.

    With ``reach`` (px) the base is the end whose neighbourhood touches the most ``rest`` pixels (the other foreground:
    body, hair), when that is clearly more than at the other end: a tail that HANGS from the head touches the head at its
    top and nothing at its tip, whatever the distances to the rest of the silhouette are. Otherwise the end clearly nearer
    ``rest`` (``ATTACH_CLEAR`` of the length) is the base."""
    if rest is None or length <= 0:
        return None
    if reach is not None:
        ca = int((np.hypot(*(rest - end_a).T) <= reach).sum())
        cb = int((np.hypot(*(rest - end_b).T) <= reach).sum())
        if max(ca, cb) >= CONTACT_MIN_POINTS and max(ca, cb) >= CONTACT_CLEAR * max(min(ca, cb), 1):
            return cb > ca
    da = float(np.sqrt(((rest - end_a) ** 2).sum(axis=1)).min())
    db = float(np.sqrt(((rest - end_b) ** 2).sum(axis=1)).min())
    if abs(da - db) < ATTACH_CLEAR * length:
        return None
    return db < da


def _elongation(mask: np.ndarray) -> float:
    """sqrt(largest / smallest eigenvalue) of the pixel scatter of ``mask`` (about 1 for a disc, large for a rod)."""
    rr, cc = np.nonzero(mask)
    if rr.size < 3:
        return 1.0
    eig = np.linalg.eigvalsh(np.cov(np.stack([cc, rr]).astype(float)))
    return float(math.sqrt(max(eig[1], 1e-9) / max(eig[0], 1e-3)))


def _rod_anchors(crop: np.ndarray, kind: str, origin: tuple[int, int], base_hint: Optional[Sequence[float]],
                 rest: Optional[np.ndarray] = None, min_ratio: float = ROD_RATIO) -> Optional[dict[str, Any]]:
    """Anchors of an elongated mask (staff, spear, thin limb) from its principal axis; None when it is not a rod.

    Endpoints are the extreme pixels along the first principal axis (so a twist, a head or a skeleton spur at one
    end cannot move them). The radius is the median over a stretch of the per-slice maximum of the distance
    transform (the medial-axis radius), clipped to the mask's thickness. base = the lower end (larger image y) when
    the axis is mostly vertical, else the wider end; ``base_hint`` ``[x, y]`` wins, then the end clearly nearer ``rest``
    (``[x, y]`` points of the body: a limb or horn is attached there). ``confidence`` falls with a short axis ratio and
    a curved centre line.
    """
    ratio = _elongation(crop)
    if ratio < min_ratio:
        return None
    rr, cc = np.nonzero(crop)
    pts = np.stack([cc + 0.5 + origin[0], rr + 0.5 + origin[1]], axis=1).astype(float)
    centre = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    axis, normal = vecs[:, 1], vecs[:, 0]
    t, q = (pts - centre) @ axis, (pts - centre) @ normal
    length = float(t.max() - t.min())
    if length < 3.0:
        return None
    dt = edt(~np.pad(crop, 2), cap=int(min(max(crop.shape), 128))) - 0.5
    dt = dt[2:-2, 2:-2][rr, cc]
    n_bins = int(length) + 1
    bins = np.clip((t - t.min()).astype(int), 0, n_bins - 1)
    slice_r = np.zeros(n_bins)       # medial-axis radius per slice = max distance transform in the slice
    slice_q = np.full(n_bins, np.nan)
    np.maximum.at(slice_r, bins, dt)
    for b in np.unique(bins):
        sel = bins == b
        slice_q[b] = q[sel][np.argmax(dt[sel])]
    frac = (np.arange(n_bins) + 0.5) / n_bins
    thick = float(dt.max())

    def radius(lo: float, hi: float) -> float:
        sel = (frac >= lo) & (frac <= hi) & (slice_r > 0)
        med = np.median(slice_r[sel]) if sel.any() else np.median(slice_r[slice_r > 0])
        return float(min(max(med, MIN_RADIUS_PX), thick))

    def end_point(sign: float) -> np.ndarray:
        near = t >= t.max() - 1.0 if sign > 0 else t <= t.min() + 1.0
        return pts[near].mean(axis=0)

    lo_pt, hi_pt = end_point(-1.0), end_point(1.0)
    r_lo, r_hi = radius(*ROD_END_BAND), radius(1.0 - ROD_END_BAND[1], 1.0 - ROD_END_BAND[0])
    if base_hint is not None:
        hint = np.asarray(base_hint, dtype=float)
        hi_is_base, base_by = bool(np.linalg.norm(hint - hi_pt) < np.linalg.norm(hint - lo_pt)), "hint"
    elif (attached := attach_base(lo_pt, hi_pt, length, rest, contact_reach(thick, length))) is not None:
        hi_is_base, base_by = attached, "attach"
    elif abs(hi_pt[1] - lo_pt[1]) >= ROD_VERTICAL * length:
        hi_is_base, base_by = bool(hi_pt[1] > lo_pt[1]), "lower"
    else:
        hi_is_base, base_by = bool(r_hi > r_lo or (r_hi == r_lo and hi_pt[1] > lo_pt[1])), "width"
    base, tip = (hi_pt, lo_pt) if hi_is_base else (lo_pt, hi_pt)
    rb, rt = (r_hi, r_lo) if hi_is_base else (r_lo, r_hi)
    mid_r = radius(0.25, 0.75)
    mid = pts[np.abs(t - (t.max() + t.min()) / 2.0) <= 1.0].mean(axis=0)
    valid = ~np.isnan(slice_q)
    inner = slice(n_bins // 10, n_bins - n_bins // 10)
    curve = float(np.ptp(slice_q[inner][valid[inner]])) / length if valid[inner].sum() > 4 else 0.0
    warnings = []
    if max(rb, rt) > ROD_ENDS_RATIO_WARN * min(rb, rt):
        warnings.append(f"base / tip radii {rb:.1f} / {rt:.1f} px differ a lot: the label may include a head or twist; "
                        "check the overlay and the end points")
    if curve > 0.08:
        warnings.append(f"the part bends ({100 * curve:.0f}% of its length off the straight axis): a straight anchor line is approximate")

    def xy(p: np.ndarray) -> list[float]:
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    out: dict[str, Any] = {"kind": kind, "base_by": base_by, "base": xy(base)}
    if kind == "tube":
        out["mid"] = xy(mid)
    out["tip"] = xy(tip)
    if kind == "tube":
        out["radius_px"] = [round(rb, 1), round(rt, 1)]
    elif kind == "capsule":
        out["radius_px"] = round(mid_r, 1)
    else:
        out["width_px"] = round(2.0 * mid_r, 1)
    out["length_px"] = round(float(np.linalg.norm(tip - base)), 1)
    out["confidence"] = round(min(1.0, ratio / ROD_FULL_CONFIDENCE_RATIO) * max(0.0, 1.0 - 4.0 * curve), 2)
    if warnings:
        out["warnings"] = warnings
    return out


def harmonize_base_tip(found: dict[str, dict[str, Any]], calibs: dict[str, ViewCalibration],
                       min_dz_frac: float = 0.2) -> list[str]:
    """Make base / tip run the same way in every upright view (flips ``found`` in place; returns the flipped views).

    Per view the base is decided on local evidence (``base_by``: the base hint, contact with the body, the wider end, the
    lower end), which can disagree between views. Here the weighted majority of the views whose axis has a clear height
    (``|dz|`` >= ``min_dz_frac`` of its length) decides which end is higher; a hint counts 3, body contact 2, the rest 1,
    each times the view's confidence. A view that disagrees swaps base and tip (a radius list is reversed)."""
    weights = {"hint": 3.0, "attach": 2.0}
    sign: dict[str, float] = {}
    vote = 0.0
    for view, a in found.items():
        c = calibs.get(view)
        if c is None or view_frame(c.camera_axis).up_axis[1] != "Z" or "base" not in a or "tip" not in a:
            continue
        zb, zt = (float(c.px_to_plane(np.array([a[k]]))[0, 2]) for k in ("base", "tip"))
        length = float(np.hypot(*(np.array(a["base"]) - np.array(a["tip"])))) * c.m_per_px
        if length > 0 and abs(zb - zt) >= min_dz_frac * length:
            sign[view] = 1.0 if zb > zt else -1.0
            vote += sign[view] * weights.get(a.get("base_by"), 1.0) * max(float(a.get("confidence", 1.0)), 0.05)
    flipped = []
    for view, sg in sign.items():
        if vote != 0.0 and sg != (1.0 if vote > 0 else -1.0):
            a = found[view]
            a["base"], a["tip"] = a["tip"], a["base"]
            if isinstance(a.get("radius_px"), list):
                a["radius_px"] = a["radius_px"][::-1]
            a["base_by"] = "harmonized"
            flipped.append(view)
    return flipped


SYMMETRIC_ELONGATION = 2.0   # a part whose axis is shorter than this many widths (a ball, a stubby fin) has no real base / tip end


def _elongation_of(found: dict[str, dict[str, Any]]) -> float:
    """Mean axis length over mean width of a part's anchors in the views (``inf`` when no view gives a width): a low value
    means the shape is nearly symmetric end to end, so which end is the base is a convention, not a measurement."""
    ratios = []
    for a in found.values():
        w = a.get("width_px")
        if w is None and a.get("radius_px") is not None:
            r = a["radius_px"]
            w = 2.0 * (max(r) if isinstance(r, (list, tuple)) else r)
        if not w or "base" not in a or "tip" not in a:
            continue
        ratios.append(float(np.hypot(*(np.array(a["base"], float) - np.array(a["tip"], float)))) / max(float(w), 1e-9))
    return float(np.mean(ratios)) if ratios else float("inf")


def check_anchor_views(found: dict[str, dict[str, Any]], calibs: dict[str, ViewCalibration],
                       z_tol_frac: float = 0.08) -> dict[str, Any]:
    """Cross-view check of one part's anchors (``found`` = view -> :func:`propose_anchors` result).

    Compares the world height z of base and tip over the upright views: the base must be the same end in every view
    and the base / tip heights must agree within ``z_tol_frac`` of the part length. Returns
    ``{"confidence": min over views (reduced on disagreement), "warnings": [...]}``.
    """
    zs: dict[str, tuple[float, float]] = {}
    for view, a in found.items():
        c = calibs.get(view)
        if c is None or view_frame(c.camera_axis).up_axis[1] != "Z" or "base" not in a or "tip" not in a:
            continue
        zs[view] = tuple(float(c.px_to_plane(np.array([a[k]]))[0, 2]) for k in ("base", "tip"))   # type: ignore[assignment]
    warnings: list[str] = []
    for view, a in found.items():
        warnings += [f"{view}: {w}" for w in a.get("warnings", [])]
    conf = min((a.get("confidence", 1.0) for a in found.values()), default=1.0)
    if len(zs) > 1:
        base_low = {v: z[0] < z[1] for v, z in zs.items()}
        mean_len = float(np.mean([abs(z[0] - z[1]) for z in zs.values()]))
        if len(set(base_low.values())) > 1 and _elongation_of(found) >= SYMMETRIC_ELONGATION:
            warnings.append("base and tip are swapped between views (base is the lower end in "
                            + ", ".join(v for v, low in base_low.items() if low) + " but the upper end in "
                            + ", ".join(v for v, low in base_low.items() if not low) + "); pass --base-hint X,Y per view")
            conf *= 0.5
        for i, name in enumerate(("base", "tip")):
            spread = max(z[i] for z in zs.values()) - min(z[i] for z in zs.values())
            if mean_len > 0 and spread > z_tol_frac * mean_len:
                warnings.append(f"{name} height disagrees between views by {spread * 1000:.0f} mm "
                                f"({100 * spread / mean_len:.0f}% of the part length): "
                                + ", ".join(f"{v} {z[i] * 1000:.0f} mm" for v, z in zs.items()))
                conf *= 0.7
    return {"confidence": round(conf, 2), "warnings": warnings}


def _auto_anchors(crop: np.ndarray, origin: tuple[int, int], base_hint: Optional[Sequence[float]],
                  rest: Optional[np.ndarray], force_type: Optional[str]) -> dict[str, Any]:
    """``kind="auto"``: rods through :func:`_rod_anchors` (a capsule, or a cone when the end radii differ), everything else
    through :func:`~bbr.core.partshape.suggest`."""
    from . import partshape     # imports this module: not at load time
    rest_pts = rest_points(rest)
    contact = None
    if rest_pts is not None:
        x0, y0 = origin
        near = imageops.dilate(np.asarray(rest, dtype=bool)[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]], CONTACT_PX)
        contact = crop & near if near.shape == crop.shape else None
    if force_type != partshape.SPHERE:
        rod = _rod_anchors(crop, "tube", origin, base_hint, rest_pts)
        if rod is not None:
            rb, rt = rod["radius_px"]
            as_cone = force_type == partshape.CONE or (force_type is None and max(rb, rt) >= ROD_CONE_RATIO * min(rb, rt))
            if not as_cone:
                rod = _rod_anchors(crop, "capsule", origin, base_hint, rest_pts)
            rod["suggested_type"] = partshape.CONE if as_cone else partshape.CAPSULE
            return rod
    return partshape.suggest(crop, origin, rest_pts, base_hint, force_type, contact)


def _strand_anchors(crop: np.ndarray, origin: tuple[int, int], base_hint: Optional[Sequence[float]],
                    rest: Optional[np.ndarray], n: int) -> dict[str, Any]:
    """``kind="strand"`` of a bool ``crop`` whose pixel (0, 0) is image pixel ``origin``: the centre-favouring geodesic between
    the two geodesically farthest pixels (:func:`~bbr.core.partshape._curved`: follows the waves and reaches both real ends),
    oriented from the base (``base_hint``, else the end touching the most ``rest`` pixels, the end clearly nearer ``rest``,
    the wider end, the lower end) and cut into ``n`` points: ``p0`` (base) ... ``p<n-1>`` (tip), at equal height steps when
    the strand runs steadily up or down (every upright view then agrees on z), else at equal arc length; ``radius_px`` per
    point is the median medial radius around it, the end values the fitted base / tip radii."""
    from . import partshape     # imports this module: not at load time
    dt = edt(~np.pad(crop, 2), cap=int(min(max(crop.shape), 128)))[2:-2, 2:-2] - 0.5
    cur = partshape._curved(crop, origin, dt)
    if cur is None:
        raise BBRError("the part is too small to find an axis in")
    path, length, r_a, r_b = cur["path"], float(cur["length"]), float(cur["r_a"]), float(cur["r_b"])
    if base_hint is not None:
        hint = np.asarray(base_hint, dtype=float)
        flip, base_by = bool(np.linalg.norm(hint - path[-1]) < np.linalg.norm(hint - path[0])), "hint"
    elif (attached := attach_base(path[0], path[-1], length, rest_points(rest), contact_reach(max(r_a, r_b), length))) is not None:
        flip, base_by = attached, "attach"
    elif abs(r_a - r_b) > 0.15 * max(r_a, r_b):
        flip, base_by = r_b > r_a, "width"
    else:
        flip, base_by = bool(path[-1][1] > path[0][1]), "lower"
    if flip:
        path, r_a, r_b = path[::-1], r_b, r_a
    s = _arclength(path)
    rows = np.clip(np.round(path[:, 1] - 0.5 - origin[1]).astype(int), 0, dt.shape[0] - 1)
    cols = np.clip(np.round(path[:, 0] - 0.5 - origin[0]).astype(int), 0, dt.shape[1] - 1)
    rad = dt[rows, cols]
    fr = np.linspace(0.0, 1.0, n)
    at = fr * length                                            # equal arc length ...
    sign = 1.0 if path[-1, 1] >= path[0, 1] else -1.0
    u = sign * path[:, 1]                                       # ... unless the strand runs steadily up or down: then equal
    env = np.maximum.accumulate(u)                              # height steps
    if u[-1] - u[0] >= STRAND_VERTICAL * length and float((env - u).max()) <= STRAND_MONOTONE_TOL * length:
        at = np.interp(u[0] + fr * (u[-1] - u[0]), env + 1e-6 * np.arange(len(env)), s)
    half = 0.5 / (n - 1) * length
    radii = []
    for t in at:
        sel = (np.abs(s - t) <= half) & (rad > 0)
        radii.append(float(np.median(rad[sel])) if sel.any() else float(np.interp(t, s, rad)))
    radii[0], radii[-1] = r_a, r_b
    out: dict[str, Any] = {"kind": "strand", "suggested_type": "strand", "base_by": base_by}
    out.update({f"p{i}": [round(float(np.interp(t, s, path[:, 0])), 1), round(float(np.interp(t, s, path[:, 1])), 1)] for i, t in enumerate(at)})
    out["radius_px"] = [round(max(r, MIN_RADIUS_PX), 1) for r in radii]
    out["length_px"] = round(length, 1)
    out["confidence"] = round(0.5 * min(1.0, _elongation(crop) / ROD_RATIO), 2)
    return out


def strand_points(anchors: dict[str, Any]) -> list[str]:
    """Names ``p0, p1, ...`` of the points of a strand anchor dict, in order."""
    return sorted((k for k in anchors if re.fullmatch(r"p\d+", k)), key=lambda k: int(k[1:]))


def flip_strand(a: dict[str, Any]) -> None:
    """Reverse a strand anchor dict in place (base <-> tip): the points and the radius list."""
    names = strand_points(a)
    pts = [a[k] for k in names][::-1]
    for k, p in zip(names, pts):
        a[k] = p
    if isinstance(a.get("radius_px"), list):
        a["radius_px"] = a["radius_px"][::-1]


def propose_anchors(mask: np.ndarray, bbox: Sequence[float], kind: str = "tube",
                    base_hint: Optional[Sequence[float]] = None, rest: Optional[np.ndarray] = None,
                    force_type: Optional[str] = None, n_points: int = STRAND_POINTS) -> dict[str, Any]:
    """Pixel anchors of the part of a foreground ``mask`` inside the pixel box ``bbox = [x0, y0, x1, y1]``.

    The mask is cut to the box and the part is treated as continuing past the box edges: a part that merges into the
    body (a horn on a helmet) is delimited by putting the box edge at the junction. Medial axis = Zhang-Suen skeleton,
    longest path; free ends are extended along their tangent to the mask boundary; radii come from the distance
    transform along the path (a line fit evaluated at both ends, so a cone gets a base and a tip radius).

    ``kind``: ``tube`` -> base, mid, tip, radius_px [base, tip]; ``capsule`` -> base, tip, radius_px; ``blade`` -> base,
    tip, width_px; ``disc`` / ``sphere`` -> center, radius_px (largest inscribed circle: only meaningful for a part that
    leaves the silhouette). base = the wider end (equal ends: the lower one) unless ``base_hint`` ``[x, y]`` names it.
    An elongated mask (axis ratio >= ``ROD_RATIO``: staff, spear) takes :func:`_rod_anchors` instead: principal-axis
    ends, median medial radius, base = the lower end. The result carries ``confidence`` (0-1, heuristic).
    Coordinates are pixels with pixel centres at +0.5, like :func:`pixel_to_world`.

    ``kind="auto"`` picks the primitive: a rod (axis ratio >= ``ROD_RATIO``) is a capsule, or a cone when its ends differ
    in radius; any other mask goes to :func:`~bbr.core.partshape.suggest` (sphere / capsule / straight or curved cone,
    scored by the IoU of the rendered shape with the mask, which is the ``confidence``). The result carries
    ``suggested_type`` (``sphere`` / ``capsule`` / ``cone``); ``force_type`` fixes the type (one type for all views of
    a part). ``rest`` = bool image mask of what the part is attached to (the body label): for ``tube`` / ``auto`` the end
    nearer to it is the base unless ``base_hint`` says otherwise.

    The base of a part is, in order: ``base_hint``; the end whose neighbourhood touches the most of ``rest`` (a tail that
    hangs from the head: the top, not the lower end); the end clearly nearer ``rest``; the wider end; the lower end.
    ``kind="strand"`` returns ``n_points`` (>= 3) points ``p0`` (base) ... ``p<n-1>`` (tip) at equal arc length along the
    medial axis of the mask plus ``radius_px`` per point (taper), for hair locks, braids, cables and long tails.
    """
    if kind not in ANCHOR_KINDS:
        raise BBRError(f"kind must be one of {ANCHOR_KINDS}, got {kind!r}")
    if kind == "strand" and int(n_points) < MIN_STRAND_POINTS:
        raise BBRError(f"a strand needs at least {MIN_STRAND_POINTS} points, got {n_points}")
    if kind == "plate":      # root edge (base, the end touching the body) and tip: the tube's axis without radii; the outline is the label
        out = propose_anchors(mask, bbox, "tube", base_hint, rest, force_type, n_points)
        return {"kind": "plate", "suggested_type": "plate", "base": out["base"], "tip": out["tip"],
                **{k: out[k] for k in ("length_px", "confidence", "warnings") if k in out}}
    if len(bbox) != 4:
        raise BBRError(f"bbox needs x0,y0,x1,y1, got {list(bbox)}")
    h, w = mask.shape
    x0, y0 = max(int(math.floor(bbox[0])), 0), max(int(math.floor(bbox[1])), 0)
    x1, y1 = min(int(math.ceil(bbox[2])), w), min(int(math.ceil(bbox[3])), h)
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=bool) if x1 > x0 and y1 > y0 else np.zeros((0, 0), bool)
    if not crop.any():
        raise BBRError(f"no foreground inside bbox {list(bbox)}")
    labels, n = imageops.label_components(crop)
    if n > 1:
        crop = labels == (1 + int(np.argmax(np.bincount(labels.ravel())[1:])))
    if kind == "strand":
        return _strand_anchors(crop, (x0, y0), base_hint, rest, int(n_points))
    ch, cw = crop.shape
    pad = int(min(max(ch, cw) // 2 + 2, 128))
    work = np.pad(crop, pad, mode="edge")                       # the part continues past the box edge
    dt = edt(~work, cap=pad + max(ch, cw) // 2 + 2) - 0.5       # inscribed radius at foreground pixel centres

    def px(rc: Sequence[float]) -> list[float]:                 # (row, col) of the padded array -> image pixel
        return [round(float(rc[1]) - pad + x0 + 0.5, 1), round(float(rc[0]) - pad + y0 + 0.5, 1)]

    if kind == "auto":
        return _auto_anchors(crop, (x0, y0), base_hint, rest, force_type)
    rest_pts = rest_points(rest) if kind == "tube" else None
    if kind in ("disc", "sphere"):
        inner = np.zeros_like(work)
        inner[pad:pad + ch, pad:pad + cw] = True
        top = (dt >= dt[inner].max() - 0.5) & inner & work
        rr, cc = np.nonzero(top)
        return {"kind": kind, "center": px([rr.mean(), cc.mean()]), "radius_px": round(float(dt[top].max()), 1)}
    if kind in ("tube", "capsule", "blade"):
        rod = _rod_anchors(crop, kind, (x0, y0), base_hint, rest_pts)
        if rod is not None:
            return rod
    skeleton = thin(work)
    skeleton_path = _longest_path(skeleton)
    if len(skeleton_path) < 2:
        raise BBRError("the part is too small to find an axis in")
    path = _smooth(np.array(skeleton_path, dtype=float))
    lo = np.array([pad - 0.5, pad - 0.5])
    hi = np.array([pad + ch - 0.5, pad + cw - 0.5])
    inside = np.all((path >= lo) & (path <= hi), axis=1)       # drop the padded continuation
    if inside.sum() < 2:
        raise BBRError("the part's axis does not enter the bbox")
    idx = np.flatnonzero(inside)
    path = path[idx[0]:idx[-1] + 1]
    (start, i0), (stop, i1) = _extend_end(path, work, lo, hi, True), _extend_end(path, work, lo, hi, False)
    path = np.vstack([start[None], path[i0:i1 + 1], stop[None]])
    s = _arclength(path)
    length = float(s[-1])
    rows = np.clip(path[:, 0].astype(int), 0, dt.shape[0] - 1)
    cols = np.clip(path[:, 1].astype(int), 0, dt.shape[1] - 1)
    rad = np.maximum(dt[rows, cols], MIN_RADIUS_PX)
    keep = (s >= END_FIT_TRIM * length) & (s <= (1.0 - END_FIT_TRIM) * length)
    if keep.sum() >= 3 and length > 0:
        slope, icpt = np.polyfit(s[keep], rad[keep], 1)
        r_start, r_end = max(float(icpt), MIN_RADIUS_PX), max(float(icpt + slope * length), MIN_RADIUS_PX)
    else:
        r_start = r_end = float(np.median(rad))
    mid_r = float(np.median(rad[keep])) if keep.any() else float(np.median(rad))
    for end, small, big in ((0, r_start, r_end), (-1, r_end, r_start)):         # a pointed end: snap to the farthest pixel
        if small < POINT_RATIO * big:
            reach = max(POINT_REACH_MIN_PX, POINT_REACH * length)
            anchor = path[len(path) // 2]
            rr, cc = np.nonzero(work & (dt <= POINT_MAX_DT_PX))
            near = np.hypot(rr - path[end][0], cc - path[end][1]) <= reach
            near &= (rr >= lo[0]) & (rr <= hi[0]) & (cc >= lo[1]) & (cc <= hi[1])
            if near.any():
                far = np.argmax(np.where(near, np.hypot(rr - anchor[0], cc - anchor[1]), -1.0))
                path[end] = (rr[far], cc[far])
    s = _arclength(path)
    length = float(s[-1])
    attached = None if rest_pts is None else attach_base(
        np.array([path[0][1] - pad + x0 + 0.5, path[0][0] - pad + y0 + 0.5]),
        np.array([path[-1][1] - pad + x0 + 0.5, path[-1][0] - pad + y0 + 0.5]), length, rest_pts,
        contact_reach(max(r_start, r_end), length))
    if base_hint is not None:
        hint = np.array([base_hint[1] - y0 + pad - 0.5, base_hint[0] - x0 + pad - 0.5])
        flip = bool(np.linalg.norm(hint - path[-1]) < np.linalg.norm(hint - path[0]))
    elif attached is not None:
        flip = attached
    elif abs(r_start - r_end) > 0.15 * max(r_start, r_end):
        flip = r_end > r_start                                   # base = the wider end
    else:
        flip = bool(path[-1][0] > path[0][0])                    # equal ends: base = the lower one
    base_i, tip_i = (-1, 0) if flip else (0, -1)
    rb, rt = (r_end, r_start) if flip else (r_start, r_end)
    out: dict[str, Any] = {"kind": kind, "base": px(path[base_i])}
    if kind == "tube":
        out["mid"] = px(_at(path, s, length / 2.0))
    out["tip"] = px(path[tip_i])
    if kind == "tube":
        out["radius_px"] = [round(rb, 1), round(rt, 1)]
    elif kind == "capsule":
        out["radius_px"] = round(mid_r, 1)
    else:
        out["width_px"] = round(2.0 * mid_r, 1)
    out["length_px"] = round(length, 1)
    out["confidence"] = round(0.5 * min(1.0, _elongation(crop) / ROD_RATIO), 2)
    return out
