"""Numbered region proposals of one reference view (numpy only), the "set of marks" an agent labels.

The agent decides what each region is (horn, hand, shield...), the tools do the rest (:mod:`labels`). A view is cut
into at most ``max_regions`` regions:

1. colour regions: k-means in Lab on the in-mask pixels, 4-connected components of every cluster;
2. thin protrusions: ``mask - opening(mask, PROTRUSION_FRAC * height)`` (horns, arms, a sword); each large piece is its
   own region, cut out of the colour regions;
3. small regions (``min_area_frac`` of the mask) and, while there are too many, the smallest ones are merged into the
   neighbour they share the longest, most similar-coloured border with.

Everything is deterministic (seeded k-means, fixed tie-breaks) and the regions are numbered by area, biggest first
(ties: top-most, then left-most), so a labels file stays valid when the proposal is regenerated from the same image.
The overlay draws the number of every region at its inner-most point with an outline in a region-specific colour.
"""
from __future__ import annotations

import heapq
from typing import Any, Optional, Sequence

import numpy as np

from .imageops import dilate, erode, label_components

MAX_REGIONS = 40
DEFAULT_K = 14
MIN_AREA_FRAC = 0.002          # regions smaller than this share of the mask are merged away
PROTRUSION_FRAC = 0.03         # opening radius as a fraction of the mask height
PROTRUSION_MIN_FRAC = 0.002    # smallest protrusion piece (share of the mask) that becomes a region of its own
PROTRUSION_CRUMB_PX = 2        # radius of the opening that drops crumbs left along curved outlines
KMEANS_ITERATIONS = 20
KMEANS_SAMPLE = 20000
COLOR_SIMILARITY_DE = 25.0     # merge preference falls off by exp(-deltaE / this)
OVERLAY_SCALE = 2
CROP_SCALE = 4
DIGIT_FRAC = 0.018             # digit height as a fraction of the image height (source pixels, before scaling)
TINT = 0.25
TICK_PX = 100                  # ruler tick spacing in view pixels
CROP_GRID = (2, 3)             # columns x rows of the crop tiles
CROP_MIN_REGIONS = 8           # a tile with at least this many region numbers gets an upscaled crop

_GLYPHS = {  # 3x5 bitmap digits
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111"}


# --------------------------------------------------------------------------- colour
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1], shape ``(..., 3)`` -> CIE Lab (D65), same shape."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def kmeans(points: np.ndarray, k: int, seed: int = 0, iterations: int = KMEANS_ITERATIONS) -> np.ndarray:
    """Deterministic k-means (k-means++ seeding, Lloyd steps) on ``(N, D)`` points; returns the ``(k', D)`` centres."""
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.RandomState(seed)
    k = max(1, min(int(k), len(pts)))
    centres = [pts[rng.randint(len(pts))]]
    d2 = ((pts - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:
            break
        centres.append(pts[np.searchsorted(np.cumsum(d2), rng.uniform(0, total))])
        d2 = np.minimum(d2, ((pts - centres[-1]) ** 2).sum(axis=1))
    centres = np.array(centres)
    for _ in range(iterations):
        assign = _nearest(pts, centres)
        new = np.array([pts[assign == j].mean(axis=0) if (assign == j).any() else centres[j] for j in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    return centres


def _nearest(pts: np.ndarray, centres: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(pts), dtype=np.int32)
    for lo in range(0, len(pts), chunk):
        d = ((pts[lo:lo + chunk, None, :] - centres[None]) ** 2).sum(axis=2)
        out[lo:lo + chunk] = d.argmin(axis=1)
    return out


# --------------------------------------------------------------------------- proposals
def protrusions(mask: np.ndarray, frac: float = PROTRUSION_FRAC, min_frac: float = PROTRUSION_MIN_FRAC) -> np.ndarray:
    """Pieces of ``mask`` an opening of radius ``frac * height`` removes (thin parts sticking out), crumbs dropped."""
    mask = np.asarray(mask, dtype=bool)
    ys = np.flatnonzero(mask.any(axis=1))
    if len(ys) == 0:
        return np.zeros_like(mask)
    r = max(2, int(round(frac * (ys[-1] - ys[0] + 1))))
    thin = mask & ~dilate(erode(mask, r), r)
    thin = dilate(erode(thin, PROTRUSION_CRUMB_PX), PROTRUSION_CRUMB_PX) & mask
    labels, n = label_components(thin, full=False)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas >= min_frac * mask.sum()
    keep[0] = False
    return keep[labels]


def _adjacency(lab: np.ndarray) -> dict[int, dict[int, int]]:
    """Border length (4-adjacent pixel pairs) between regions ``a != b`` (both non-zero)."""
    adj: dict[int, dict[int, int]] = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        sel = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[sel], b[sel]).astype(np.int64), np.maximum(a[sel], b[sel]).astype(np.int64)
        keys, counts = np.unique(lo * (1 << 20) + hi, return_counts=True)
        for key, cnt in zip(keys.tolist(), counts.tolist()):
            x, y = key >> 20, key & ((1 << 20) - 1)
            adj.setdefault(x, {})[y] = adj.setdefault(x, {}).get(y, 0) + cnt
            adj.setdefault(y, {})[x] = adj.setdefault(y, {}).get(x, 0) + cnt
    return adj


def _merge_regions(lab: np.ndarray, lab_img: np.ndarray, max_regions: int, min_area: int) -> np.ndarray:
    """Merge the smallest regions into their best neighbour until all are ``>= min_area`` and at most ``max_regions``
    remain. Returns the relabelled map (ids are not contiguous)."""
    n = int(lab.max())
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    colour = np.zeros((n + 1, 3))
    for c in range(3):
        colour[:, c] = np.bincount(lab.ravel(), weights=lab_img[..., c].ravel(), minlength=n + 1)
    adj = _adjacency(lab)
    ys, xs = np.mgrid[0:lab.shape[0], 0:lab.shape[1]]
    cent = np.stack([np.bincount(lab.ravel(), weights=xs.ravel(), minlength=n + 1),
                     np.bincount(lab.ravel(), weights=ys.ravel(), minlength=n + 1)], axis=1)
    alive = {int(i) for i in np.flatnonzero(area[1:]) + 1}
    parent = {i: i for i in alive}
    heap = [(int(area[i]), i) for i in alive]
    heapq.heapify(heap)
    while heap:
        a_area, r = heapq.heappop(heap)
        if r not in alive or a_area != area[r]:
            continue
        if a_area >= min_area and len(alive) <= max_regions:
            break
        if len(alive) <= 1:
            break
        nbrs = adj.get(r) or {}
        if nbrs:
            mean_r = colour[r] / area[r]
            target = max(sorted(nbrs), key=lambda t: nbrs[t] * np.exp(-np.linalg.norm(mean_r - colour[t] / area[t]) / COLOR_SIMILARITY_DE))
        else:   # detached piece: the nearest region by centroid
            c = cent[r] / area[r]
            target = min(sorted(alive - {r}), key=lambda t: float(np.hypot(*(cent[t] / area[t] - c))))
        for other, cnt in (adj.pop(r, {}) or {}).items():
            adj[other].pop(r, None)
            if other != target:
                adj[other][target] = adj[other].get(target, 0) + cnt
                adj.setdefault(target, {})[other] = adj[other][target]
        area[target] += area[r]
        colour[target] += colour[r]
        cent[target] += cent[r]
        alive.discard(r)
        parent[r] = target
        heapq.heappush(heap, (int(area[target]), target))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    lut = np.zeros(n + 1, dtype=np.int32)
    for i in parent:
        lut[i] = root(i)
    return lut[lab]


def inner_point(region: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Pixel ``(x, y)`` (centre, +0.5) of the region's deepest interior point (max distance to the region border)."""
    from .distance import edt
    x0, y0, x1, y1 = bbox
    crop = np.pad(region[y0:y1, x0:x1], 1)
    dist = edt(~crop, cap=max(crop.shape))
    iy, ix = np.unravel_index(int(np.argmax(np.where(crop, dist, -1.0))), crop.shape)
    return float(x0 + ix - 1 + 0.5), float(y0 + iy - 1 + 0.5)


def propose_regions(rgb: np.ndarray, mask: np.ndarray, max_regions: int = MAX_REGIONS, k: int = DEFAULT_K,
                    min_area_frac: float = MIN_AREA_FRAC, seed: int = 0) -> dict[str, Any]:
    """Numbered regions of the in-``mask`` pixels of ``rgb`` (``(H, W, 3+)`` floats in [0, 1]).

    Returns ``{"id_map": uint8 (H, W) (0 = background, 1..N = regions, biggest first), "regions": [...], "params": {...}}``;
    a region is ``{id, area, centroid [x, y], bbox [x0, y0, x1, y1] (end exclusive), label_px [x, y], kind
    ("protrusion" | "colour"), color [r, g, b] 0-255}``. ``label_px`` is where the overlay draws the number.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("propose_regions: empty mask")
    max_regions = max(1, min(int(max_regions), 255))
    lab_img = srgb_to_lab(np.asarray(rgb)[..., :3])
    pts = lab_img[mask]
    sample = pts[:: max(1, len(pts) // KMEANS_SAMPLE)]
    clusters = np.zeros(mask.shape, dtype=np.int32)
    clusters[mask] = _nearest(pts, kmeans(sample, k, seed)) + 1
    lab = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    thin = protrusions(mask)
    for c in range(1, int(clusters.max()) + 1):
        comps, n = label_components((clusters == c) & ~thin, full=False)
        lab[comps > 0] = comps[comps > 0] + count
        count += n
    comps, n = label_components(thin, full=False)
    lab[comps > 0] = comps[comps > 0] + count
    lab = _merge_regions(lab, lab_img, max_regions, max(1, int(round(min_area_frac * mask.sum()))))
    ids = np.unique(lab[lab > 0])
    area = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)[ids]
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    top = np.array([ys[lab == i].min() * 100000 + xs[lab == i].min() for i in ids])
    order = np.lexsort((top, -area))   # biggest first, ties top-most then left-most
    lut = np.zeros(int(lab.max()) + 1, dtype=np.uint8)
    lut[ids[order]] = np.arange(1, len(ids) + 1)
    id_map = lut[lab]
    regions = []
    for rid in range(1, len(ids) + 1):
        region = id_map == rid
        ry, rx = np.nonzero(region)
        bbox = (int(rx.min()), int(ry.min()), int(rx.max()) + 1, int(ry.max()) + 1)
        regions.append({
            "id": rid, "area": int(region.sum()), "centroid": [round(float(rx.mean()) + 0.5, 1), round(float(ry.mean()) + 0.5, 1)],
            "bbox": list(bbox), "label_px": list(inner_point(region, bbox)),
            "kind": "protrusion" if thin[region].mean() > 0.5 else "colour",
            "color": [int(round(v * 255)) for v in np.asarray(rgb)[..., :3][region].mean(axis=0)]})
    return {"id_map": id_map, "regions": regions,
            "params": {"max_regions": max_regions, "k": int(k), "min_area_frac": min_area_frac, "seed": int(seed),
                       "protrusion_frac": PROTRUSION_FRAC}}


# --------------------------------------------------------------------------- overlay
def region_color(rid: int) -> tuple[float, float, float]:
    """Distinct saturated outline colour of a region (golden-ratio hue walk)."""
    h = (rid * 0.61803398875) % 1.0
    s, v = (1.0, 1.0) if rid % 2 else (0.85, 0.9)
    i, f = int(h * 6), h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def _digits_mask(text: str, height: int) -> np.ndarray:
    """Bool bitmap of ``text`` (digits only) ``height`` px tall (3x5 glyph grid scaled by ``height // 5``)."""
    s = max(1, height // 5)
    cells = np.zeros((5, 4 * len(text) - 1), dtype=bool)
    for i, ch in enumerate(text):
        cells[:, 4 * i:4 * i + 3] = np.array([c == "1" for c in _GLYPHS[ch]]).reshape(5, 3)
    return np.kron(cells, np.ones((s, s), dtype=bool))


def _stamp(img: np.ndarray, glyph: np.ndarray, cx: float, cy: float, fill: tuple, edge: tuple, outline: int) -> None:
    pad = outline + 1
    box = np.pad(glyph, pad)
    ring = dilate(box, outline) & ~box
    h, w = img.shape[:2]
    y0, x0 = int(round(cy - box.shape[0] / 2)), int(round(cx - box.shape[1] / 2))
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1, xs1 = min(y0 + box.shape[0], h), min(x0 + box.shape[1], w)
    if ys1 <= ys0 or xs1 <= xs0:
        return
    sub = (slice(ys0 - y0, ys1 - y0), slice(xs0 - x0, xs1 - x0))
    view = img[ys0:ys1, xs0:xs1]
    view[ring[sub]] = edge
    view[box[sub]] = fill


def render_overlay(rgb: np.ndarray, id_map: np.ndarray, regions: list[dict[str, Any]], scale: int = OVERLAY_SCALE,
                   tint: float = TINT, digit_scale: float = 1.0, box: Optional[Sequence[int]] = None) -> np.ndarray:
    """RGBA float image: the view upscaled by ``scale``, regions tinted and outlined in their own colour, each number
    drawn big (white, black edge) at its ``label_px``, pixel rulers (yellow, in VIEW pixels) on the top and left edge.
    ``box = [x0, y0, x1, y1]`` (view pixels) renders only that part."""
    h, w = id_map.shape
    base = np.repeat(np.repeat(np.asarray(rgb)[..., :3].astype(np.float64), scale, axis=0), scale, axis=1)
    ids = np.repeat(np.repeat(id_map, scale, axis=0), scale, axis=1)
    palette = np.zeros((int(id_map.max()) + 1, 3))
    for r in regions:
        palette[r["id"]] = region_color(r["id"])
    out = np.where((ids > 0)[..., None], base * (1 - tint) + palette[ids] * tint, base)
    p = np.pad(ids, 1, mode="edge")
    edge = (p[1:-1, 1:-1] > 0) & ((p[:-2, 1:-1] != ids) | (p[2:, 1:-1] != ids) | (p[1:-1, :-2] != ids) | (p[1:-1, 2:] != ids))
    edge = dilate(edge, 1) & (ids > 0)
    out[edge] = palette[ids[edge]]
    digit_h = max(10, int(round(DIGIT_FRAC * h * scale * digit_scale)))
    for r in regions:
        glyph = _digits_mask(str(r["id"]), digit_h)
        _stamp(out, glyph, r["label_px"][0] * scale, r["label_px"][1] * scale, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), max(1, digit_h // 8))
    x0, y0, x1, y1 = (0, 0, w, h) if box is None else (max(int(box[0]), 0), max(int(box[1]), 0), min(int(box[2]), w), min(int(box[3]), h))
    out = out[y0 * scale:y1 * scale, x0 * scale:x1 * scale].copy()
    _rulers(out, scale, x0, y0, max(10, digit_h // 2), grid=box is not None)
    return np.concatenate([np.clip(out, 0, 1), np.ones((*out.shape[:2], 1))], axis=2).astype(np.float32)


def _rulers(out: np.ndarray, scale: int, x0: int, y0: int, tick_h: int, grid: bool = False) -> None:
    """Yellow ticks on the top and left edge of ``out`` (which shows the view from ``x0, y0`` at ``scale``), numbered
    with the VIEW coordinate (every ``TICK_PX``, every ``TICK_PX // 4`` when the crop spans under 4 ticks); ``grid``
    adds faint lines across the image. Read split lines / polygons off them."""
    h, w = out.shape[0] // scale, out.shape[1] // scale
    step = TICK_PX if max(h, w) >= 4 * TICK_PX else TICK_PX // 4
    yellow, black = (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)
    for v in range(-(-(x0 + 1) // step) * step, x0 + w, step):
        px = (v - x0) * scale
        if grid:
            out[:, max(px - 1, 0):px] = out[:, max(px - 1, 0):px] * 0.65 + 0.35
        out[:tick_h // 2 + 3, max(px - 1, 0):px + 1] = yellow
        _stamp(out, _digits_mask(str(v), tick_h), px, tick_h + 4, yellow, black, 1)
    for v in range(-(-(y0 + 1) // step) * step, y0 + h, step):
        py = (v - y0) * scale
        glyph = _digits_mask(str(v), tick_h)
        if grid:
            out[max(py - 1, 0):py, :] = out[max(py - 1, 0):py, :] * 0.65 + 0.35
        out[max(py - 1, 0):py + 1, :tick_h // 2 + 3] = yellow
        _stamp(out, glyph, glyph.shape[1] // 2 + tick_h // 2 + 6, py, yellow, black, 1)


def add_rulers(img: np.ndarray, scale: int = 1, x0: int = 0, y0: int = 0, grid: bool = False) -> np.ndarray:
    """RGBA float copy of an image that shows the view from pixel ``(x0, y0)`` magnified by ``scale``, with yellow rulers in
    VIEW pixels on the top and left edge (the unit of every coordinate in labels.json, anchors_px and ``measure --px``)."""
    out = np.array(img[..., :3], dtype=np.float64)
    _rulers(out, scale, x0, y0, max(10, out.shape[0] // 70), grid)
    return np.concatenate([np.clip(out, 0, 1), np.ones((*out.shape[:2], 1))], axis=2).astype(np.float32)


def overlay_crops(rgb: np.ndarray, id_map: np.ndarray, regions: list[dict[str, Any]], scale: int = CROP_SCALE
                  ) -> dict[str, np.ndarray]:
    """Upscaled crops (``crop_<row>_<col>``) of the grid tiles that hold at least ``CROP_MIN_REGIONS`` region numbers,
    with a margin so neighbouring crops overlap; agents see images downscaled, a busy area needs the extra pixels."""
    h, w = id_map.shape
    cols, rows = CROP_GRID
    out: dict[str, np.ndarray] = {}
    for r in range(rows):
        for c in range(cols):
            x0, x1 = int(w * c / cols), int(w * (c + 1) / cols)
            y0, y1 = int(h * r / rows), int(h * (r + 1) / rows)
            n = sum(x0 <= q["label_px"][0] < x1 and y0 <= q["label_px"][1] < y1 for q in regions)
            if n >= CROP_MIN_REGIONS:
                mx, my = (x1 - x0) // 8, (y1 - y0) // 8
                out[f"crop_{r}_{c}"] = render_overlay(rgb, id_map, regions, scale, digit_scale=0.75,
                                                      box=[x0 - mx, y0 - my, x1 + mx, y1 + my])
    return out


def region_summary(regions: list[dict[str, Any]]) -> str:
    """One line per region for the terminal: ``id area bbox kind colour``."""
    return "\n".join(f"  {r['id']:>2}  area {r['area']:>6}  bbox {r['bbox']}  {r['kind']:<10} rgb {tuple(r['color'])}" for r in regions)
