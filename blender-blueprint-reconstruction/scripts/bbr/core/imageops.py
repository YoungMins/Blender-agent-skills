"""Image operations on numpy arrays: masks, bboxes, crops, resizing, morphology, sheet splitting.

Masks are bool ``H x W`` (row 0 = top). Images are float32 ``H x W x C`` in [0, 1].
Connected-component labelling is run-based and fully vectorised (no scipy).
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np

from .. import ImageError, get_logger

log = get_logger(__name__)

DEFAULT_BG_TOL = 0.12
DEFAULT_MAX_HOLE_FRAC = 0.002
DEFAULT_MIN_COMPONENT_FRAC = 0.0005
ALPHA_THRESHOLD = 0.5
ALPHA_MEANINGFUL_FRAC = 1e-4  # share of transparent pixels needed to treat alpha as a mask
DEFAULT_SHEET_MARGIN_FRAC = 0.02
DISTANCE_CAP_PX = 32
ENCLOSED_MIN_FRAC = 0.0002        # enclosed background-coloured holes smaller than this share of the image are kept
ENCLOSED_MEAN_TOL_FRAC = 0.5      # ... and so are holes whose mean colour is farther than this share of ``tol`` from the backdrop
ENCLOSED_GRADIENT_TOL = 0.03      # ... or whose median step between neighbouring pixels is above this (texture, not a flat backdrop)
CHROMA_SATURATION = 0.3           # backdrop (max - min channel) from which ``remove_enclosed`` defaults to on

Box = tuple[int, int, int, int]


# --------------------------------------------------------------------------- labelling
def label_components(mask: np.ndarray, full: bool = True) -> tuple[np.ndarray, int]:
    """Label connected components of a 2-D or 3-D bool mask.

    ``full=True``: 8-connectivity (2-D) / 26-connectivity (3-D); ``False``: 4 / 6.
    Returns ``(labels int32 same shape, n)``; background is 0, labels follow raster order of first voxel.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim not in (2, 3):
        raise ImageError("label_components supports 2-D and 3-D masks")
    shape = mask.shape
    run_len = shape[-1]
    col_shape = shape[:-1]
    n_cols = int(np.prod(col_shape))
    width = run_len + 1  # one False separator per column so runs never cross columns
    padded = np.zeros((n_cols, width), dtype=np.int8)
    padded[:, :run_len] = mask.reshape(n_cols, run_len)
    delta = np.diff(padded.ravel(), prepend=0)
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    n_runs = len(starts)
    if n_runs == 0:
        return np.zeros(shape, dtype=np.int32), 0
    col = starts // width
    s = starts % width
    e = ends % width
    col_ids = np.arange(n_cols)
    first = np.searchsorted(col, col_ids, side="left")
    last = np.searchsorted(col, col_ids, side="right")
    coords = np.stack(np.unravel_index(col_ids, col_shape), axis=1)  # (n_cols, k)
    k = len(col_shape)
    if full:
        offsets = [o for o in itertools.product((-1, 0, 1), repeat=k) if any(o) and next(x for x in o if x) > 0]
    else:
        offsets = [tuple(1 if i == j else 0 for i in range(k)) for j in range(k)]
    edges_a: list[np.ndarray] = []
    edges_b: list[np.ndarray] = []
    run_ids = np.arange(n_runs)
    for off in offsets:
        nbr = coords + np.asarray(off)
        valid = np.all((nbr >= 0) & (nbr < np.asarray(col_shape)), axis=1)
        nbr_id = np.full(n_cols, -1)
        nbr_id[valid] = np.ravel_multi_index(tuple(nbr[valid].T), col_shape)
        target = nbr_id[col]
        sel = target >= 0
        lo = first[target[sel]]
        cnt = last[target[sel]] - lo
        ia = np.repeat(run_ids[sel], cnt)
        if ia.size == 0:
            continue
        excl = np.cumsum(cnt) - cnt
        ib = np.repeat(lo, cnt) + (np.arange(ia.size) - np.repeat(excl, cnt))
        if full:
            hit = (s[ia] <= e[ib]) & (s[ib] <= e[ia])
        else:
            hit = (s[ia] < e[ib]) & (s[ib] < e[ia])
        edges_a.append(ia[hit])
        edges_b.append(ib[hit])
    labels = np.arange(n_runs)
    if edges_a:
        ea = np.concatenate(edges_a)
        eb = np.concatenate(edges_b)
        while True:
            la, lb = labels[ea], labels[eb]
            new = labels.copy()
            np.minimum.at(new, la, lb)
            np.minimum.at(new, lb, la)
            while True:
                jumped = new[new]
                if np.array_equal(jumped, new):
                    break
                new = jumped
            if np.array_equal(new, labels):
                break
            labels = new
    _, comp = np.unique(labels, return_inverse=True)
    comp = comp.astype(np.int32) + 1
    lens = e - s
    total = int(lens.sum())
    excl_len = np.cumsum(lens) - lens
    pos = np.repeat(starts, lens) + (np.arange(total) - np.repeat(excl_len, lens))
    out = np.zeros(n_cols * width, dtype=np.int32)
    out[pos] = np.repeat(comp, lens)
    return out.reshape(n_cols, width)[:, :run_len].reshape(shape), int(comp.max())


def _border_labels(labels: np.ndarray) -> np.ndarray:
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    uniq = np.unique(border)
    return uniq[uniq > 0]


def fill_small_holes(mask: np.ndarray, max_hole_frac: float = DEFAULT_MAX_HOLE_FRAC) -> np.ndarray:
    """Fill background regions not touching the border and smaller than ``max_hole_frac`` of the image."""
    labels, n = label_components(~mask, full=False)
    if n == 0:
        return mask.copy()
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    fill = areas <= max_hole_frac * mask.size
    fill[0] = False
    fill[_border_labels(labels)] = False
    return mask | fill[labels]


def remove_small_components(mask: np.ndarray, min_frac: float = DEFAULT_MIN_COMPONENT_FRAC) -> np.ndarray:
    """Drop 8-connected components smaller than ``min_frac`` of the image area (always keeps the largest)."""
    labels, n = label_components(mask, full=True)
    if n == 0:
        return mask.copy()
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    areas[0] = 0
    keep = areas >= min_frac * mask.size
    keep[int(np.argmax(areas))] = True
    keep[0] = False
    return keep[labels]


# --------------------------------------------------------------------------- masks
def _as_rgba(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=2)
    return arr


def border_background_color(rgb: np.ndarray) -> np.ndarray:
    """Median RGB of the image border pixels."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]], axis=0)
    return np.median(border[:, :3], axis=0)


def extract_mask(rgba: np.ndarray, mode: str = "auto", color: Optional[Sequence[float]] = None,
                 tol: float = DEFAULT_BG_TOL, max_hole_frac: float = DEFAULT_MAX_HOLE_FRAC,
                 min_component_frac: float = DEFAULT_MIN_COMPONENT_FRAC,
                 remove_enclosed: Optional[bool] = None) -> np.ndarray:
    """Foreground mask of an image.

    ``alpha``: alpha >= 0.5. ``color``: background = ``color`` (or the border median) and the mask is
    NOT(4-connected flood fill of background-coloured pixels from the border), so a white object with a
    dark outline on a white background keeps its interior. ``auto``: alpha if it is meaningful else ``color``.
    Small holes (alpha mode) and specks are cleaned up. ``remove_enclosed`` (colour mode): also drop flat,
    backdrop-coloured regions the border flood fill cannot reach (:func:`remove_enclosed_background`);
    ``None`` = on for a chroma backdrop (saturation >= ``CHROMA_SATURATION``), off for white / grey / black ones.
    """
    arr = _as_rgba(rgba)
    if mode not in ("auto", "alpha", "color"):
        raise ImageError(f"unknown mask mode {mode!r}")
    alpha = arr[..., 3]
    if mode == "auto":
        mode = "alpha" if float((alpha < ALPHA_THRESHOLD).mean()) > ALPHA_MEANINGFUL_FRAC else "color"
    if mode == "alpha":
        fg = fill_small_holes(alpha >= ALPHA_THRESHOLD, max_hole_frac)
    else:
        rgb = arr[..., :3]
        bg_color = border_background_color(rgb) if color is None else np.asarray(color, np.float32)[:3]
        candidate = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2)) < tol
        labels, _ = label_components(candidate, full=False)
        bg = np.isin(labels, _border_labels(labels)) & candidate
        fg = ~bg
        if remove_enclosed is None:
            remove_enclosed = float(bg_color.max() - bg_color.min()) >= CHROMA_SATURATION
        if remove_enclosed:
            fg = remove_enclosed_background(rgb, fg, bg_color, tol)
    return remove_small_components(fg, min_component_frac)


def remove_enclosed_background(rgb: np.ndarray, fg: np.ndarray, bg_color: np.ndarray, tol: float,
                               min_frac: float = ENCLOSED_MIN_FRAC) -> np.ndarray:
    """Foreground mask without enclosed backdrop: components of ``fg`` that are backdrop coloured and flat.

    A component of backdrop-coloured pixels (within ``tol``) that the border flood fill could not reach (the
    gap between a staff, hair and a hat brim, a hand on a hip) is dropped when it covers at least ``min_frac``
    of the image, its mean colour is within ``ENCLOSED_MEAN_TOL_FRAC * tol`` of the backdrop and its median
    colour step between neighbouring pixels is <= ``ENCLOSED_GRADIENT_TOL`` (a flat backdrop, not texture), so
    shaded or textured look-alike regions stay. Logs a warning with the count and area removed.
    """
    dist = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2))
    labels, n = label_components((dist < tol) & fg, full=False)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    drop = np.zeros(n + 1, bool)
    for k in np.flatnonzero(areas >= max(min_frac * fg.size, 1)):
        if k == 0:
            continue
        rows, cols = np.nonzero(labels == k)
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        comp = (labels[y0:y1, x0:x1] == k)
        patch = rgb[y0:y1, x0:x1]
        steps = np.concatenate([
            np.sqrt(((patch[:, 1:] - patch[:, :-1]) ** 2).sum(axis=2))[comp[:, 1:] & comp[:, :-1]],
            np.sqrt(((patch[1:] - patch[:-1]) ** 2).sum(axis=2))[comp[1:] & comp[:-1]]])
        flat = steps.size == 0 or float(np.median(steps)) <= ENCLOSED_GRADIENT_TOL
        if flat and float(dist[y0:y1, x0:x1][comp].mean()) <= ENCLOSED_MEAN_TOL_FRAC * tol:
            drop[k] = True
    if not drop.any():
        return fg
    removed = drop[labels]
    log.warning("removed %d enclosed background hole(s) (%d px, %.2f%% of the image) from the mask; "
                "set background.remove_enclosed=false if they are part of the object",
                int(drop.sum()), int(removed.sum()), 100.0 * removed.mean())
    return fg & ~removed


def mask_bbox(mask: np.ndarray) -> Box:
    """``(x0, y0, x1, y1)`` with exclusive end. Raises ImageError for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ImageError("mask is empty")
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``img[y0:y1, x0:x1]`` (box clipped to the image)."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def pad_to(img: np.ndarray, size: tuple[int, int], value: float = 0, align: str = "center") -> np.ndarray:
    """Pad ``img`` to ``size=(W, H)``; ``align`` is ``'center'`` or ``'topleft'``."""
    w, h = size
    ih, iw = img.shape[:2]
    if iw > w or ih > h:
        raise ImageError(f"image {(iw, ih)} larger than target {size}")
    out = np.full((h, w) + img.shape[2:], value, dtype=img.dtype)
    ox, oy = ((w - iw) // 2, (h - ih) // 2) if align == "center" else (0, 0)
    out[oy:oy + ih, ox:ox + iw] = img
    return out


def resize_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``size=(W, H)`` (pixel-centre aligned)."""
    w, h = size
    ih, iw = mask.shape[:2]
    ys = np.minimum(((np.arange(h) + 0.5) * ih / h).astype(int), ih - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * iw / w).astype(int), iw - 1)
    return mask[ys][:, xs]


def resize_bilinear(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of ``H x W [x C]`` float image to ``size=(W, H)`` (pixel-centre aligned)."""
    w, h = size
    ih, iw = img.shape[:2]
    src = np.asarray(img, dtype=np.float32)
    sy = np.clip((np.arange(h) + 0.5) * ih / h - 0.5, 0, ih - 1)
    sx = np.clip((np.arange(w) + 0.5) * iw / w - 0.5, 0, iw - 1)
    y0 = np.floor(sy).astype(int)
    x0 = np.floor(sx).astype(int)
    y1 = np.minimum(y0 + 1, ih - 1)
    x1 = np.minimum(x0 + 1, iw - 1)
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)
    if src.ndim == 3:
        fy = fy[:, None, None]
        fx = fx[None, :, None]
    else:
        fy = fy[:, None]
        fx = fx[None, :]
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bot = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


# --------------------------------------------------------------------------- morphology
def _window_sum(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    zero_shape = list(a.shape)
    zero_shape[axis] = 1
    c = np.cumsum(np.concatenate([np.zeros(zero_shape, a.dtype), a], axis=axis), axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, None)
    lo[axis] = slice(None, -k)
    return c[tuple(hi)] - c[tuple(lo)]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square-structuring-element dilation by radius ``r`` pixels (outside = background)."""
    if r <= 0:
        return mask.copy()
    p = np.pad(np.asarray(mask, dtype=np.int32), r)
    k = 2 * r + 1
    return _window_sum(_window_sum(p, k, 0), k, 1) > 0


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Square erosion by radius ``r``; image border does not erode (outside = foreground)."""
    if r <= 0:
        return mask.copy()
    return ~dilate(~np.asarray(mask, dtype=bool), r)


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of the mask (outside the image counts as background)."""
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def distance_to(mask: np.ndarray, cap: int = DISTANCE_CAP_PX) -> np.ndarray:
    """Approximate distance (px) to the nearest True pixel, capped at ``cap``.

    Octagonal chamfer by alternating 4- and 8-neighbour dilation steps (error < ~8%).
    """
    reached = np.asarray(mask, dtype=bool).copy()
    dist = np.full(reached.shape, float(cap), dtype=np.float32)
    dist[reached] = 0.0
    for step in range(1, cap + 1):
        p = np.pad(reached, 1)
        grown = reached | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if step % 2 == 0:
            grown |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        new = grown & ~reached
        if not new.any() and step % 2 == 0:
            break
        dist[new] = float(step)
        reached = grown
    return dist


# --------------------------------------------------------------------------- sheet splitting
def split_sheet(rgba: np.ndarray, n_views: int, bg_tol: float = DEFAULT_BG_TOL,
                margin_frac: float = DEFAULT_SHEET_MARGIN_FRAC) -> list[Box]:
    """Find ``n_views`` objects side by side on a turnaround sheet.

    Columns containing foreground are grouped; while there are more than ``n_views`` groups, the two
    neighbours separated by the smallest gap are merged (so detached arms stay with their body).
    Returns left-to-right crops ``(x0, 0, x1, H)`` padded by ``margin_frac * W`` (not overlapping each
    other); all crops keep the full image height so vertical alignment is preserved.
    """
    arr = _as_rgba(rgba)
    h, w = arr.shape[:2]
    mask = extract_mask(arr, "auto", tol=bg_tol)
    occ = np.concatenate([[False], mask.any(axis=0), [False]]).astype(np.int8)
    d = np.diff(occ)
    groups = [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if len(groups) < n_views:
        raise ImageError(f"found {len(groups)} separated object(s) on the sheet, expected {n_views}")
    while len(groups) > n_views:
        gaps = [groups[i + 1][0] - groups[i][1] for i in range(len(groups) - 1)]
        i = int(np.argmin(gaps))
        groups[i][1] = groups[i + 1][1]
        del groups[i + 1]
    margin = int(round(margin_frac * w))
    boxes: list[Box] = []
    for i, (x0, x1) in enumerate(groups):
        left = max(0, x0 - margin)
        right = min(w, x1 + margin)
        if i > 0:
            left = max(left, (groups[i - 1][1] + x0) // 2)
        if i < len(groups) - 1:
            right = min(right, (x1 + groups[i + 1][0] + 1) // 2)
        boxes.append((left, 0, right, h))
    return boxes
