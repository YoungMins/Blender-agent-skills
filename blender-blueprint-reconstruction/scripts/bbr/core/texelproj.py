"""Per-texel colour projection from reference views (numpy only): view selection by facing with a priority bias,
visibility by depth buffer, silhouette-edge distrust, seam blending, inpainting and padding.

Same sampling rules as :mod:`colorproj` (samples only from the eroded mask, no background bleed), evaluated at the
texel positions produced by :mod:`uvraster` instead of at vertices, so the colour resolution is the texture's.
A texel takes the colour of ONE primary view; neighbouring views only blend in within a narrow seam band, so the
second view's content never ghosts over a surface the primary view sees well. Texels no view sees well are inpainted
from valid neighbours (UV island first, then the nearest valid surface point in 3D), never from background pixels.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .. import get_logger
from .colorproj import DEFAULT_EDGE_ERODE_PX, FALLBACK_COLOR, ViewInput, ViewSampler
from .conventions import AXES
from .distance import edt
from .zbuffer import visible

log = get_logger(__name__)

DEFAULT_VIEW_PRIORITY = ("front", "back", "side")  # earlier = preferred; unlisted views rank last
DEFAULT_PRIORITY_MARGIN = 0.0   # score penalty per priority rank: a lower-ranked view must face this much better to win
DEFAULT_SEAM_BLEND_WIDTH = 0.1   # views within this score of the primary blend in (smoothstep); 0 = hard selection
DEFAULT_CONFLICT_THRESHOLD = 0.25  # mean abs RGB difference above which two views "disagree" (no blending, conflict stat)
DEFAULT_EDGE_DISTRUST_PX = 8     # samples closer than this to a view's silhouette boundary lose confidence
EDGE_PENALTY = 0.5               # score penalty of a sample right at the silhouette boundary
DEFAULT_SNAP_PX = 2              # a projected pixel just outside the mask may snap to the mask by this much
DEFAULT_DEPTH_TOL_PX = 3.0       # occlusion tolerance in reference pixels
MIN_FACING = 0.05                # views seeing a texel more obliquely than this do not contribute
CHUNK_TEXELS = 300_000
DEFAULT_PADDING_PX = 8
DONOR_MIN_TRUST = 0.5            # inpainting copies colour only from texels at least this far (share of edge_distrust_px) inside a silhouette
INPAINT_UV_PX = 48               # max UV distance (px) of the same-island fill
INPAINT_CELLS = 160              # voxel cells along the longest extent for the 3-D nearest-valid fill
INPAINT_QUERY_CHUNK = 256
INPAINT_SMOOTH_ITERATIONS = 8   # 3x3 diffusion passes over the inpainted texels (valid texels never change)
CELL_KEY_STRIDE = 1_000_003

# source-map colours: front red, side green, back blue, other views cycle; inpainted grey
VIEW_COLORS = {"front": (1.0, 0.0, 0.0), "side": (0.0, 1.0, 0.0), "back": (0.0, 0.0, 1.0)}
EXTRA_VIEW_COLORS = ((1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))
INPAINT_COLOR = (0.5, 0.5, 0.5)
METHOD_NONE, METHOD_UV, METHOD_3D = 0, 1, 2


MIN_TEXTURE_PX, MAX_TEXTURE_PX = 128, 1024


def suggest_texture_size(area_m2: float, m_per_px: float, lo: int = MIN_TEXTURE_PX, hi: int = MAX_TEXTURE_PX) -> int:
    """Power-of-two texture side giving about one texel per reference pixel over a surface of ``area_m2`` (the reference
    cannot resolve more), clamped to ``[lo, hi]``: a 4 cm bracket needs ``lo``, a chest body ``hi``. UV packing fills the
    whole square, so the side is ``sqrt(area / m_per_px**2)`` rounded up to a power of two."""
    if area_m2 <= 0 or m_per_px <= 0:
        return lo
    side = (area_m2 / (m_per_px * m_per_px)) ** 0.5
    return int(min(max(2 ** int(np.ceil(np.log2(max(side, 1.0)))), lo), hi))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def project_texel_colors(points: np.ndarray, normals: np.ndarray, views: Sequence[ViewInput],
                         zbuffers: Optional[Sequence[Optional[np.ndarray]]] = None,
                         depth_tol_px: float = DEFAULT_DEPTH_TOL_PX, edge_erode_px: int = DEFAULT_EDGE_ERODE_PX,
                         view_priority: Sequence[str] = DEFAULT_VIEW_PRIORITY,
                         priority_margin: float = DEFAULT_PRIORITY_MARGIN,
                         seam_blend_width: float = DEFAULT_SEAM_BLEND_WIDTH,
                         conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
                         edge_distrust_px: int = DEFAULT_EDGE_DISTRUST_PX,
                         snap_px: int = DEFAULT_SNAP_PX,
                         label_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                         view_penalty: Optional[Sequence[float]] = None,
                         min_facing: float = MIN_FACING
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pick reference colors for texel positions; returns ``(rgb (M,3) in [0,1], has (M,) bool, info)``.

    A view is a *candidate* for a texel when it sees it (``dot(normal, camera_axis) > MIN_FACING``), the sample comes
    from the eroded mask (within ``snap_px`` of it) and, with ``zbuffers`` (one per view, from
    :func:`~bbr.core.zbuffer.depth_buffer`), the texel is not hidden (``depth_tol_px`` reference pixels of tolerance).
    Candidate score = facing - rank * ``priority_margin`` (rank in ``view_priority``; unlisted views rank last)
    - ``EDGE_PENALTY`` * (1 - distance-to-silhouette / ``edge_distrust_px``). The best score is the primary view;
    candidates within ``seam_blend_width`` of it blend in with smoothstep weights unless their colour differs from the
    primary by more than ``conflict_threshold`` (mean abs RGB). Texels without any candidate have ``has`` False and
    undefined colour (gray): inpaint them with :func:`inpaint_colors`.

    ``info``: ``weights (M,V)`` (rows sum to 1 where ``has``), ``conflict (M,)`` (>= 2 candidates that disagree
    strongly, whatever the weights), ``view_keys`` (``calib.view`` per view), ``label_rejected (M,)`` (a view that
    would have seen the texel was refused because the pixel carries another object's label), ``trust (M,)`` (0..1, how far
    the primary view's sample lies inside its silhouette: a silhouette-edge sample is contaminated by the backdrop and must not
    be a colour donor for :func:`inpaint_colors`).

    ``label_masks`` (one per view, ``None`` = view not gated): bool image of the pixels labelled as THIS object (see
    :mod:`labels`). The view's mask is cut to them, so the texel only samples pixels of its own label (never a
    neighbour's, also not by snapping or filtering); a texel whose pixel has another label is rejected in that view and
    takes another view or, with none, is inpainted.

    ``view_penalty`` (one score penalty per view, default 0): subtracted from the view's candidate score. A mirrored
    virtual view (:mod:`symmetry`) gets a penalty so that it is only the primary where no real view sees the texel
    nearly as well (a tie-break, not a competitor), and blends in only inside the seam band of the best real view.

    ``min_facing`` (default ``MIN_FACING``): views seeing a texel more obliquely than this do not contribute; hard-surface
    parts use a higher value, since a flat face seen at a grazing angle only smears the silhouette-edge pixels.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    n, nv = len(points), len(views)
    gates = list(label_masks) if label_masks is not None else [None] * nv
    samplers = [ViewSampler(rgba, mask if gate is None else mask & gate, calib, edge_erode_px, snap_px)
                for (rgba, mask, calib), gate in zip(views, gates)]
    ungated = [None if gate is None else ViewSampler(rgba, mask, calib, edge_erode_px, snap_px)
               for (rgba, mask, calib), gate in zip(views, gates)]
    rejected = np.zeros(n, dtype=bool)
    penalty = np.zeros(nv) if view_penalty is None else np.asarray(view_penalty, dtype=float)
    ranks = [list(view_priority).index(s.calib.view) if s.calib.view in view_priority else len(view_priority)
             for s in samplers]
    out = np.tile(np.asarray(FALLBACK_COLOR[:3]), (n, 1))
    has = np.zeros(n, dtype=bool)
    weights = np.zeros((n, nv))
    conflict = np.zeros(n, dtype=bool)
    confidence = np.zeros(n)
    for lo in range(0, n, CHUNK_TEXELS):
        sl = slice(lo, min(lo + CHUNK_TEXELS, n))
        p, nr = points[sl], normals[sl]
        m = len(p)
        score = np.full((m, nv), -np.inf)
        color = np.zeros((m, nv, 3))
        trusts = np.zeros((m, nv))
        for i, sampler in enumerate(samplers):
            calib = sampler.calib
            facing = nr @ AXES[calib.camera_axis]
            col, _, in_mask = sampler.sample(p)
            ok = in_mask & (facing > min_facing)
            if ungated[i] is not None:
                would = ungated[i].sample(p)[2] & (facing > min_facing)
            if zbuffers is not None and zbuffers[i] is not None:
                seen = visible(p, calib, zbuffers[i], depth_tol_px * calib.m_per_px)
                ok &= seen
                if ungated[i] is not None:
                    would &= seen
            if ungated[i] is not None:
                rejected[sl] |= would & ~ok
            trust = np.clip(sampler.edge_distance(p, edge_distrust_px) / max(edge_distrust_px, 1), 0.0, 1.0)
            trusts[:, i] = trust
            score[:, i] = np.where(ok, facing - ranks[i] * priority_margin - penalty[i]
                                    - (1.0 - trust) * EDGE_PENALTY, -np.inf)
            color[:, i] = col
        cand = np.isfinite(score)
        got = cand.any(axis=1)
        best = np.where(got, score.max(axis=1), 0.0)
        primary = score.argmax(axis=1)
        rows = np.arange(m)
        diff = np.abs(color - color[rows, primary][:, None, :]).mean(axis=2)   # (m, nv) vs the primary colour
        agree = diff <= conflict_threshold
        if seam_blend_width > 0:
            w = _smoothstep((score - (best[:, None] - seam_blend_width)) / seam_blend_width)
        else:
            w = np.zeros((m, nv))
        w = np.where(cand & agree & (best[:, None] - score < seam_blend_width), w, 0.0)
        w[rows, primary] = np.where(got, 1.0, 0.0)
        w = w / np.maximum(w.sum(axis=1), 1e-12)[:, None]
        out[sl] = np.where(got[:, None], np.einsum("mv,mvc->mc", w, color), out[sl])
        has[sl], weights[sl] = got, w
        confidence[sl] = np.where(got, trusts[rows, primary], 0.0)
        conflict[sl] = (cand & ~agree).any(axis=1)
    log.info("texel projection: %d/%d texels from visible samples, %d to inpaint", int(has.sum()), n, int((~has).sum()))
    info = {"weights": weights, "conflict": conflict, "view_keys": [s.calib.view for s in samplers],
            "label_rejected": rejected, "trust": confidence}
    return np.clip(out, 0.0, 1.0), has, info


def source_map_colors(weights: np.ndarray, has: np.ndarray, view_keys: Sequence[str]) -> np.ndarray:
    """``(M,3)`` debug colour per texel: weights mixed over the view colours (front red, side green, back blue);
    texels without a valid sample (inpainted) are gray."""
    extra = iter(EXTRA_VIEW_COLORS * 4)
    palette = np.array([VIEW_COLORS.get(k) or next(extra) for k in view_keys], dtype=float).reshape(-1, 3)
    rgb = np.asarray(weights, dtype=float) @ palette
    return np.where(np.asarray(has, dtype=bool)[:, None], rgb, np.asarray(INPAINT_COLOR))


def texture_shares(weights: np.ndarray, has: np.ndarray, conflict: np.ndarray, view_keys: Sequence[str]) -> dict[str, Any]:
    """Exclusive shares of the covered texels (``views`` + ``blend_share`` + ``inpainted_share`` sum to 1):
    ``views`` = texels that come from that view alone, ``blend_share`` = >= 2 views mixed, ``inpainted_share`` = no
    valid sample; ``conflict_ratio`` = texels with >= 2 candidate views that disagree strongly."""
    weights, has = np.asarray(weights, dtype=float), np.asarray(has, dtype=bool)
    total = max(len(has), 1)
    used = (weights > 1e-3) & has[:, None]
    single = used.sum(axis=1) == 1
    shares = {k: 0.0 for k in view_keys}
    for i, k in enumerate(view_keys):
        shares[k] += float((single & used[:, i]).sum()) / total
    return {"views": {k: round(v, 4) for k, v in shares.items()},
            "blend_share": round(float((has & ~single).sum()) / total, 4),
            "inpainted_share": round(float((~has).sum()) / total, 4),
            "conflict_ratio": round(float(np.asarray(conflict, dtype=bool).sum()) / total, 4)}


def _cell_keys(pos: np.ndarray, lo: np.ndarray, cell: float) -> np.ndarray:
    k = np.floor((pos - lo) / cell).astype(np.int64)
    return (k[:, 0] * CELL_KEY_STRIDE + k[:, 1]) * CELL_KEY_STRIDE + k[:, 2]


def _cell_means(values: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unique keys, per-key mean of values (K,3))`` plus the inverse index via the second return of ``np.unique``."""
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uk))[:, None]
    mean = np.stack([np.bincount(inv, weights=values[:, c], minlength=len(uk)) for c in range(3)], axis=1) / cnt
    return mean, inv


def inpaint_colors(img: np.ndarray, valid: np.ndarray, covered: np.ndarray, island: np.ndarray, position: np.ndarray,
                   uv_px: int = INPAINT_UV_PX, cells: int = INPAINT_CELLS,
                   smooth_iterations: int = INPAINT_SMOOTH_ITERATIONS, donor: Optional[np.ndarray] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Fill covered texels without a valid sample (``covered & ~valid``) from valid ones; never from other pixels.

    ``donor`` (bool, default ``valid``): the valid texels allowed to give colour, e.g. those with ``trust >=``
    :data:`DONOR_MIN_TRUST` (a sample on a silhouette edge is mixed with the backdrop and can be near black: an unseen
    underside copied from it renders black). Without any donor every valid texel is one.

    1. nearest valid texel in UV space, accepted only inside the same UV island (``island`` id map, -1 = empty);
    2. otherwise the nearest valid texel in 3-D (``position (H,W,3)``), on a voxel grid of ``cells`` cells along the
       longest extent (colours averaged per cell), so islands without any valid texel still get a surface colour.
    The filled texels are then softened by ``smooth_iterations`` diffusion passes.

    Returns ``(img, method (H,W) int8)``: 0 = untouched, 1 = UV fill, 2 = 3-D fill. Without any valid texel nothing changes.
    """
    img = np.array(img, copy=True)
    valid, covered = np.asarray(valid, dtype=bool), np.asarray(covered, dtype=bool)
    method = np.zeros(valid.shape, dtype=np.int8)
    todo = covered & ~valid
    if not todo.any() or not valid.any():
        return img, method
    src = valid if donor is None else valid & np.asarray(donor, dtype=bool)
    if not src.any():
        src = valid
    _, iy, ix = edt(src, max(int(uv_px), 1), return_index=True)
    near = todo & (iy >= 0)
    sy, sx = np.where(near, iy, 0), np.where(near, ix, 0)
    same = near & (island[sy, sx] == island)
    img[same] = img[sy[same], sx[same]]
    method[same] = METHOD_UV
    rest = todo & ~same
    if rest.any():
        pos = np.asarray(position, dtype=float)
        lo = pos[covered].min(axis=0)
        cell = max(float((pos[covered].max(axis=0) - lo).max()) / max(cells, 1), 1e-9)
        src_pos, _ = _cell_means(pos[src], _cell_keys(pos[src], lo, cell))
        src_col, _ = _cell_means(img[src], _cell_keys(pos[src], lo, cell))
        q_pos, q_inv = _cell_means(pos[rest], _cell_keys(pos[rest], lo, cell))
        pick = np.empty(len(q_pos), dtype=np.int64)
        for a in range(0, len(q_pos), INPAINT_QUERY_CHUNK):
            d2 = ((q_pos[a:a + INPAINT_QUERY_CHUNK, None, :] - src_pos[None]) ** 2).sum(axis=2)
            pick[a:a + INPAINT_QUERY_CHUNK] = d2.argmin(axis=1)
        img[rest] = src_col[pick[q_inv]]
        method[rest] = METHOD_3D
    return _diffuse(img, method > 0, covered, smooth_iterations), method


def _diffuse(img: np.ndarray, fill: np.ndarray, covered: np.ndarray, iterations: int) -> np.ndarray:
    """Soften the filled texels with masked 3x3 box averages over covered texels (gaps between islands block leaks)."""
    if iterations <= 0 or not fill.any():
        return img
    h, w = covered.shape
    wgt = covered.astype(float)
    for _ in range(iterations):
        pad = np.pad(img * wgt[..., None], ((1, 1), (1, 1), (0, 0)))
        pw = np.pad(wgt, 1)
        acc = sum(pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3))
        cnt = sum(pw[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3))
        img = np.where(fill[..., None], acc / np.maximum(cnt, 1e-9)[..., None], img)
    return img


def dilate_colors(img: np.ndarray, mask: np.ndarray, radius: int = DEFAULT_PADDING_PX) -> tuple[np.ndarray, np.ndarray]:
    """Fill empty texels (``~mask``) with the nearest covered texel's colour within ``radius`` px (seam padding).

    Returns ``(img, filled_mask)``; texels farther than ``radius`` from any covered texel stay unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.array(img, copy=True)
    if radius <= 0 or not mask.any():
        return out, mask
    _, iy, ix = edt(mask, radius, return_index=True)
    fill = ~mask & (iy >= 0)
    out[fill] = img[iy[fill], ix[fill]]
    return out, mask | fill
