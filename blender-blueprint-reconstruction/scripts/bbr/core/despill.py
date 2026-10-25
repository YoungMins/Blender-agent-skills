"""Chroma-background decontamination of reference colour images (numpy only).

Reference views on a chroma backdrop (green screen) carry the backdrop colour in anti-aliased edge pixels and in
enclosed backdrop pockets (gap between arm and body). Sampled into a texture these show up as coloured specks.
:func:`decontaminate_view` removes the pockets from the colour mask and, within ``edge_px`` of the remaining
boundary, un-mixes the backdrop (``c = (c_obs - (1 - a) * bg) / a``) and clamps the leftover spill.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .distance import edt

DEFAULT_SPILL_EDGE_PX = 6
CHROMA_DOMINANCE = 0.15       # bg is "chroma" when its top channel exceeds the others by this much
MIN_ALPHA = 0.3               # lowest foreground coverage trusted by the un-mixing
MIN_FG_CONTRAST = 0.2         # |interior - bg| below this: the mixing line is ill-conditioned, skip the un-mixing
MAX_LINE_RESIDUAL = 0.12      # a pixel farther than this from the bg->interior line is detail, not a blend
LEAK_DISTANCE = 0.12          # background_leak_ratio: RGB distance to the backdrop colour counted as leaked


def chroma_channel(bg: Sequence[float], dominance: float = CHROMA_DOMINANCE) -> Optional[int]:
    """Index of the dominant channel of a chroma backdrop colour; None for gray / white / black backdrops."""
    bg = np.asarray(bg, dtype=float)[:3]
    ch = int(bg.argmax())
    return ch if bg[ch] - np.delete(bg, ch).max() >= dominance else None


def chroma_dominant(rgb: np.ndarray, ch: int, dominance: float = CHROMA_DOMINANCE) -> np.ndarray:
    """Pixels whose channel ``ch`` exceeds both other channels by ``dominance``."""
    others = np.delete(rgb, ch, axis=-1)
    return rgb[..., ch] - others.max(axis=-1) > dominance


def decontaminate_view(rgba: np.ndarray, mask: np.ndarray, bg_color: Sequence[float], edge_px: int = DEFAULT_SPILL_EDGE_PX,
                       tol: float = LEAK_DISTANCE) -> tuple[np.ndarray, np.ndarray]:
    """``(rgba, colour mask)`` of a view with the chroma backdrop removed; unchanged for a non-chroma backdrop.

    1. Pixels of ``mask`` within ``tol`` (RGB distance) of ``bg_color`` are backdrop pockets: dropped from the mask.
    2. Pixels within ``edge_px`` of the new mask boundary that lie on the line backdrop -> nearest interior colour are
       un-mixed (alpha clamped to ``MIN_ALPHA``..1).
    3. Those pixels then get the standard spill clamp ``c[ch] = min(c[ch], max(other channels))``.
    """
    bg = np.asarray(bg_color, dtype=np.float32)[:3]
    ch = chroma_channel(bg)
    if ch is None or edge_px <= 0:
        return rgba, mask
    rgb = rgba[..., :3]
    eff = mask & (np.sqrt(((rgb - bg) ** 2).sum(axis=-1)) >= tol)
    dist = edt(~eff, edge_px + 1)                       # 1 = pixel touching the outside
    near = eff & (dist <= edge_px)
    out = np.array(rgb, copy=True)
    core = eff & ~near
    if core.any() and near.any():
        _, iy, ix = edt(core, 3 * edge_px, return_index=True)
        ny, nx = iy[near], ix[near]
        found = ny >= 0
        fg = rgb[np.where(found, ny, 0), np.where(found, nx, 0)]
        c = rgb[near]
        line = fg - bg
        den = (line ** 2).sum(axis=1)
        t = ((c - bg) * line).sum(axis=1) / np.maximum(den, 1e-9)
        resid = np.sqrt(((c - bg - t[:, None] * line) ** 2).sum(axis=1))
        ok = found & (den >= MIN_FG_CONTRAST ** 2) & (resid <= MAX_LINE_RESIDUAL) & (t < 1.0)
        a = np.clip(t, MIN_ALPHA, 1.0)[:, None]
        un = (c - (1.0 - a) * bg) / a
        out[near] = np.where(ok[:, None], np.clip(un, 0.0, 1.0), c)
    px = out[near]
    px[:, ch] = np.minimum(px[:, ch], np.delete(px, ch, axis=1).max(axis=1))
    out[near] = px
    res = np.array(rgba, copy=True)
    res[..., :3] = out
    return res, eff


def background_leak_ratio(rgb: np.ndarray, covered: np.ndarray, bg_colors: Sequence[Sequence[float]],
                          tol: float = LEAK_DISTANCE) -> float:
    """Fraction of the ``covered`` texels within ``tol`` of a backdrop colour or dominated by a chroma backdrop's channel."""
    covered = np.asarray(covered, dtype=bool)
    if not covered.any() or not len(bg_colors):
        return 0.0
    px = np.asarray(rgb, dtype=float)[covered]
    leak = np.zeros(len(px), dtype=bool)
    for bg in bg_colors:
        bg = np.asarray(bg, dtype=float)[:3]
        leak |= np.sqrt(((px - bg) ** 2).sum(axis=1)) < tol
        ch = chroma_channel(bg)
        if ch is not None:
            leak |= chroma_dominant(px, ch)
    return float(leak.mean())
