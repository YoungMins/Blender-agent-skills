"""Euclidean distance transform and signed distance of 2-D masks (numpy only, exact within a cap)."""
from __future__ import annotations

import numpy as np

DEFAULT_CAP_PX = 24


def edt(fg: np.ndarray, cap: int = DEFAULT_CAP_PX, return_index: bool = False):
    """Distance (px) from every pixel to the nearest ``True`` pixel of ``fg``, clamped to ``cap``.

    Exact for distances <= ``cap`` (two-pass column/row minimisation, vectorised over the other axis).
    With ``return_index`` also returns ``(iy, ix)`` of the nearest foreground pixel (-1 where none within ``cap``).
    """
    fg = np.asarray(fg, dtype=bool)
    h, w = fg.shape
    cap = int(max(1, cap))
    big = cap + 1
    d = np.where(fg, 0, big).astype(np.int32)
    row = np.where(fg, np.arange(h)[:, None], -1).astype(np.int32)
    for y in range(1, h):  # vertical pass, downward then upward
        cand = d[y - 1] + 1
        better = cand < d[y]
        d[y] = np.where(better, cand, d[y])
        row[y] = np.where(better, row[y - 1], row[y])
    for y in range(h - 2, -1, -1):
        cand = d[y + 1] + 1
        better = cand < d[y]
        d[y] = np.where(better, cand, d[y])
        row[y] = np.where(better, row[y + 1], row[y])
    g = np.minimum(d, big).astype(np.float32) ** 2
    out = g.copy()
    best = np.zeros((h, w), dtype=np.int16)
    for s in range(1, cap + 1):
        s2 = float(s * s)
        if s2 >= out.max():
            break
        for sign in (1, -1):  # candidate column j + sign*s
            if sign > 0:
                cand = g[:, s:] + s2
                cur = out[:, : w - s]
                better = cand < cur
                cur[better] = cand[better]
                best[:, : w - s][better] = s
            else:
                cand = g[:, : w - s] + s2
                cur = out[:, s:]
                better = cand < cur
                cur[better] = cand[better]
                best[:, s:][better] = -s
    dist = np.minimum(np.sqrt(out), cap).astype(np.float32)
    if not return_index:
        return dist
    xs = np.clip(np.arange(w)[None, :] + best, 0, w - 1)
    iy = np.take_along_axis(row, xs, axis=1)
    ix = np.where(iy >= 0, xs, -1)
    iy = np.where(out <= cap * cap, iy, -1)
    ix = np.where(iy >= 0, ix, -1)
    return dist, iy, ix


def signed_distance(mask: np.ndarray, cap: int = DEFAULT_CAP_PX) -> np.ndarray:
    """Signed distance (px, float32) to the mask boundary: negative inside, positive outside, 0 on the boundary.

    The boundary lies half a pixel from the nearest centres on each side; magnitudes clamp at ``cap``.
    """
    mask = np.asarray(mask, dtype=bool)
    outside = edt(mask, cap) - 0.5      # for background pixels: distance to the nearest foreground centre
    inside = edt(~mask, cap) - 0.5      # for foreground pixels: distance to the nearest background centre
    return np.where(mask, -inside, outside).astype(np.float32)
