"""Background-safe colour sampling of the reference views (numpy only)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .calibration import ViewCalibration
from .distance import edt
from .imageops import erode

DEFAULT_EDGE_ERODE_PX = 3
NEAREST_SEARCH_PX = 24
FALLBACK_COLOR = (0.5, 0.5, 0.5, 1.0)

ViewInput = tuple[np.ndarray, np.ndarray, ViewCalibration]  # (rgba, mask, calibration)


class ViewSampler:
    """Background-safe colour sampling of one reference view (used by the texel projection).

    Samples never touch pixels outside the mask eroded by ``edge_erode_px`` (no background bleed): a projected pixel
    outside it is replaced by the nearest pixel inside it (within ``snap_px``, default ``NEAREST_SEARCH_PX``;
    farther points get no sample), and a bilinear tap outside it falls back to the nearest pixel.
    """

    def __init__(self, rgba: np.ndarray, mask: np.ndarray, calib: ViewCalibration,
                 edge_erode_px: int = DEFAULT_EDGE_ERODE_PX, snap_px: int = NEAREST_SEARCH_PX) -> None:
        self.rgba, self.calib, self.snap_px = rgba, calib, int(snap_px)
        self.eroded = erode(mask, edge_erode_px) if edge_erode_px > 0 else mask
        self._nearest: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._edge: Optional[tuple[int, np.ndarray]] = None

    def edge_distance(self, points: np.ndarray, cap_px: int) -> np.ndarray:
        """Distance (px, clamped to ``cap_px``) from the projected pixel to the eroded silhouette boundary; 0 outside."""
        if self._edge is None or self._edge[0] != cap_px:
            self._edge = (cap_px, edt(~self.eroded, cap_px))
        uv = self.calib.world_to_px(points)
        h, w = self.eroded.shape
        px = np.floor(uv[:, 0]).astype(int)
        py = np.floor(uv[:, 1]).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        out = np.zeros(len(points))
        out[inside] = self._edge[1][py[inside], px[inside]]
        return out

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(color (N,3), inside_image (N,), valid (N,))`` for world ``points``; ``valid`` = sample came from the mask."""
        n = len(points)
        uv = self.calib.world_to_px(points)
        h, w = self.eroded.shape
        px = np.floor(uv[:, 0]).astype(int)
        py = np.floor(uv[:, 1]).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        valid = np.zeros(n, dtype=bool)
        valid[inside] = self.eroded[py[inside], px[inside]]
        out_idx = inside & ~valid
        if out_idx.any() and self.snap_px > 0:
            if self._nearest is None:
                _, iy, ix = edt(self.eroded, max(self.snap_px, 1), return_index=True)
                self._nearest = (iy, ix)
            ny, nx = self._nearest[0][py[out_idx], px[out_idx]], self._nearest[1][py[out_idx], px[out_idx]]
            found = ny >= 0
            sel = np.flatnonzero(out_idx)[found]
            uv[sel, 0], uv[sel, 1] = nx[found] + 0.5, ny[found] + 0.5
            valid[sel] = True
        return _sample_bilinear(self.rgba, uv, self.eroded), inside, valid


def _sample_bilinear(img: np.ndarray, uv: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Bilinear RGB sample at continuous pixel coordinates (pixel centres at +0.5), clamped to the image.

    With ``mask``, a point with any of its four taps outside the mask takes the colour of the pixel it lies in
    (nearest) instead, so no background pixel is ever blended in.
    """
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    rgb = img[..., :3]
    top = rgb[y0, x0] * (1 - fx) + rgb[y0, x1] * fx
    bot = rgb[y1, x0] * (1 - fx) + rgb[y1, x1] * fx
    out = top * (1 - fy) + bot * fy
    if mask is not None:
        bad = ~(mask[y0, x0] & mask[y0, x1] & mask[y1, x0] & mask[y1, x1])
        if bad.any():
            nx = np.clip(np.floor(uv[bad, 0]).astype(int), 0, w - 1)
            ny = np.clip(np.floor(uv[bad, 1]).astype(int), 0, h - 1)
            out[bad] = rgb[ny, nx]
    return out
