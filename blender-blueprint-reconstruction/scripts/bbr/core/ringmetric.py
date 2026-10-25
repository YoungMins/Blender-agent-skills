"""Ring ("lampshade") metric of a hull field (numpy only).

A per-slice cross-section carve can turn a spiky silhouette (hair, fur) into a stack of horizontal rings: every spike
row inflates the whole cross-section, not just the outline. The references only see the hull along the carve axes, so
a ledge that runs around the *whole* circumference is not supported by them. The metric looks at the other azimuths
only: per Z slice the outer radius from the slice centroid in ``bins`` azimuth sectors, high-passed along Z, taken as
the median over the sectors (a ring has the same sign everywhere, a local fin or plate does not), RMS over the body
slices, relative to the mean radius. A smooth sphere scores ~0; a lampshade stack scores several percent.
"""
from __future__ import annotations

import warnings

import numpy as np

BINS = 48
SKIP_BINS = 3                 # sectors within this many bins of a carve axis (+-X, +-Y) are supported by a silhouette
HIGH_PASS_FRAC = 0.015        # Gaussian sigma of the high-pass along Z, share of the height
BODY_MIN_RADIUS = 0.4         # slices with a mean radius below this share of the maximum are not scored
BODY_MIN_FILL = 0.7           # nor are slices whose area is below this share of the disc of the mean radius (legs, gaps)
FEATURE_FRAC = 0.1            # slices whose high-passed radius exceeds this share of the mean radius are a part (brim), not a ring
FLAG_PCT = 2.4                # ``ring_pct`` above this is flagged by the bench (see bench/BASELINE.md for the cases' values)


def radial_profile(field: np.ndarray, bins: int = BINS) -> tuple[np.ndarray, np.ndarray]:
    """Outer radius (in cells) of the inside voxels (``field < 0``) per Z slice and azimuth sector, ``[nz, bins]``
    (NaN where a sector is empty), measured from the slice's centroid; and the slice areas (cells)."""
    inside = field < 0
    nx, ny, nz = inside.shape
    gx, gy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    out = np.full((nz, bins), np.nan)
    area = inside.sum(axis=(0, 1)).astype(float)
    for k in range(nz):
        m = inside[:, :, k]
        if m.sum() < 20:
            continue
        dx, dy = gx[m] - gx[m].mean(), gy[m] - gy[m].mean()
        sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        radius = np.full(bins, -1.0)
        np.maximum.at(radius, sector, np.hypot(dx, dy))
        out[k] = np.where(radius >= 0, radius, np.nan)
    return out, area


def _gauss_z(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(int(np.ceil(3 * sigma)), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, "valid") for j in range(x.shape[1])], axis=1)


def ring_score(field: np.ndarray, cell: float, height: float, bins: int = BINS, high_pass_frac: float = HIGH_PASS_FRAC,
               skip_bins: int = SKIP_BINS) -> float:
    """Ring energy of a hull ``field`` (m, negative inside; voxel size ``cell``) as a fraction of the mean radius:
    RMS over the body slices of the azimuth-median, Z-high-passed outer radius of the sectors no carve axis sees."""
    radius, area = radial_profile(field, bins)
    ok = np.isfinite(radius)
    sigma = max(high_pass_frac * height / cell, 1.0)
    weight = _gauss_z(ok.astype(float), sigma)
    smooth = _gauss_z(np.where(ok, radius, 0.0), sigma) / np.maximum(weight, 1e-9)
    resid = np.where(ok & (weight > 0.99), radius - smooth, np.nan)
    centre = (np.arange(bins) + 0.5) / bins * 360.0 - 180.0
    axis_dist = np.min(np.abs(((centre[:, None] - np.array([-180.0, -90.0, 0.0, 90.0, 180.0])[None, :]))), axis=1)
    free = axis_dist > (skip_bins + 0.5) * 360.0 / bins
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_r = np.nanmean(np.where(free[None, :], radius, np.nan), axis=1)
        ring = np.nanmedian(np.where(free[None, :], resid, np.nan), axis=1)
    body = np.isfinite(mean_r) & np.isfinite(ring) & (np.isfinite(resid[:, free]).mean(axis=1) > 0.8)
    if not body.any():
        return 0.0
    body &= (mean_r > BODY_MIN_RADIUS * mean_r[body].max()) & (area > BODY_MIN_FILL * np.pi * mean_r ** 2)
    body &= np.abs(np.where(np.isfinite(ring), ring, np.inf)) < FEATURE_FRAC * mean_r
    if not body.any():
        return 0.0
    return float(np.sqrt(np.mean(ring[body] ** 2)) / np.mean(mean_r[body]))
