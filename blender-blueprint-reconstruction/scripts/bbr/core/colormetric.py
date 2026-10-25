"""Soft colour metric: mean colour difference between a reference view and a flat-shaded render (numpy only)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .imageops import erode

DEFAULT_ERODE_PX = 2   # keep silhouette-edge pixels (antialiasing, projection bleed guard) out of the comparison


def color_error(ref_rgb: np.ndarray, out_rgb: np.ndarray, ref_mask: np.ndarray, out_mask: np.ndarray,
                erode_px: int = DEFAULT_ERODE_PX) -> dict[str, Any]:
    """Mean absolute RGB difference (0..1, sRGB-encoded) inside the eroded intersection of both masks.

    ``error`` is None when the intersection is empty. ``pixels`` = number of compared pixels.
    """
    both = np.asarray(ref_mask, dtype=bool) & np.asarray(out_mask, dtype=bool)
    region = erode(both, erode_px) if erode_px > 0 else both
    if not region.any():
        region = both
    n = int(region.sum())
    if n == 0:
        return {"error": None, "pixels": 0}
    diff = np.abs(np.asarray(ref_rgb, dtype=float)[..., :3][region] - np.asarray(out_rgb, dtype=float)[..., :3][region])
    return {"error": float(diff.mean()), "pixels": n}


def format_color_error_line(view: str, error: Optional[float]) -> str:
    """``Front Color Error : 7.3%`` (``n/a`` without a value)."""
    value = "n/a" if error is None else f"{error * 100:.1f}%"
    return f"{view.capitalize()} Color Error : {value}"
