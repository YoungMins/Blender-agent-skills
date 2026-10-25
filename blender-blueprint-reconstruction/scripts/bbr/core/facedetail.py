"""Face detail check (numpy only): the second pass of a character, after the silhouettes and colours pass.

The first pass matches silhouettes and the per-view colour; a face can pass it and still look wrong up close: the nose,
lips and chin seen in profile extruded across the whole face width (a visual hull makes ridges), or a texture too coarse
for the eyes and mouth. For every face layer (``layer_hull`` without ``shell``) the report gets ``face_detail``:

* ``texels_per_px``: texture texels per reference pixel over the front-facing surface (eyes need >= MIN_TEXELS_PER_PX);
* ``profile_relief_m``: depth of the profile features (nose, lips, chin) and whether ``params.relief`` puts them on the
  midline (else they form ridges across the face);
* ``color_error``: mean absolute colour difference over the face label in the front view (0..1);
* ``shells``: glass shells covering the face; ``closeup``: a reference | model sheet of the face to look at.

:func:`warnings` turns that into soft warnings (``face_ridge``, ``face_texture``): they never fail a run, they are the
cue for the detail pass.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

MIN_TEXELS_PER_PX = 1.5
RIDGE_MIN_M = 0.002          # profile features deeper than this, without relief, form a visible ridge


def texels_per_px(uv_area: np.ndarray, world_area: np.ndarray, facing: np.ndarray, texture_size: int, m_per_px: float,
                  min_facing: float = 0.5) -> float:
    """Texel density of the faces looking at the front camera (``facing`` = cos to it >= ``min_facing``): sqrt of
    (their UV area in texels) over (their projected area in reference pixels)."""
    use = np.asarray(facing) >= min_facing
    if not use.any():
        return 0.0
    texels = float(np.asarray(uv_area)[use].sum()) * texture_size ** 2
    pixels = float((np.asarray(world_area)[use] * np.asarray(facing)[use]).sum()) / m_per_px ** 2
    return float(np.sqrt(texels / pixels)) if pixels > 0 else 0.0


def color_error(ref_rgb: np.ndarray, model_rgb: np.ndarray, region: np.ndarray) -> float:
    """Mean absolute RGB difference (0..1) over ``region``."""
    region = np.asarray(region, dtype=bool)
    if not region.any():
        return 0.0
    diff = np.abs(np.asarray(ref_rgb, dtype=float)[..., :3] - np.asarray(model_rgb, dtype=float)[..., :3])
    return float(diff[region].mean())


def warnings(block: Mapping[str, Any]) -> list[str]:
    """Soft warning strings of a ``face_detail`` report block."""
    out: list[str] = []
    for lid, f in ((block or {}).get("layers") or {}).items():
        if not isinstance(f, Mapping):
            continue
        depth = float(f.get("profile_relief_m") or 0.0)
        if not f.get("relief_applied") and depth > RIDGE_MIN_M:
            out.append(f"face_ridge {lid} (profile features up to {depth * 1000:.0f} mm deep are extruded across the whole face; "
                       f"set params.relief on the layer to keep nose, lips and chin on the midline)")
        tpp = f.get("texels_per_px")
        if tpp is not None and tpp < MIN_TEXELS_PER_PX:
            out.append(f"face_texture {lid} ({tpp:.1f} texels per reference px, below {MIN_TEXELS_PER_PX}; "
                       f"raise params.texture_size)")
    return out
