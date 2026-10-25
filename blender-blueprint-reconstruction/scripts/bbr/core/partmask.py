"""Hull exclusion (numpy only): remove thin parts (horns, a sword) from the silhouettes the hull is carved from.

With part labels (:mod:`labels`) the removed pixels ARE the labelled pixels of the excluded parts
(:func:`exclude_label_pixels`: rules 2 and 3 below; no fitted mesh is involved). The mesh-silhouette path
(:func:`exclude_parts_from_masks`, all four rules) is the fallback for a view without labels.

A visual hull of front + side silhouettes cannot represent a part that sticks out of the body: it becomes a wide fan
or a blob. The part is modelled as its own primitive instead; its silhouette is rasterized in every carve view at the
reference calibration and taken out of the carve mask by :func:`exclude_part_pixels`.

Rule (``cut = dilate(part, dilate_px)``, ``rest = mask & ~cut``):

1. keep the connected component(s) of the body: everything at least ``keep_frac`` of the largest component, so a
   sliver the removal cut off (a horn tip) is dropped but a real second body part survives;
2. never punch holes: a removed pixel that ended up ENCLOSED by the kept body (the part overlaps the body interior,
   e.g. a sword in front of the torso) is restored. Only pixels that stay connected to the image border after the
   removal, i.e. outside the body, remain removed. Holes that already exist in the original mask stay holes;
3. never cut slots: a part that leaves the body (a horn on a helmet) is connected to the outside through its own
   protruding half, so rule 2 alone would also remove the wedge where it overlaps the body outline, and the hull
   would get a slot through the whole depth. Each part silhouette (connected component of ``cut``) therefore gets a
   morphological closing of ``rest`` with its own inscribed radius: removed pixels in concavities narrower than the
   part itself (the wedge between the dome and the horn) are restored, the protruding half stays removed;
4. no fins: structures thinner than ``2 * SLIVER_PX + 1`` px that touch the cut (what an imperfect part model
   leaves of the reference part, a thin crescent along a horn) are dropped, so the hull gets no fin from them.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .calibration import ViewCalibration
from .imageops import dilate, distance_to, erode, label_components
from .uvraster import rasterize_triangles

DEFAULT_EXCLUDE_DILATE_PX = 3
DEFAULT_LABEL_DILATE_PX = 0   # label masks are exact: no safety margin by default
DEFAULT_KEEP_FRAC = 0.05   # of the largest remaining component
SLIVER_PX = 4              # radius of the opening that removes thin leftovers next to a cut
MAX_CLOSE_PX = 64          # cap of the closing radius (a part thicker than this is not a thin part)


def triangles_mask(verts: np.ndarray, tris: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """Orthographic silhouette of a triangle mesh (world vertices) on the view's pixel grid, bool ``(H, W)``."""
    w, h = calib.image_size
    tris = np.asarray(tris, dtype=np.int64)
    if len(tris) == 0:
        return np.zeros((h, w), dtype=bool)
    xy = calib.world_to_px(np.asarray(verts, dtype=float))
    return rasterize_triangles(xy[tris], (w, h)) >= 0


def _enclosed(mask: np.ndarray) -> np.ndarray:
    """Background pixels of ``mask`` that cannot reach the image border (4-connected)."""
    labels, n = label_components(~mask, full=False)
    if n == 0:
        return np.zeros_like(mask)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.zeros(n + 1, dtype=bool)
    outside[border] = True
    outside[0] = True
    return ~outside[labels]


def _close_restore(rest: np.ndarray, mask: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """``rest`` plus the pixels of ``cut & mask`` that a closing of ``rest`` (radius = inscribed radius of each cut
    component + 1) fills: concavities of the body outline narrower than the part."""
    labels, n = label_components(cut, full=True)
    out = rest.copy()
    h, w = cut.shape
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labels == k)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        comp = np.pad(labels[y0:y1, x0:x1] == k, 1)
        radius = int(np.ceil(distance_to(~comp, cap=MAX_CLOSE_PX).max())) + 1
        ry0, ry1, rx0, rx1 = max(y0 - 2 * radius, 0), min(y1 + 2 * radius, h), max(x0 - 2 * radius, 0), min(x1 + 2 * radius, w)
        window = rest[ry0:ry1, rx0:rx1]
        closed = erode(dilate(window, radius), radius)
        out[ry0:ry1, rx0:rx1] |= closed & mask[ry0:ry1, rx0:rx1] & (labels[ry0:ry1, rx0:rx1] == k)
    return out


def exclude_part_pixels(mask: np.ndarray, part: np.ndarray, dilate_px: int = DEFAULT_EXCLUDE_DILATE_PX,
                        keep_frac: float = DEFAULT_KEEP_FRAC) -> np.ndarray:
    """``mask`` without the (dilated) ``part`` silhouette, keeping the body connected and hole-free (see module doc)."""
    mask = np.asarray(mask, dtype=bool)
    cut = dilate(np.asarray(part, dtype=bool), int(dilate_px))
    rest = mask & ~cut
    if not rest.any():
        return mask.copy()
    rest &= ~(rest & ~dilate(erode(rest, SLIVER_PX), SLIVER_PX) & dilate(cut, 2 * SLIVER_PX + 1))   # rule 4 first: a
    labels, n = label_components(rest, full=True)                                                   # sliver must not
    areas = np.bincount(labels.ravel(), minlength=n + 1)                                            # be bridged by 3
    areas[0] = 0
    keep = areas >= keep_frac * areas.max()
    keep[0] = False
    rest = keep[labels]
    rest = rest | (_enclosed(rest) & mask)
    return _close_restore(rest, mask, cut)


def exclude_label_pixels(mask: np.ndarray, cut: np.ndarray, dilate_px: int = DEFAULT_LABEL_DILATE_PX) -> np.ndarray:
    """``mask`` without the ``cut`` pixels (union of the excluded parts' label masks), keeping the body hole-free and
    slot-free: rule 2 (a removed pixel the body encloses, e.g. a hand over the torso, is restored) and rule 3 (a part
    seen against the body outline, a horn over the helmet dome, is connected to the outside through its protruding half;
    each part component gets a closing of the rest with its inscribed radius, so the wedge over the dome is restored and
    the protruding half stays removed). Exact labels need no rule 1 (component filter) or 4 (sliver filter): nothing is
    left over from an imperfect part model, and a real detached piece (a cape end) must survive."""
    mask = np.asarray(mask, dtype=bool)
    cut = dilate(np.asarray(cut, dtype=bool), int(dilate_px)) & mask
    rest = mask & ~cut
    if not rest.any():
        return mask.copy()
    rest = rest | (_enclosed(rest) & mask)
    return _close_restore(rest, mask, cut)


def exclude_parts_from_masks(masks: Mapping[str, np.ndarray], calibs: Mapping[str, ViewCalibration],
                             parts: Sequence[tuple[np.ndarray, np.ndarray]], views: Sequence[str],
                             dilate_px: int = DEFAULT_EXCLUDE_DILATE_PX
                             ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """New mask dict with the ``parts`` (world ``(verts, tris)`` meshes) removed from every view in ``views``.

    Returns ``(masks, stats)`` with per-view ``mask_px_before`` / ``mask_px_after`` / ``part_px``; masks of other views
    are passed through untouched (the input dict is never modified).
    """
    out = dict(masks)
    stats: dict[str, dict[str, int]] = {}
    for key in views:
        if key not in masks or key not in calibs:
            continue
        part = np.zeros_like(masks[key], dtype=bool)
        for verts, tris in parts:
            part |= triangles_mask(verts, tris, calibs[key])
        out[key] = exclude_part_pixels(masks[key], part, dilate_px)
        stats[key] = {"mask_px_before": int(masks[key].sum()), "mask_px_after": int(out[key].sum()),
                      "part_px": int(part.sum())}
    return out, stats
