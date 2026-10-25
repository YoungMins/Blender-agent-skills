"""Where does each model object's texture come from, and is the object really visible? (numpy only)

Both metrics work on surface samples of every model object (see :func:`geomscore.sample_surface`), tested against
the colour views:

* **Misattribution**: the sample's source pixel (the pixel of the view that colours it) belongs, in that view's
  object-ID render, to a different object. Reported per object as a share of its sampled texels. Texels are
  depth-tested before a source view is chosen, so this fires at boundaries (snapping, the depth tolerance), not for
  a part hidden inside another one; the occlusion share covers that case.
* **Camera-facing occlusion**: the sample faces a colour camera (``normal . camera_axis > FACING``) but fails that
  view's depth test: something else is in front. A correctly mounted part is unoccluded in the view that sees it
  best, so the headline number is the share in the *best* view (the minimum over views); an embedded or hidden
  part is occluded in every view. ``pooled_share`` (all facing pairs) is reported too but is no warning basis: the
  inner face of a surface-mounted part legitimately faces a camera that the body hides.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import ViewCalibration
from .conventions import AXES
from .zbuffer import visible

FACING = 0.3              # normal . camera_axis above which a surface sample "faces" a colour camera
OCCLUSION_WARN = 0.3      # best-view occlusion share above which a part is flagged embedded / occluded
MISATTRIBUTION_WARN = 0.05
MIN_VIEW_SAMPLES = 30     # a view needs this many facing samples of an object to count for its best-view share
MIN_VIEW_FRACTION = 0.05  # ... and this fraction of the object's samples
DEPTH_TOL_PX = 3.0        # same occlusion tolerance (reference pixels) as the texture projection


def id_at_pixels(id_map: np.ndarray, calib: ViewCalibration, points: np.ndarray) -> np.ndarray:
    """Object id (0 = background) of the ID render at the pixel each world point projects to (clamped to the image)."""
    h, w = id_map.shape
    uv = calib.world_to_px(points)
    px = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 1)
    return id_map[py, px]


def occlusion_by_object(points: np.ndarray, normals: np.ndarray, owner: np.ndarray, n_objects: int,
                        calibs: Sequence[ViewCalibration], zbuffers: Sequence[np.ndarray],
                        facing: float = FACING, depth_tol_px: float = DEPTH_TOL_PX) -> dict[int, dict[str, Any]]:
    """Per object index: ``{samples, per_view: {view: {facing, occluded, share}}, occlusion_share, best_view,
    pooled_share, hidden_share}``.

    ``share`` = occluded / facing samples of the object in that view. ``occlusion_share`` = the smallest ``share`` over
    the views that see enough of the object (``MIN_VIEW_SAMPLES`` / ``MIN_VIEW_FRACTION``; the best-seen view when
    none does); ``None`` when no view faces the object. ``hidden_share`` = share of the samples facing some view that
    are visible in none of the views they face.
    """
    out: dict[int, dict[str, Any]] = {}
    nv = len(calibs)
    faces = np.zeros((len(points), nv), dtype=bool)
    occ = np.zeros((len(points), nv), dtype=bool)
    for j, (calib, zbuf) in enumerate(zip(calibs, zbuffers)):
        faces[:, j] = normals @ AXES[calib.camera_axis] > facing
        occ[:, j] = faces[:, j] & ~visible(points, calib, zbuf, depth_tol_px * calib.m_per_px)
    for k in range(n_objects):
        sel = owner == k
        total = int(sel.sum())
        views: dict[str, dict[str, Any]] = {}
        for j, calib in enumerate(calibs):
            nf, no = int(faces[sel, j].sum()), int(occ[sel, j].sum())
            views[calib.view] = {"facing": nf, "occluded": no, "share": (no / nf) if nf else None}
        seen = [(v, i) for v, i in views.items() if i["facing"] >= max(MIN_VIEW_SAMPLES, MIN_VIEW_FRACTION * total)]
        if not seen:
            seen = [(v, i) for v, i in views.items() if i["facing"] > 0]
            seen = [max(seen, key=lambda t: t[1]["facing"])] if seen else []
        best = min(seen, key=lambda t: t[1]["share"]) if seen else None
        any_face = faces[sel].any(axis=1)
        pooled_f, pooled_o = int(faces[sel].sum()), int(occ[sel].sum())
        shown = (faces[sel] & ~occ[sel]).any(axis=1)
        out[k] = {"samples": total, "per_view": views,
                  "occlusion_share": None if best is None else float(best[1]["share"]),
                  "best_view": None if best is None else best[0],
                  "pooled_share": (pooled_o / pooled_f) if pooled_f else None,
                  "hidden_share": float((any_face & ~shown).sum() / any_face.sum()) if any_face.any() else None}
    return out


def misattribution_by_object(points: np.ndarray, owner: np.ndarray, n_objects: int, source: np.ndarray,
                             calibs: Sequence[ViewCalibration], id_maps: Sequence[np.ndarray]
                             ) -> dict[int, dict[str, Any]]:
    """Per object index: ``{samples, share, top_confusion}``.

    ``source``: index (into ``calibs`` / ``id_maps``) of the view that colours each sample, ``-1`` = none (inpainted,
    not counted). ``id_maps``: object-ID renders where the value ``index + 1`` marks object ``index`` and 0 the
    background (a source pixel on the background is not a mismatch). ``top_confusion`` = the most frequent other
    object ``{object: index, share: of the object's samples}``.
    """
    other = np.full(len(points), -1, dtype=np.int64)    # object index at the source pixel (-1 none)
    for j, (calib, ids) in enumerate(zip(calibs, id_maps)):
        sel = source == j
        if sel.any():
            other[sel] = id_at_pixels(ids, calib, points[sel]).astype(np.int64) - 1
    out: dict[int, dict[str, Any]] = {}
    for k in range(n_objects):
        sel = (owner == k) & (source >= 0)
        n = int(sel.sum())
        bad = sel & (other >= 0) & (other != k)
        info: dict[str, Any] = {"samples": n, "share": (int(bad.sum()) / n) if n else None, "top_confusion": None}
        if bad.any():
            counts = np.bincount(other[bad], minlength=n_objects)
            top = int(counts.argmax())
            info["top_confusion"] = {"object": top, "share": float(counts[top] / n)}
        out[k] = info
    return out


def warning_details(attribution: Mapping[str, Any] | None, parts: Mapping[str, Any] | None = None) -> list[str]:
    """Short soft-warning strings for a report: embedded / occluded parts, misattributed textures, parts sticking out of the
    reference silhouette (blocks as produced by ``blender.attribution.analyze`` and ``projection.compare_parts``)."""
    out: list[str] = []
    objects = (attribution or {}).get("objects") or {}
    for name in (attribution or {}).get("occluded_objects", []):
        o = objects[name]
        out.append(f"occluded_part {name} ({o['occlusion_share'] * 100:.0f}% of camera-facing samples hidden in its "
                   f"best view {o.get('best_view')})")
    for name in (attribution or {}).get("misattributed_objects", []):
        o = objects[name]
        with_ = o.get("top_confusion")
        out.append(f"misattribution {name} ({o['misattribution'] * 100:.0f}% of texels sourced from "
                   f"{with_['object'] if with_ else 'another object'})")
    for name in (parts or {}).get("parts_outside_reference", []):
        out.append(f"part_outside_reference {name} (only {parts['parts'][name]['min_precision'] * 100:.0f}% of it on the reference silhouette)")
    return out
