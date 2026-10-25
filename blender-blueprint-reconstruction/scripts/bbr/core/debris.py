"""Hull debris (numpy only): detached fragments of the carved hull that belong to no part.

The hull is the intersection of the carve silhouettes, so a detached piece of silhouette in ONE view (the tip of a tail
beside a foot in the front view) meets whatever the other view has at that height (a foot, a leg) and leaves an island
of the carved volume: a speck floating by the feet. ``island_min_frac`` keeps detached islands because real ones exist
(a staff, a hat over a head, the legs of a creature); this module tells the two apart, per island of the carved field:

* (a) **one-view support**: in one carve view the island's silhouette is a blob of its own (not connected to the blob(s)
  the main hull projects onto), in another carve view it is part of the main silhouette, and the labelled part under
  it differs between the two views (a tail label in one, a foot or nothing in the other): two different things met. The
  same labelled part under both (a staff in both views) is a real, detached part: kept.
* (b) **a small unlabelled speck**: smaller than :data:`VOLUME_FRAC` of the hull volume and not overlapped by any
  labelled part in any carve view. Needs part labels: without them the scene says nothing about what is a part.

A dropped island is reported (``reason``, size, position) so that the agent can see what was removed (hull param
``drop_debris: false`` turns the check off); a kept detached island is reported as ``floating_part`` by the part check
(:func:`bbr.core.gapcheck.floating_parts`: the pieces of the hull mesh are parts of their own).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .calibration import ViewCalibration
from .imageops import label_components

VOLUME_FRAC = 0.003         # rule (b): an island below this share of the hull's carved volume is a speck (the smallest real parts, the feet of a hatted figure, are 0.6 %)
LABEL_SHARE = 0.5           # a label is "under" an island when it covers at least this share of its projection
OVERLAP_SHARE = 0.2         # rule (b): ... and "overlapped" by a labelled part from this share
MIN_BLOB_PX = 10            # a mask blob counts as part of the main hull's silhouette from this many projected main-hull pixels
MAX_SAMPLE = 200_000        # main-hull voxels projected per view (a regular subsample)


def _pixels(calib: ViewCalibration, pts: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    uv = calib.world_to_px(pts)
    x = np.clip(np.floor(uv[:, 0]).astype(int), 0, shape[1] - 1)
    y = np.clip(np.floor(uv[:, 1]).astype(int), 0, shape[0] - 1)
    return x, y


def _dominant_label(names: Sequence[str], masks: Sequence[np.ndarray], x: np.ndarray, y: np.ndarray, share: float) -> Optional[str]:
    """The labelled part (not ``body`` / ``_ignored``) under at least ``share`` of the points, else None."""
    best, best_share = None, 0.0
    for name, m in zip(names, masks):
        s = float(np.mean(m[y, x]))
        if s > best_share:
            best, best_share = name, s
    return best if best_share >= share else None


def find_debris(field: np.ndarray, origin: np.ndarray, cell: float, masks: Mapping[str, np.ndarray],
                calibs: Mapping[str, ViewCalibration], views: Sequence[str],
                labels: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None, volume_frac: float = VOLUME_FRAC
                ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Islands of ``field < 0`` (26-connected) other than the largest, classified by rules (a) / (b) above.

    Returns ``(drop mask (bool, field's shape; True = voxel of a debris island), report)`` where every report entry
    describes one island: ``{island, voxels, volume_pct, box_m, dropped, rule, reason, views}`` (``dropped`` False =
    kept as a detached part). ``views`` = the carve views used; ``labels`` = ``{view: {part: mask}}`` (None = unlabelled).
    """
    inside = np.asarray(field) < 0
    drop = np.zeros(inside.shape, dtype=bool)
    lab, n = label_components(inside, full=True)
    if n <= 1:
        return drop, []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    main = int(sizes.argmax())
    total = float(sizes.sum())
    use = [v for v in views if v in masks and v in calibs]
    blobs = {v: label_components(np.asarray(masks[v], bool), full=True)[0] for v in use}
    parts = {v: [(k, np.asarray(m, bool)) for k, m in (labels or {}).get(v, {}).items() if k != "body" and not k.startswith("_")] for v in use}
    main_idx = np.argwhere(lab == main)
    step = max(1, len(main_idx) // MAX_SAMPLE)
    main_pts = origin + (main_idx[::step] + 0.5) * cell
    main_blobs: dict[str, set[int]] = {}
    for v in use:
        x, y = _pixels(calibs[v], main_pts, masks[v].shape)
        counts = np.bincount(blobs[v][y, x], minlength=int(blobs[v].max()) + 1)
        main_blobs[v] = {int(b) for b in np.flatnonzero(counts >= MIN_BLOB_PX) if b > 0}
    have_labels = bool(labels) and any(parts[v] for v in use)
    report: list[dict[str, Any]] = []
    for isl in np.argsort(-sizes):
        if isl == 0 or isl == main or sizes[isl] == 0:
            continue
        idx = np.argwhere(lab == isl)
        pts = origin + (idx + 0.5) * cell
        info: dict[str, dict[str, Any]] = {}
        for v in use:
            x, y = _pixels(calibs[v], pts, masks[v].shape)
            ids = blobs[v][y, x]
            ids = ids[ids > 0]
            blob = int(np.bincount(ids).argmax()) if len(ids) else 0
            overlap = max((float(np.mean(m[y, x])) for _, m in parts[v]), default=0.0)
            info[v] = {"separate": blob not in main_blobs[v],
                       "label": _dominant_label([k for k, _ in parts[v]], [m for _, m in parts[v]], x, y, LABEL_SHARE),
                       "overlap": round(overlap, 3)}
        frac = float(sizes[isl]) / total
        entry: dict[str, Any] = {"island": int(isl), "voxels": int(sizes[isl]), "volume_pct": round(100.0 * frac, 4),
                                 "box_m": [[round(float(c), 4) for c in (origin + idx.min(axis=0) * cell)],
                                           [round(float(c), 4) for c in (origin + (idx.max(axis=0) + 1) * cell)]],
                                 "views": {v: {k: i[k] for k in ("separate", "label")} for v, i in info.items()},
                                 "dropped": False, "rule": None, "reason": "a detached part of the silhouette"}
        one_view = [(a, b) for a in use for b in use if a != b and info[a]["separate"] and not info[b]["separate"]
                    and info[a]["label"] != info[b]["label"] and (info[a]["label"] or info[b]["label"])]
        if have_labels and one_view:
            a, b = one_view[0]
            entry.update(dropped=True, rule="a", reason=(
                f"supported by a blob of its own in view {a} ({info[a]['label'] or 'unlabelled'}) but by the main silhouette in "
                f"view {b} ({info[b]['label'] or 'unlabelled'}): two different things meet there"))
        elif have_labels and frac < volume_frac and max(i["overlap"] for i in info.values()) < OVERLAP_SHARE:
            entry.update(dropped=True, rule="b", reason=f"{entry['volume_pct']:.2f}% of the hull volume (< {100 * volume_frac:.1f}%) "
                                                        f"and under no labelled part")
        if entry["dropped"]:
            drop[tuple(idx.T)] = True
        report.append(entry)
    return drop, report


def drop_debris(field: np.ndarray, drop: np.ndarray) -> np.ndarray:
    """``field`` with the debris voxels turned to outside (their sign flipped; the distances stay)."""
    out = np.array(field, copy=True)
    out[drop] = np.abs(out[drop]) + 1e-6
    return out


def format_debris_report(entries: Sequence[Mapping[str, Any]], hull_id: str = "body", show: int = 3) -> list[str]:
    """Warning lines for a :func:`find_debris` report: one line for everything dropped (count, share of the volume, the
    ``show`` largest with position and rule); kept islands are the part check's business (``floating_part``) and give none."""
    gone = sorted((e for e in entries if e["dropped"]), key=lambda e: -e["voxels"])
    if not gone:
        return []
    def where(e: Mapping[str, Any]) -> str:
        lo, hi = e["box_m"]
        return f"x {lo[0]:.3f}..{hi[0]:.3f} y {lo[1]:.3f}..{hi[1]:.3f} z {lo[2]:.3f}..{hi[2]:.3f} (rule {e['rule']}: {e['reason']})"
    total = sum(e["volume_pct"] for e in gone)
    head = (f"{hull_id} dropped {len(gone)} floating hull fragment{'s' if len(gone) > 1 else ''} ({total:.2f}% of the hull volume): "
            + "; ".join(where(e) for e in gone[:show]) + (f"; and {len(gone) - show} smaller" if len(gone) > show else ""))
    return [head + " [label the piece as a part if it should stay, or set the hull param drop_debris: false]"]
