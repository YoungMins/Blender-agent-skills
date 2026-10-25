"""Whole-model review (numpy only): per-component scorecard and a ranked list of issues with the fix to try.

The pipeline's pass / fail looks at the whole silhouette and colour; a run can pass with one part clearly wrong. After a
run, :func:`build` gathers per component what the report already measured (label IoU, misattribution, occlusion,
inpainted texture, face detail, the silhouette mismatch region) plus its colour error over its own label per view, and
:func:`rank` turns every metric past its threshold into an issue ``{component, check, value, threshold, severity,
detail, action}``, worst first (severity = how far past the threshold). The agent reads the issues and
``previews/review_sheet.png`` (reference | model per part and view) with the turntable, writes a plan and corrects.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

BODY = "body"
MIN_LABEL_PX = 50                 # a label smaller than this in a view is not scored there

# check -> (threshold, higher is worse, action)
CHECKS: dict[str, tuple[float, bool, str]] = {
    "color_error": (0.08, True, "its colour is off in that view: check its label (foreign pixels), its texture "
                                "(inpainted / misattributed) and its shape (a wrong surface takes the colour)"),
    "label_iou": (0.6, False, "its shape or pose misses its label: fix anchors, the part type (kind) or fit_locked"),
    "misattribution": (0.15, True, "it takes colour that belongs to another object: fix the label border, or the part "
                                   "sits in front of / inside another one"),
    "occlusion": (0.3, True, "most of it is hidden: it is embedded in the body or behind another part; check its "
                             "placement and exclude_components"),
    "inpainted_share": (0.45, True, "much of its texture is guessed: no view sees it well; add mirror_color_views / a "
                                    "colour view, or accept it if it is truly hidden"),
    "texels_per_px": (1.0, False, "its texture is coarser than the reference: raise params.texture_size"),
    "silhouette_excess": (0.02, True, "the silhouette misses the reference here (largest mismatch region): fix the "
                                      "part's size / position, its label, or the hull's carve masks"),
    "face_ridge": (0.002, True, "profile features stretch across the face as ridges: set params.relief on the face layer"),
    "slab": (0.45, False, "a limb of the hull is a flat slab (thin across, deep along the other axis): the views disagree on "
                          "it (an AI sheet draws the arm out in front, hanging in the side) and the hull took the torso's "
                          "depth; make it a part (capsule / sphere) from the view that shows it clearly, exclude it from "
                          "the hull. Leave it if it is truly flat (cape, wing)"),
}
SLAB_MIN_HEIGHT = 0.05            # share of the model height a slab must span
SLAB_MIN_AREA = 0.02              # cross-section side, share of the model height, below which a blob is ignored


def label_color_errors(ref_rgb: np.ndarray, model_rgb: np.ndarray, ref_mask: np.ndarray, model_mask: np.ndarray,
                       labels: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Mean absolute RGB difference (0..1) per label of one view, over the label's pixels the model also covers;
    ``body`` = reference foreground no label claims. ``{label: {"error", "px"}}``, labels under MIN_LABEL_PX skipped."""
    ref_mask, model_mask = np.asarray(ref_mask, bool), np.asarray(model_mask, bool)
    named = {k: np.asarray(m, bool) for k, m in labels.items() if not k.startswith("_") and k != BODY}
    claimed = np.logical_or.reduce(list(named.values())) if named else np.zeros_like(ref_mask)
    regions = {**named, BODY: ref_mask & ~claimed}
    diff = np.abs(np.asarray(ref_rgb, float)[..., :3] - np.asarray(model_rgb, float)[..., :3]).mean(axis=2)
    out: dict[str, dict[str, float]] = {}
    for name, region in regions.items():
        use = region & model_mask
        if use.sum() >= MIN_LABEL_PX:
            out[name] = {"error": round(float(diff[use].mean()), 4), "px": int(use.sum())}
    return out


def slab_regions(inside: np.ndarray, origin: np.ndarray, cell: float, height: float,
                 max_ratio: float = 0.45) -> list[dict[str, Any]]:
    """Flat slabs among the hull's limbs: per horizontal slice (z) of the carved volume ``inside`` ``[nx, ny, nz]``, every
    cross-section blob except the largest (the torso) gives its x and y extent; blobs of consecutive slices whose centres
    stay within their own size form a track; a track spanning SLAB_MIN_HEIGHT of the model whose median thin / thick
    extent ratio is below ``max_ratio`` is a slab. Returns ``[{x, y, z: [lo, hi], width_x, depth_y, ratio}]`` (m)."""
    from .imageops import label_components
    inside = np.asarray(inside, dtype=bool)
    nx, ny, nz = inside.shape
    min_cells = max(4, int((SLAB_MIN_AREA * height / cell) ** 2))
    tracks: list[list[tuple[int, float, float, float, float]]] = []
    open_tracks: list[list[tuple[int, float, float, float, float]]] = []
    for k in range(nz):
        comps, n = label_components(inside[:, :, k], full=True)
        if n < 2:
            tracks += open_tracks
            open_tracks = []
            continue
        areas = np.bincount(comps.ravel(), minlength=n + 1)
        areas[0] = 0
        blobs = []
        for cid in np.nonzero(areas >= min_cells)[0]:
            if cid == int(np.argmax(areas)):
                continue
            xs, ys = np.nonzero(comps == cid)
            blobs.append((k, xs.mean(), ys.mean(), (np.ptp(xs) + 1) * cell, (np.ptp(ys) + 1) * cell))
        nxt = []
        for b in blobs:
            for t in open_tracks:
                last = t[-1]
                if last[0] == k - 1 and abs(last[1] - b[1]) * cell < max(last[3], b[3]) and abs(last[2] - b[2]) * cell < max(last[4], b[4]):
                    t.append(b)
                    nxt.append(t)
                    open_tracks.remove(t)
                    break
            else:
                nxt.append([b])
        tracks += open_tracks
        open_tracks = nxt
    tracks += open_tracks
    out = []
    for t in tracks:
        if len(t) * cell < SLAB_MIN_HEIGHT * height:
            continue
        w = np.median([b[3] for b in t])
        d = np.median([b[4] for b in t])
        ratio = float(min(w, d) / max(w, d))
        if ratio < max_ratio:
            out.append({"x": round(float(origin[0] + (np.median([b[1] for b in t]) + 0.5) * cell), 3),
                        "y": round(float(origin[1] + (np.median([b[2] for b in t]) + 0.5) * cell), 3),
                        "z": [round(float(origin[2] + (t[0][0] + 0.5) * cell), 3), round(float(origin[2] + (t[-1][0] + 0.5) * cell), 3)],
                        "width_x": round(float(w), 4), "depth_y": round(float(d), 4), "ratio": round(ratio, 3)})
    return out


def build(components: Mapping[str, str], report: Mapping[str, Any],
          color_errors: Mapping[str, Mapping[str, Mapping[str, float]]],
          texels: Optional[Mapping[str, float]] = None) -> dict[str, dict[str, Any]]:
    """Scorecard per component id. ``components``: id -> label name; ``color_errors``: view -> label -> {error, px}
    (:func:`label_color_errors`); ``texels``: id -> texels per reference px."""
    parts = ((report.get("part_iou") or {}).get("parts")) or {}
    objects = ((report.get("attribution") or {}).get("objects")) or {}
    hull = report.get("hull") or {}
    faces = ((report.get("face_detail") or {}).get("layers")) or {}
    per_view = ((report.get("reference_match") or {}).get("per_view")) or {}
    cards: dict[str, dict[str, Any]] = {}
    for cid, label in components.items():
        c: dict[str, Any] = {}
        errs = {v: e[label]["error"] for v, e in color_errors.items() if label in e}
        if errs:
            c["color_error"] = errs
        p = parts.get(cid) or {}
        if "mean_label_iou" in p:
            c["label_iou"] = p["mean_label_iou"]
        o = objects.get(cid) or {}
        for key, src in (("misattribution", "misattribution"), ("occlusion", "occlusion_share")):
            if o.get(src) is not None:
                c[key] = round(float(o[src]), 4)
        tex = (hull.get(cid) or {}).get("texture") if isinstance(hull.get(cid), Mapping) else None
        if tex and "inpainted_share" in tex:
            c["inpainted_share"] = round(float(tex["inpainted_share"]), 4)
        if texels and cid in texels:
            c["texels_per_px"] = texels[cid]
        f = faces.get(cid) or {}
        if f and not f.get("relief_applied") and f.get("profile_relief_m") is not None:
            c["face_ridge"] = f["profile_relief_m"]
        slabs = (hull.get(cid) or {}).get("slabs") if isinstance(hull.get(cid), Mapping) else None
        if slabs:
            c["slab"] = {f"x {s['x']:+.2f} z {s['z'][0]:.2f}-{s['z'][1]:.2f}": s["ratio"] for s in slabs}
        for view, info in per_view.items():
            if cid in (info.get("components_in_region") or []) and info.get("excess_error") is not None:
                c.setdefault("silhouette_excess", {})[view] = round(float(info["excess_error"]), 4)
        cards[cid] = c
    return cards


def rank(cards: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Issues of every scorecard metric past its threshold (per view for per-view metrics), worst first."""
    issues = []
    for cid, card in cards.items():
        for check, value in card.items():
            if check not in CHECKS:
                continue
            threshold, higher_worse, action = CHECKS[check]
            items = value.items() if isinstance(value, Mapping) else [(None, value)]
            for view, v in items:
                v = float(v)
                bad = v > threshold if higher_worse else v < threshold
                if not bad:
                    continue
                severity = v / threshold if higher_worse else threshold / max(v, 1e-6)
                issues.append({"component": cid, "check": check, "view": view, "value": round(v, 4),
                               "threshold": threshold, "severity": round(severity, 2), "action": action})
    issues.sort(key=lambda i: -i["severity"])
    return issues


def format_issue(i: Mapping[str, Any]) -> str:
    """One line: ``<component> <check>[ (view)] <value> (limit <threshold>): <action>``."""
    where = f" ({i['view']})" if i.get("view") else ""
    return f"{i['component']} {i['check']}{where} {i['value']:g} (limit {i['threshold']:g}): {i['action']}"
