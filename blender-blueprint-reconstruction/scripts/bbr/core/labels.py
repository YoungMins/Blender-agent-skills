"""Part labels (numpy only): the agent says which numbered regions of :mod:`proposals` belong to which part; this turns
that into per-part, per-view pixel masks, checks them across views and hands them to the hull and the texture projection.

``labels.json``::

    {"_regions": "regions",                       # optional: where <view>/id_map.npy live, relative to this file
     "front": {"horn_l": [32, 20], "hand_l": [9, "-12", {"polygon": [[40, 560], [90, 560], [80, 640]]}],
               "sword_grip": [{"split": 2, "line": [[330, 530], [380, 600]], "side": "left"}], "body": []},
     "side": {...}}

* a part is a component id; ``body`` is the silhouette hull. Every in-mask pixel nobody labelled belongs to ``body``
  (parts whose name starts with ``_`` hold pixels nobody samples: ``"_ignore": [14]``);
* list items: ``N`` / ``"+N"`` add region N, ``"-N"`` removes it again; ``{"polygon": [[x, y], ...]}`` adds that polygon
  (view pixels; ``"within": N`` or ``[N, ...]`` keeps only its pixels of those regions); ``{"rect": [x0, y0, x1, y1]}``
  is the same for an axis-aligned rectangle (``"within"`` works too); ``{"split": N, "line": [[x0, y0], [x1, y1]], "side": "left" | "right"}`` adds the part of region N on
  that side of the line (walking from the first to the second point, as seen in the image: ``right`` is down when going
  right). Whole regions are assigned first (a later part wins a shared region), then polygons / splits override them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import BBRError
from .calibration import ViewCalibration
from .conventions import view_frame
from .symmetry import character_side

BODY = "body"
REGIONS_KEY = "_regions"
DEFAULT_REGIONS_DIR = "regions"
IGNORE_PREFIX = "_"
SIDE_TOL_FRAC = 0.02         # a _l / _r part's centroid may sit this far (of the object height) on the wrong side of X = 0
DEFAULT_Z_TOL_FRAC = 0.04    # a part's z-range may differ this much (of the object height) between views
_REGION_TOKEN = re.compile(r"^([+-]?)(\d+)$")


def load_labels(path: Path | str) -> dict[str, Any]:
    """Read and shape-check a labels file; returns the raw dict."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BBRError(f"cannot read labels file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BBRError(f"{path}: labels must be an object {{view: {{part: [regions]}}}}")
    for view, parts in data.items():
        if view == REGIONS_KEY:
            if not isinstance(parts, str):
                raise BBRError(f"{path}: {REGIONS_KEY} must be a directory name")
            continue
        if not isinstance(parts, dict) or not all(isinstance(v, list) for v in parts.values()):
            raise BBRError(f"{path}: view {view!r} must map part ids to lists of regions / edits")
    return data


def regions_dir(labels_path: Path | str, data: Mapping[str, Any]) -> Path:
    """Directory holding ``<view>/id_map.npy`` (the labels file's ``_regions``, default ``regions`` beside it)."""
    return Path(labels_path).resolve().parent / str(data.get(REGIONS_KEY, DEFAULT_REGIONS_DIR))


def _parse_items(items: Sequence[Any], where: str) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    add: set[int] = set()
    remove: set[int] = set()
    edits: list[dict[str, Any]] = []
    for item in items:
        m = _REGION_TOKEN.match(str(item)) if isinstance(item, (int, str)) and not isinstance(item, bool) else None
        if m:
            (remove if m.group(1) == "-" else add).add(int(m.group(2)))
        elif isinstance(item, dict) and ("polygon" in item or "rect" in item or "split" in item):
            edits.append(item)
        else:
            raise BBRError(f"{where}: unsupported item {item!r} (use N, '+N', '-N', {{'polygon': ...}} or {{'split': ...}})")
    return add, remove, edits


def polygon_mask(shape: tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    """Even-odd fill of a polygon given in pixel coordinates (pixel centres at +0.5)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise BBRError(f"a polygon needs at least 3 [x, y] points, got {points!r}")
    h, w = shape
    x0, x1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, w)
    y0, y1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, h)
    out = np.zeros(shape, dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px, py = xx + 0.5, yy + 0.5
    inside = np.zeros(px.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(pts, np.roll(pts, -1, axis=0)):
        crosses = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (px < at)
    out[y0:y1, x0:x1] = inside
    return out


def _rect_points(rect: Sequence[float]) -> list[list[float]]:
    """Corner points of ``[x0, y0, x1, y1]`` (any corner order)."""
    if len(rect) != 4:
        raise BBRError(f"a rect needs [x0, y0, x1, y1], got {rect!r}")
    x0, x1, y0, y1 = min(rect[0], rect[2]), max(rect[0], rect[2]), min(rect[1], rect[3]), max(rect[1], rect[3])
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def side_of_line(shape: tuple[int, int], line: Sequence[Sequence[float]], side: str) -> np.ndarray:
    """Pixels on the ``left`` / ``right`` of the line from ``line[0]`` to ``line[1]`` (image view, y down)."""
    if side not in ("left", "right"):
        raise BBRError(f"split side must be 'left' or 'right', got {side!r}")
    (ax, ay), (bx, by) = line
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cross = (bx - ax) * (yy + 0.5 - ay) - (by - ay) * (xx + 0.5 - ax)
    return cross > 0 if side == "right" else cross < 0


def _as_list(value: Any) -> list[int]:
    return [int(v) for v in (value if isinstance(value, (list, tuple)) else [value])]


def resolve_view(entries: Mapping[str, Sequence[Any]], id_map: np.ndarray, mask: np.ndarray, view: str = "") -> dict[str, np.ndarray]:
    """Per-part bool masks of one view (see the module doc); ``body`` always exists and holds the unlabelled pixels."""
    mask = np.asarray(mask, dtype=bool)
    top = int(id_map.max())
    names = [BODY] + [p for p in entries if p != BODY]   # body first: any explicit part wins a shared region
    label = np.zeros(id_map.shape, dtype=np.int32)
    parsed: dict[str, tuple[set[int], list[dict[str, Any]]]] = {}
    for part in names:
        add, remove, edits = _parse_items(entries.get(part, []), f"view {view!r} part {part!r}")
        regions = sorted(add - remove)
        bad = [r for r in regions if not 1 <= r <= top]
        if bad:
            raise BBRError(f"view {view!r} part {part!r}: regions {bad} do not exist (the view has 1..{top})")
        parsed[part] = (set(regions), edits)
    for i, part in enumerate(names, start=1):
        regions = parsed[part][0]
        if regions:
            label[np.isin(id_map, sorted(regions))] = i
    for i, part in enumerate(names, start=1):
        for edit in parsed[part][1]:
            if "polygon" in edit or "rect" in edit:
                inside = mask if "within" not in edit else np.isin(id_map, _as_list(edit["within"]))
                label[polygon_mask(id_map.shape, _rect_points(edit["rect"]) if "rect" in edit else edit["polygon"]) & inside] = i
            else:
                n = int(edit["split"])
                if not 1 <= n <= top or "line" not in edit:
                    raise BBRError(f"view {view!r} part {part!r}: bad split {edit!r} (needs an existing region and a 'line')")
                label[(id_map == n) & side_of_line(id_map.shape, edit["line"], str(edit.get("side", "left")))] = i
    label[(label == 0) & mask] = names.index(BODY) + 1
    label[~mask] = 0
    return {part: label == i for i, part in enumerate(names, start=1)}


def load_label_masks(labels_path: Path | str, masks: Mapping[str, np.ndarray], regions: Optional[Path | str] = None
                     ) -> dict[str, dict[str, np.ndarray]]:
    """``{view: {part: bool mask}}`` for the labelled views that exist in ``masks`` (the loaded reference masks).

    Needs ``<regions dir>/<view>/id_map.npy`` of each labelled view, computed from the same images (same size);
    ``regions`` overrides the directory named by the labels file."""
    data = load_labels(labels_path)
    base = Path(regions) if regions else regions_dir(labels_path, data)
    out: dict[str, dict[str, np.ndarray]] = {}
    for view, entries in data.items():
        if view == REGIONS_KEY or view not in masks:
            continue
        file = base / view / "id_map.npy"
        if not file.is_file():
            raise BBRError(f"labels for view {view!r} need {file} (run prepare_references.py propose first)")
        id_map = np.load(file)
        if id_map.shape != masks[view].shape:
            raise BBRError(f"{file}: shape {id_map.shape} does not match the {view} reference {masks[view].shape}; "
                           f"re-run propose on the current images")
        out[view] = resolve_view(entries, id_map, masks[view], view)
    return out


def label_names(label_masks: Mapping[str, Mapping[str, np.ndarray]]) -> set[str]:
    """Part names that own at least one pixel in some view (``_`` parts and ``body`` included)."""
    return {p for parts in label_masks.values() for p, m in parts.items() if m.any()}


def allowed_masks(label_masks: Mapping[str, Mapping[str, np.ndarray]], name: str, views: Sequence[str]
                  ) -> list[Optional[np.ndarray]]:
    """Per view of ``views``: pixels carrying label ``name`` (an empty mask when the view is labelled but ``name`` is
    absent there), or ``None`` for a view without labels (not gated)."""
    out: list[Optional[np.ndarray]] = []
    for view in views:
        parts = label_masks.get(view)
        if parts is None:
            out.append(None)
        else:
            out.append(parts.get(name, np.zeros_like(next(iter(parts.values())))))
    return out


def part_masks(label_masks: Mapping[str, Mapping[str, np.ndarray]], name: str) -> dict[str, np.ndarray]:
    """``{view: mask}`` of label ``name`` over the labelled views where it owns at least one pixel (a layer's carve masks)."""
    return {view: parts[name] for view, parts in label_masks.items() if name in parts and parts[name].any()}


def union(label_masks: Mapping[str, np.ndarray], names: Sequence[str]) -> Optional[np.ndarray]:
    """Union of the masks of ``names`` in one view, None when none of them is labelled there."""
    found = [label_masks[n] for n in names if n in label_masks and label_masks[n].any()]
    return np.logical_or.reduce(found) if found else None


# --------------------------------------------------------------------------- left / right naming check
def side_warning(part: str, view: str, mask: np.ndarray, calib: ViewCalibration, height_m: float,
                 tol_frac: float = SIDE_TOL_FRAC) -> Optional[str]:
    """Warning when a ``_l`` / ``_r`` part sits on the wrong side of X = 0 in a front / back view, else None.

    ``_l`` / ``_r`` are the CHARACTER's left / right; the character faces -Y, so its left is +X: image-RIGHT in the front
    view, image-LEFT in the back view. Only the sign of the label centroid's world x counts (dead zone ``tol_frac`` of the
    height); side views are not checked (a part may hide behind the body there)."""
    side = character_side(part)
    if side is None or part.startswith(IGNORE_PREFIX) or calib.camera_axis not in ("-Y", "+Y") or not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    x = float(calib.px_to_plane(np.array([[cols.mean() + 0.5, rows.mean() + 0.5]]))[0, 0])
    if abs(x) <= tol_frac * height_m or (x > 0) == (side == "l"):
        return None
    want = "image-right" if (side == "l") == (calib.camera_axis == "-Y") else "image-left"
    own = ("left", "+X") if side == "l" else ("right", "-X")
    return (f"{view}: '{part}' is on the {'+X' if x > 0 else '-X'} side (x = {x * 1000:+.0f} mm) but _l/_r name the CHARACTER's own "
            f"side (its {own[0]} = {own[1]} = {want} in the {view} view): swap the _l/_r names of this pair or fix the label")


def side_warnings(label_masks: Mapping[str, Mapping[str, np.ndarray]], calibs: Mapping[str, ViewCalibration],
                  height_m: float, tol_frac: float = SIDE_TOL_FRAC) -> list[str]:
    """:func:`side_warning` of every ``_l`` / ``_r`` part in every labelled front / back view."""
    out = []
    for view, parts in label_masks.items():
        calib = calibs.get(view)
        if calib is not None:
            out += [w for part, m in parts.items() if (w := side_warning(part, view, m, calib, height_m, tol_frac))]
    return out


# --------------------------------------------------------------------------- cross-view check
def part_z_ranges(label_masks: Mapping[str, Mapping[str, np.ndarray]], calibs: Mapping[str, ViewCalibration]
                  ) -> dict[str, dict[str, list[float]]]:
    """``{part: {view: [z_min, z_max]}}`` in meters from the pixel rows of each label mask (views whose image up axis
    is world Z only: front / back / side / left / right)."""
    out: dict[str, dict[str, list[float]]] = {}
    for view, parts in label_masks.items():
        calib = calibs.get(view)
        if calib is None or abs(float(view_frame(calib.camera_axis).up[2])) < 0.99:
            continue
        for part, mask in parts.items():
            rows = np.flatnonzero(mask.any(axis=1))
            if len(rows) == 0 or part.startswith(IGNORE_PREFIX):
                continue
            z = calib.px_to_plane(np.array([[0.0, rows[0]], [0.0, rows[-1] + 1.0]]))[:, 2]
            out.setdefault(part, {})[view] = [round(float(z.min()), 4), round(float(z.max()), 4)]
    return out


def check_z_ranges(ranges: Mapping[str, Mapping[str, Sequence[float]]], height_m: float,
                   tol_frac: float = DEFAULT_Z_TOL_FRAC) -> dict[str, Any]:
    """Compare each part's z-range over the views that see it. ``spread_m`` = largest disagreement of its lower or its
    upper end between any two views; flagged when above ``tol_frac * height_m``. A part seen in one view cannot be
    checked (``views`` = 1). Returns ``{parts: {part: {...}}, flagged: [part...], tolerance_m}``."""
    tol = tol_frac * height_m
    parts: dict[str, Any] = {}
    for part, per_view in sorted(ranges.items()):
        lo = [v[0] for v in per_view.values()]
        hi = [v[1] for v in per_view.values()]
        spread = max(max(lo) - min(lo), max(hi) - min(hi)) if len(per_view) > 1 else 0.0
        parts[part] = {"z_range": {v: list(r) for v, r in per_view.items()}, "views": len(per_view),
                       "spread_m": round(float(spread), 4), "spread_frac": round(float(spread / height_m), 4) if height_m else None,
                       "flagged": bool(len(per_view) > 1 and spread > tol)}
    return {"parts": parts, "flagged": [p for p, r in parts.items() if r["flagged"]], "tolerance_m": round(tol, 4)}


def format_z_check(result: Mapping[str, Any]) -> str:
    """Terminal table of :func:`check_z_ranges`."""
    lines = [f"z-range per view (m), tolerance {result['tolerance_m']} m:"]
    for part, r in result["parts"].items():
        views = "  ".join(f"{v} [{a:.3f}, {b:.3f}]" for v, (a, b) in r["z_range"].items())
        mark = "  <-- DISAGREE (mislabel or generator inconsistency)" if r["flagged"] else ""
        lines.append(f"  {part:<14} spread {r['spread_m']:.3f} m  {views}{mark}")
    return "\n".join(lines)
