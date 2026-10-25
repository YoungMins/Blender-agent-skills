"""Whole-model review (bpy): per-component scorecards, ranked issues and a reference | model sheet per part.
Metrics and ranking: :mod:`bbr.core.review`; it runs after the other measurements of the report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import get_logger
from ..core import imageio, imageops, labels as labels_core
from ..core import facedetail as facedetail_core
from ..core import review as core
from ..core.conventions import AXES
from . import facedetail, partstex, render, scene
from .components import HULL_TYPE

log = get_logger(__name__)

REVIEW_SHEET = "review_sheet.png"
TILE_PX = 160          # height of one reference | model tile in the sheet
TILE_PAD = 0.15        # margin around a label, share of its size


def _labels(inp: Any) -> dict[str, dict[str, np.ndarray]]:
    spec = inp.scene
    if not spec.parts_labels or inp.hull is None:
        return {}
    path = Path(spec.parts_labels)
    if not path.is_absolute() and spec.source_path is not None:
        path = Path(spec.source_path).parent / path
    return labels_core.load_label_masks(path, inp.hull.masks)


def _texels(inp: Any, views: list[str]) -> dict[str, float]:
    """Texels per reference px of every textured object, each face measured against the colour view it faces most."""
    out: dict[str, float] = {}
    hi = inp.hull
    dirs = np.array([AXES[hi.calibs[v].camera_axis] for v in views])
    m_per_px = min(hi.calibs[v].m_per_px for v in views)
    for cid, st in (inp.hull_stats or {}).items():
        size = ((st or {}).get("texture") or {}).get("size") if isinstance(st, dict) else None
        obj = scene.find_by_agent_id(inp.scene.agent_id(cid))
        if not size or obj is None or not obj.data.uv_layers:
            continue
        uv_area, area, normals = facedetail.mesh_arrays(obj)
        out[cid] = round(facedetail_core.texels_per_px(uv_area, area, (normals @ dirs.T).max(axis=1), int(size), m_per_px), 2)
    return out


def _tile(ref: np.ndarray, model: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = imageops.mask_bbox(mask)
    pad = int(TILE_PAD * max(x1 - x0, y1 - y0)) + 2
    h, w = mask.shape
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
    pair = np.concatenate([ref[y0:y1, x0:x1, :3], np.ones((y1 - y0, 3, 3)), model[y0:y1, x0:x1, :3]], axis=1)
    h2 = TILE_PX
    w2 = max(1, round(pair.shape[1] * h2 / pair.shape[0]))
    return np.clip(imageops.resize_bilinear(pair, (w2, h2)), 0.0, 1.0)


def analyze(inp: Any, report: dict[str, Any], out: Path) -> dict[str, Any]:
    """``review`` block of the report: ``components`` (scorecards), ``issues`` (ranked), ``sheet`` (PNG path)."""
    hi = inp.hull
    if hi is None:
        return {"skipped": "no reference views"}
    hull = partstex.hull_component(inp.scene)
    views = [k for k in (partstex.effective_params(inp.scene, hull) if hull else {}).get("color_views")
             or [v.key for v in hi.view_spec.enabled_views("color")] if k in hi.calibs and k in hi.rgba]
    labels = _labels(inp)
    see_through = {o.get(scene.PROP_AGENT_ID) for o in scene.model_meshes() if o.get(scene.PROP_TRANSPARENT)}
    names = {c.id: (core.BODY if c.type == HULL_TYPE else partstex.label_name(c)) for c in inp.scene.components
             if not c.boolean and inp.scene.agent_id(c.id) not in see_through}   # glass: the reference shows what is behind
    colors: dict[str, dict[str, dict[str, float]]] = {}
    renders: dict[str, np.ndarray] = {}
    for v in views:
        model = render.render_color(v, hi.calibs[v], transparent=True)
        ref = np.asarray(hi.rgba[v], dtype=float)
        ref = ref / 255.0 if ref.max() > 1.0 else ref
        renders[v] = model
        colors[v] = core.label_color_errors(ref, model, hi.masks[v], model[..., 3] > 0.5, labels.get(v, {}))
    cards = core.build(names, report, colors, _texels(inp, views))
    issues = core.rank(cards)
    block: dict[str, Any] = {"components": cards, "issues": issues}
    rows, row_ids = [], []
    worst = {i["component"]: i["severity"] for i in reversed(issues)}
    for cid in sorted(names, key=lambda c: -worst.get(c, 0.0)):
        tiles = []
        for v in views:
            mask = labels.get(v, {}).get(names[cid]) if names[cid] != core.BODY else None
            if mask is None and names[cid] == core.BODY:
                mask = hi.masks[v]
            if mask is None or mask.sum() < core.MIN_LABEL_PX:
                continue
            ref = np.asarray(hi.rgba[v], dtype=float)
            ref = ref / 255.0 if ref.max() > 1.0 else ref
            tiles.append(_tile(ref, renders[v], mask))
        if tiles:
            row_ids.append(cid)
            rows.append(np.concatenate([np.concatenate([t, np.ones((TILE_PX, 8, 3))], axis=1) for t in tiles], axis=1))
    if rows:
        width = max(r.shape[1] for r in rows)
        sheet = np.concatenate([np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0)), constant_values=1.0) for r in rows])
        path = out / "previews" / REVIEW_SHEET
        imageio.save_rgba(path, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,))], axis=2).astype(np.float32))
        block["sheet"] = str(path)
        block["sheet_rows"] = row_ids     # top to bottom: worst component first; each tile reference | model
    return block
