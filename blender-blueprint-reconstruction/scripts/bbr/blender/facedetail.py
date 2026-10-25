"""Face detail check of a character (bpy): texel density, profile ridge, face colour error and a close-up sheet per face
layer. Metrics and warnings: :mod:`bbr.core.facedetail`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import get_logger
from ..core import facedetail as core
from ..core import imageio, imageops, layers as layers_core, labels as labels_core
from . import partstex, render, scene
from .components import LAYER_TYPE

log = get_logger(__name__)

CLOSEUP_PREVIEW = "face_closeup.png"
CLOSEUP_PAD = 0.15          # margin around the face label, share of its size
CLOSEUP_SCALE = 3           # magnification of the sheet (nearest neighbour)


def _mesh_arrays(obj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per polygon of ``obj``: UV area (0..1 square), world area, cos of the angle to the front camera (-Y)."""
    me = obj.data
    mw = np.array(obj.matrix_world)
    n = len(me.polygons)
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    starts = np.zeros(n, dtype=np.int64)
    totals = np.zeros(n, dtype=np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    uv_area = np.zeros(n)
    for i in range(n):         # shoelace per polygon (small face meshes)
        p = uv[starts[i]:starts[i] + totals[i]]
        uv_area[i] = 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))
    area = np.zeros(n)
    normal = np.zeros(n * 3)
    me.polygons.foreach_get("area", area)
    me.polygons.foreach_get("normal", normal)
    scale = float(abs(np.linalg.det(mw[:3, :3]))) ** (1.0 / 3.0)
    world_n = normal.reshape(-1, 3) @ np.linalg.inv(mw[:3, :3])
    world_n /= np.maximum(np.linalg.norm(world_n, axis=1, keepdims=True), 1e-12)
    return uv_area, area * scale ** 2, -world_n[:, 1]


def _closeup(ref: np.ndarray, model: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    pair = np.concatenate([ref[y0:y1, x0:x1, :3], np.ones((y1 - y0, 4, 3)), model[y0:y1, x0:x1, :3]], axis=1)
    return np.repeat(np.repeat(pair, CLOSEUP_SCALE, axis=0), CLOSEUP_SCALE, axis=1)


def analyze(inp: Any, out: Path) -> dict[str, Any]:
    """``face_detail`` block of the report (see :mod:`bbr.core.facedetail`); ``{"skipped": reason}`` without a face layer."""
    spec, hi = inp.scene, inp.hull
    faces = [c for c in spec.components if c.type == LAYER_TYPE and not c.params.get("shell")]
    if not faces:
        return {"skipped": "no face layer (layer_hull)"}
    if hi is None or not spec.parts_labels:
        return {"skipped": "no reference views or part labels"}
    path = Path(spec.parts_labels)
    if not path.is_absolute() and spec.source_path is not None:
        path = Path(spec.source_path).parent / path
    labels = labels_core.load_label_masks(path, hi.masks)
    layers: dict[str, Any] = {}
    sheets = []
    for comp in faces:
        f: dict[str, Any] = {}
        layers[comp.id] = f
        try:
            label = str(comp.params.get("label", comp.id))
            views = [str(v) for v in comp.params.get("views") or ("front", "side")]
            masks = labels_core.part_masks(labels, label)
            front, side = views
            f["shells"] = [c.id for c in spec.components if c.type == LAYER_TYPE and c.params.get("shell")
                           and comp.id in (c.params.get("cover") or [])]
            stats = (inp.hull_stats or {}).get(comp.id) or {}
            f["relief_applied"] = "relief" in stats
            if side in masks:
                f["profile_relief_m"] = round(layers_core.profile_relief(masks[side], hi.calibs[side]).amplitude, 4)
            obj = scene.find_by_agent_id(spec.agent_id(comp.id))
            tex = (stats.get("texture") or {})
            if obj is not None and obj.data.uv_layers and tex.get("size"):
                uv_area, area, facing = _mesh_arrays(obj)
                f["texels_per_px"] = round(core.texels_per_px(uv_area, area, facing, int(tex["size"]), hi.calibs[front].m_per_px), 2)
            if front in masks and front in hi.rgba:
                model = render.render_color(front, hi.calibs[front], bg=(0.0, 0.0, 0.0))
                ref = np.asarray(hi.rgba[front], dtype=float)
                ref = ref / 255.0 if ref.max() > 1.0 else ref
                f["color_error"] = round(core.color_error(ref, model, masks[front]), 4)
                x0, y0, x1, y1 = imageops.mask_bbox(masks[front])
                pad = int(CLOSEUP_PAD * max(x1 - x0, y1 - y0))
                h, w = masks[front].shape
                sheets.append(_closeup(ref, model, (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h))))
        except Exception as exc:  # noqa: BLE001 - one face must not break the report
            log.exception("face detail of %s failed", comp.id)
            f["error"] = f"{type(exc).__name__}: {exc}"
    block: dict[str, Any] = {"layers": layers}
    if sheets:
        width = max(s.shape[1] for s in sheets)
        sheet = np.concatenate([np.pad(s, ((0, 4), (0, width - s.shape[1]), (0, 0)), constant_values=1.0) for s in sheets])
        path_out = out / "previews" / CLOSEUP_PREVIEW
        imageio.save_rgba(path_out, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,))], axis=2).astype(np.float32))
        block["closeup"] = str(path_out)
    return block
