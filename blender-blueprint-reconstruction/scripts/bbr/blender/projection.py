"""Reference-vs-render comparison per view and the comparison images (no PIL: written through imageio)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .. import get_logger
from ..core import imageio, imageops, pose
from ..core.calibration import ViewCalibration
from ..core.colormetric import color_error
from ..core.config import PipelineConfig
from ..core.metrics import ViewComparison, compare_view, iou
from . import render

log = get_logger(__name__)

PANEL_GAP_PX = 4
GAP_COLOR = 0.15
BG_COLOR = 0.08
REF_BG_COLOR = 0.9
MAX_ROW_WIDTH_PX = 1800
SHEET_WIDTH_PX = 1200
COLOR_BOTH = (0.1, 0.8, 0.2)     # green: reference and render agree
COLOR_REF_ONLY = (0.9, 0.15, 0.15)  # red: reference only (missing in the model)
COLOR_OUT_ONLY = (0.2, 0.35, 0.95)  # blue: render only (excess)


def validate_view_keys(view_spec, calibs: Mapping[str, ViewCalibration], ref_masks: Mapping[str, np.ndarray]) -> list[str]:
    """Views that take part in validation: role ``validate``, calibrated, with a reference mask."""
    return [v.key for v in view_spec.enabled_views("validate")
            if v.key in calibs and v.key in ref_masks and v.is_orthographic_usable]


def compare_all(view_spec, calibs: Mapping[str, ViewCalibration], ref_masks: Mapping[str, np.ndarray],
                config: Optional[PipelineConfig] = None) -> dict:
    """Render the model mask of every validation view and compare with the reference mask.

    Returns ``{"results": {view: ViewComparison}, "out_masks": {view: bool array}}``.
    """
    cfg = config or PipelineConfig()
    results: dict[str, ViewComparison] = {}
    out_masks: dict[str, np.ndarray] = {}
    for key in validate_view_keys(view_spec, calibs, ref_masks):
        out_masks[key] = render.render_mask(key, calibs[key], render_scale=cfg.render_scale)
        results[key] = compare_view(ref_masks[key], out_masks[key], calibs[key], cfg.edge_distance_cap_px)
        log.info("view %s: IoU %.4f (error %.2f%%)", key, results[key].iou, results[key].error * 100)
    return {"results": results, "out_masks": out_masks}


def color_view_keys(view_spec, calibs: Mapping[str, ViewCalibration], ref_masks: Mapping[str, np.ndarray],
                    ref_rgba: Mapping[str, np.ndarray]) -> list[str]:
    """Views with role ``color`` (or ``validate``) that are calibrated and have a reference image + mask."""
    keys: list[str] = []
    for v in view_spec.enabled_views():
        if ("color" in v.roles or "validate" in v.roles) and v.is_orthographic_usable and v.key in calibs                 and v.key in ref_masks and v.key in ref_rgba and v.key not in keys:
            keys.append(v.key)
    return keys


def compare_colors(view_spec, calibs: Mapping[str, ViewCalibration], ref_masks: Mapping[str, np.ndarray],
                   ref_rgba: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Soft colour metric per view: flat (unlit) base-colour render vs the reference, inside both silhouettes.

    Returns ``{view: {"error": mean abs RGB difference 0..1 or None, "pixels": n}}``.
    """
    out: dict[str, dict] = {}
    for key in color_view_keys(view_spec, calibs, ref_masks, ref_rgba):
        img = render.render_color(key, calibs[key], bg=(0.0, 0.0, 0.0), flat=True, transparent=True)
        out[key] = color_error(ref_rgba[key], img, ref_masks[key], img[..., 3] > render.MASK_ALPHA_THRESHOLD)
        log.info("view %s: color error %s", key, "n/a" if out[key]["error"] is None else f"{out[key]['error'] * 100:.2f}%")
    return out


def _gray(mask: np.ndarray) -> np.ndarray:
    g = np.where(mask, 1.0, BG_COLOR).astype(np.float32)
    return np.stack([g, g, g], axis=-1)


def overlay(ref: np.ndarray, out: np.ndarray) -> np.ndarray:
    """RGB overlay: green = both, red = reference only (missing), blue = render only (excess)."""
    img = np.full(ref.shape + (3,), BG_COLOR, dtype=np.float32)
    img[ref & out] = COLOR_BOTH
    img[ref & ~out] = COLOR_REF_ONLY
    img[~ref & out] = COLOR_OUT_ONLY
    return img


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return imageops.resize_bilinear(img, (width, max(1, round(h * width / w))))


def compare_image(ref_rgba: Optional[np.ndarray], ref_mask: np.ndarray, out_mask: np.ndarray) -> np.ndarray:
    """Row image ``reference | render | overlay`` as RGB float."""
    if ref_rgba is not None:
        alpha = ref_rgba[..., 3:4].astype(np.float32)
        ref_panel = ref_rgba[..., :3] * alpha + REF_BG_COLOR * (1.0 - alpha)  # transparent references over light gray
    else:
        ref_panel = _gray(ref_mask)
    panels = [ref_panel, _gray(out_mask), overlay(ref_mask, out_mask)]
    gap = np.full((ref_panel.shape[0], PANEL_GAP_PX, 3), GAP_COLOR, dtype=np.float32)
    row = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
    return _fit_width(row, MAX_ROW_WIDTH_PX)


def write_comparison_images(view_spec, ref_masks: Mapping[str, np.ndarray], out_masks: Mapping[str, np.ndarray],
                            out_dir: Path | str, ref_rgba: Optional[Mapping[str, np.ndarray]] = None,
                            preview: Optional[np.ndarray] = None) -> dict[str, str]:
    """Write ``compare_<view>.png`` per view and ``comparison_sheet.png`` (all rows + the 3/4 preview tile)."""
    out = Path(out_dir)
    paths: dict[str, str] = {}
    rows = []
    for key in out_masks:
        row = compare_image(None if ref_rgba is None else ref_rgba.get(key), ref_masks[key], out_masks[key])
        path = out / f"compare_{key}.png"
        imageio.save_rgba(path, row)
        paths[key] = str(path)
        rows.append(_fit_width(row, SHEET_WIDTH_PX))
    if preview is not None:
        rows.append(_fit_width(preview[..., :3].astype(np.float32), SHEET_WIDTH_PX))
    if rows:
        width = max(r.shape[1] for r in rows)
        spacer = np.full((PANEL_GAP_PX, width, 3), GAP_COLOR, dtype=np.float32)
        padded: list[np.ndarray] = []
        for r in rows:
            padded.append(imageops.pad_to(r, (width, r.shape[0]), BG_COLOR))
            padded.append(spacer)
        sheet = out / "comparison_sheet.png"
        imageio.save_rgba(sheet, np.concatenate(padded[:-1], axis=0))
        paths["sheet"] = str(sheet)
    return paths


# --------------------------------------------------------------------------- per-part IoU (soft)
PART_IOU_TYPES = ("cone", "blade", "disc", "capsule", "strand")
PART_REGION_PAD_FRAC = 0.15
PART_REGION_PAD_MIN_PX = 4
PART_OUTSIDE_WARN = 0.7        # a part with less of its silhouette on the reference than this sticks out of it


def part_components(scene_spec) -> list:
    """Components that get a per-part IoU: ``stage: detail`` ones and explicit part primitives (never the hull or cutters)."""
    return [c for c in scene_spec.components
            if c.type != "silhouette_hull" and not c.boolean
            and (c.stage == "detail" or c.type in PART_IOU_TYPES or (c.type == "plate" and "outline" in c.params))]


def _region(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[slice, slice]:
    x0, y0, x1, y1 = box
    pad = max(PART_REGION_PAD_MIN_PX, round(PART_REGION_PAD_FRAC * max(x1 - x0, y1 - y0)))
    return slice(max(0, y0 - pad), min(shape[0], y1 + pad)), slice(max(0, x0 - pad), min(shape[1], x1 + pad))


def compare_parts(view_spec, scene_spec, calibs: Mapping[str, ViewCalibration], ref_masks: Mapping[str, np.ndarray],
                  model_masks: Optional[Mapping[str, np.ndarray]] = None) -> dict:
    """Cheap per-part check until label masks exist: render each part alone in every carve / validate view and compare
    it with the reference mask restricted to the part bbox region (padded).

    Per part and view: ``iou`` (part vs reference inside the region), ``precision`` (share of the part pixels that lie
    on the reference silhouette; low = the part sticks out of the reference) and, when ``model_masks`` has the view,
    ``model_iou`` (whole model vs reference inside the region). Soft: nothing here fails a run.
    """
    from . import scene
    views = [v.key for v in view_spec.enabled_views() if ("carve" in v.roles or "validate" in v.roles)
             and v.is_orthographic_usable and v.key in calibs and v.key in ref_masks]
    out: dict = {}
    for comp in part_components(scene_spec):
        obj = scene.find_by_agent_id(scene_spec.agent_id(comp.id))
        if obj is None:
            continue
        per_view: dict = {}
        for key in views:
            part = render.render_mask(key, calibs[key], objects=[obj])
            if not part.any():
                per_view[key] = {"visible": False}
                continue
            reg = _region(imageops.mask_bbox(part), part.shape)
            ref, p = ref_masks[key][reg], part[reg]
            info = {"visible": True, "iou": float(iou(ref, p)), "precision": float((p & ref).sum() / max(p.sum(), 1)),
                    "part_px": int(part.sum())}
            if model_masks is not None and key in model_masks:
                info["model_iou"] = float(iou(ref, model_masks[key][reg]))
            per_view[key] = info
        seen = [i for i in per_view.values() if i.get("visible")]
        out[comp.id] = {"type": comp.type, "stage": comp.stage, "views": per_view,
                        "mean_iou": float(np.mean([i["iou"] for i in seen])) if seen else None,
                        "min_precision": float(min(i["precision"] for i in seen)) if seen else None}
    return {"parts": out, "outside_warn": PART_OUTSIDE_WARN,
            "parts_outside_reference": sorted(k for k, v in out.items()
                                              if v["min_precision"] is not None and v["min_precision"] < PART_OUTSIDE_WARN)}


# --------------------------------------------------------------------------- held-out views
HOLDOUT_CANVAS_PX = 512
HOLDOUT_CANVAS_MARGIN = 1.15


def compare_holdout(view_spec, config: Optional[PipelineConfig] = None, out_dir: Optional[Path | str] = None,
                    azimuth_range: float = pose.DEFAULT_AZIMUTH_RANGE, azimuth_step: float = pose.DEFAULT_AZIMUTH_STEP,
                    scale_range: float = pose.DEFAULT_SCALE_RANGE) -> dict:
    """Score the model against every ``holdout`` view (never used to build it): silhouette IoU after a pose fit, plus colour error.

    The model silhouette is rendered orthographically from the view direction on a canonical canvas and fitted to the
    reference mask by :func:`bbr.core.pose.search_pose` (bbox bottom-centre alignment, scale ``1 +- scale_range`` and, for
    azimuth/elevation views, azimuth ``+- azimuth_range`` in ``azimuth_step`` steps: the angle found is only good to that
    step). Axis views search the scale only. The fitted camera is rendered again at the reference resolution for the
    reported ``iou`` and ``color_error`` (flat unlit model colour vs the reference inside both masks); ``iou_nominal`` is the
    fit at the nominal angle, ``azimuth_offset_deg`` how far the best azimuth is from it.
    Returns ``{"views": {key: {...}}, "mean_iou", "mean_color_error"}`` (``{}`` without holdout views).
    """
    from . import mesh, scene
    entries = view_spec.holdout_views()
    if not entries:
        return {}
    box = mesh.world_bbox(scene.model_meshes())
    if box is None:
        return {"error": "AI_MODEL has no mesh geometry"}
    cfg = config or PipelineConfig()
    centre = (box[0] + box[1]) / 2.0
    m0 = HOLDOUT_CANVAS_MARGIN * float(np.linalg.norm(box[1] - box[0])) / HOLDOUT_CANVAS_PX
    canvas = (HOLDOUT_CANVAS_PX, HOLDOUT_CANVAS_PX)
    cache: dict = {}

    def canonical(az: float, el: float) -> np.ndarray:
        key = (round(az, 3), round(el, 3))
        if key not in cache:
            cache[key] = render.render_mask_dir(az, el, canvas, m0, centre)
        return cache[key]

    results: dict = {}
    for entry in entries:
        img = imageio.load_view_image(entry)
        bg = entry.background
        ref = imageops.extract_mask(img, bg.mode, bg.color, bg.tolerance, cfg.max_hole_frac, cfg.min_component_frac,
                                  bg.remove_enclosed)
        az, el = entry.camera_angles or pose.axis_angles(entry.camera_axis)
        searching = entry.camera_angles is not None and abs(el) < 90.0
        best = pose.search_pose(ref, canonical, m0, az, el, azimuth_range if searching else 0.0, azimuth_step,
                                scale_range)
        h, w = ref.shape
        _, right, up = pose.view_basis(best["azimuth"], el)
        cam_centre = centre + right * best["p_r"] + up * best["p_up"]
        fitted = best["k"] > 0
        out_mask = render.render_mask_dir(best["azimuth"], el, (w, h), best["k"], cam_centre) if fitted else np.zeros_like(ref)
        value = float(iou(ref, out_mask))
        item = {"camera": {"azimuth": az, "elevation": el}, "iou": value, "error": 1.0 - value,
                "iou_nominal": best["nominal_iou"], "best_azimuth": best["azimuth"],
                "azimuth_offset_deg": best["azimuth"] - az, "best_scale": best["scale"], "azimuth_searched": searching,
                "pose_search": {"azimuth_range_deg": azimuth_range if searching else 0.0, "azimuth_step_deg": azimuth_step,
                                "scale_range": scale_range}, "color_error": None}
        if fitted:
            col = render.render_color_dir(best["azimuth"], el, (w, h), best["k"], cam_centre, bg=(0.0, 0.0, 0.0),
                                          transparent=True)
            item["color_error"] = color_error(img, col, ref, col[..., 3] > render.MASK_ALPHA_THRESHOLD)["error"]
        if out_dir is not None:
            path = Path(out_dir) / f"holdout_{entry.key}.png"
            imageio.save_rgba(path, compare_image(img, ref, out_mask))
            item["image"] = str(path)
        results[entry.key] = item
        log.info("holdout %s: IoU %.4f (nominal %.4f) azimuth %.1f -> %.1f", entry.key, value, best["nominal_iou"], az,
                 best["azimuth"])
    colors = [r["color_error"] for r in results.values() if r["color_error"] is not None]
    return {"views": results, "mean_iou": float(np.mean([r["iou"] for r in results.values()])),
            "mean_color_error": float(np.mean(colors)) if colors else None}
