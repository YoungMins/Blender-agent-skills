"""The reconstruction workflow: 16 logged steps, blockout gate, axis-scale correction loop, validation, outputs."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import bpy
import numpy as np

from .. import BBRError, get_logger
from ..core import attribution as attribution_core, consistency, dimcheck, gapcheck, imageio, measure
from ..core import facedetail as facedetail_core
from ..core import labels as labels_core
from ..core.calibration import ViewCalibration, calibrate_views
from ..core.config import PipelineConfig
from ..core.conventions import view_frame
from ..core import correction
from ..core.correction import apply_axis_scale, axis_scale_factors, diagnose
from ..core.metrics import aggregate, extent_gate
from ..core.summary import format_summary, merge_label_iou  # noqa: F401  (format_summary re-exported: used by run_pipeline)
from ..core.spec import SceneSpec, ViewSpec, load_scene_spec, load_view_spec, parse_scene_spec, project_textures
from . import (attribution, camera, checkpoint, components, facedetail, material, mesh, partstex, primitive, projection, reference, render,
               scene, validation)
from .components import HULL_TYPE, BuildContext, HullInputs

log = get_logger(__name__)

STOP_AFTER_CHOICES = ("blockout", "primary", "secondary", "final")
STAGE_PREVIEW_FILES = {"blockout": "stage_A_blockout.png", "primary": "stage_B_primary.png",
                       "secondary": "stage_B2_secondary.png",
                       "final": "stage_C_final.png"}
PRIMARY_STAGES = ("primary",)
DETAIL_STAGES = ("secondary", "detail")
MIN_CORRECTION = 0.002   # axis scale factors closer to 1 than this are not worth a rebuild
MIN_IMPROVEMENT = 1e-6   # a correction must lower the total error by at least this much
PERSPECTIVE_PREVIEW = "perspective.png"
PERSPECTIVE_BACK_PREVIEW = "perspective_back.png"
TOP_PREVIEW = "top.png"
TURNTABLE_PREVIEW = "turntable.png"
LIT_VIEW_PREVIEWS = (("front_lit", "front_lit.png", (0.0, -1.0, 0.05)),    # pure front / side (the object's right, -X)
                     ("side_lit", "side_lit.png", (-1.0, 0.0, 0.05)))      # lit renders: inspect the projected texture
BACK_PREVIEW_DIRECTION = (0.75, 1.0, 0.6)  # back-left-above
TOP_PREVIEW_PX = 800
BBOX_AGENT_ID = "bbr:bbox"
BBOX_BLOCKOUT_ID = "__bbox_blockout"
BBOX_COLOR = (0.7, 0.72, 0.78, 1.0)
WORKFLOW_KEYS = ("max_iterations", "error_threshold", "blockout_threshold", "auto_correct", "consistency_gate", "dimension_tolerance")


@dataclass
class Inputs:
    """Parsed specs and reference data of a run (mutable: the correction loop replaces ``scene``)."""

    scene: SceneSpec
    scene_dict: dict[str, Any]
    view_spec: Optional[ViewSpec] = None
    rgba: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    calibs: dict[str, ViewCalibration] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    out_dir: Optional[Path] = None                      # where hull textures are written (``<out_dir>/textures``)
    hull_stats: dict[str, Any] = field(default_factory=dict)  # per-component post-processing stats (cleanup, retopo, uv, texture)

    @property
    def dims(self) -> dict[str, float]:
        """Overall width/depth/height: calibrated from the views when available, else the SceneSpec dimensions."""
        if self.info.get("dimensions"):
            return dict(self.info["dimensions"])
        return {k: float(v) for k, v in self.scene.dimensions.items() if v}

    @property
    def hull(self) -> Optional[HullInputs]:
        """Reference data for silhouette hulls (None without a ViewSpec)."""
        if self.view_spec is None:
            return None
        return HullInputs(self.view_spec, self.rgba, self.masks, self.calibs, self.dims)

    def set_scene(self, scene_dict: dict[str, Any]) -> None:
        """Replace the SceneSpec (validated) with ``scene_dict``."""
        self.scene = parse_scene_spec(scene_dict, self.scene.source_path)
        self.scene_dict = scene_dict


class StepLog:
    """Records the numbered workflow steps (name, status, duration) into ``records``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    @contextlib.contextmanager
    def step(self, number: int, name: str) -> Iterator[dict[str, Any]]:
        """Context manager logging ``STEP n: name`` and timing it."""
        log.info("STEP %d: %s", number, name)
        rec: dict[str, Any] = {"n": number, "name": name, "status": "running"}
        self.records.append(rec)
        t0 = time.perf_counter()
        try:
            yield rec
            rec["status"] = "ok"
        except Exception:
            rec["status"] = "failed"
            raise
        finally:
            rec["duration_s"] = round(time.perf_counter() - t0, 3)

    def skip(self, number: int, name: str, reason: str) -> None:
        """Record a step that does not apply."""
        log.info("STEP %d: %s (skipped: %s)", number, name, reason)
        self.records.append({"n": number, "name": name, "status": "skipped", "reason": reason, "duration_s": 0.0})


# --------------------------------------------------------------------------- inputs
def resolve_config(config: Optional[PipelineConfig], workflow: Mapping[str, Any],
                   overrides: Optional[Mapping[str, Any]] = None) -> PipelineConfig:
    """``config`` (or defaults) < SceneSpec ``workflow`` (only when ``config`` is None) < explicit ``overrides``."""
    cfg = config
    if cfg is None:
        data = {k: workflow[k] for k in WORKFLOW_KEYS if k in workflow}
        cons = workflow.get("consistency") or {}
        if cons.get("ignore_flags"):
            data["consistency_ignore_flags"] = list(cons["ignore_flags"])
        if cons.get("roles_override"):
            data["consistency_roles_override"] = {k: list(v) for k, v in cons["roles_override"].items()}
        cfg = PipelineConfig.from_dict(data)
    clean = {k: v for k, v in (overrides or {}).items() if v is not None}
    if "consistency_ignore_flags" in clean:   # --ignore-flag adds to workflow.consistency.ignore_flags, never replaces them
        clean["consistency_ignore_flags"], _ = consistency.merge_overrides(
            {"ignore_flags": cfg.consistency_ignore_flags}, {"ignore_flags": clean["consistency_ignore_flags"]})
    return dataclasses.replace(cfg, **clean) if clean else cfg


def load_view_data(view_spec: ViewSpec, config: PipelineConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load (crop + flip applied) images and foreground masks of every enabled, usable view."""
    for view in view_spec.enabled_views():
        if not view.is_orthographic_usable:
            log.info("view %s is preview-only (roles %s): stored, not used for carving/validation", view.key, view.roles)
    return measure.load_view_data(view_spec, config)


def load_inputs(view_spec_path: Optional[Path | str], scene_spec_path: Path | str,
                config: Optional[PipelineConfig] = None) -> Inputs:
    """Load both specs, reference images, masks and calibrations (steps 1-3 without logging)."""
    spec = load_scene_spec(scene_spec_path)
    cfg = config or PipelineConfig()
    inp = Inputs(spec, spec.to_dict())
    if view_spec_path:
        inp.view_spec = load_view_spec(view_spec_path)
        inp.rgba, inp.masks = load_view_data(inp.view_spec, cfg)
        inp.calibs, inp.info = calibrate_views(inp.view_spec, inp.masks, _symmetry_labels(spec, inp.masks)[0])
    return inp


# --------------------------------------------------------------------------- building
def scene_objects(spec: SceneSpec) -> dict[str, bpy.types.Object]:
    """Existing objects of the spec's components by component id."""
    found = {}
    for comp in spec.components:
        obj = scene.find_by_agent_id(spec.agent_id(comp.id))
        if obj is not None:
            found[comp.id] = obj
    return found


def prune_orphans(spec: SceneSpec) -> int:
    """Remove AI_MODEL objects of this scene (agent_id ``<name>:...``) whose component left the spec."""
    prefix = f"{spec.name}:"
    keep = {spec.agent_id(c.id) for c in spec.components}
    removed = 0
    for obj in scene.objects_in(scene.COLL_MODEL):
        aid = obj.get(scene.PROP_AGENT_ID, "")
        if aid.startswith(prefix) and aid not in keep:
            scene.remove_object(obj)
            removed += 1
    return removed


def build_context(inp: Inputs, config: PipelineConfig) -> BuildContext:
    """Build context of a run; ``defer_texture`` leaves the hull texture to :func:`texture_parts` when parts are textured."""
    colls = scene.ensure_structure()
    return BuildContext(inp.scene, config, colls[scene.COLL_MODEL], inp.hull,
                        {"out_dir": inp.out_dir, "hull_stats": inp.hull_stats,
                         "defer_texture": inp.view_spec is not None and bool(partstex.textured_parts(
                             inp.scene, project_textures(inp.scene.style, inp.view_spec)))})


def texture_parts(inp: Inputs, config: PipelineConfig, only: Optional[str] = None) -> dict[str, Any]:
    """Projected textures of the hull (when deferred) and of every textured part primitive, against the final scene
    (see :mod:`partstex`). No-op without a ViewSpec or textured parts."""
    if inp.view_spec is None:
        return {}
    return partstex.texture_pass(build_context(inp, config), only)


def build_stages(inp: Inputs, config: PipelineConfig, stages: Optional[Sequence[str]] = None,
                 only: Optional[str] = None) -> dict[str, bpy.types.Object]:
    """Build (or rebuild in place) components of ``stages`` (all when None), each with its Mirror modifier."""
    ctx = build_context(inp, config)
    built: dict[str, bpy.types.Object] = {}
    for comp in inp.scene.components_by_stage():
        if (stages is None or comp.stage in stages) and (only is None or comp.id == only):
            obj = components.build_component(comp, ctx)
            components.apply_symmetry(obj, comp)
            built[comp.id] = obj
    return built


def apply_booleans(inp: Inputs) -> int:
    """Step 10: boolean cuts between components."""
    return components.apply_booleans(inp.scene, scene_objects(inp.scene))


def apply_bevels(inp: Inputs, only: Optional[str] = None) -> None:
    """Step 11: bevel modifiers."""
    for comp in inp.scene.components:
        obj = scene.find_by_agent_id(inp.scene.agent_id(comp.id))
        if obj is not None and (only is None or comp.id == only):
            components.apply_bevel(obj, comp, inp.scene)


def apply_normals(inp: Inputs, only: Optional[str] = None) -> None:
    """Step 12: normal cleanup, smooth shading and Weighted Normal."""
    for comp in inp.scene.components:
        obj = scene.find_by_agent_id(inp.scene.agent_id(comp.id))
        if obj is not None and (only is None or comp.id == only):
            components.apply_normal_cleanup(obj, comp, inp.scene)


def rebuild_all(inp: Inputs, config: PipelineConfig) -> dict[str, bpy.types.Object]:
    """Idempotent full build: all stages, symmetry, booleans, bevel, normals."""
    prune_orphans(inp.scene)
    built = build_stages(inp, config)
    apply_booleans(inp)
    apply_bevels(inp)
    apply_normals(inp)
    texture_parts(inp, config)
    return built


def ensure_bounding_volume(dims: Mapping[str, float]) -> bpy.types.Object:
    """Wire box of the overall bounding volume in AI_DEBUG (never rendered)."""
    w, d, h = dims["width"], dims["depth"], dims["height"]
    corners = np.array([[x, y, z] for z in (0.0, h) for y in (-d / 2, d / 2) for x in (-w / 2, w / 2)])
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    data = mesh.get_or_create_mesh(BBOX_AGENT_ID)
    data.clear_geometry()
    data.from_pydata(corners.tolist(), edges, [])
    data.update()
    obj = scene.ensure_object(BBOX_AGENT_ID, scene.ensure_collection(scene.COLL_DEBUG), data)
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def build_bbox_blockout(inp: Inputs) -> bpy.types.Object:
    """Temporary box of the resolved overall dimensions (agent_id ``<name>:__bbox_blockout``, confidence inferred)."""
    dims = inp.dims
    w, d, h = dims["width"], dims["depth"], dims["height"]
    agent_id = inp.scene.agent_id(BBOX_BLOCKOUT_ID)
    data = mesh.get_or_create_mesh(agent_id)
    primitive.create_box(data, (w, d, h))
    obj = scene.ensure_object(agent_id, scene.ensure_collection(scene.COLL_MODEL), data)
    obj.location = (0.0, 0.0, h / 2.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    obj[scene.PROP_CONFIDENCE] = "inferred"
    obj[scene.PROP_TYPE] = "box"
    obj[scene.PROP_STAGE] = "blockout"
    material.assign_material(obj, material.ensure_material(f"{inp.scene.name}:{BBOX_BLOCKOUT_ID}", BBOX_COLOR))
    return obj


def remove_bbox_blockout(inp: Inputs) -> None:
    """Delete the temporary bbox blockout (no-op when absent)."""
    obj = scene.find_by_agent_id(inp.scene.agent_id(BBOX_BLOCKOUT_ID))
    if obj is not None:
        scene.remove_object(obj)


# --------------------------------------------------------------------------- comparison / outputs
def compare(inp: Inputs, config: PipelineConfig) -> dict[str, Any]:
    """Render and compare all validation views; adds ``aggregate`` (total error, passed)."""
    if inp.view_spec is None:
        raise BBRError("comparison needs a ViewSpec")
    res = projection.compare_all(inp.view_spec, inp.calibs, inp.masks, config)
    res["aggregate"] = aggregate(res["results"], config.view_weights, config.error_threshold) if res["results"] else None
    return res


def _per_view_errors(cmp: Mapping[str, Any]) -> dict[str, float]:
    return {v: round(float(r.error), 5) for v, r in cmp["results"].items()}


def _top_calibration(dims: Mapping[str, float]) -> ViewCalibration:
    """Synthetic orthographic top-down camera framing the footprint (preview only)."""
    w, d = dims["width"], dims["depth"]
    m_per_px = 1.15 * max(w, d) / TOP_PREVIEW_PX
    size = (round(1.15 * w / m_per_px), round(1.15 * d / m_per_px))
    return ViewCalibration("top", "+Z", size, m_per_px, (size[0] / 2.0, size[1] / 2.0))


def _stage_preview(out: Path, stage: str) -> str:
    path = out / "previews" / STAGE_PREVIEW_FILES[stage]
    render.render_preview(path)
    return str(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write ``report.json``."""
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def confidence_summary(inp: Inputs) -> dict[str, Any]:
    """Component and view confidences (observed / inferred / unknown)."""
    comps = {c.id: c.confidence for c in inp.scene.components}
    counts: dict[str, int] = {}
    for value in comps.values():
        counts[value] = counts.get(value, 0) + 1
    views = {k: v.confidence for k, v in (inp.view_spec.views.items() if inp.view_spec else [])}
    return {"components": comps, "counts": counts, "views": views}


# --------------------------------------------------------------------------- the workflow
def run_reconstruction(view_spec_path: Path | str, scene_spec_path: Path | str, out_dir: Path | str,
                       config: Optional[PipelineConfig] = None, overrides: Optional[Mapping[str, Any]] = None,
                       stop_after: Optional[str] = None, force: bool = False) -> dict[str, Any]:
    """Run the full workflow and return the report dict (also written to ``<out_dir>/report.json``).

    ``config`` < SceneSpec workflow < ``overrides``. ``stop_after`` in blockout|primary|secondary (body plus parts) stops after that stage
    (stage previews are always written; ``final`` runs everything). ``force`` continues after a failed blockout gate.
    """
    if stop_after is not None and stop_after not in STOP_AFTER_CHOICES:
        raise BBRError(f"stop_after must be one of {STOP_AFTER_CHOICES}")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    report: dict[str, Any] = {"name": None, "status": "running", "passed": None, "steps": [], "outputs": {}}
    try:
        _run(Path(view_spec_path), Path(scene_spec_path), out, config, overrides, stop_after, force, report)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = round(time.perf_counter() - t_start, 2)
        write_report(report, out / "report.json")
    return report


def _run(view_spec_path: Path, scene_spec_path: Path, out: Path, config: Optional[PipelineConfig],
         overrides: Optional[Mapping[str, Any]], stop_after: Optional[str], force: bool,
         report: dict[str, Any]) -> None:
    steps = StepLog(report["steps"])
    outputs = report["outputs"]
    outputs["report"] = str(out / "report.json")
    spec0 = load_scene_spec(scene_spec_path)
    cfg = resolve_config(config, spec0.workflow, overrides)
    report["name"] = spec0.name
    report["config"] = cfg.to_dict()

    with steps.step(1, "import reference views"):
        view_spec = load_view_spec(view_spec_path)
        inp = Inputs(spec0, spec0.to_dict(), view_spec, out_dir=out)
        inp.rgba, inp.masks = load_view_data(view_spec, cfg)
        report["views"] = {k: {"path": str(view_spec.views[k].path), "camera_axis": view_spec.views[k].camera_axis,
                               "roles": view_spec.views[k].roles, "confidence": view_spec.views[k].confidence}
                           for k in inp.masks}

    with steps.step(2, "alignment (axis conventions)") as rec:
        for key, axis in view_spec.view_axes().items():
            frame = view_frame(axis)
            log.info("view %s: camera_axis %s, image right = %s, image up = %s", key, axis, frame.right_axis, frame.up_axis)
        rec["views"] = view_spec.view_axes()

    with steps.step(3, "scale calibration"):
        labels, asym = _symmetry_labels(spec0, inp.masks)       # labels also place the body axis (X origin)
        inp.calibs, inp.info = calibrate_views(view_spec, inp.masks, labels)
        report["dimensions"] = inp.info["dimensions"]
        report["provenance"] = inp.info["provenance"]
        report["calibrations"] = {k: c.to_dict() for k, c in inp.calibs.items()}
        report["frame_offset_m"] = inp.info["frame_offset_m"]
        log.info("dimensions %s provenance %s", inp.info["dimensions"], inp.info["provenance"])
        ignore_flags, roles_override = consistency.merge_overrides(   # view spec < scene workflow < --ignore-flag
            view_spec.consistency, {"ignore_flags": cfg.consistency_ignore_flags, "roles_override": cfg.consistency_roles_override})
        report["consistency"] = consistency.analyze(inp.masks, inp.calibs, view_spec.view_axes(),
                                                    {k: v.roles for k, v in view_spec.views.items()},
                                                    labels=labels, asymmetric_parts=asym,
                                                    ignore_flags=ignore_flags, roles_override=roles_override)
        log.info("\n%s", consistency.format_consistency(report["consistency"]))
        if not report["consistency"]["passed"]:
            log.warning("reference views are inconsistent: validate errors cannot go below the floors above")
    if cfg.consistency_gate == "strict" and not report["consistency"]["passed"] and not force:
        report["status"] = "consistency_gate_failed"
        report["passed"] = False
        log.error("consistency gate (strict) failed: fix the references or the roles (use force to continue)")
        return

    with steps.step(4, "bounding volume + reference planes"):
        scene.ensure_structure()
        ensure_bounding_volume(inp.dims)
        reference.setup_reference_views(view_spec, inp.calibs, inp.dims, cfg, inp.rgba, out / "refs")
        checkpoint.save_checkpoint(checkpoint.CHECKPOINT_NAMES[0], out)

    blockout_comps = [c for c in inp.scene.components if c.stage == "blockout"]
    report["blockout"] = {"gate": "skipped", "mode": None}
    with steps.step(5, "blockout") as rec:
        if blockout_comps:
            build_stages(inp, cfg, ("blockout",))
            apply_normals(inp)
            cmp = compare(inp, cfg)
            err = cmp["aggregate"]["total_error"] if cmp["aggregate"] else 1.0
            report["blockout"] = {"gate": "passed" if err <= cfg.blockout_threshold else "failed", "mode": "components",
                                  "error": err, "threshold": cfg.blockout_threshold}
        else:
            build_bbox_blockout(inp)
            cmp = compare(inp, cfg)
            gate = extent_gate(cmp["results"], cfg.blockout_threshold)
            report["blockout"] = {"gate": "passed" if gate["passed"] else "failed", "mode": "bbox_fallback",
                                  "error": gate["error"], "threshold": gate["threshold"], "ratios": gate["ratios"]}
            rec["note"] = "no blockout components: temporary bbox blockout from the resolved dimensions"
    if blockout_comps:
        with steps.step(6, "major proportions (axis scale)") as rec:
            gate = report["blockout"]
            if gate["gate"] == "failed" and cfg.auto_correct:
                factors = axis_scale_factors(cmp["results"])
                inp.set_scene(apply_axis_scale(inp.scene_dict, factors))
                build_stages(inp, cfg, ("blockout",))
                apply_normals(inp)
                cmp = compare(inp, cfg)
                err = cmp["aggregate"]["total_error"] if cmp["aggregate"] else 1.0
                rec["factors"] = factors
                gate.update(error=err, gate="passed" if err <= cfg.blockout_threshold else "failed", corrected=True)
    else:
        steps.skip(6, "major proportions (axis scale)", "bbox blockout: proportions come from the calibrated dimensions")
    outputs["stage_blockout_preview"] = _stage_preview(out, "blockout")
    if report["blockout"]["gate"] == "failed" and not force:
        remove_bbox_blockout(inp)
        report["status"] = "blockout_gate_failed"
        report["passed"] = False
        log.error("blockout gate failed (error %.3f > %.3f); stopping before details (use force to continue)",
                  report["blockout"]["error"], cfg.blockout_threshold)
        return
    checkpoint.save_checkpoint(checkpoint.CHECKPOINT_NAMES[1], out)
    remove_bbox_blockout(inp)
    if stop_after == "blockout":
        _stop(report, "blockout")
        return

    with steps.step(7, "symmetry (mirror modifiers)") as rec:
        mirrored = [c.id for c in inp.scene.components if any(c.mirror.values())]
        rec["mirrored"] = mirrored

    with steps.step(8, "primary components"):
        build_stages(inp, cfg, PRIMARY_STAGES)
        checkpoint.save_checkpoint(checkpoint.CHECKPOINT_NAMES[2], out)
        outputs["stage_primary_preview"] = _stage_preview(out, "primary")
    if stop_after == "primary":
        _stop(report, "primary")
        return

    with steps.step(9, "secondary + detail components") as rec:
        build_stages(inp, cfg, DETAIL_STAGES)
        checkpoint.save_checkpoint(checkpoint.CHECKPOINT_NAMES[3], out)
        report["part_gaps"] = rec["part_gaps"] = part_gaps(inp, out)
        for gap in report["part_gaps"]:
            log.warning("part_gap: %s", gapcheck.format_gap(gap))
        report["floating_parts"] = rec["floating_parts"] = floating_check(inp)
        for fl in report["floating_parts"]:
            log.warning("floating_part: %s", gapcheck.format_floating(fl))
        report["dimension_check"] = rec["dimension_check"] = dimension_check(inp, cfg)
        for line in dimcheck.format_check(report["dimension_check"]):
            log.info("%s", line)
        outputs["stage_secondary_preview"] = _secondary_preview(inp, out)
    if stop_after == "secondary":
        _stop(report, "secondary")
        return
    with steps.step(10, "boolean / cut") as rec:
        rec["booleans"] = apply_booleans(inp)
    with steps.step(11, "bevel"):
        apply_bevels(inp)
    with steps.step(12, "normal cleanup + part textures"):
        apply_normals(inp)
        texture_parts(inp, cfg)

    with steps.step(13, "reference comparison"):
        cmp = compare(inp, cfg)
        report["iterations"] = [{"iteration": 0, "total_error": cmp["aggregate"]["total_error"],
                                 "per_view": _per_view_errors(cmp), "accepted": True}]

    with steps.step(14, "correction loop") as rec:
        cmp = _correction_loop(inp, cfg, cmp, report["iterations"], out, rec)

    with steps.step(15, "validation") as rec:
        outputs["perspective_preview"] = str(out / "previews" / PERSPECTIVE_PREVIEW)
        preview = render.render_preview(outputs["perspective_preview"])
        outputs["stage_final_preview"] = str(out / "previews" / STAGE_PREVIEW_FILES["final"])
        imageio.save_rgba(outputs["stage_final_preview"], preview)
        outputs["perspective_back_preview"] = str(out / "previews" / PERSPECTIVE_BACK_PREVIEW)
        render.render_preview(outputs["perspective_back_preview"], direction=BACK_PREVIEW_DIRECTION)
        for key, fname, direction in LIT_VIEW_PREVIEWS:
            outputs[f"{key}_preview"] = str(out / "previews" / fname)
            render.render_preview(outputs[f"{key}_preview"], direction=direction)
        outputs["top_preview"] = str(out / "previews" / TOP_PREVIEW)
        render.render_color("top", _top_calibration(inp.dims), outputs["top_preview"], bg=render.PREVIEW_BG, flat=False)
        images = projection.write_comparison_images(view_spec, inp.masks, cmp["out_masks"], out, inp.rgba, preview)
        outputs.update({("comparison_sheet" if k == "sheet" else f"compare_{k}"): v for k, v in images.items()})
        report["validation"] = validation.validate_reconstruction(
            cmp["results"], inp.calibs, inp.dims, {**inp.scene.symmetry}, cfg, inp.scene_dict,
            dimension_tolerance=dimension_tolerance(inp, cfg)[0], dimension_context=dimension_context(inp))
        report["reference_match"] = report["validation"]["reference_match"]
        colors = projection.compare_colors(view_spec, inp.calibs, inp.masks, inp.rgba)
        for view, info in colors.items():
            report["reference_match"]["per_view"].setdefault(view, {})["color_error"] = (
                None if info["error"] is None else round(info["error"], 5))
        report["color_match"] = {"soft": True, "per_view": colors}
        per_view = report["reference_match"]["per_view"]
        consistency.add_excess(report["consistency"], {v: i["error"] for v, i in per_view.items() if i.get("error") is not None})
        for view, info in report["consistency"]["views"].items():
            if info.get("excess_error") is not None and view in per_view:
                per_view[view].update(floor=info["floor"], excess_error=info["excess_error"])
        if inp.hull_stats:
            report["hull"] = inp.hull_stats
        report["passed"] = report["validation"]["passed"]
        report["status"] = "passed" if report["passed"] else "failed_validation"
        report["confidence"] = confidence_summary(inp)
        report["recommendation"] = report["validation"]["recommendation"]
        report["diagnosis"] = report["validation"]["diagnosis"]
        report["warnings"] = list(report["validation"]["warnings"])
        measure_extras(inp, cfg, cmp, report, out)
        if stop_after == "final":
            report["stop_after"] = "final"

    with steps.step(16, "save outputs"):
        checkpoint.save_checkpoint(checkpoint.CHECKPOINT_NAMES[4], out)
        corrected = out / "scene_spec_corrected.json"
        fitted_path = out / "fitted_parts.json"
        if fitted_path.exists():   # keep the fitted pose / primitive values beside each anchored part (params.fitted)
            fitted = json.loads(fitted_path.read_text(encoding="utf-8"))
            for c in inp.scene_dict.get("components", []):
                if c.get("id") in fitted:
                    c.setdefault("params", {})["fitted"] = {k: v for k, v in fitted[c["id"]].items() if k != "type"}
        corrected.write_text(json.dumps(inp.scene_dict, indent=2, default=_json_default), encoding="utf-8")
        outputs["scene_spec_corrected"] = str(corrected)
        outputs["blend"] = str(checkpoint.save_blend(out / f"{inp.scene.name}.blend"))


def _symmetry_labels(spec: SceneSpec, masks: Mapping[str, Any]) -> tuple[Any, list[str]]:
    """Label masks and the hull's ``asymmetric_parts`` for the symmetry suggestion (so a sword does not count against the
    mirror IoU); ``(None, [])`` without ``parts_labels``."""
    if not spec.parts_labels:
        return None, []
    path = Path(spec.parts_labels)
    if not path.is_absolute() and spec.source_path is not None:
        path = Path(spec.source_path).parent / path
    hull = next((c for c in spec.components if c.type == HULL_TYPE), None)
    return labels_core.load_label_masks(path, masks), [str(a) for a in (hull.params.get("asymmetric_parts") or [])] if hull else []


def part_gaps(inp: Inputs, out: Path) -> list[dict[str, Any]]:
    """Parts that should touch (``params.adjacent_to``, labels touching in every shared view, or anchors typed on the same
    pixel) but are apart in 3D, from the placements written to ``<out>/placement.json`` (:mod:`gapcheck`)."""
    path = out / "placement.json"
    if not path.is_file():
        return []
    ids = {c.id for c in inp.scene.components}
    placed = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k in ids and v.get("keypoints")}
    declared = [(c.id, str(o)) for c in inp.scene.components for o in c.params.get("adjacent_to") or []]
    declared += [(c.id, str(c.params["joint"]["part"])) for c in inp.scene.components if c.params.get("joint")]
    labels, _ = _symmetry_labels(inp.scene, inp.masks)
    touching = gapcheck.label_adjacent_pairs(labels, placed) if labels else []
    tols = {c.id: float(c.params["gap_tol_m"]) for c in inp.scene.components if c.params.get("gap_tol_m")}
    anchors = {c.id: c.params["anchors_px"] for c in inp.scene.components if c.id in placed and c.params.get("anchors_px")}
    return gapcheck.find_gaps(placed, declared, touching, tols, joints=gapcheck.shared_anchors(anchors))


def dimension_tolerance(inp: Inputs, cfg: PipelineConfig) -> tuple[float, str]:
    """``(tolerance, why)`` of the overall-dimension check: ``workflow.dimension_tolerance``, else by reference quality
    (:func:`dimcheck.resolve_tolerance`: 3 % synthetic / blueprint, 5 % AI-generated)."""
    vs = inp.view_spec
    modes = {k: v.calibration.mode for k, v in vs.views.items()} if vs else {}
    return dimcheck.resolve_tolerance(cfg.dimension_tolerance, vs.reference_quality if vs else {}, modes)


def dimension_context(inp: Inputs) -> dict[str, Any]:
    """What names the limiting part of a failed dimension check: every built component's world box, the reference
    silhouette's extents per axis and the hull ids."""
    boxes = {}
    for comp in inp.scene.components:
        obj = None if comp.boolean else scene.find_by_agent_id(inp.scene.agent_id(comp.id))
        box = mesh.world_bbox([obj]) if obj is not None and obj.type == "MESH" else None
        if box is not None:
            boxes[comp.id] = box
    return {"boxes": boxes, "reference": dimcheck.reference_extents(inp.masks, inp.calibs),
            "hull_ids": [c.id for c in inp.scene.components if c.type == HULL_TYPE]}


def dimension_check(inp: Inputs, cfg: PipelineConfig) -> dict[str, Any]:
    """The overall-dimension check on the model as built so far (the parts-only stage, ``--stop-after secondary``): the
    same result as validation's ``dimensions`` check, with the limiting parts named. Soft: never raises."""
    tol, why = dimension_tolerance(inp, cfg)
    try:
        out = validation.validate_dimensions(inp.dims, tol, dimension_context(inp))
    except Exception as exc:  # noqa: BLE001 - a soft check must never turn a finished stage into an error
        log.exception("dimension check failed")
        return {"passed": True, "reason": f"{type(exc).__name__}: {exc}"}
    out["tolerance_source"] = why
    return out


def floating_check(inp: Inputs) -> list[dict[str, Any]]:
    """``floating_part`` check: every built part's minimum distance to the union of the other parts' meshes against
    max(1 % of the height, 25 % of its own minor dimension) (:func:`gapcheck.floating_parts`); detached islands of the hull count
    as parts (``body#1``). Soft: a failure of the check is logged and gives no entries."""
    from . import hullpost
    meshes, tols, skip = {}, {}, []
    try:
        for comp in inp.scene.components:
            obj = None if comp.boolean else scene.find_by_agent_id(inp.scene.agent_id(comp.id))
            if obj is None or obj.type != "MESH":
                continue
            verts, tris = hullpost.world_triangles([obj])
            if len(tris) and comp.type == HULL_TYPE:        # detached islands of the hull are parts of their own: <hull>#1, #2 ...
                pieces = gapcheck.mesh_components(verts, tris)
                ids = [comp.id] + [f"{comp.id}#{k}" for k in range(1, len(pieces))]
                meshes.update(zip(ids, pieces))
                if comp.params.get("floating_ok"):
                    skip += ids
                continue
            if len(tris):
                meshes[comp.id] = (verts, tris)
            if comp.params.get("gap_tol_m"):
                tols[comp.id] = float(comp.params["gap_tol_m"])
            if comp.params.get("floating_ok"):
                skip.append(comp.id)
        return gapcheck.floating_parts(meshes, float(inp.dims["height"]), tols=tols, skip=skip)
    except Exception:  # noqa: BLE001 - a soft check must never turn a finished reconstruction into an error
        log.exception("floating_part check failed")
        return []


def _part_colours(inp: Inputs) -> dict[str, tuple[float, float, float]]:
    """Linear RGB per component for the secondary-stage preview: the mean reference colour under its label (in the first of
    front / back / side that labels it); the hull = its silhouette minus every label. Components without pixels are left out."""
    labels = _symmetry_labels(inp.scene, inp.masks)[0] or {}
    views = [v for v in ("front", "back", "side") if v in inp.rgba] or list(inp.rgba)
    out: dict[str, tuple[float, float, float]] = {}
    for comp in inp.scene.components:
        view, mask = next(((v, m) for v in views for m in [(labels.get(v) or {}).get(comp.id)] if m is not None and m.any()), (None, None))
        if mask is None and comp.type == HULL_TYPE and views:
            view = views[0]
            parts = [m for k, m in (labels.get(view) or {}).items() if not k.startswith("_")]
            mask = inp.masks[view] & ~np.logical_or.reduce(parts + [np.zeros_like(inp.masks[view])])
        if mask is None or not mask.any():
            continue
        rgb = np.asarray(inp.rgba[view])[..., :3][mask].mean(axis=0)
        out[comp.id] = tuple(render._srgb_to_linear(float(c)) for c in rgb)
    return out


def _secondary_preview(inp: Inputs, out: Path) -> str:
    """``stage_B2_secondary.png``: the model with flat part colours (the mean reference colour of each part's label; the
    texture projection only runs later) from the front 3/4 and the back 3/4 side by side, lit like the other previews.
    Materials are restored afterwards."""
    path = out / "previews" / STAGE_PREVIEW_FILES["secondary"]
    saved: list[tuple[Any, list[Any]]] = []
    temp: list[Any] = []
    try:
        for comp_id, col in _part_colours(inp).items():
            obj = scene.find_by_agent_id(inp.scene.agent_id(comp_id))
            if obj is None or obj.type != "MESH":
                continue
            saved.append((obj, list(obj.data.materials)))
            mat = material.ensure_material(f"bbr_preview_{comp_id}", (*col, 1.0))
            temp.append(mat)
            material.assign_material(obj, mat)
        views = [render.render_preview(None, direction=d)[..., :3] for d in (camera.PREVIEW_DIRECTION, BACK_PREVIEW_DIRECTION)]
        sheet = np.concatenate(views, axis=1)
        imageio.save_rgba(path, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,), dtype=sheet.dtype)], axis=2))
    finally:
        for obj, mats in saved:
            obj.data.materials.clear()
            for m in mats:
                obj.data.materials.append(m)
        for m in temp:
            bpy.data.materials.remove(m)
    return str(path)


def measure_extras(inp: Inputs, cfg: PipelineConfig, cmp: Mapping[str, Any], report: dict[str, Any], out: Path) -> None:
    """Measurements that look past the in-sample silhouette error: per-object misattribution and camera-facing occlusion,
    per-part IoU, held-out views and the turntable sheet. Each block is independent; a failure is recorded as
    ``{"error": ...}`` in its own block and never fails the run. Soft: only adds ``warnings``."""
    steps = (("attribution", lambda: attribution.analyze(inp, cfg)),
             ("part_iou", lambda: projection.compare_parts(inp.view_spec, inp.scene, inp.calibs, inp.masks, cmp["out_masks"])),
             ("holdout", lambda: projection.compare_holdout(inp.view_spec, cfg, out)),
             ("face_detail", lambda: facedetail.analyze(inp, out)))
    for name, fn in steps:
        try:
            report[name] = fn()
            if name == "part_iou" and (out / "partfit_report.json").is_file():
                merge_label_iou(report[name], json.loads((out / "partfit_report.json").read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 - a metric must never turn a finished reconstruction into an error
            log.exception("%s measurement failed", name)
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    outputs = report.setdefault("outputs", {})
    try:
        outputs["turntable_preview"] = str(out / "previews" / TURNTABLE_PREVIEW)
        render.render_turntable(outputs["turntable_preview"])
    except Exception as exc:  # noqa: BLE001
        log.exception("turntable failed")
        outputs["turntable_preview"] = f"error: {type(exc).__name__}: {exc}"
    details = attribution_core.warning_details(report.get("attribution"), report.get("part_iou"))
    details += facedetail_core.warnings(report.get("face_detail"))
    if isinstance(report.get("face_detail"), dict) and report["face_detail"].get("closeup"):
        outputs["face_closeup_preview"] = report["face_detail"]["closeup"]
    details += [f"part_gap {g['a']}~{g['b']} ({g['gap_m'] * 1000:.0f} mm apart, tolerance {g['tol_m'] * 1000:.0f} mm; pin with params.joint)"
                for g in report.get("part_gaps") or []]
    details += [f"floating_part {g['part']} ({g['gap_m'] * 1000:.0f} mm from {g['nearest']}, tolerance {g['tol_m'] * 1000:.0f} mm; attach it)"
                for g in report.get("floating_parts") or []]
    for hid, st in (inp.hull_stats or {}).items():
        gone = [e for e in (st.get("debris") or []) if e.get("dropped")] if isinstance(st, dict) else []
        if gone:
            details.append(f"hull_debris {hid} ({len(gone)} floating fragment{'s' if len(gone) > 1 else ''} dropped, "
                           f"{sum(e['volume_pct'] for e in gone):.2f}% of the hull volume; label a real part so it stays)")
    if details:
        report.setdefault("warnings", []).extend(d.split(" ", 1)[0] + ":" + d.split(" ", 2)[1] for d in details)
        report["warning_details"] = list((report.get("validation") or {}).get("warning_details", [])) + details
        report["recommendation"] = correction.with_warnings(report.get("recommendation", ""), details)


def _stop(report: dict[str, Any], stage: str) -> None:
    report["status"] = f"stopped_after_{stage}"
    log.info("stopped after %s (preview %s)", stage, report["outputs"][f"stage_{stage}_preview"])


def _correction_loop(inp: Inputs, cfg: PipelineConfig, cmp: dict[str, Any], iterations: list[dict[str, Any]],
                     out: Path, rec: dict[str, Any]) -> dict[str, Any]:
    """Axis-scale correction: compare -> factors -> rebuild -> re-render; stop at threshold / no improvement."""
    correctable = not any(c.type == HULL_TYPE for c in inp.scene.components) and any(c.size for c in inp.scene.components)
    if not cfg.auto_correct or not correctable:
        rec["note"] = "auto_correct disabled" if not cfg.auto_correct else "silhouette hull scenes: the hull and its parts are not axis-scaled"
        return cmp
    for i in range(1, cfg.max_iterations + 1):
        total = cmp["aggregate"]["total_error"]
        if total <= cfg.error_threshold:
            rec["note"] = f"error {total:.4f} <= threshold {cfg.error_threshold}"
            break
        factors = axis_scale_factors(cmp["results"])
        if all(abs(f - 1.0) < MIN_CORRECTION for f in factors.values()):
            rec["note"] = "axis scale factors are ~1: remaining error is shape, not proportion"
            break
        for issue in diagnose(cmp["results"], inp.calibs)[:3]:
            log.info("diagnosis: %s", issue["recommendation"])
        previous = inp.scene_dict
        inp.set_scene(apply_axis_scale(previous, factors))
        rebuild_all(inp, cfg)
        candidate = compare(inp, cfg)
        new_total = candidate["aggregate"]["total_error"]
        accepted = new_total < total - MIN_IMPROVEMENT
        iterations.append({"iteration": i, "total_error": new_total, "per_view": _per_view_errors(candidate),
                           "factors": factors, "accepted": accepted})
        log.info("correction %d: total error %.4f -> %.4f (%s)", i, total, new_total,
                 "accepted" if accepted else "rejected")
        if not accepted:
            inp.set_scene(previous)
            rebuild_all(inp, cfg)
            rec["note"] = "no improvement; correction reverted"
            break
        cmp = candidate
    return cmp
