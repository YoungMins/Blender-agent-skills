"""Semantic tool interface (JSON-able in/out) for agents / MCP. Every function returns a dict with ``ok`` and
never raises: failures become ``{"ok": false, "error": "<Type>: <message>"}``. Requires Blender (bpy)."""
from __future__ import annotations

import functools
import inspect
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import get_logger

log = get_logger(__name__)

Result = dict[str, Any]

_TEXT_ARGS = ("name", "out_dir", "view_spec_path", "scene_spec_path")   # when given, a non-empty str


def _jsonable(value: Any) -> Any:
    from .blender.reconstruction import _json_default
    return json.loads(json.dumps(value, default=_json_default))


def tool(fn: Callable[..., Result]) -> Callable[..., Result]:
    """Wrap a tool: add ``ok: true`` to dict results, convert any exception into ``{"ok": false, "error": ...}``."""
    params = inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Result:
        try:
            for key, value in inspect.signature(fn).bind(*args, **kwargs).arguments.items():
                optional = params[key].default is None
                if key in _TEXT_ARGS and not (optional and value is None) and not (isinstance(value, str) and value.strip()):
                    return {"ok": False, "error": f"ValueError: {key} must be a non-empty string, got {value!r}"}
            result = fn(*args, **kwargs)
            return _jsonable({"ok": True, **result})
        except Exception as exc:  # tools must never raise to the agent
            log.error("%s failed: %s\n%s", fn.__name__, exc, traceback.format_exc())
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return wrapper


@tool
def inspect_scene() -> Result:
    """Summary of collections, generated objects (agent_id, type, size, confidence, modifiers) and checkpoints."""
    from .blender import scene
    return {"scene": scene.inspect_scene()}


@tool
def setup_reference_views(view_spec_path: str) -> Result:
    """Create/update the reference image planes (REF_BLUEPRINT) for every calibrated view of a ViewSpec."""
    from .blender import reconstruction as rc, reference, scene
    from .core.config import PipelineConfig
    from .core.spec import load_view_spec
    cfg = PipelineConfig()
    view_spec = load_view_spec(view_spec_path)
    rgba, masks = rc.load_view_data(view_spec, cfg)
    from .core.calibration import calibrate_views
    calibs, info = calibrate_views(view_spec, masks)
    scene.ensure_structure()
    planes = reference.setup_reference_views(view_spec, calibs, info["dimensions"], cfg, rgba)
    return {"planes": sorted(planes), "dimensions": info["dimensions"], "provenance": info["provenance"]}


@tool
def build_from_scene_spec(scene_spec_path: str, view_spec_path: Optional[str] = None) -> Result:
    """Build (or rebuild in place) every component of a SceneSpec. ``view_spec_path`` is needed for silhouette hulls."""
    from .blender import reconstruction as rc, scene
    inp = rc.load_inputs(view_spec_path, scene_spec_path)
    cfg = rc.resolve_config(None, inp.scene.workflow)
    built = rc.rebuild_all(inp, cfg)
    return {"objects": [scene.find_by_agent_id(inp.scene.agent_id(cid)).name for cid in built],
            "duplicate_agent_ids": scene.duplicate_agent_ids()}


def _merge(target: dict[str, Any], changes: Mapping[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


@tool
def update_component(scene_spec_path: str, component_id: str, changes: dict, view_spec_path: Optional[str] = None
                     ) -> Result:
    """Merge ``changes`` into one component of the SceneSpec JSON file (validated first; the file is only written
    when valid) and rebuild just that object. If the rebuild fails the original file is restored."""
    from .blender import reconstruction as rc
    from .core.spec import validate_scene_spec
    path = Path(scene_spec_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    comp = next((c for c in data.get("components", []) if c.get("id") == component_id), None)
    if comp is None:
        raise KeyError(f"no component with id {component_id!r}")
    _merge(comp, changes)
    errors = validate_scene_spec(data)
    if errors:
        return_error = "; ".join(errors)
        raise ValueError(f"changes make the SceneSpec invalid, file not modified: {return_error}")
    backup = path.read_bytes()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        inp = rc.load_inputs(view_spec_path, path)
        cfg = rc.resolve_config(None, inp.scene.workflow)
        rc.build_stages(inp, cfg, only=component_id)
        rc.apply_booleans(inp)
        rc.apply_bevels(inp, only=component_id)
        rc.apply_normals(inp, only=component_id)
        rc.texture_parts(inp, cfg, only=component_id)
    except Exception:
        path.write_bytes(backup)   # the rebuild failed: the file goes back to what it was
        raise
    return {"component": component_id, "agent_id": inp.scene.agent_id(component_id), "spec_updated": str(path)}


@tool
def measure_reference(view_spec_path: str, view: str, points: Optional[list] = None, bbox: Optional[list] = None
                      ) -> Result:
    """World coordinates of image pixels ``points`` (``[[x, y], ...]``) and/or a pixel ``bbox`` (``[x0, y0, x1, y1]``)
    of a reference view, on that view's plane (two world axes; depth is not measurable), using the pipeline's own
    calibration (silhouette bbox + known dimensions). Use it to place parts from the images instead of guessing."""
    from .core import measure
    return {"measurement": measure.measure_reference(view_spec_path, view, points or [], bbox)}


@tool
def measure_points(view_spec_path: str, points: dict, labels_path: Optional[str] = None) -> Result:
    """Batch :func:`measure_reference`: ``points`` = ``{view: {name: [x, y] | [x0, y0, x1, y1]}}`` (pixels; two numbers a point,
    four a box) -> ``views`` ``{view: {name: metres}}`` plus each view's ``calibration`` (m_per_px, origin_px, pixel -> metre
    formula). ``labels_path`` = the scene's parts_labels, when the pipeline calibrates with labels."""
    from .core import measure
    return {"measurement": measure.measure_points(view_spec_path, points, labels_path=labels_path)}


@tool
def suggest_parts(view_spec_path: str, labels_path: Optional[str] = None, out_dir: Optional[str] = None,
                  regions_dir: Optional[str] = None) -> Result:
    """Part proposals from thin protrusions (ears, horns, tails, hair locks, limbs): per view, mask minus its opening, matched
    across views by z-range, fitted like ``anchors --kind auto``. Returns ``suggestions`` (type, confidence, views),
    ready-to-paste ``labels`` (region ids or polygons), ``components`` (with ``anchors_px``) and ``exclude_components``;
    with ``out_dir`` also ``suggestions.json`` and one numbered overlay per view (``files``)."""
    from .core import suggest
    return suggest.suggest_from_files(view_spec_path, labels_path, regions_dir, out_dir=out_dir)


@tool
def render_reference_views(view_spec_path: str, out_dir: str, scene_spec_path: Optional[str] = None) -> Result:
    """Render the model from every calibrated view camera into ``<out_dir>/renders/<view>.png``.

    Source of the model: ``scene_spec_path`` when given (built first); else the scene already built in this
    session (AI_MODEL has meshes; rendered as is, not rebuilt); else a ``scene_spec.json`` next to the ViewSpec.
    """
    from .blender import reconstruction as rc, render, scene
    from .core import imageio
    from .core.calibration import calibrate_views
    from .core.config import PipelineConfig
    from .core.spec import load_view_spec
    sibling = Path(view_spec_path).resolve().parent / "scene_spec.json"
    spec_path = scene_spec_path or (None if scene.model_meshes() else (sibling if sibling.is_file() else None))
    if spec_path is not None:
        inp = rc.load_inputs(view_spec_path, spec_path)
        rc.rebuild_all(inp, rc.resolve_config(None, inp.scene.workflow))
        calibs, source = inp.calibs, ("scene_spec" if scene_spec_path else "sibling_scene_spec")
    elif scene.model_meshes():
        view_spec = load_view_spec(view_spec_path)
        _, masks = rc.load_view_data(view_spec, PipelineConfig())
        calibs, _ = calibrate_views(view_spec, masks)
        source = "current_scene"
    else:
        raise FileNotFoundError(
            "nothing to render: pass scene_spec_path, build a scene first (build_from_scene_spec), "
            f"or put a scene_spec.json next to {view_spec_path}")
    out = Path(out_dir) / "renders"
    paths = {}
    for key, calib in calibs.items():
        paths[key] = str(out / f"{key}.png")
        imageio.save_rgba(paths[key], render.render_color(key, calib))
    return {"renders": paths, "source": source}


@tool
def validate_reconstruction(view_spec_path: str, scene_spec_path: str, out_dir: str) -> Result:
    """Compare the current model with the references and run all validation checks; writes compare_<view>.png."""
    from .blender import projection, reconstruction as rc, validation
    inp = rc.load_inputs(view_spec_path, scene_spec_path)
    cfg = rc.resolve_config(None, inp.scene.workflow)
    cmp = rc.compare(inp, cfg)
    images = projection.write_comparison_images(inp.view_spec, inp.masks, cmp["out_masks"], out_dir, inp.rgba)
    report = validation.validate_reconstruction(cmp["results"], inp.calibs, inp.dims, inp.scene.symmetry, cfg,
                                                inp.scene_dict, dimension_tolerance=rc.dimension_tolerance(inp, cfg)[0],
                                                dimension_context=rc.dimension_context(inp))
    return {"passed": report["passed"], "report": report, "images": images, "diagnosis": report["diagnosis"],
            "warnings": report["warnings"], "recommendation": report["recommendation"]}


@tool
def audit_reconstruction(view_spec_path: str, scene_spec_path: str, out_dir: str) -> Result:
    """Measurements beyond the silhouette error on the CURRENT scene (not rebuilt): per-object misattribution and
    camera-facing occlusion (``occluded_objects`` = parts embedded in or hidden by others), per-part IoU, held-out
    (``holdout`` role) views, and ``previews/turntable.png`` under ``out_dir``. Soft: nothing here fails."""
    from .blender import reconstruction as rc
    inp = rc.load_inputs(view_spec_path, scene_spec_path)
    cfg = rc.resolve_config(None, inp.scene.workflow)
    cmp = rc.compare(inp, cfg)
    report: dict = {}
    rc.measure_extras(inp, cfg, cmp, report, Path(out_dir))
    return {k: report.get(k) for k in ("attribution", "part_iou", "holdout", "warnings", "warning_details")} | {
        "turntable": report["outputs"].get("turntable_preview")}


@tool
def save_checkpoint(name: str, out_dir: str) -> Result:
    """Save a named checkpoint (.blend + JSON sidecar)."""
    from .blender import checkpoint
    return {"path": str(checkpoint.save_checkpoint(name, out_dir))}


@tool
def restore_checkpoint(name: str, out_dir: str) -> Result:
    """Open a named checkpoint (replaces the current session data)."""
    from .blender import checkpoint
    return {"path": str(checkpoint.restore_checkpoint(name, out_dir))}


@tool
def run_reconstruction(view_spec_path: str, scene_spec_path: str, out_dir: str, max_iterations: Optional[int] = None,
                       stop_after: Optional[str] = None, force: bool = False, render_scale: Optional[float] = None
                       ) -> Result:
    """Run the whole 16-step workflow; returns the report (``passed`` tells whether validation passed)."""
    from .blender import reconstruction as rc
    report = rc.run_reconstruction(view_spec_path, scene_spec_path, out_dir,
                                   overrides={"max_iterations": max_iterations, "render_scale": render_scale},
                                   stop_after=stop_after, force=force)
    return {"passed": report["passed"], "status": report["status"], "report": report}
