# blender-blueprint-reconstruction — Implementation Contracts (v0.1)

Source requirements: `D:\Git\skills\blender-agent-skills\prompt.md` (read it fully).
Skill root: `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\` (abbrev. `SK/`).
Do NOT modify `prompt.md` or anything under `generated-images/`.

Environment facts (verified):
- Blender 5.1.1 at `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`; bundled Python 3.13 + numpy 2.3.
  Blender Python has NO PIL, NO jsonschema, NO scipy/cv2.
- Headless renders work: `BLENDER_WORKBENCH` (0.3s), `CYCLES`, `BLENDER_EEVEE`.
- System Python 3.13 (`python3`) has numpy, PIL, jsonschema, pytest.
- Shell is Git Bash on Windows. Run Blender headless:
  `"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" -b --factory-startup -P SK/scripts/run_pipeline.py -- <args>`

## 0. Principles (from prompt.md, non-negotiable)
Reference fidelity first; agent freedom minimal; deterministic functions; intermediate specs (ViewSpec, SceneSpec);
validation mandatory; re-run safe (find→update by `agent_id`); stage-wise editable; extensible (component registry, GN later).
Code rules: type hints, docstrings, `logging` (logger name `bbr.<module>`), clear errors (`BBRError` subclasses), no magic
numbers (module-level named constants / config dataclass), prefer `bpy.data`/`bmesh`/modifier API over `bpy.ops`.

## 1. Layout
```
SK/
  SKILL.md, README.md                      (written later by orchestrator — do not write)
  schemas/view_spec.schema.json            (JSON Schema draft 2020-12)
  schemas/scene_spec.schema.json
  scripts/
    run_pipeline.py        CLI for `blender -b -P` (Blender side)
    prepare_references.py  CLI, plain python3 (PIL): split sheet, extract masks, draft view_spec
    bbr/__init__.py        version, BBRError hierarchy, get_logger
    bbr/core/              PURE PYTHON + numpy ONLY (must import with no bpy, no PIL at import time)
      __init__.py
      config.py            dataclass PipelineConfig (thresholds, iterations, resolutions) + load from dict
      spec.py              load/validate ViewSpec & SceneSpec, defaults, minimal built-in validator
      conventions.py       axis conventions per view (THE single source of truth, see §2)
      imageio.py           load_rgba(path)->np.float32 HxWx4 top-down (PIL if importable, else bpy), save_rgba
      imageops.py          masks, bbox, crop, resize, sheet splitting, morphology (numpy only)
      calibration.py       ViewCalibration, resolve_dimensions, pixel<->world
      metrics.py           IoU, bbox deltas, edge diff, symmetry, per-view + total error
      correction.py        error -> axis/component diagnosis, axis scale correction of SceneSpec
      carve.py             visual-hull voxel carving + per-view depth/visibility (numpy)
      colorproj.py         per-vertex color projection math (numpy) given vertices/normals/occupancy
    bbr/blender/           requires bpy
      __init__.py
      scene.py             collections REF_BLUEPRINT / AI_MODEL / AI_DEBUG, find_or_create by agent_id, origin empty
      reference.py         setup_reference_views(view_spec) -> image planes
      primitive.py         create_box, create_rounded_box, create_cylinder, create_sphere, create_capsule ... (bmesh)
      mesh.py              mesh from voxels / from arrays, set mesh data in place, normals cleanup
      modifiers.py         ensure_modifier(obj, name, type, **props) idempotent; mirror/bevel/weighted normal stack
      components.py        COMPONENT_REGISTRY: type -> builder(spec_component, ctx) ; semantic: leg, panel, plate, frame...
      material.py          ensure_material (principled color / color-attribute material / reference image material)
      camera.py            ensure_ortho_camera(view, calibration) exact framing; perspective 3/4 preview camera
      render.py            render_mask(view) / render_color(view) / render_perspective(); engine setup, visibility
      projection.py        compare reference vs render per view, comparison sheet image writer
      validation.py        validate_* functions + validate_reconstruction()
      checkpoint.py        save_checkpoint / restore_checkpoint / list_checkpoints
      hull.py              build silhouette hull component (uses core.carve + core.colorproj)
      reconstruction.py    run_reconstruction(): the 16-step workflow + correction loop + report
    bbr/api.py             semantic tool interface (JSON-able in/out), for MCP
  examples/
    simple_table/scene_spec.json, view_spec.json, references/ (generated synthetic refs)
    simple_chair/scene_spec.json, view_spec.json, references/
    robot_character/view_spec.json, scene_spec.json (orchestrator fills references)
  tests/                   pytest, plain python3, no Blender
    conftest.py (adds SK/scripts to sys.path)
    test_view_spec.py, test_scene_spec.py, test_geometry_utils.py (+ test_metrics.py, test_carve.py, test_imageops.py)
    blender/test_blender_smoke.py  (runs inside blender: `blender -b --factory-startup -P tests/blender/test_blender_smoke.py`)
  references/  (docs — written later by orchestrator; coordinate-system.md must match §2 exactly)
```
Any Python file meant to be run inside Blender must insert `SK/scripts` into `sys.path` based on `__file__`.

## 2. Coordinate system (single source of truth = core/conventions.py)
Blender world, meters, Z up. The object's FRONT faces **-Y**. Object's own right side is at **-X**.
World origin = object footprint center at floor: X/Y bbox center = 0, lowest point Z = 0.

Each view is defined by `camera_axis` = the world-axis direction FROM the origin TO the camera.
Image coords: u to the right, v DOWN (pixel row 0 = top). Mapping (image_right, image_up) in world:

| view key (default axis) | camera_axis | camera looks along | image_right | image_up |
|---|---|---|---|---|
| front  | -Y | +Y | +X | +Z |
| back   | +Y | -Y | -X | +Z |
| right  (camera at +X, Blender numpad 3) | +X | -X | +Y | +Z |
| left   (camera at -X, Blender ctrl+numpad 3) | -X | +X | -Y | +Z |
| top    | +Z | -Z | +X | +Y |
| bottom | -Z | +Z | +X | -Y |

`side` is allowed as a view key but REQUIRES explicit `camera_axis` (+X or -X).
Rule for agents: in a profile image, if the object's front/face points to image-RIGHT → camera_axis `-X`; image-LEFT → `+X`.
`flip_horizontal: true` mirrors the image before use (for references known to be mirrored).
Extra views (`three_quarter_front`, `perspective`, `detail_*`) are accepted in the spec with `role: ["preview"]` only; they are
not used for carving/validation in v0.1 (store but skip with a log message).

`conventions.py` API:
```python
AXES: dict[str, np.ndarray]  # "+X" -> (1,0,0) ...
DEFAULT_CAMERA_AXIS: dict[str, str]  # front:-Y back:+Y right:+X left:-X top:+Z bottom:-Z
@dataclass(frozen=True) class ViewFrame: camera_axis: str; right: np.ndarray; up: np.ndarray; forward: np.ndarray  # forward = camera look dir = -axis
def view_frame(camera_axis: str) -> ViewFrame
def axis_index(axis: str) -> tuple[int, int]   # "-Y" -> (1, -1)
```

## 3. ViewSpec (schemas/view_spec.schema.json)
```json
{
  "spec_version": "0.1",
  "name": "robot_character",
  "base_dir": null,                       // paths relative to the spec file's dir unless absolute
  "views": {
    "front": {"path": "references/front.png", "enabled": true, "camera_axis": "-Y",
               "flip_horizontal": false, "crop_px": null,            // [x0,y0,x1,y1] optional
               "roles": ["carve", "color", "validate"],            // subset of carve|color|validate|preview
               "background": {"mode": "auto", "color": null, "tolerance": 0.12}, // auto|alpha|color
               "calibration": {"mode": "silhouette_bbox", "px_per_meter": null, "origin_px": null},
               "camera_type": "orthographic", "confidence": "observed", "notes": ""}
  },
  "known_dimensions": {"width": null, "depth": null, "height": 0.9},   // meters, X/Y/Z extents
  "symmetry": {"x": true, "y": false, "z": false},
  "reference_quality": {"orthographic": true, "notes": ""}
}
```
Defaults: enabled=true, camera_axis from key, roles: front/back/side/left/right → ["carve","color","validate"] except back →
["color","validate"]; top/bottom → ["validate"]; background auto tol 0.12; calibration silhouette_bbox.
At least one enabled view required. Unknown view keys allowed only with explicit camera_axis or role preview.
`spec.py`: `load_view_spec(path) -> ViewSpec` (dataclasses, resolved absolute paths, defaults applied),
`validate_view_spec(dict) -> list[str]` (errors; uses jsonschema if importable else built-in checks covering required keys,
enums, types, ranges). Same for scene spec. Raise `SpecError` on load if errors.

## 4. SceneSpec (schemas/scene_spec.schema.json)
```json
{
  "spec_version": "0.1",
  "name": "simple_table",
  "category": "furniture",            // furniture|product|mechanical|prop|hard_surface|character|other
  "object_type": "table",
  "units": "meters",
  "dimensions": {"width": 1.2, "depth": 0.6, "height": 0.72},  // target overall bbox; null allowed (then from views)
  "symmetry": {"x": true, "y": true, "z": false},
  "style": {"hard_surface": true, "bevel": true, "bevel_width": 0.003, "bevel_segments": 2, "weighted_normals": true},
  "materials": {"wood": {"base_color": [0.55,0.38,0.22,1.0], "roughness": 0.6}},
  "components": [
    {"id": "top", "type": "rounded_box", "stage": "primary", "confidence": "observed",
     "location": [0,0,0.7025], "rotation_deg": [0,0,0], "size": [1.2,0.6,0.035],
     "params": {"radius": 0.01, "segments": 3}, "material": "wood",
     "mirror": {"x": false, "y": false}, "bevel": null},
    {"id": "leg_fl", "type": "box", "stage": "primary", "confidence": "observed",
     "location": [0.555,-0.255,0.3425], "size": [0.05,0.05,0.685], "mirror": {"x": true, "y": true}, "material": "wood"}
  ],
  "workflow": {"max_iterations": 3, "error_threshold": 0.05, "blockout_threshold": 0.12, "auto_correct": true}
}
```
Component `stage`: blockout|primary|secondary|detail (build order; blockout first). `location` = component center (world).
`size` = full extents [x,y,z] in local axes before rotation. Mirror = Mirror modifier using the AI_ORIGIN empty at world origin.
Component types for v0.1 (registry, extensible): `box`, `rounded_box`, `cylinder` (params: axis "X|Y|Z", vertices),
`sphere` (size = ellipsoid extents), `capsule` (axis), `leg` (= box/cylinder by params.profile "square|round", optional taper),
`panel`/`plate` (= box with thin axis), `frame` (rectangular tube frame: params thickness), `silhouette_hull` (§7).
Each created object gets custom props: `agent_id` = `<scene name>:<component id>`, `reconstruction_confidence`,
`bbr_component_type`, `bbr_stage`. Re-run = find by agent_id and rebuild mesh data in place (object identity kept).

## 5. Calibration (core/calibration.py)
```python
@dataclass class ViewCalibration:
    view: str; camera_axis: str; image_size: tuple[int,int]  # (W,H) after crop
    m_per_px: float; origin_px: tuple[float,float]           # pixel (u,v) where world origin projects (u right, v down)
    def world_to_px(self, pts: np.ndarray) -> np.ndarray     # (N,3)->(N,2) float u,v
    def px_to_plane(self, uv: np.ndarray) -> np.ndarray      # (N,2)->(N,3) world points on the plane through origin
    def image_extent_m(self) -> tuple[float,float]
def resolve_dimensions(bboxes_px: dict[str, tuple], axes: dict[str,str], known: dict) -> tuple[dict, dict[str,str]]
    # returns ({"width","depth","height"} meters, provenance {"width": "known"|"derived:front"|"default"})
    # front/back give width/height ratio, left/right/side give depth/height, top gives width/depth.
    # If nothing known: height default 1.0 (provenance "default", log WARNING).
def calibrate_views(view_spec, masks: dict[str, np.ndarray]) -> tuple[dict[str, ViewCalibration], dict]
    # silhouette_bbox mode: m_per_px from the view's vertical extent (Z views) or horizontal extent (top/bottom) vs resolved dims;
    # origin_px: horizontal = bbox center; vertical = bbox bottom (for Z-up views); top/bottom views: bbox center both.
    # manual mode: uses given px_per_meter/origin_px.
```
Pixel centers: pixel (i,j) center is at u=j+0.5, v=i+0.5.

## 6. Image ops / metrics (pure numpy)
imageops: `extract_mask(rgba, mode="auto", color=None, tol=0.12) -> bool HxW`:
  alpha mode if image has meaningful alpha (any alpha<0.5); else background color = median of border pixels;
  mask = NOT(background region), where background region = flood fill (4-connected, iterative numpy/BFS) from all border pixels over
  pixels whose RGB distance to bg color < tol; then fill enclosed holes smaller than `max_hole_frac` (default 0.002 of image area);
  keep only components larger than `min_component_frac` (default 0.0005) — remove specks.
`mask_bbox(mask) -> (x0,y0,x1,y1)` exclusive end; `crop`, `pad_to`, `resize_nearest(mask, (W,H))`, `resize_bilinear(rgb,(W,H))`,
`split_sheet(rgba, n_views, bg_tol) -> list[(x0,y0,x1,y1)]` (columns separated by background gaps; pick the n widest groups;
return crops padded by margin, all crops SAME height = full image height so vertical alignment is preserved),
`dilate/erode(mask, r)`, `mask_edges(mask)`.
metrics: `iou(a,b)`, `bbox_delta_px(ref, out)` -> dict left/right/top/bottom (px, signed: positive = output extends beyond ref),
`edge_distance(ref, out)` mean symmetric boundary distance in px (chamfer via distance transform approximation by repeated
dilation up to cap), `symmetry_score(mask, axis_u_px)`, `compare_view(ref_mask, out_mask, calib) -> ViewComparison`
(dataclass: iou, error=1-iou, bbox_delta_m, width_ratio, height_ratio, edge_px, largest_mismatch_region (bbox px + world center)),
`aggregate(view_results, weights) -> {"per_view":..., "total_error":..., "passed":...}`.
Error % reported like `Front Error : 4.3%`.

## 7. Silhouette hull (characters / organic / any object)  — core/carve.py + blender/hull.py
`carve_visual_hull(masks: dict[str,np.ndarray], calibs: dict[str,ViewCalibration], dims, resolution:int, margin=0.03,
   carve_views: list[str]) -> VoxelGrid` (occupancy bool [nx,ny,nz], origin (min corner world), cell size).
For each carve view, project voxel centers to pixels (vectorize: front/back depend on (x,z), side views on (y,z), top on (x,y) —
build 2D lookup per view then broadcast), occupied = AND over views; outside image = empty. Keep the largest 26-connected
component (or all components above min size). `depth_visibility(grid, camera_axis) -> np.ndarray` first-hit index along view dir
for occlusion tests.
Top view participates in carving only if listed AND its bbox aspect agrees with width/depth within `top_consistency_tol` (0.15);
else log and use for validation only.
blender/hull.py: build mesh from occupancy: boundary quads (numpy face extraction, shared vertices) → Remesh VOXEL
(voxel_size = cell) applied → smoothing that does not shrink (e.g. Laplacian smooth with `use_normalized`/volume preservation,
or Corrective Smooth with rest = only_smooth off) — choose by measured IoU: the smoothed hull's front & side IoU vs reference
masks must be ≥ 0.95 at resolution 192 on the robot test; validation must run AFTER smoothing.
Color projection (core/colorproj.py): for each final vertex: for each `color` view, weight = max(0, dot(normal, camera_axis_vec))^p
(p=2), sample reference RGB bilinearly at projected pixel only if inside mask AND vertex is visible from that view
(depth test against hull depth map with tolerance 2 cells); weighted average; fallback nearest-view color. Store as color
attribute `bbr_ref_color` (POINT, FLOAT_COLOR); material `M_ref_color` uses Color Attribute → Base Color (emission-free,
roughness 0.6). Parameters in component params: `resolution`, `carve_views`, `smooth_iterations`, `color_projection` bool,
`color_views`.

## 8. Blender side contracts
scene.py: `ensure_collection(name, parent=None)`, `COLL_REF="REF_BLUEPRINT"`, `COLL_MODEL="AI_MODEL"`, `COLL_DEBUG="AI_DEBUG"`,
`find_by_agent_id(agent_id) -> Object|None`, `ensure_object(agent_id, data, collection) -> Object` (find→update else create),
`ensure_origin_empty() -> Object` ("AI_ORIGIN", agent_id "bbr:origin"), `inspect_scene() -> dict` (JSON summary: collections,
objects with agent_id/type/dims/confidence/modifiers, checkpoints), `clear_generated(scope)`.
reference.py: `setup_reference_views(view_spec, calibs) -> dict[str, Object]` — one plane per enabled view in REF_BLUEPRINT,
image material (emission/unlit, alpha blend 0.5 opacity), sized & positioned so each pixel lands on its world position on the plane
through origin, then pushed back along -camera_axis by `ref_plane_offset` (= 0.75 * max dim + 0.1) so it sits behind the model,
`hide_select=True`, `hide_render=True`, custom prop agent_id `ref:<view>`. Idempotent.
camera.py: `ensure_ortho_camera(view, calib, distance) -> Object` in AI_DEBUG, agent_id `cam:<view>`; ortho_scale and position
derived from calib so that rendering at calib.image_size (optionally scaled by `render_scale`) reproduces the reference pixel grid
exactly (sensor_fit AUTO uses larger dimension; handle portrait vs landscape). `ensure_preview_camera()` 3/4 perspective
(front-right-above, frames bbox).
render.py: `render_mask(view, calib, out_path, render_scale) -> np.ndarray bool` (WORKBENCH, film_transparent, only AI_MODEL
visible to render; flat lighting; alpha>0.5), `render_color(view, ..)` (WORKBENCH: color_type ATTRIBUTE if hull else MATERIAL,
studio light, white world), `render_preview(out_path)`. Must restore scene render settings/visibility after (context manager).
projection.py: `compare_all(view_spec, calibs, ref_masks, out_dir) -> dict` renders masks for validate views, metrics, and writes
`compare_<view>.png` (ref | render | overlay: green=both, red=ref only (missing), blue=render only (excess)) plus
`comparison_sheet.png` (all views + 3/4 preview tile). Images saved with bpy (image.pixels / save_render) — no PIL.
validation.py: `validate_dimensions, validate_scale (all AI_MODEL objects scale ~1 or reported), validate_symmetry,
validate_normals (consistent outward; recalculated mesh compare), validate_non_manifold (bmesh), validate_modifiers (order Mirror
→ Boolean → Bevel → WeightedNormal), validate_materials (every mesh has a material), validate_reference_projection,
validate_scene_structure (collections exist, every generated object has agent_id)` each -> dict {"passed": bool, ...};
`validate_reconstruction(...) -> dict` matching prompt.md §18 shape + "recommendation" string from core.correction.
checkpoint.py: `save_checkpoint(name, out_dir) -> path` (bpy.ops.wm.save_as_mainfile(filepath, copy=True)) + JSON sidecar of
specs/report; `restore_checkpoint(name, out_dir)` (wm.open_mainfile); names `checkpoint_00_initial`, `_01_blockout`,
`_02_primary`, `_03_secondary`, `_04_final`.
reconstruction.py: `run_reconstruction(view_spec_path, scene_spec_path, out_dir, config: PipelineConfig) -> dict report`.
Steps (log each as STEP n): 1 import refs (load images, masks) 2 alignment (conventions) 3 scale calibration 4 bounding volume
(debug wire box in AI_DEBUG) 5 blockout (stage=blockout components; if none, a bbox box per spec dims) 6 major proportions
(axis scale correction from validation) 7 symmetry (mirror modifiers) 8 primary 9 secondary 10 boolean/cut (registry hook; v0.1
supports component `boolean: {"target": id, "operation": "DIFFERENCE"}`) 11 bevel 12 normal cleanup 13 reference comparison
14 correction loop (≤ max_iterations; each iteration: compare all → largest mismatch → axis/component → modify → re-render;
stop when total_error ≤ threshold or no improvement) 15 validation 16 save (.blend + report.json + comparison images).
Blockout gate: after step 5/6 compute errors; if blockout error > blockout_threshold record `blockout_gate: failed` and
(unless `--force`) stop before details with report. Approval points: `--stop-after blockout|primary|final` writes stage preview
`previews/stage_<A|B|C>_*.png` and stops.
Report JSON: steps (name, status, duration), dimensions + provenance, calibrations, per-iteration errors, validation dict,
confidence summary (component → observed/inferred/unknown), outputs (paths).
api.py (JSON-able, for MCP): `inspect_scene()`, `setup_reference_views(view_spec_path)`, `build_from_scene_spec(scene_spec_path,
view_spec_path=None)`, `update_component(scene_spec_path, component_id, changes: dict)` (updates the JSON spec file AND rebuilds
that one object), `render_reference_views(view_spec_path, out_dir)`, `validate_reconstruction(view_spec_path, scene_spec_path,
out_dir)`, `save_checkpoint(name, out_dir)`, `restore_checkpoint(name, out_dir)`, `run_reconstruction(...)`. Each returns a dict
with `ok: bool` and never raises to the caller (errors -> {"ok": false, "error": msg}).
run_pipeline.py CLI: `-- --view-spec P --scene-spec P --out DIR [--max-iterations N] [--stop-after S] [--force]
[--render-scale F] [--make-synthetic-references]` (the last: render the scene spec through ortho cameras into
`<view_spec dir>/references/<view>.png` with a solid background — used to create the table/chair test refs). Exit code 0 when
validation passed, 2 when it ran but did not pass, 1 on error. Print a final summary block with per-view error %.

## 9. Tests / acceptance
- pytest (python3) green for tests/ (no Blender).
- Blender smoke test green.
- Table: synthetic refs from ground-truth spec → rebuild → total error < 0.01; perturbed spec (width ×1.25, depth ×0.8) with
  auto_correct → converges below 0.02 within 3 iterations. Same for chair.
- Second run on same .blend creates no duplicate objects (count objects by agent_id).
