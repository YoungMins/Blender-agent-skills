# blender-blueprint-reconstruction

A Claude Code skill and Python SDK that rebuilds a 3D model in Blender from front / side / back reference images (blueprints, or AI-generated turnaround sheets of characters and props), then **checks the result with orthographic projection measurements** instead of judging it by eye.

```
Reference images -> ViewSpec -> SceneSpec -> deterministic bbr builders -> 3D model
  -> orthographic renders -> reference comparison (IoU, floor, per-part, occlusion) -> fix loop -> report
```

The agent decides what the parts are (which, labels, relations). The tools do the metric work: calibration, pixel measuring, triangulation, part fitting, rendering, validation.

Two tracks:

- **Hard-surface** (furniture, products, machines): one component per part (`box`, `rounded_box`, `cylinder`, `sphere`, `capsule`, `leg`, `panel`, `plate`, `frame`) with Mirror, Boolean, Bevel and Weighted Normal.
- **Character / organic**: a `silhouette_hull` body (signed-distance visual hull, Surface Nets, voxel cleanup, retopo, UV unwrap, per-texel colour projection into a base-colour texture), plus labelled primitive parts, all textured by default in a character scene, (`cone`, `blade`, `disc`, `gem`, `capsule`, `sphere`, `cylinder`) that are triangulated from pixel anchors and auto-fitted to their label masks. Optional extras: a `layer_hull` for recessed faces, and a mirrored virtual colour view for symmetric characters.

The agent instructions are in [SKILL.md](SKILL.md); details are in `references/`.

## Requirements

- Blender 5.1+ (tested with 5.1.1). It uses its bundled Python and numpy; nothing is installed into Blender.
- Python 3.10+ with numpy and Pillow for `scripts/prepare_references.py` and the tests; pytest to run the tests. jsonschema is optional (a built-in validator is used without it).

## Install

Copy or link this folder to `~/.claude/skills/blender-blueprint-reconstruction` (user level) or `.claude/skills/` (project level).

## Quick start

```bash
B="/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"   # adjust
PY=python                                                         # python3 on Linux/macOS

# Hard-surface example with known ground truth
"$B" -b --factory-startup -P scripts/run_pipeline.py -- \
  --view-spec examples/simple_table/view_spec.json --scene-spec examples/simple_table/scene_spec.json \
  --out examples/simple_table/output

# Character example: hull plus labelled, anchor-placed parts
"$B" -b --factory-startup -P scripts/run_pipeline.py -- \
  --view-spec examples/sd_warrior/view_spec.json --scene-spec examples/sd_warrior/scene_spec.json \
  --out examples/sd_warrior/output

# Your own turnaround sheet: split, draft the view spec, check the views agree
$PY scripts/prepare_references.py split-sheet --image sheet.png --out-dir work/char/references --names front,side,back
$PY scripts/prepare_references.py draft-view-spec --dir work/char/references --height 0.6   # complete: axes, roles, background, chroma, symmetry.x; or --width W / --depth D [--size-of body|silhouette]
$PY scripts/prepare_references.py check --view-spec work/char/references/view_spec.json
```

`prepare_references.py` subcommands: `split-sheet`, `masks`, `draft-view-spec`, `check`, `measure`, `propose`, `labels`, `anchors`. Run `$PY scripts/prepare_references.py <subcommand> --help` for flags. `check --ignore-flag/--roles-override ... --write` records a false-alarm judgement in the view spec's `consistency` block (read by `check` and the pipeline). `anchors` picks sphere / capsule / cylinder / cone (`suggested_type`) and warns when a `_l`/`_r` label is on the wrong side (`_l` = the character's left = +X, see [coordinate system](references/coordinate-system.md)). A run takes 13 s-5 min depending on the track (a flat-colour hard-surface kit about 15 s, the textured chest about 1 min, a character with parts 3-5 min): run it in the background or with a long timeout. A ready prompt for generating a consistent turnaround sheet is in [references/turnaround-prompt.md](references/turnaround-prompt.md).

Review in the Blender UI: `"$B" <out>/<name>.blend --python scripts/open_result.py`.

**Exit codes:** 0 validation passed, 2 ran but did not pass (the `recommendation` and the `validation:` line name the failed check, e.g. `FAILED: dimensions (width -9.7% vs tolerance ±3%)`), 1 error.

**`run_pipeline.py` options:**
- `--stop-after blockout|primary|secondary|final` stops at an approval stage and writes its preview (`primary` = hull body only, `secondary` = body plus parts).
- `--verbose` also prints Blender's own per-render `Saved:` lines (quiet by default); the `===== BBR RESULT` ... `===== END BBR RESULT` block is the last thing printed.
- `--blend FILE` re-runs on an existing scene, updating objects in place.
- `--max-iterations N` overrides the correction loop limit (hard-surface).
- `--force` continues past a failed blockout or strict consistency gate.
- `--consistency-gate warn|strict` overrides `workflow.consistency_gate`.
- `--render-scale F` renders comparison masks at a multiple of the reference resolution.
- `--make-synthetic-references` renders the SceneSpec into the ViewSpec image paths and exits (how the synthetic fixtures were made).

## Outputs

In the `--out` directory:

| File | Content |
|---|---|
| `report.json` | status, per-view error / floor / excess, consistency, blockout, part IoU, attribution (occlusion, misattribution), held-out views, hull and texture statistics, warnings, recommendation |
| `comparison_sheet.png`, `compare_<view>.png` | reference, render mask and overlay per view (green both, red missing, blue excess) |
| `previews/` | `perspective.png`, `perspective_back.png`, `top.png`, `front_lit.png`, `side_lit.png`, `turntable.png` (8 azimuths), `stage_A/B/C_*.png` |
| `textures/` | `<name>_basecolor.png` per textured object (also packed into the `.blend`), `<name>_source_map.png` (which view each texel came from) |
| `hull_carve_masks/` | carve masks after part exclusion |
| `placement.json`, `partfit_report.json`, `fitted_parts.json` | triangulated and fitted part poses with per-view IoU |
| `<name>.blend`, `checkpoints/`, `scene_spec_corrected.json` | the model, checkpoints, the spec as corrected |

## Bench

`bench/` measures the whole skill against ground truth.

- `$PY bench/run_bench.py [--cases chibi,sd_warrior] [--no-variants] [--jobs 2] [--blender PATH] [--out DIR] [--score-only RUN_DIR]` runs the synthetic ground-truth cases (built by `bench/make_synthetic.py`, with `__scale2` and `__noacc` perturbed variants), plus `sd_warrior` and `robot_character`, and writes `summary.md` under `bench/runs/<timestamp>/`. Past run directories are archived outside the package (`.omc/archive/bench_runs/` in the parent repo) and are regenerated by `bench/run_bench.py`.
- `bench/score.py --case DIR --run DIR` scores one run (excess over floor, held-out IoU, F-score and Chamfer against the ground-truth mesh, colour PSNR, flags).
- bench: `ring %` column and `ring:<id>` flag (`core/ringmetric.py`); regression case `sd_fox_blind`.
- [`bench/BASELINE.md`](bench/BASELINE.md) is the before/after table of the hull-only synthetic cases (held-out IoU, F-score, Chamfer) with the root causes of the disconnected-silhouette fix (`island_min_frac`).
- `bench/blind_run.md` is the protocol for a blind run: a fresh agent gets only the skill package and the images, with a fixed run and time budget.

## Blender MCP integration

No dependency on a particular MCP server: any server that can run Python inside Blender works. The agent sends one-line semantic calls, not modelling code:

```python
import sys; sys.path.insert(0, r"D:/path/to/blender-blueprint-reconstruction/scripts")
from bbr import api
api.run_reconstruction(r".../view_spec.json", r".../scene_spec.json", r".../output")
```

| Semantic tool | Function |
|---|---|
| inspect_scene | `api.inspect_scene()` |
| setup_reference_views | `api.setup_reference_views(view_spec_path)` |
| build_from_scene_spec | `api.build_from_scene_spec(scene_spec_path, view_spec_path=None)` |
| update_component | `api.update_component(scene_spec_path, component_id, changes, view_spec_path=None)` |
| measure_reference | `api.measure_reference(view_spec_path, view, points=None, bbox=None)`: pixels to world metres |
| render_reference_views | `api.render_reference_views(view_spec_path, out_dir, scene_spec_path=None)` |
| validate_reconstruction | `api.validate_reconstruction(view_spec_path, scene_spec_path, out_dir)` |
| audit_reconstruction | `api.audit_reconstruction(view_spec_path, scene_spec_path, out_dir)`: occlusion, misattribution, part IoU, held-out views, turntable |
| save_checkpoint / restore_checkpoint | `api.save_checkpoint(name, out_dir)` / `api.restore_checkpoint(name, out_dir)` |
| run_reconstruction | `api.run_reconstruction(view_spec_path, scene_spec_path, out_dir, max_iterations=None, stop_after=None, force=False, render_scale=None)` |

Every function returns a JSON-able dict with `ok` and none of them raise.

## Tests

```bash
$PY -m pytest -q tests                                          # pure Python, no Blender
"$B" -b --factory-startup -P tests/blender/test_blender_smoke.py   # inside Blender
```

## Layout

| Path | Role |
|---|---|
| `SKILL.md` | Agent contract: tracks, workflow with exact commands, hard rules |
| `references/` | Workflow, validation rules, turnaround prompt, input image guidelines, coordinate system, hard-surface rules |
| `schemas/` | JSON Schemas for ViewSpec and SceneSpec |
| `scripts/prepare_references.py` | Reference preparation CLI (see subcommands above) |
| `scripts/run_pipeline.py` | Headless pipeline CLI (run inside Blender) |
| `scripts/open_result.py` | Open a result in the Blender UI |
| `scripts/bbr/core/` | Pure Python and numpy: specs, calibration, masks, metrics, consistency, labels, triangulation, part fitting, carving, surface extraction, colour projection |
| `scripts/bbr/blender/` | bpy SDK: scene, references, primitives, components registry, hull and post-processing, part textures, rendering, validation, attribution, checkpoints |
| `scripts/bbr/api.py` | Semantic tool interface for MCP |
| `examples/` | `simple_table`, `simple_chair` (synthetic, with perturbed specs), `treasure_chest` (hard-surface: arch / torus kit, projected colour), `mushroom_house` (hard-surface: measured `lathe` bodies, `attach_at` door and windows, `_chimney` label, `color_views`), `robot_character` (hull only), `sd_warrior` (labels, anchors, parts, face layer), `sd_astronaut` (face layer with relief, clear glass visor shell, antenna parts) |
| `bench/` | Benchmark harness and blind-run protocol |
| `tests/` | pytest suites and the Blender smoke test |

`examples/*/output/` are intentional demo outputs (about 90 MB in total). They can be regenerated with the pipeline and may be omitted to ship a smaller package.

## Licenses

See `THIRD_PARTY_NOTICES.md` (hull post-processing adapted from EL4CTEO/blender-skills, MIT).
