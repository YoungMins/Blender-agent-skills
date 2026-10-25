---
name: blender-blueprint-reconstruction
description: Rebuild a 3D model in Blender from front/side/back reference images and verify it with measurements. Use for blueprints, orthographic views, or an AI-generated turnaround sheet of a character, creature or prop wanting a matching Blender model (textured for characters). Not for free-form modelling without references.
---

# Blender Blueprint Reconstruction

Run from the skill root:

```bash
B="/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
PY=python                  # python3 on Linux/macOS
W=work/mymodel
```

## Principle

You decide semantics (parts, labels, relations); tools do the metric work. "Done" = good numbers and previews you looked at.

## Choose a track

- **Hard-surface**: furniture, props, machines. One component per visible part. Templates `examples/simple_table`, `examples/treasure_chest`, [hard-surface rules](references/hard-surface-rules.md) (type table; `lathe` for caps, vases).
- **Character/organic**: soft outlines, colour texture. A `silhouette_hull` body; templates `examples/robot_character` (hull only), `examples/sd_warrior` (with parts).
  - Hull-only for props disconnected in EVERY carve view; else parts.
  - Labelled parts (thin parts, limbs, wings, accessories): read the [character cheatsheet](references/character-cheatsheet.md) first; details in the [workflow](references/reconstruction-workflow.md).

## Workflow

1. **References.** Use the user's images, or generate ONE sheet ([turnaround prompt](references/turnaround-prompt.md)), never separate views. Split, draft the view spec:
   ```bash
   $PY scripts/prepare_references.py split-sheet --image SHEET.png --out-dir $W/references --names front,side,back
   $PY scripts/prepare_references.py draft-view-spec --dir $W/references --height 0.6 --side-axis -X
   ```
   No `split-sheet` for one PNG per view. Height in metres, or `--width W` / `--depth D` (`--size-of body|silhouette`). Image-right: front `-Y` = +X, back `+Y` = -X; side `-X` (faces image-right) = -Y, `+X` = +Y.
2. **Gate.** Do views agree; do masks have holes?
   ```bash
   $PY scripts/prepare_references.py check --view-spec $W/references/view_spec.json --out-dir $W/check
   $PY scripts/prepare_references.py masks --view-spec $W/references/view_spec.json --out-dir $W/masks
   ```
   Apply the recommended roles unless the XOR overlay shows agreement; then record once (`check --ignore-flag scale_drift --roles-override back=color,validate --write`). One part differs (tail pose): label it `_x` there ([recipe](references/reconstruction-workflow.md)).
3. **Hard-surface only.** Measure every size and position in one batch, pixels in, metres out (also `lathe`, `attach_at`, labels, `color_views`: [rules](references/hard-surface-rules.md)), then step 6:
   ```bash
   $PY scripts/prepare_references.py measure --view-spec $W/references/view_spec.json --points-file pts.json --out metres.json
   ```
   `pts.json` = `{view: {name: [x, y] | [x0, y0, x1, y1]}}`.
4. **Character with parts: propose, label.** Read only `$W/regions/<view>/compact.png` (`--zoom VIEW:x0,y0,x1,y1`), write `$W/labels.json` like `examples/sd_warrior/labels.json` (keep `"_regions": "regions"`). Label every colour view and part. `_l`/`_r` = the CHARACTER's sides: its left is +X, image-RIGHT in the front view, image-left in the back ([diagram](references/coordinate-system.md)); `labels` warns on a mismatch. The side view (camera -X) shows the RIGHT (`_r`) flank; `_l` parts are hidden there: use `mirror_of`. Read `preview_<view>.png`, fix; one-sided parts: `asymmetric_parts`:
   ```bash
   $PY scripts/prepare_references.py propose --view-spec $W/references/view_spec.json --out-dir $W/regions --max-regions 20 --compact
   $PY scripts/prepare_references.py labels --view-spec $W/references/view_spec.json --labels $W/labels.json --out-dir $W/labelcheck
   ```
   - Thin parts (ears, horns, tail, hair): `suggest-parts --view-spec $W/references/view_spec.json --out-dir $W/suggest` gives overlays and paste-ready `labels`, `components`, `exclude_components`.
5. **Character with parts: anchors.** Per part and view, merge pixel anchors into `params.anchors_px`:
   ```bash
   $PY scripts/prepare_references.py anchors --view-spec $W/references/view_spec.json --view front --view side --view back --labels $W/labels.json --part staff --overlay $W/ov.png
   ```
   `--kind auto` prints `suggested_type` ([which type](references/reconstruction-workflow.md); bat wings, ear fins, leaves: `plate`); base = the end touching the most other foreground. Read overlays, `WARNING`s; Twin tails, braids, long locks: `--kind strand --points 6`, never a cone.
   - Under 0.6, or a part inside the body outline: type pixels read off the overlay rulers:
     ```bash
     $PY scripts/prepare_references.py anchors --from-px front:base=118,543 front:tip=147,443 front:radius_px=8 side:base=280,560 --id handle
     ```
     Copy it into the component. A point hidden in a view: omit it there. An unmeasured coordinate comes from the `_l`/`_r` partner, body contact, else the centre line (`inferred`).
   - Parts that meet: `params.joint: {"part": "head", "at": "base", "self": "base"}` pins keypoints; `part_gap` / `floating_part` warn.
6. **Scene spec.** Copy a template to `$W/scene_spec.json` and edit ([schema](schemas/scene_spec.schema.json)): `dimensions.height`, `workflow.error_threshold` 0.08 for AI references, `parts_labels: "labels.json"` when labelled.
7. **Run.** 13 s-5 min; run in background. First `--stop-after secondary` (body plus parts; prints the dimension, floating and gap checks), then without it:
   ```bash
   "$B" -b --factory-startup -P scripts/run_pipeline.py -- --view-spec $W/references/view_spec.json --scene-spec $W/scene_spec.json --out $W/out --stop-after secondary
   "$B" -b --factory-startup -P scripts/run_pipeline.py -- --view-spec $W/references/view_spec.json --scene-spec $W/scene_spec.json --out $W/out
   ```
   Exit 0 passed, 2 failed, 1 error; read the `===== BBR RESULT` block (`next:` says what to do). Hard-surface: `--stop-after blockout` first.
8. **Read results** ([validation rules](references/validation-rules.md); occlusion and misattribution warnings are noisy for kits). Open `out/comparison_sheet.png`, `previews/perspective.png`, `turntable.png`. In `out/report.json` read `status`, `warnings`, `recommendation`, and:
   - `consistency.views.<view>`: `error`, `floor`, `excess_error` (fix only the excess).
   - `part_iou.parts.<id>.mean_label_iou` (vs its own label; `mean_iou` is low for thin parts), `attribution.occluded_objects`, `attribution.misattributed_objects`.
   - `hull.<id>.texture`: `inpainted_share`, `background_leak_ratio`.
9. **Fix loop.** Status `failed`: read `recommendation` and the failing check, fix the biggest error source (label, anchors, `fit_locked`, size, a `floating_part` WARNING), re-run until it passes or the 6-run budget (runs without `--stop-after`) is spent. Never stop while budget remains; two runs without gain: change approach.
10. **Finish.** Report per-view error vs floor, pass or fail, every warning, what is inferred, visible defects, `.blend` and preview paths. Never claim success over a defect.

## Hard rules

1. Reference fidelity beats aesthetics. Add no unseen detail; mark guesses `"confidence": "inferred"`.
2. Never type a 3D value a tool can measure (positions, angles, depths): batch `measure --points-file`, no hand conversion or own script. No raw bpy; edit specs.
3. A wrong profile `camera_axis` mirrors front to back, invisible to IoU: check perspective.
4. Never carve from back, a generated top view or a single generated view.
5. Do not chase the floor.
6. Symmetric character, one side view: hull param `mirror_color_views: ["side"]`, one-sided parts in `asymmetric_parts`.
7. A face behind a visor, glass, mask opening or helmet rim needs a `layer_hull` face layer, else front face and side profile project onto one surface: [cheatsheet](references/character-cheatsheet.md) section 9.
8. Set `material` / `class` per part ([classes](references/hard-surface-rules.md)), or defaults look like plastic.

## Missing or inconsistent data

- Only front and side: build; back/top `unknown`; hidden depth `inferred`.
- Part visible in one view: `mirror_of`, `attach_to` or `depth_m`.
- No dimension: height 1.0 m; say it is relative.
- Views disagree (`check` flags): regenerate the sheet, else carve front+side, report.

[Image guidelines](references/input-image-guidelines.md).
