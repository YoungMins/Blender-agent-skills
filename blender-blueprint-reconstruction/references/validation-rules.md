# Validation Rules

`validate_reconstruction()` (`scripts/bbr/blender/validation.py`) returns:

```json
{
  "passed": false,
  "geometry": {"passed": true, "checks": {...}},
  "reference_match": {"passed": false, "front_error": 0.032, "side_error": 0.087, "top_error": 0.041, "total_error": 0.05,
                      "per_view": {"side": {"iou": ..., "bbox_delta_m": {...}, "largest_mismatch_region": {...},
                                            "components_in_region": ["rear_support"]}}},
  "warnings": ["non_manifold"], "warning_details": [...],
  "diagnosis": [...],
  "recommendation": "Side view: depth too large by 3.2 cm → Y axis ..."
}
```
In `report.json`, `diagnosis`, `warnings` and `recommendation` are also copied to the top level. When soft warnings exist, `recommendation` names them, e.g. `...; warnings: non_manifold (25 edges)`. A failed run (`status: failed_validation`) starts the `recommendation` and the summary's `validation:` line with `FAILED: <check> (<number vs tolerance>)`, e.g. `FAILED: dimensions (width -3.7% (limit 3%); model extreme set by fin_r (-X side), 14 mm short of the reference edge; consider fit_locked on fin_r position)` (the dimension tolerance per axis is 3% for synthetic / blueprint references, 5% for AI-generated ones (`reference_quality.ai_generated`, views cut from one sheet, `orthographic: false`); set `workflow.dimension_tolerance` to override; the message names the part whose extreme sets the edge) or `FAILED: reference_match (total error 9.1% > threshold 8.0%; side 12.0%)`; a total error under the threshold can still fail on `dimensions`.

## Reference match (projection validation)

The comparison is computed per `validate` view. The render and the reference share one `ViewCalibration`, so their pixel grids are identical.

| Metric | Meaning |
|---|---|
| `iou` and `error = 1 - iou` | Silhouette overlap. This is the primary score. |
| `bbox_delta_m` (left/right/top/bottom) | Signed outline offset in meters. Positive means the model extends beyond the reference. |
| `width_ratio`, `height_ratio` | Model extent divided by reference extent, on the view's image axes |
| `edge_px` | Mean symmetric boundary distance, capped at `edge_distance_cap_px` |
| `largest_mismatch_region` | The biggest connected missing or excess blob, with its world bbox |

- **Total error** is the weighted mean of the per-view errors (`config.view_weights`).
- **Pass**: total error ≤ `workflow.error_threshold`. The default is 0.05. Use about 0.08 for AI-generated references, whose views disagree by a few percent on their own.
- **Blockout gate**: blockout error ≤ `workflow.blockout_threshold` (default 0.12) before any detail is built. It scores the **blockout stage only**: the components with `stage: "blockout"`, built with their Mirror modifiers, against the validate-view silhouettes (`report.blockout.error`, `stage_A_blockout.png`). Booleans, bevel and textures are not applied at that point, so a hidden cutter changes nothing there (a cylinder-plus-cutter lid scores as the full cylinder; an `arch` scores as built); primary / secondary / detail parts are not in it, only in the final comparison. If the spec has no blockout components, a bbox fallback compares the width and height ratios per view instead (`report.blockout.mode = "bbox_fallback"`).

Typical results with this version:

| Case | Expected error |
|---|---|
| Synthetic hard-surface refs | < 1% |
| AI-generated character, front and side (carved) | 1–3% |
| AI-generated character, back (validated only) | 5–10% (generator noise) |

## Reference consistency gate

Opposite orthographic views of one object have mirror-image silhouettes, so how well they mirror bounds how well any model can match them. `scripts/bbr/core/consistency.py` measures it before building. Run it standalone with `prepare_references.py check --view-spec VS [--out-dir D] [--strict] [--json] [--ignore-flag TYPE[:view]] [--roles-override view=role,role]`; it prints the table and writes one XOR overlay PNG per pair (green both, red first view only, blue mirrored second view only).

| Quantity | Meaning |
|---|---|
| Pair mirror IoU (`iou_raw`) | front vs mirrored back, side vs mirrored right/left, on the calibrated world grid. `iou_aligned` is the best scale + shift fit: it shows whether the disagreement is placement or shape. |
| `floor` = 1 - pair IoU | Predicted irreducible error of a validate view. `report.consistency.views.<view>.excess_error = error - floor` is what the reconstruction could still improve. |
| Recommended roles | pair IoU >= 0.95 keeps `carve`; >= 0.92 `color`/`validate` only; otherwise `drop`. In a failed pair only the lower-priority view (front > side > left > right > back) is dropped; its partner is `unverified`. Thresholds live in `ConsistencyThresholds` (calibratable). |
| Flags | `scale_drift` (core height vs the other views > 2%), `vertical_shift` (baseline off by > 1% of the height), `missing_accessory` (a one-sided blob > 1% of the silhouette after rigid alignment), `mirrored_view` (matches better unmirrored: wrong `camera_axis` sign or missing `flip_horizontal`). Core = mask minus thin protrusions (opening at 3% of the height); it also eats legs thinner than that, so for views on one shared scale (`sheet` calibration) `scale_drift` / `vertical_shift` are NOT raised when the full bbox heights agree within 1.5% (bbox heights are printed as `(bbox ...)`). |
| Judged false alarm | One rule: apply the recommended roles unless the XOR overlay shows the views agree; then record it, never silently ignore it: `check --ignore-flag scale_drift[:view] --roles-override back=color,validate --write`, which stores them in the view spec's `consistency` block `{"ignore_flags": [...], "roles_override": {view: roles}}`; `check` and the pipeline read that block, merged with `workflow.consistency` and `run_pipeline.py --ignore-flag` (flags are added, never replaced). Without `--write` the options affect that `check` run only. Ignored flags move to `ignored_flags`, overrides to `overrides`; both are printed and do not count against `passed`. |

Measured: `sd_warrior` front/back 0.945 (floor 5.5%); side/right 0.846 (0.859 even after the best shift and scale; flag `missing_accessory`, the back shield exists in `right` only; recommend dropping `right`); `robot_character` front/back 0.926 (floor 7.4%, back error 6.9%).

Pipeline: the gate runs after calibration, writes `report.consistency` (pairs, floors, recommended roles, flags; every validated view gets `error` and `excess_error`) and prints a "Consistency" block with `error - floor` next to each validate view error. It only **warns**. `consistency_gate: "strict"` (`PipelineConfig.consistency_gate`, `workflow.consistency_gate`) stops the run before the blockout (`status: "consistency_gate_failed"`; `--force` continues) when a view must lose a role or an error flag is raised.

Fix inconsistent references at the source (regenerate one sheet, see [turnaround-prompt.md](turnaround-prompt.md)). `calibration.mode: "sheet"` removes the scale and baseline part of the disagreement for views cut from one sheet; it cannot repair a missing accessory.

## Geometry checks

| Check | Hard or soft | Rule |
|---|---|---|
| `validate_dimensions` | hard | Model bbox within tolerance of the resolved dimensions (3% / 5% by reference quality); a failure names the limiting part. `--stop-after secondary` runs it too (`Dimensions :` line) |
| `validate_scale` | hard | Every generated object has scale (1, 1, 1) |
| `validate_modifiers` | hard | Order is Mirror → Boolean → Bevel → WeightedNormal |
| `validate_materials` | hard | Every mesh has a material |
| `validate_scene_structure` | hard | `REF_BLUEPRINT`, `AI_MODEL` and `AI_DEBUG` exist, and every generated object has an `agent_id` |
| `validate_symmetry` | soft | When the spec claims symmetry, the bbox center offset ratio on that axis |
| `validate_normals` | soft | Consistent outward normals |
| `validate_non_manifold` | soft | Non-manifold edge count (a touching primitive is normal in hard-surface kits) |

Soft checks are reported as `warnings` and do not fail the run.

### Color error (soft, reported only)

For every view with role `color` or `validate` that has a reference image, the model is rendered flat (unlit, so lighting cannot bias it) with its base color / texture on a transparent background at the reference resolution. `color_error` = mean absolute RGB difference (sRGB-encoded, 0..1) inside the intersection of the reference mask and the render mask, eroded by 2 px to ignore silhouette edges. It is stored as `reference_match.per_view.<view>.color_error` and `color_match.per_view.<view>.{error, pixels}` and printed as `Front Color Error : 3.1%`. It does not enter `total_error` or `passed`.

Colour sampling on a chroma backdrop (green screen) is decontaminated first (`spill_suppression`, default true): enclosed backdrop pockets leave the colour mask, pixels within `spill_edge_px` (6) of the edge are un-mixed from the backdrop and their spill channel is clamped; bilinear taps never reach outside the eroded mask. Gray/white backdrops are left untouched.

### Texture statistics (silhouette hull)
`report.json -> hull.<component id>.texture` and `output/textures/<name>_source_map.png`:
- `views.<view>`: share of covered texels whose color comes from that view alone (`mirror_side` = the `mirror_color_views` virtual view); `blend_share`: texels mixed from >= 2 views (the seam band, width `seam_blend_width`); `inpainted_share`: texels no view sees well (hidden, too oblique, near a silhouette edge); these sum to 1. `conflict_ratio`: texels whose candidate views disagree by more than `conflict_threshold` (not blended). `inpainted_uv_share` / `inpainted_3d_share` split the inpainted texels into same-UV-island fill and 3D nearest-surface fill.
- `background_leak_ratio`: share of covered texels within 0.12 RGB distance of a view's backdrop colour or dominated by a chroma backdrop's channel (> 0.15 over both others). Should be ~0 for a character without backdrop-coloured content; a higher value means green-screen spill reached the texture (check `spill_suppression` / `spill_edge_px`, or whether the object really contains that colour). Also printed as `bg-leak` in the texture summary line.
- Source map: front red, side green, back blue, inpainted gray, outside the islands near-black. Large gray patches mark unseen surface.
- Reading it: `inpainted_share` > 30% or a high `conflict_ratio` deserve a look at the previews (`front_lit.png`, `side_lit.png`, `perspective*.png`).
- Lit-preview defects and the usual knobs: streaks where a view faces the surface obliquely -> the best-facing view already wins, so streaks mean no view sees that region well (hull and photo disagree); smear near an outline -> raise `edge_distrust_px`; ghosting where views disagree -> lower `conflict_threshold`; hard seam -> raise `seam_blend_width`.

## Beyond the in-sample silhouette

The silhouette error above is in-sample for a hull: it carves from the same masks it is scored on, and it cannot see a part that is hidden inside another. `run_reconstruction` therefore adds four soft blocks to `report.json` (step 15, after the validation; each is independent, a failure becomes `{"error": ...}` in its own block, none can fail a run). Offending items are appended to `warnings` and `recommendation`.

### Camera-facing occlusion (`report.attribution.objects.<id>.occlusion_share`)
Each AI_MODEL object is sampled on its surface (area-uniform, about 150 k points in total, normals from the evaluated mesh and flipped outward for inverted objects). A sample *faces* a colour view when `normal . camera_axis > 0.3` and is *occluded* when it fails that view's depth test (something else is in front, 3 px tolerance like the texture projection). `occlusion_share` is the share of occluded samples among the facing ones **in the object's best view** (the minimum over the colour views that see enough of it); `best_view` names it. The warning threshold is 0.3 (`occluded_part <id>`).
- A correctly mounted part (a shield on the back, a sword at the hip) is unoccluded in the view that sees it: its inner face legitimately faces a camera the body hides, so the pooled share (`pooled_occlusion`, every facing pair) sits near 0.5 and is NOT a warning basis. An embedded or hidden part is occluded in every view.
- `occlusion_per_view` and `hidden_share` (facing some view but visible in none) are there for diagnosis.
- Measured on the v6 `sd_warrior` parts: shield 0.38 (best view `right`; flagged), both forearms 0.40 / 0.46 (flagged: they are mostly inside the body), body 0.03, every other part 0.0-0.09.

### Misattribution (`report.attribution.objects.<id>.misattribution`)
For each sample the colour view that would paint it is picked with the texture projection itself (same views, depth buffers and settings as the bake); its source pixel is looked up in an **object-ID render** of that view (every AI_MODEL object a flat unique colour, orthographic camera = the view's calibration, no antialiasing). The share of an object's painted samples whose source pixel belongs to a different object is its `misattribution`; `top_confusion` names that object. Warning threshold 0.05 (`misattribution <id>`).
- Limitation: samples are depth-tested before a source view is chosen, so a visible sample's pixel is normally its own object. The metric fires at object boundaries (hand on forearm, sword hilt on belt: snapping, the depth tolerance, ID/z-buffer pixel-coverage differences), NOT for a part buried inside another one (the hull is front-most at those pixels); the occlusion share catches that. Treat it as a boundary-bleed indicator.

### Per-part IoU (`report.part_iou.parts.<id>`, soft)
For `stage: "detail"` components and the part primitives (cone, blade, disc, capsule) each part is rendered alone in every carve / validate view and compared with the reference mask **restricted to the part's padded bbox region**: `iou` (part vs reference there; low is normal because the reference region also holds the body), `precision` (share of the part's pixels on the reference silhouette; below 0.7 the part sticks out of the reference, warning `part_outside_reference`) and `model_iou` (whole model vs reference in the region). A cheap proxy until label masks exist.

**Which number to trust.** For a labelled part trust the **label IoU**: `part_iou.parts.<id>.label_iou` (per view) and `mean_label_iou`, copied from `partfit_report.json` `iou_vis` (the part as built vs its OWN label mask, ignoring pixels over other parts' labels; `iou` there is the strict version). The summary prints it first as `label IoU`, then `on-reference` (precision) and `region IoU` (= `mean_iou`, the proxy above). A thin part (staff, tail) has a region IoU near 10-15% next to a passing label IoU of 55-70%: that is expected, not a defect. A low label IoU (< ~40% in a view that sees the part) means wrong anchors, a wrong label or a bad `kind`; look at the overlay.

### Hard-surface kits: reading the occlusion and misattribution warnings
A kit is many small parts that touch (corner brackets made of two boxes, a plate under a ring, bands over a lid, a shackle half inside a lock). Both warnings fire on exactly that, and mostly do not mean a defect. On `examples/treasure_chest` the run reports `occluded_part` for `ring_plate` (68 %: the ring hangs in front of it), `padlock_shackle` (49 %: its lower half sits inside the lock body by design), the iron bands (48 %: hasp and padlock cover part of them), the bracket arms (50 %), and `misattribution` of 35-50 % between the two boxes of a bracket (flush parts share their pixels), with the status `passed`.
- **Ignore, but say so in the report,** when: the part is visible in the preview / turntable, its `part_iou` precision is >= 0.7 (`part_outside_reference` is absent), and the partner it is occluded by or confused with is the neighbour you modelled it against (`top_confusion`).
- **Act** when: a part you can see in a reference sits at ~100 % occlusion in every view (it is buried inside another: wrong location or size; the smoke test's `embedded` part gives 1.0), or a part that touches nothing has a high `top_confusion`. Merging flush boxes into one component also silences the pair; do it only when they share a material and a stage.
- Hidden boolean cutters are excluded from all of these metrics and from the depth buffers: `scene.model_meshes()` leaves out every object with `bbr_cutter`, and that list feeds the occlusion / misattribution samples, the object-ID renders, the visibility test of the texture projection, the previews and the bbox; `part_iou` and the texture pass skip `boolean` components. (Tested in `tests/blender/test_blender_smoke.py` and `tests/test_hard_surface.py`.)

### Held-out views (`roles: ["holdout"]`, `report.holdout`)
A view with role `holdout` is stored but never used for carving, colour, calibration, consistency or correction (it cannot be combined with other roles except `preview`). It is scored on its own:
- `camera_axis` may be an axis (`"-X"`) or, for holdout / preview views, an azimuth / elevation object `{"azimuth": 45, "elevation": 0}`. Azimuth 0 = front (camera on -Y), 90 = +X, 180 = back (+Y), 270 = -X; elevation in degrees above the horizon. Omitting `roles` for such a view gives `["holdout"]`.
- The model silhouette is rendered orthographically from that direction and fitted to the reference mask (model bbox bottom-centre on the reference bbox bottom-centre, scale 1 +- 5 %); for azimuth / elevation views the azimuth is searched +- 15 degrees in 5 degree steps (an AI 3/4 image is rarely at exactly 45). Reported: `iou` (best fit), `iou_nominal` (at the stated angle), `best_azimuth`, `azimuth_offset_deg`, `best_scale`, `pose_search` (the step sizes) and `color_error` (flat model colour vs the reference inside both masks, the same metric as above). The angle is only as good as the IoU curve is sharp: for rounded characters it is flat and the found angle can sit anywhere inside the range, so do not read more precision into it than the step.
- `holdout_<view>.png` (reference | render | overlay) is written next to the compare images.
- Ortho assumption: a perspective 3/4 image lowers every IoU a little; compare holdout numbers between runs of the same references, not across cases.

### Turntable
`previews/turntable.png`: the lit model from 8 azimuths (0, 45, ... 315; 0 = front, counter-clockwise seen from above; row-major, 4 per row, elevation 15 degrees). Check it for a second shield, a hand on the torso, a missing back: things no silhouette view shows.

## Benchmark harness (`bench/`)

`bench/make_synthetic.py` builds ground-truth character cases (chibi with horns, creature with a tail, hatted figure with a staff) from primitives into `bench/cases/<case>/` (GT SceneSpec, `gt.obj`, front / side / back references, 8 held-out azimuths, a hull-only reconstruction spec) plus perturbed variants `<case>__scale2` (side view 2 % short) and `<case>__noacc` (accessory missing from the side view). `bench/run_bench.py` runs the pipeline on every case (the synthetic ones, `sd_warrior`, `sd_warrior_v6`, `robot_character`, and the hard-surface `treasure_chest`) into `bench/runs/<timestamp>/<case>/` and writes `summary.md`; `bench/score.py` scores one run: training IoU and excess over the reference floor, held-out IoU over 8 azimuths with the exact ground-truth camera, F-score at 1 % / 2 % of the height and normalised Chamfer against the GT mesh, held-out colour PSNR, and the flags above. `bench/blind_run.md` is the protocol for a fresh-agent blind run. [`bench/BASELINE.md`](../bench/BASELINE.md) is the before/after table of the hull-only synthetic cases (train error, held-out IoU, F@2%, Chamfer; before and after the disconnected-island fix) with the root causes found.

## What validation cannot see

- **A front/back mirror caused by a wrong side `camera_axis`.** Always look at `previews/perspective.png`.
- **Concavities invisible in every silhouette.** (Occlusion share and the turntable show parts hidden inside others; neither shows a hollow.)
- **Color accuracy is only a soft metric** (below); it never fails a run. Check the comparison sheet and previews as well.

## Human or agent visual check (mandatory before "done")

Open `comparison_sheet.png` and `previews/perspective*.png` and confirm:
1. The front of the object faces -Y (front preview).
2. No large red (missing) or blue (excess) areas remain.
3. The proportions and major parts read the same as the reference.
