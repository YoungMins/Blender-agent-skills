# Character with parts: cheatsheet

The character track in one page. Details: [workflow](reconstruction-workflow.md) (54 KB: search it, do not read it whole), [validation](validation-rules.md), [coordinates](coordinate-system.md). Shell setup (`$B`, `$PY`, `$W`) as in SKILL.md.

## 1. Idea

A `silhouette_hull` body is carved from the front + side silhouettes. Everything the hull cannot represent (horns, ears, wings, fins, arms, tails, props) is a labelled part: its label mask is cut out of the carve masks (`exclude_components`), and it is built as a primitive placed from pixel anchors. Parts get a projected texture from the colour views.

## 2. Coordinates and sides

- Image-right in the front view = +X = the character's LEFT (`_l`). Back view: `_l` is image-left.
- The side view (camera -X) shows the character's RIGHT (`_r`) flank. `_l` parts are hidden there: give them `mirror_of` (the `_r` partner) instead of side anchors.
- Pixel anchors only; never type metres. A point hidden in a view is omitted in that view.

## 3. Part types

| shape | type | anchors in `anchors_px.<view>` | params |
|---|---|---|---|
| horn, ear, fang, tapered limb | `cone` | `base`, `mid` (the curve), `tip`, `radius_px` [base, tip] | `flatten` 0.3-0.6 flat |
| long / wavy tail, braid, twin tail, hair lock | `strand` | `p0` (base) ... `pN-1` + `radius_px` | `anchors --kind strand --points 6`; `flatten` |
| arm, leg, staff, finger | `capsule` | `base`, `tip`, `radius_px` | fit tapers it |
| ball, fist, pom-pom, round fin | `sphere` (ellipsoid) | `center`, `radius_px` [h, v] | |
| crystal | `gem` | `base`, `tip`, `radius_px` | `facets`, `waist` |
| brim, shield, coin | `disc` | `center`, `radius_px` (face-on view) | `thickness`, `radius_ratio`, `dome_height` |
| sword, straight flat blade | `blade` | `base`, `tip`, `width_px` | `thickness`, `tip_length` |
| bat wing, ear fin, leaf, flat curved-outline part | `plate` | `base` (root edge), `tip` in 2+ views | `thickness` (required), `face_view`, `bend`, `edge_on_views`. A `plate` with `anchors_px` or `mirror_of` is a contour plate; otherwise it is a `box` alias (hard-surface flat boxes: use `panel`/`box`). |
| spiky ridge / back plates | row of small `cone`s, or one `plate` | see the ridge recipe in the workflow | excluded from the hull |
| layer in a helmet (face, visor) | `layer_hull` | label | `layers` on the hull (section 9) |

`anchors --kind auto` prints `suggested_type`; `--kind plate` gives base / tip only. The table wins over a sloppy label. A `plate`'s outline is its label mask in `face_view`: that view matches the label exactly, the other views fit the tilt (an edge-on line in `edge_on_views` sets it directly).

## 4. Labels (`labels.json`)

1. `propose --compact`, read `regions/<view>/compact.png`, label every colour view: `{"horn_l": [13], "wing_r": [{"polygon": [[x, y], ...]}], "tail": [10, {"split": ...}], "body": []}`.
2. Unlabelled in-mask pixels are `body`. `_name` parts (`_tail_back`) are ignored: no colour, no gating.
3. `labels` checks sides (`_l` = +X), z ranges across views and writes `preview_<view>.png`; fix what it warns about.
4. Thin parts: `suggest-parts` proposes labels, components, `exclude_components`.
5. One-sided parts go in the hull's `asymmetric_parts`.

## 5. Scene spec skeleton

```json
{"parts_labels": "labels.json", "workflow": {"error_threshold": 0.08},
 "components": [
  {"id": "body", "type": "silhouette_hull", "params": {"resolution": 320, "carve_views": ["front", "side"],
     "exclude_components": ["horn_l", "horn_r", "wing_l", "wing_r"], "mirror_color_views": ["side"], "asymmetric_parts": ["tail*"]}},
  {"id": "horn_r", "type": "cone", "params": {"anchors_px": {"front": {"base": [127, 142], "tip": [115, 58], "radius_px": [25, 13]}}}},
  {"id": "horn_l", "type": "cone", "params": {"mirror_of": "horn_r", "anchors_px": {"front": {"base": [310, 142], "tip": [321, 58], "radius_px": [25, 13]}}}},
  {"id": "wing_r", "type": "plate", "params": {"thickness": 0.004, "face_view": "back", "anchors_px": {"back": {"base": [270, 392], "tip": [392, 425]}, "front": {"base": [150, 385], "tip": [52, 418]}}}}]}
```

Joints: `params.joint {"part": "tail", "at": "tip", "self": "base"}`. A part resting on the body: `attach_to: "body"` + `attach_side`. Parts built from `attach_to` are NOT in `exclude_components`.

## 6. Run and fix loop

1. `--stop-after secondary` first (2 min; cheap checks, they do not use the budget): body + parts, the `Dimensions :` check (limit 3 % blueprint / 5 % AI sheets, `workflow.dimension_tolerance`), `floating_part`, `part_gap`, `hull_debris`. Fix those before the full run.
2. Full run = a run without `--stop-after` (5 min, background). Read `BBR RESULT`, `comparison_sheet.png`, `perspective.png`, `turntable.png` (look for floating bits, wrong tilt, unattached fins).
3. Fix the biggest error: label, anchors, `fit_locked`, thickness, then re-run. Two runs without gain: change approach. Budget 6 full runs.

## 7. Common warnings

| warning | meaning / fix |
|---|---|
| `dimensions (width -3.7% (limit 5%); model extreme set by fin_r, 14 mm short ...)` | an outer part was shrunk by the fit: `fit_locked` on its outer position, or fix anchors; hull named = check the silhouette |
| `floating_part X` | gap to every other mesh > max(1 % height, 25 % of its minor size): add anchors in a second view, `joint`, `attach_to`, or `floating_ok` |
| `part_gap A~B` | pin with `params.joint` |
| `hull_debris` | floating hull fragments were dropped (tail tips by the feet); label a real part so it stays, or `drop_debris: false` |
| `views disagree on z by N mm` | an anchor differs between views: re-measure; lock that coordinate |
| `occluded_part` / `misattributed` | noisy for overlapping parts; trust `label_iou` (`part_iou.parts.<id>.mean_label_iou`) |
| `a single part may differ (e.g. tail pose)` | views disagree in one place: recipe "views disagree on one part" (label it `_x` in that view) |
| `inferred` placement | a coordinate no view measures came from the partner / body: add `depth_m`, `mirror_of` or a second view |
| `reference views are inconsistent` | apply the recommended roles unless the XOR overlay shows agreement; record with `check --write` |

## 9. Recipes

**Face behind a visor, glass, mask opening or helmet rim.** Label it `face` in front and side (all that sits inside the opening); the rim stays `body`. Without the layer, the front face and the side profile can land on one surface ("double face").

Labels: `{"front": {"face": [4]}, "side": {"face": [7, 30]}}`. Spec:

```json
{"id": "body", "type": "silhouette_hull", "params": {"layers": ["face"]}},
{"id": "face", "type": "layer_hull", "stage": "secondary", "params": {"label": "face", "views": ["front", "side"], "color_projection": true}}
```

Never list `face` in `exclude_components`. Details: workflow, "Layers".

**Material per part.** Without a class every part renders like plastic. Set `"material"` on the component to a class name (`metal_steel` blade and armour plates, `cloth`, `leather`, `skin`, `hair`, `rubber`, `painted`, `emissive`; armour parts such as horns are not metal unless the reference shows it) or to a `materials` key with `"class"`. The body hull stays the default unless you set one. Table and values: [hard-surface rules](hard-surface-rules.md), "Material classes". Visor recipe: give the face layer `"material": "visor"` (a coat over the face colour; the face stays visible):

```json
{"id": "face", "type": "layer_hull", "stage": "secondary", "material": "visor", "params": {"label": "face", "views": ["front", "side"], "color_projection": true}}
```

**Where `size` goes.** `box`, `rounded_box`, `cylinder` and the other size-based types take `size` on the component (`"size": [x, y, z]` next to `type`), never in `params`. Anchored parts need none. Only `cone capsule cylinder blade plate disc sphere gem strand` can have `anchors_px` (`disc` yes, `rounded_box` and `box` no: validation rejects them).

**Plain box accessory (backpack).** `box` (or `rounded_box`) with `size` from `measure --points-file` (width from the back view, height and depth from the side view; corners as `[x0, y0, x1, y1]`), then either `location` from the same batch, or sit it on the body: `params: {"attach_to": "body", "attach_at": {"view": "back", "px": [cx, cy]}, "face": "+Y", "standoff": 0}` (an `attach_to` part is built after the hull: not in `exclude_components` unless labelled in a carve view).

**Label edits** (`labels.json`, view pixels):

```json
"head": [3, "-5", {"rect": [120, 40, 260, 180], "within": 3}, {"polygon": [[40, 560], [90, 560], [80, 640]]},
         {"split": 2, "line": [[330, 530], [380, 600]], "side": "left"}]
```

`within` (region id or list) applies to `rect` and `polygon`: only the pixels inside those regions. `side` is as seen walking from the first to the second point (`right` is down when going right).

**Limb with very different front and side widths.** `radius_px` is averaged over the views for `cone`, `capsule`, `cylinder` (one round radius). Give it in the view whose width you need and use `cone` / `strand` with `flatten` (thins the depth; the fit searches 0.2-1.5), or an ellipsoid `sphere` (`radius_px [h, v]` per view fills each axis).

**Hull exclusion per part.** `report.json` `hull.body.exclusion.views.<view>.parts.<id>` has `removed_px` and `restored_px` (label pixels cut from the carve mask / put back because the body encloses them): a part with mostly `restored_px` does not shape the hull there.

## 8. Finish

Report per-view error vs floor, pass or fail, every warning, what is inferred, visible defects, `.blend` and preview paths. Never claim success over a visible defect.
