# Hard Surface Rules (v0.2)

These rules are enforced by the component builders in `scripts/bbr/blender/` and checked by `validate_modifiers()`. Agents do not hand-roll alternatives. Worked example: `examples/treasure_chest` (arch lid and bands, torus handles and shackle, mirror, projected reference colour; no boolean cutters).

## Non-destructive first

- Geometry comes from component builders (`primitive.py`, `partgeom.py` and `components.py`). Shape changes are made by editing the SceneSpec component and rebuilding it with `update_component`, never by editing vertices by hand.
- Modifiers are created with `modifiers.ensure_modifier()`. It is idempotent: it finds the modifier by name and updates its properties.
- Curved shapes have their own types (`arch`, `torus`, `rounded_box` with a per-axis radius). Do not fake them with a cylinder plus a boolean cutter: the cutter is invisible to the blockout gate and to every metric (see Mirror and Boolean).

## Modifier order

```
Mirror  →  Boolean  →  Bevel  →  Weighted Normal
```

- **Mirror** about the `AI_ORIGIN` empty at the world origin, or, with `mirror_center`, about a per-component `AI_PIVOT_<id>` empty. Components keep their own centre as object origin.
- **Boolean**: a component with `"boolean": {"target": "<id>", "operation": "DIFFERENCE"}` becomes a hidden cutter on its target.
- **Bevel**: width and segments come from `style.bevel_width` and `style.bevel_segments`, or a per-component `bevel`: `null` scene style, `true` scene style, `false` none, a number (width in metres) or `{"width", "segments"}`. Limit method is ANGLE. `rounded_box`, `torus`, `gem` and the thin-part types (`cone`, `blade`, `disc`) are never auto-bevelled.
- **Weighted Normal** is used when `style.weighted_normals` is true, with *Keep Sharp* enabled.

## Component types

`size` is the full extent `[x, y, z]` in local axes, `location` the centre of that extent. Ranges are what the builder accepts; outside them the build reports an error (`ok: false`), it does not guess.

| Type | `size` / `params` (valid range) | Use for |
|---|---|---|
| `box`, `panel` | `size` | Blocks, boards, plates, straps, brackets. `panel` is an alias of `box` (name the intent). For flat boxes use `panel` or `box`, not `plate`: a `plate` with `params.anchors_px` or `params.mirror_of` is a contour plate (character parts); otherwise it builds as a `box`. |
| `rounded_box` | `size`; `params.radius`: number >= 0 (all edges) or `[rx, ry, rz]` (reach of the rounding along each axis, 0 = sharp that way; an edge needs two non-zero axes to be rounded); clamped below 0.49 x the extent; `segments` >= 1 (3) | Soft-edged housings, cushions, table tops. `[0, ry, rz]` rounds only the edges parallel to X. |
| `cylinder` | `size` (elliptic when x != y for axis Z); `params.axis` X\|Y\|Z (Z); `vertices` >= 3 (24) | Pipes, posts, wheels, drums, pins. |
| `sphere` | `size` (ellipsoid); `params.segments` >= 4 (24) | Knobs, balls, domes (with the lower half hidden). |
| `capsule` | `size`; `params.axis` X\|Y\|Z (Z); `segments` >= 4 (16); axis Z only: `tip_radius` > 0 (tapered) | Handles, limbs, pill shapes. |
| `leg` | `size` = the TOP footprint `[x, y, height]`; `params.profile` square\|round; `taper` > 0 multiplies the BOTTOM footprint (1 = straight, 1.07 = the foot is 7 % wider than the top, 0.8 = narrower); `vertices` >= 3 | Table and chair legs, feet, tapered posts. |
| `frame` | `size`; `params.thickness` > 0 (0.02, clamped to 0.49 x the in-plane extent); `axis` X\|Y\|Z (the thinnest extent) = window axis | Picture / window frames, rails. Four bars that touch, not welded. |
| `arch` (alias `half_cylinder`) | `size` (required); `params.axis` X\|Y\|Z long axis (X); `bulge` +X..-Z perpendicular to the axis (+Z; +X for axis Z); `thickness` 0 <= t < min(half width, height) (0 = solid); `segments` >= 2 (32) | Chest lids, vaults, tunnels, fenders. Half-ellipse section: half the extent across, the FULL extent along `bulge` (flat base on the opposite bounding-box face). With `thickness` a half-ring band (iron strap over a lid, a pipe clamp), closed ends. |
| `torus` | no `size`; `params.major_radius` number or `[r1, r2]` > 0 (elliptic, along the two axes other than `axis` in X, Y, Z order), required; `minor_radius` > 0 and below min(r)^2 / max(r), required; `axis` X\|Y\|Z ring-plane normal (Z); `major_segments` >= 3 (48); `minor_segments` >= 3 (16). Origin = ring centre | Ring handles, hoops, padlock shackles, washers, O-rings. Outer extent = major + minor radius. |
| `gem` | `size` `[2r, 2r, length]`; `params.facets` >= 3 (6); `waist` in (0, 1) (0.5) | Faceted crystals, studs. Flat shaded. |
| `lathe` | no `size`; `params.profile` `[[z, r], ...]` metres bottom to top (z local, explicit) or `profile_from` (below); `depth_ratio` Y over X extent (auto from the side view, else 1); `segments` (48); `cap_top` / `cap_bottom` (true; false = open end) | Bodies of revolution: mushroom caps, vases, lamps, bottles, columns, round towers. Smooth, sharp at the caps; never auto-bevelled. |
| `disc` | no `size`; `params.radius`, `thickness` required (true size); optional `dome_height` > 0 (spherical-cap dome on the +`axis` face), `dome_radius` (default the whole face), `radius_ratio` (extent along Y = radius x ratio), `axis` (Z), `segments`, `dome_rings`. Origin = the CENTRE of the cylinder (mid-thickness); the dome rises from `+thickness/2` | Shields, hat brims, round plates, a flat domed roof (a cap with a profile is better a `lathe`). Textured by default. |
| `cone`, `blade` | true-size thin parts, see [workflow](reconstruction-workflow.md) | Horns, blades. Textured by default. |
| `silhouette_hull`, `layer_hull` | character track | Not hard-surface. |

Every component also takes `stage`, `confidence`, `material`, `rotation_deg`, `mirror`, `mirror_center`, `bevel`, `boolean`.

## Scale

- Builders create mesh data at true size, so object scale is always (1, 1, 1).
- If an object ever has a non-unit scale (manual edits, or a correction step), `validate_scale()` reports it. The fix is to apply the scale through the data API, not to leave it unapplied.

## API preference

`bpy.data`, then `bmesh`, then the Object Data API and Modifier API, and `bpy.ops` only when unavoidable: saving or opening files, rendering, and the silhouette-hull post-processing that has no data-API equivalent (`quadriflow_remesh`, `uv.smart_project`, `uv.pack_islands`). The voxel remesh, decimate and shrinkwrap are modifiers collapsed with `bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))` (headless-safe; Sculpt Mode is never used because it crashes with `-b`). Operators run inside `bpy.context.temp_override(...)` and return to OBJECT mode. Rendering runs inside `bpy.context.temp_override(scene=...)`. File save and open are window-manager operators that do not depend on a scene context.

## Detail discipline

- Never add details that are not visible in any reference, unless they are structurally necessary. A structurally necessary hidden part, such as a rear support, is marked `confidence: "inferred"`.
- Bevel and secondary details are added only **after** the blockout gate has passed.
- Sizes come from the references and dimensions, not from taste. Reference fidelity is more important than aesthetics.

## Hard-surface spec recipe

1. **Copy the template.** Copy `examples/simple_table/scene_spec.json` (flat colours, synthetic references) or `examples/treasure_chest/scene_spec.json` (projected colour, curved parts) to `$W/scene_spec.json` and edit it. Set `dimensions` (`width` = X, `depth` = Y, `height` = Z, metres), `symmetry`, `materials` and the components. Keep your own `view_spec.json`; the templates' references are not yours.
2. **Size the object from what the task gives.** `draft-view-spec --height H`, or `--width W` / `--depth D`: the height is derived from the front (width) or side (depth) view and recorded in `view_spec.json` (`known_dimensions.height` plus `dimension_provenance`, printed next to the pipeline's `provenance`). `--size-of silhouette` (default) = the number is the whole outline, handles, locks and all; the width is then also stored as known. `--size-of body` = the body without protrusions (the typical row extent, so rings, a padlock or a raised arm do not count); only the derived height is stored. Both candidates are printed, so pick the one the task means: on the chest 0.8486 m is the silhouette (height 0.567 m), the same number read as body width would give 0.606 m. A width-only `known_dimensions` calibrates exactly like a height: the pipeline derives the other dimensions from the silhouette bbox.
3. **Measure sizes, never guess them.** For each visible part, pick pixels on its edges and let the tool convert them to metres (flags: `--view-spec`, `--view`, repeatable `--px X,Y`, or `--bbox X0,Y0,X1,Y1` for a part's extents, `--json`):
   ```bash
   $PY scripts/prepare_references.py measure --view-spec $W/references/view_spec.json --view front --px 118,190 --px 125,45
   $PY scripts/prepare_references.py measure --view-spec $W/references/view_spec.json --view side --bbox 40,60,220,95
   ```
   Front/back views give x and z, side views give y and z. The side view's horizontal origin is its bbox centre: a protrusion at one end (a padlock) shifts it from the body axis (chest: 5.5 px = 7 mm; the body-axis origin of `calibration.py` only applies to front / back views and above 3 % of the width). Measure y from the same view and, if it matters, centre on the body (`YC = sy(228)` in the chest example).
   **Batch**: with 3+ points write them once and convert in one call; never convert by hand or with your own script:
   ```bash
   # pts.json: {"front": {"door_top": [340, 378], "door": [270, 405, 410, 578]}, "side": {"win": [321.5, 461.5]}}   (2 numbers = pixel, 4 = box)
   $PY scripts/prepare_references.py measure --view-spec $W/references/view_spec.json --points-file pts.json --out metres.json
   ```
   `metres.json` has `{view: {name: {axis: metres}}}` (a box gives `min` / `max` / `size`), each view's calibration and its formula. By hand (`check` and `masks` print the same numbers): `m` = m/px and `(ox, oy)` = origin_px of the view, `u` right and `v` down in pixels; the image-right axis follows the camera:

   | view (camera_axis) | image right | horizontal metres | vertical metres |
   |---|---|---|---|
   | front (-Y) | +X | `x = (u - ox) * m` | `z = (oy - v) * m` |
   | back (+Y) | -X | `x = (ox - u) * m` | `z = (oy - v) * m` |
   | side `-X` (faces image-right) | -Y | `y = (ox - u) * m` | `z = (oy - v) * m` |
   | side `+X` (faces image-left) | +Y | `y = (u - ox) * m` | `z = (oy - v) * m` |

   Pass `--labels labels.json` to `measure` when the scene has `parts_labels` (the body label can move the origin; `check --labels` prints the same).
4. **One component per visible part**, no unseen detail, types from the table above.
5. **Mirror.** Model one instance and set `"mirror": {"x": true, "y": true}` (any of `x`, `y`, `z`); `location` is the positive-side instance: `leg_fl` at `[0.575, -0.275, ...]` with `x` and `y` mirrored gives four legs. By default the planes pass through the world origin. An object whose axis is not the origin sets `"mirror_center": [x, y, z]` (world metres; only the mirrored axes matter): a lamp at `y = 0.2` with `"mirror": {"y": true}, "mirror_center": [0, 0.2, 0]` gets its twin at `y = 0.1` when the instance is at `0.3`. `mirror_center` without a `mirror` axis is an error.
6. **Mirror and boolean.** Mirror runs BEFORE boolean, so the cutter acts on the mirrored result: an unmirrored cutter at `+x` cuts only the `+x` instance and leaves the twin whole. Make the cutter cover both instances (centred on the plane, or wide enough) or give it the same `mirror`; a mirrored cutter cuts both (checked in `tests/blender/test_blender_smoke.py`). Better: avoid the cutter, use `arch` / `torus` / `thickness`. A cutter is hidden: it is never rendered, textured, counted in any metric or an occluder (below), and in the blockout stage it is not applied yet (see validation rules).
7. **Stages.** `blockout` -> `primary` -> `secondary` -> `detail`, built in that order. Put the main volumes in `primary`, small features in `secondary`/`detail`; bevel and details come only after the blockout gate. Run with `--stop-after blockout` first, check `stage_A_blockout.png` and the numbers, then `--stop-after primary`, then the full run.
8. **Confidence.** `observed` for what a view shows, `inferred` for structurally necessary hidden parts, `unknown` otherwise.

## Material classes

`material` on a component is a key of `materials` or a class name; `materials.<key>.class` plus explicit keys (`roughness`, `metallic`, `coat`, `coat_roughness`, `sheen`, `specular`, `emission_strength`, `ior`, `transmission`; explicit wins) tunes one. Projected textures use the stylized column, flat materials the physical one (`style.pbr`: `auto` | `stylized` | `physical`). Colour always comes from the texture / `base_color`. No class = roughness 0.6, metallic 0 (plastic look).

| class (aliases) | physical | stylized |
|---|---|---|
| `metal_steel` (iron, steel) / `metal_gold` (gold) / `metal_brass` (brass) | metallic 1, R 0.4 / 0.25 / 0.3 | metallic 0.6 / 0.65 / 0.6, R 0.45 / 0.35 / 0.4 |
| `cloth` (fabric) | R 0.9, sheen 0.5 | R 0.9, sheen 0.3, specular 0.3 |
| `skin` | R 0.5, IOR 1.4 | R 0.55 |
| `hair` / `leather` / `wood` / `stone` | R 0.45 / 0.55 / 0.7 / 0.88 | R 0.5 / 0.6 / 0.7 / 0.9 |
| `plastic` / `painted` (paint) | R 0.4 / R 0.4 + coat 1 | R 0.45 / R 0.45 + coat 0.3 |
| `glass` (opaque glossy, no transmission) | R 0.05 + coat 1 (coat R 0.02) | same |
| `glass_clear` (see-through: transmission 1, R 0.03, IOR 1.45) | transmission >= 0.5 makes a part see-through: the colour metric, the texture projection and the object-ID renders look through it, the lit previews show it | same |
| `visor` | R 0.45 + coat 1 (coat R 0.03) | same |
| `emissive` | emission 1.0 x colour | 0.6 |
| `rubber` | R 0.88, specular 0.3 | R 0.9, specular 0.3 |

Example: `"materials": {"iron": {"class": "metal_steel", "base_color": [0.1, 0.1, 0.11]}}`. The result is in `hull_stats.<id>.material`.

## Reference colour (texture projection)

`style.texture`: `auto` (default), `project`, `flat`.

- `auto`: every non-cutter component gets a projected texture when the view spec has a `color` view and its references are real images (not `reference_quality.synthetic: true`, which the table and chair examples set: their references are renders of the spec, a flat material is exact). `project` forces it, `flat` turns it off. `params.color_projection` (`true` / `false`) overrides one component; a hidden boolean cutter is never textured.
- What it does: each component is UV-unwrapped (smart project, boxy island packing), then every texel takes the colour of the colour view that sees it best, with the depth buffers of ALL visible meshes as occluders (a bracket hides the wall behind it; cutters occlude nothing), seam blending, silhouette-edge distrust, green-screen despill and inpainting of texels no view sees. Faces seen more obliquely than `min_facing` (default 0.3 here) are not sampled, and face normals are flat, so a flank is not painted from a grazing view. Mirrored components are baked once (the base instance) and the Mirror modifier reuses the UVs.
- A scene with `symmetry.x` also uses its side view mirrored through X = 0 (`mirror_color_views`) for the flank no real view sees; `params.mirror_color_views: []` opts out. The texture side is chosen per part from its surface area over the reference pixel size (128-1024 px; `params.texture_size` overrides it).
- Read the result in `report.json` -> `hull.<id>.texture` (`inpainted_share`, `background_leak_ratio`), the color error lines and the lit previews / turntable. Colour error is measured against the same references the colour was projected from: low numbers show the projection works, not that unseen sides are right (check the turntable). What it cannot give: geometry (rivets, plank gaps and cracks are painted, not modelled) and anything no view shows (a flat top under a lid is inpainted).

## Lathe: a body of revolution measured from the masks

`lathe` reads its profile, axis and depth from the references (about Z):

```json
{"id": "cap",  "type": "lathe", "params": {"profile_from": {"part": "cap"}, "segments": 64}},
{"id": "slab", "type": "lathe", "params": {"profile_from": {"view": "front", "bbox_px": [60, 568, 625, 616], "bottom_z": 0}}}
```

- Per pixel row of the front mask (a pixel box, or the label `part`) the larger half-width about the axis is the radius; the side silhouette gives the axis y and the elliptic `depth_ratio` (median over the rows). A one-sided cut (a chimney or spout labelled `_x`, an occluder) does not shrink it; spikes of 3 rows or less are filtered. `location` is derived (the axis position; profile z are world heights). Result: `out/lathe_report.json`.
- A base whose rim the camera saw from slightly above curls in at the bottom: cut those rows with `bbox_px` and set `bottom_z: 0`. Open top, bottom or mouth: `cap_top` / `cap_bottom` false.
- An explicit `profile` (metres, local z) wins and keeps your `location`.

## Place a part on a curved surface: `attach_at`

For windows, doors, plaques on a body, lathe or cylinder (`attach_to` + `attach_at`, no `anchors_px`):

```json
{"id": "win", "type": "torus", "params": {"axis": "Y", "major_radius": 0.108, "minor_radius": 0.017,
  "attach_to": "body", "attach_at": {"view": "front", "px": [498, 461.5]}, "standoff": -0.017}}
```

- `attach_at` = `{"view": V, "px": [x, y]}` (cast along the view's camera through the pixel) or `{"azimuth_deg": a, "height": z}` (0 = front, 90 = +X, 180 = back; cast at height z toward the target's axis). The target's own mesh is ray-cast, so it must be built earlier (earlier stage or earlier in the list); a flat door works too: `attach_to: "door_lo"`.
- The part's local `face` axis (`-Y` default, or `"+Y"`) turns along the outward normal, local Z stays upright; a part facing the front keeps rotation 0. `standoff` (m, default 0) = gap from the surface to the part's BACK face: 0 rests it on the surface, `-half thickness` centres it on the surface, `-t` sinks `t` into the wall. Parts that must line up share one attach point. Placement: `out/surface_attach.json`.
- A flat part on a curved wall touches only at its centre: sink it by the curvature gap `r - sqrt(r^2 - w^2/4)`.

## Labels and per-component colour views

- `parts_labels` works in the hard-surface track. Each component samples only pixels labelled with its id (the component `body` also gets every unlabelled pixel); a component with no label in any view samples everything (log line only). A label starting with `_` (`_x`, `_chimney`) is sampled by nobody. Labels do not switch a hard-surface scene (`style.hard_surface` true) to character texturing: flat face normals, boxy unwrap and `min_facing` stay.
- When ONE view contradicts the others (the side view draws the chimney on the near cap, front and back put it on the far side; `check` still says "consistent" because masks only compare silhouettes), label that region `_x` in that view instead of editing the reference, and give the part `params.color_views: ["front", "back"]` (per-component override of the colour views). Label a part in every view or in none: in a labelled view a part missing there samples nothing.
- Cross-check: `labels` prints each part's z-range per view (flagged when views disagree by 4 % of the height). It cannot see x / y contradictions: compare the part's front x and side y against the side camera's near side by eye.
