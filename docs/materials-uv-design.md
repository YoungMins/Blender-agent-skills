# Materials + UV design (Claude session, 2026-10-23)

> Source: Opus deep-reasoner design review in the Claude Code session `2710d383` (2026-10-23 13:25 KST).
> Paths under `C:/Users/nanyo/AppData/Local/Temp/claude/...` are local scratch from that session and do not exist in the repo.
> WI-1 was started (partially implemented, unverified) when the session hit its weekly rate limit. See `HANDOFF.md`.

## Conclusion

The fastest way to "much better" is not a new UV layout. It is two cheaper fixes:

- **Wire up the materials the specs already declare.** Add class presets and a real preview environment.
- **Fix the hard-surface texel budget.** 17 of the 24 chest parts get less than 1 texel per reference pixel.

The character hull already gets about 1.5–2 texels per reference pixel, so view-aligned hull UVs would buy almost no sharpness. Keep smart-project there and only add measurement. Per-texel material classes come third; the warrior needs them because armour, cloth and skin share one hull.

**Confidence:** high on the diagnosis and on the material work leaving the colour metric unchanged (measured, checked in code, and probed in Blender 5.1.1). Moderate on how metallic will look on AI textures that have lighting painted in; one A/B render settles it.

| # | Work item | Effort | Expected visible gain | Changes metrics? | Default |
|---|---|---|---|---|---|
| WI-1 | Spec materials and class presets feed the projected-texture material; gradient preview world; visor coat | 1–1.5 d | Medium–high on kits (chest bands, brackets, padlock; mushroom glass and brass) and character parts; visor reads as glass | No; lit previews change | On, but land after the bench run |
| WI-2 | UV statistics, then hard-surface texel budget | 1.5–2 d | Medium on props: bands, ring, shackle, feet go from 0.33–0.58 to ≥1.0 texels/px (1.7–3× sharper); chest body 0.85 → ~1.2 | Yes (textures change) | Behind a flag; flip for hard-surface after a bench A/B |
| WI-3 | Per-texel material classes from `materials.json`, plus a suggestion tool | 2.5–3 d | High on characters with mixed materials on one hull (warrior); astronaut helmet, suit and lights | No | Opt-in via `material_labels` |
| (WI-4) | View-aligned UV charts | 2–3 d | Low | Yes | Only if WI-2's numbers call for it |

**Per track:**
- **Hard-surface:** keep smart-project; fix the texel budget (WI-2); take materials from the existing `material` keys (WI-1).
- **Hull:** keep smart-project at 2048 and only measure; per-texel materials come from WI-3.
- **View-aligned islands:** held back as WI-4.
- **Rejected:** direct camera-projection UVs, seams from sharp edges, and `ANGLE_BASED` / `MINIMUM_STRETCH` unwrapping.

## Evidence

- **Spec materials are silently dropped (measured, in code).**
  - `treasure_chest` declares iron (metallic 0.9, roughness 0.45), wood and brass. `mushroom_house` declares stone, wood, brass and glass.
  - `components.finalize_object` assigns these materials.
  - Then `partstex.bake_object` → `hullpost.assign_texture_material` → `material.ensure_texture_material` replaces them with roughness 0.6, metallic 0.
- **The preview ambient never reaches Cycles (measured, Blender probe).** In 5.1.1 with `--factory-startup`, the world already uses a node tree (Background 0.051, strength 1). `render._configure` only sets `world.color = PREVIEW_AMBIENT (0.12)`, which Cycles ignores. Metal would reflect a 0.05 world and render nearly black.
- **The colour metric and bench PSNR can't see the material graph (checked in code).**
  - Both render Workbench flat with `color_type="TEXTURE"`. That shows each material's active image node; `_color_type` looks for the `M_ref_texture` name prefix.
  - The bench path is `bench/render_holdout.py` → `synthetic.render_azimuths` → `render.render_color_dir`, the same route.
  - So the material graph is invisible to both metrics as long as `bbr_basecolor` stays the active node.
- **Hard-surface parts are undersampled (measured from spec geometry and report coverage, ±10%).** Texels per reference pixel, k:

  | Parts | k |
  |---|---|
  | chest body | 0.85 |
  | lid | 1.00 |
  | rim_band | 1.14 |
  | band_centre / band_side | 0.43 / 0.42 |
  | feet | 0.58 |
  | ring_plate | 0.52 |
  | ring (torus) | 0.33 |
  | padlock_shackle | 0.45 |
  | padlock_body | 0.74 |
  | brackets and hasp | 0.81–1.46 |

  Causes:
  - `texelproj.suggest_texture_size` assumes the UV islands fill the whole texture; actual coverage is 10–65%.
  - The margin is a fixed 8 px, which is 6.25% per side on a 128 px texture.
  - 44–87% of covered texels are surfaces no view sees, stored at full density.
- **Estimated (areas guessed):** warrior hull k ≈ 1.5, astronaut hull ≈ 2.0, character parts ≥ 1.4, warrior face layer ≈ 1.1.
- **Simulation (sphere, k = 1.5):** the smart-project island-axis mismatch gives p05 of 1.25 (6 axes) or 1.28 (14 directions), against 1.50 for view-aligned, with about 0% undersampled. So the hull is very unlikely to be undersampled.
- **Observed hull defects come from the projection, not the UVs:**
  - a ghost forearm painted on the torso in `side_lit.png`;
  - streaks on the cape and arms where views see the surface at a grazing angle;
  - blocky fill on unseen surfaces;
  - the mushroom cap spots, where the views disagree.

## WI-1: material classes and preview world

**Files**
- New `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\core\pbr.py`:
  - `CLASSES` with two value columns, plus `ALIASES` (iron/steel → `metal_steel`, gold, brass, fabric → cloth, paint → painted).
  - `resolve(entry, column)`: preset first, explicit keys always win.
  - `column_for(projected, style)`.
  - MIT attribution header.
- `...\scripts\bbr\blender\material.py`:
  - `ensure_texture_material(image, name, roughness, pbr=None, maps=None)`. With `pbr=None` the node graph is identical to today's.
  - Set sockets via `bsdf.inputs.get()`; the 5.1 Principled names were verified.
  - Link Emission Color from the base image.
  - Set `nt.nodes.active = tex` **last**.
  - `ensure_material(..., pbr=None)`: same for flat materials; keep `diffuse_color`.
- `...\scripts\bbr\blender\hullpost.py`: `assign_texture_material(obj, image, name, pbr=None, maps=None)`.
- `...\scripts\bbr\blender\partstex.py`: `bake_object` resolves the component's `material` and records `stats["material"]`.
- `...\scripts\bbr\blender\components.py`: `material_for` passes the resolved values.
- `...\scripts\bbr\blender\render.py`: new `preview_world(sc)`.
  - Node chain: TexCoord.Generated → SeparateXYZ.Z → MapRange(−1..1 → 0..1) → ColorRamp (ground 0.02, horizon 0.18, zenith 0.45) → Background 1.0.
  - Swap it in only inside `_render_lit_preview`, within `render_session`; restore the old world and delete the temporary one in `finally`.
  - Retune `RIG_POWER_PER_M2` by A/B.
  - Don't use the Sky Texture node: its types changed in 5.x.
- `...\schemas\scene_spec.schema.json`:
  - `materials.&lt;key&gt;`: add `class`, `coat`, `coat_roughness`, `sheen`, `specular`, `emission_strength`, `ior`.
  - `style.pbr`: `auto | stylized | physical`. `auto` means stylized for projected textures and physical for flat ones.
- `...\THIRD_PARTY_NOTICES.md`: add a bullet for the presets. The external LICENSE confirms EL4CTEO, MIT.
- Docs:
  - `references\hard-surface-rules.md`
  - `references\character-cheatsheet.md`: visor recipe is `"material": "visor"` on the `face` layer.

**Presets.** R = roughness, M = metallic. The stylized column is the default for projected textures, because their base colour already has lighting painted in.

| class | physical (from pbr-values.md) | stylized |
|---|---|---|
| metal_steel / gold / brass | M1, R 0.4 / 0.25 / 0.3 | M 0.6 / 0.65 / 0.6, R 0.45 / 0.35 / 0.4 |
| cloth | R 0.9, sheen 0.5 | R 0.9, sheen 0.3, specular 0.3 |
| skin | R 0.5, IOR 1.4, subsurface off (the table says 1; not ported) | R 0.55 |
| hair / leather / wood / stone | R 0.45 / 0.55 / 0.7 / 0.88 | R 0.5 / 0.6 / 0.7 / 0.9 |
| plastic / painted | R 0.4 / R 0.4 + coat 1, coat R 0.03 | R 0.45 / R 0.45 + coat 0.3 |
| glass (opaque, glossy) | R 0.05 + coat 1, coat R 0.02 | same |
| visor (coat over a layer) | R 0.45 + coat 1, coat R 0.03 | same |
| emissive | emission 1.0 × base colour | 0.6 |
| rubber | R 0.88, specular 0.3 | R 0.9, specular 0.3 |
| default | R 0.6 (today) | same |

The table's metal base colours are only used for flat components that have no `base_color`.

**Report:** `hull.&lt;id&gt;.material = {key, class, column, params}`, plus a summary line per object.

**Tests**
- `tests\test_pbr.py`: preset-plus-override resolution, aliases, column choice; an unknown class is rejected (`tests\test_scene_spec.py`).
- `tests\blender\test_blender_smoke.py`:
  - (a) A part with a `material` gets Metallic and Roughness equal to the spec values, keeps the `M_ref_texture:` name prefix, and `nodes.active.name == "bbr_basecolor"`.
  - (b) `render.render_color(view, calib, aa=CRISP_AA)` with and without classes gives exactly equal arrays (`np.array_equal`; this path is deterministic).
  - (c) After a lit preview, `scene.world` is restored and no `bbr:preview_world` datablock is left.

**Acceptance**
- `color_match.per_view.*.error` is identical on `sd_warrior`, `treasure_chest` and the astronaut; bench holdout PSNR is identical.
- Judge on `perspective.png` (the 320 px turntable tiles are too small):
  - the share of model pixels with luminance &lt; 0.05 rises by no more than 2 points (no black metal);
  - mean model luminance on `robot_character` stays within ±15%.
- Visually: bands, brackets and padlock show the reflected gradient; wood stays matte; the visor shows a glossy highlight.
- Lit render time ≤ 1.2×.

**Risks**
- Metallic on textures with painted-in lighting can look dark or dirty.
  - The stylized column and the world are the mitigation.
  - If the chest's explicit 0.9 iron looks dark in the A/B, switch the example specs to `class: metal_steel`.
- Every lit preview changes (this also fixes the dead ambient).
- Chest and mushroom are bench cases, so land this after the current run.

## WI-2: UV statistics and hard-surface texel budget

**Statistics.** New `...\scripts\bbr\core\uvstats.py`, computed in `hullpost.bake_texture` from the raster and the projection info.

| stat | definition |
|---|---|
| `texel_per_px` p05 / p50 | Per seen texel: J = A_v · [P(x+1) − P(x), P(y+1) − P(y)], using same-island neighbours. A_v is the linear part of the primary view's `world_to_px`. r = 1/σmax(J). Weighted by \|det J\| (reference pixels per texel). |
| `undersampled_share` | Weight share with r &lt; 1 |
| `conformal_p95` (angle stretch) | p95 of σmax/σmin of the 3D → texel map, from the same differences |
| `density_cv` (area stretch) | Area-weighted spread (CV) of per-island density over islands that are at least 50% seen |
| `islands`, `islands_tiny` | Island count; tiny = &lt; 0.1% of covered texels |
| `fill`, `seen_fill` | Covered texels and seen texels as a share of the texture |
| `seam_m`, `seam_seen_m` | 3D length of edges whose two faces have different UVs at the shared corners; `_seen` = both faces seen |

These go into `report.hull.&lt;id&gt;.uv`. The `texture &lt;id&gt;:` summary line gains `k p05 0.41 under 88%`.

**Per-object colour error.** This is the check that doesn't come from my own new metric; per-view error is dominated by the body and lid, so a sharper band won't show there.
- In `...\scripts\bbr\blender\projection.py`, render `render.render_object_ids(view, calib, objects)`.
- Compute `color_error(ref, flat, ref_mask &amp; ids==i, render_mask &amp; ids==i)`.
- Store as `color_match.per_object.&lt;id&gt;.&lt;view&gt;`.

**Budget algorithm.** Hard-surface parts only; the hull is unchanged.
1. **Visibility pass before unwrapping.** In `bake_object`, compute inputs, mirror views and depth buffers before the unwrap. Run `project_texel_colors` on 5 points per face. Each face is then one of:
   - **seen** (most points have a sample);
   - **hidden** (some view faces it but the depth test fails, or the normal points down, z &lt; −0.5);
   - **oblique** (no view faces it above `min_facing`).
2. **Unwrap and scale.** Run `smart_project` as today. Then scale each island about its centroid:
   - 1.0 if at least 5% of its faces are seen;
   - 0.25 if hidden;
   - 0.5 if oblique (the lid's top strip is exposed in previews).
   Then `pack_islands(scale=True)`.
3. **Size.** Start from the effective area (seen + 0.25² × hidden + 0.5² × oblique) with an assumed fill of 0.5 and a k target of 1.25. After packing, measure k per face. If the seen p05 is below `texel_per_px`, double the size, up to two retries; a part unwraps in under 0.5 s.
4. **Margin.** Parts default to `clamp(size // 128, 2, 8)` px; the hull keeps 8. Dilation radius = margin + 2.

**Files**
- New `...\scripts\bbr\core\uvstats.py`.
- New `...\scripts\bbr\core\uvbudget.py`: face visibility classes, island scale factors, size loop, margin rule.
- `hullpost.unwrap(..., island_scale=None)`.
- `hullpost.bake_texture`: computes the statistics.
- `partstex.bake_object`: reorder, sizing loop.
- `texelproj.suggest_texture_size(fill=1.0)`: unchanged by default.
- `projection.py`: per-object colour error.
- `summary.format_summary`.

**New parameters (schema)**

| Parameter | Default |
|---|---|
| `uv_budget` | false (true for hard-surface after A/B) |
| `texel_per_px` | 1.0 |
| `unseen_hidden_scale` | 0.25 |
| `unseen_oblique_scale` | 0.5 |
| `texture_max_px` | 2048 |
| `texture_padding_px` | size-based rule above (parts) |

**Tests** (pure numpy: `tests\test_uvstats.py`, `tests\test_uvbudget.py`)
- A plane facing −Y with UV = front projection × k gives r = k exactly.
- The same plane projected along an axis tilted 60° gives r = k·cos 60°.
- UV scaled (2, 1) gives conformal = 2.
- Two islands at densities 1 and 2 give the analytic `density_cv`.
- A cylinder: unrolled UV gives r ≥ k on the visible half; view-planar UV gives r = k.
- An icosphere with 6-axis planar islands gives p05 ≈ 0.83·k (±0.03).
- A quad split on its diagonal gives `seam_m` = diagonal length.
- A box on the floor: bottom classified hidden, top oblique; the size loop and margin rule behave as specified.
- Blender smoke:
  - the bottom island's area is 0.25² × a side island's (±10%);
  - `pack_islands` keeps relative island scale (±1%).

**Acceptance**
- Chest and mushroom with `uv_budget` on:
  - `texel_per_px` p05 ≥ 1.0 on every part with seen texels;
  - `undersampled_share` ≤ 5%.
- Per-object colour error goes down on every part with baseline k &lt; 1 (bands, ring, ring_plate, shackle, feet, padlock_body, body). It gets no worse than +0.002 on any other part; per-view error no worse than +0.001.
- Texture-pass time ≤ 1.2× (fewer unseen texels to fill offsets the larger seen ones).
- With the flag off, hull and character-part textures are bitwise identical.
- The hull's measured p05 is expected ≥ 1.1. If it comes out below 1.0 with more than 5% undersampled, open WI-4 for the hull.
- Visually: rivets on the bands, the ring and the shackle are crisper in `perspective.png`; no new seams in the turntable.

**Risks**
- Oblique but exposed faces lose some density (bounded at 0.5).
- Islands that mix seen and unseen faces get no gain.
- Textures change, so the colour metric and holdout PSNR shift; bench A/B before flipping the default.

## WI-3: per-texel material classes

**How classes are assigned**
- A part's `material` key (one class) always wins.
- Hull and multi-material parts use `materials.json`:
  - same syntax as `labels.json`: `_regions`, region ids, polygon/rect/split/within;
  - keys are scene `materials` keys or class names;
  - unlabelled pixels (`body`) take the object's own `material`, else `default`.
- It must be a separate file: `labels.load_labels` rejects non-list values, and `labels.json` keys are parts.
- **Agent cost:** label the front view using the same regions it already read for parts, then review the overlays.

**Tool:** `prepare_references.py materials --view-spec VS --materials M.json [--labels L.json] --out-dir D [--suggest]`.
- Writes `&lt;view&gt;_classes.png` overlays and per-view class shares.
- Warns when a class covers more than 2% of one colour view and 0% of another.
- `--suggest` writes `materials_suggested.json` from per-region statistics:
  - Lab mean, chroma and hue; L* spread (texture variance); highlight ratio; area.
  - Rule order: emissive → gold/brass → steel → skin → cloth → rubber → plastic → painted.
  - Part-name hints, e.g. blade → steel, cape → cloth, face/hand → skin, belt → leather.
  - Side and back regions within ΔE 12 of a labelled front region inherit its class.
- Mark suggestions low confidence: steel vs white plastic vs white cloth, and leather vs hair vs wood, can't be told apart by colour.

**Projection.** New `...\scripts\bbr\core\matclass.py`.
- Per-view class maps come from `labels.load_label_masks`. Mirror views use the same `[:, ::-1]` flip as `symmetry.mirror_view`, ANDed with the mirrored mask.
- Each texel takes its class from the primary view only (`info["weights"].argmax(1)`), snapped to the nearest labelled pixel within 4 px. Classes are never blended.
- `view_disagreement`: share of texels with two or more candidate views whose classes differ.
- Unseen texels are filled by copying a class index only: the nearest seen texel in the same island, else the nearest 3D donor. Never `inpaint_colors`: its smoothing pass would blend class ids.
- Two parameter textures at half resolution, dilated like the colour texture:
  - `pbr_a` = roughness, metallic, coat;
  - `pbr_b` = specular, sheen, emission/2.
  These are Non-Color image nodes feeding the sockets. One material per object; `bbr_basecolor` stays the active node.
- If one class covers at least 95% of an object, use constants instead of textures.

**Files**
- `matclass.py`.
- `hullpost.bake_texture(class_maps=…)`.
- `partstex`: load the class maps once per build; mirror them in `mirror_extras`.
- `material.ensure_texture_material(maps=…)`.
- `prepare_references.py` subcommand.
- Schema: top-level `material_labels`.
- `spec.py`: resolve the path like `parts_labels`.

**Report:** `hull.&lt;id&gt;.material = {source, shares, view_disagreement, maps}` and `textures\&lt;name&gt;_material_map.png`.

**Tests** (`tests\test_matclass.py`)
- Weights 0.6/0.4 on two classes give the primary view's class.
- The fill never creates a class that isn't among its donors.
- The mirror flip is correct.
- Map lookup and dilation work; shares sum to 1.
- Smoke: a two-class hull has the `bbr_pbr_a`/`bbr_pbr_b` nodes and a flat render exactly equal to the single-class build.

**Acceptance**
- Metrics unchanged, as in WI-1.
- On `sd_warrior`: class shares within ±5 points of the labelled pixel shares; `view_disagreement` ≤ 10%; no class seam at the front/side boundary in `material_map.png`.
- Visually: armour reflective, cape matte, skin distinct.
- Astronaut: helmet glossy, suit matte, chest light emissive.

**Risks**
- Inconsistent labels between views show up as material seams; `view_disagreement` and the overlays catch them.
- Thin gold trims get merged into neighbouring regions.

## WI-4 (held back)

Open it only if WI-2 shows p05 &lt; 1.0 after budgeting on curved parts or the hull.
- Group faces into same-view charts.
- UV = the primary view's `world_to_px` × k; no angle-based relax, which would break the 1:1 link to the source pixels.
- Check for overlaps by rasterising the UVs; demote overlapping faces.
- Unwrap the remainder with smart-project on a bmesh face selection.
- Pack with `pack_islands(rotate_method="CARDINAL")`.

## What will NOT help

- **Re-unwrapping the hull or a 4096 hull texture:** it's already about 1.5–2 texels per pixel; the limit is the 500–900 px reference.
- **Direct camera-projection UVs or one material per view:**
  - Workbench shows one active image node, so the colour metric and bench PSNR break;
  - it loses despill, label gating, edge distrust and inpainting;
  - surfaces no view sees are left unhandled.
- **`ANGLE_BASED` / `MINIMUM_STRETCH` unwrap or seams from sharp edges:** the hull has no seams and collapses; boxes already get one island per face.
- **Normalising 3D texel density (`texel_density.py --set`):** it spends texels on hidden faces, the opposite of what's needed.
- **Cycles baking (`bake_maps.py`):** the numpy texel projection already is the bake.
- **Exact 1:1 pixel alignment:** negligible gain at k ≥ 1.2.
- **Bigger mip-safe margins for Cycles:** Cycles doesn't mip-map.
- **Physical metallic 1.0 on painted textures.**
- **Transmission on a painted visor or glass:** it shows the hull or wall behind.
- **A separate visor shell:** it enters the depth buffers, ID renders and the flat metric.
- **Subsurface scattering on painted faces:** blurs eyes and adds noise.
- **Bump maps from painted shading.**
- **UV or material changes for the mushroom cap spots:** the views disagree there.

Out of scope, but these would help the hull: a better fill for unseen surfaces, handling of grazing-view streaks, and the ghost-forearm label gating.

## Unknowns and next steps

**Unknowns**
- That Workbench TEXTURE mode uses the active image node: smoke test (b) settles it.
- That `pack_islands(scale=True)` keeps relative island scale: the WI-2 smoke assertion settles it.
- The hull and face-layer k values are estimates; the WI-2 statistics measure them. If the warrior face layer comes out below 1.2, budget it too.

**Steps, in order**
1. After the bench finishes: a 10-minute A/B on copies of the chest and warrior `.blend` files. Set metallic/roughness, swap in the gradient world, render `perspective.png`. No repo changes.
2. WI-1, with tests (a)–(c).
3. WI-2 statistics (read-only metric), then the budget behind `uv_budget`; bench A/B; then flip the default for hard-surface.
4. WI-3 once step 1 shows metal reads well; it can run in parallel with step 3.
5. WI-4 only if WI-2's numbers call for it.

**Scratchpad scripts, to re-run:**
- Sphere simulation: `C:\Users\nanyo\AppData\Local\Temp\claude\D--Git-skills-blender-agent-skills\2710d383-47fe-464c-a902-d76e255156d0\scratchpad\sim\uvsim.py`
- Blender 5.1 API probe: `C:\Users\nanyo\AppData\Local\Temp\claude\D--Git-skills-blender-agent-skills\2710d383-47fe-464c-a902-d76e255156d0\scratchpad\sim\probe_api.py`

**Main files:**
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\blender\material.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\blender\partstex.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\blender\hullpost.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\blender\render.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\blender\projection.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\core\texelproj.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\core\uvraster.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\scripts\bbr\core\labels.py`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\schemas\scene_spec.schema.json`
- `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction\tests\blender\test_blender_smoke.py`

---

# Appendix: WI-1 implementation brief (as sent to the executor agent)

Implement **WI-1: PBR material classes and a preview world** in the Blender skill at `D:\Git\skills\blender-agent-skills\blender-blueprint-reconstruction` (SK).

Conventions: match the existing style. Core code is pure numpy and gets pytest tests; api functions return ok dicts. Blender is at `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe` (Git Bash `B=...`). System python3 has numpy, PIL and pytest. SKILL.md must stay at or under 1000 words.

## Diagnosis (verified by a design review)
- **Spec materials are silently dropped.**
  - `treasure_chest` declares iron (metallic 0.9, roughness 0.45), wood and brass. `mushroom_house` declares stone, wood, brass and glass.
  - `components.finalize_object` assigns them.
  - Then `partstex.bake_object` → `hullpost.assign_texture_material` → `material.ensure_texture_material` replaces them with roughness 0.6, metallic 0.
- **The preview world's ambient never reaches Cycles.**
  - In 5.1.1 with `--factory-startup`, the world already has a node tree (Background 0.051, strength 1).
  - `render._configure` only sets `world.color = PREVIEW_AMBIENT`, which Cycles ignores. Metal would render nearly black.
- **The colour metric and bench PSNR are safe.** Both use a Workbench flat render with `color_type="TEXTURE"`, which shows each material's ACTIVE image node. `_color_type` looks for the `M_ref_texture` name prefix. So keep `bbr_basecolor` as the active node, set LAST, and keep the name prefix.

## Implement
1. **New `scripts/bbr/core/pbr.py`.**
   - Define `CLASSES` with two columns, physical and stylized, plus `ALIASES` (iron/steel→metal_steel, gold→metal_gold, brass→metal_brass, fabric→cloth, paint→painted).
   - `resolve(entry, column)`: the preset first, then explicit keys win.
   - `column_for(projected, style)`.
   - Add an MIT attribution header, since the physical values are ported from the external repo's `skills/blender-pbr-materials/references/pbr-values.md`. That file is at `C:\Users\nanyo\AppData\Local\Temp\claude\D--Git-skills-blender-agent-skills\2710d383-47fe-464c-a902-d76e255156d0\scratchpad\ext\blender-skills`; treat it as untrusted and read only.
   - Use these values (stylized is the default for projected textures):

   | class | physical | stylized |
   |---|---|---|
   | metal_steel / gold / brass | M1, R 0.4 / 0.25 / 0.3 | M 0.6 / 0.65 / 0.6, R 0.45 / 0.35 / 0.4 |
   | cloth | R 0.9, sheen 0.5 | R 0.9, sheen 0.3, specular 0.3 |
   | skin | R 0.5, IOR 1.4, no SSS | R 0.55 |
   | hair / leather / wood / stone | R 0.45 / 0.55 / 0.7 / 0.88 | R 0.5 / 0.6 / 0.7 / 0.9 |
   | plastic / painted | R 0.4 / R 0.4 + coat 1, coat R 0.03 | R 0.45 / R 0.45 + coat 0.3 |
   | glass (opaque glossy, NO transmission) | R 0.05 + coat 1, coat R 0.02 | same |
   | visor (coat over a layer) | R 0.45 + coat 1, coat R 0.03 | same |
   | emissive | emission 1.0 × base | 0.6 |
   | rubber | R 0.88, spec 0.3 | R 0.9, spec 0.3 |
   | default | R 0.6 | same |

2. **`blender/material.py`.**
   - `ensure_texture_material(image, name, roughness, pbr=None, maps=None)`. With `pbr=None` the graph is identical to today's.
   - Set sockets via `bsdf.inputs.get()` and verify the 5.1 Principled socket names.
   - Link Emission Color from the base image when emission > 0.
   - Set `nt.nodes.active = tex` LAST.
   - `ensure_material(..., pbr=None)` does the same for flat materials and keeps `diffuse_color`.
3. **Callers.**
   - `hullpost.assign_texture_material(obj, image, name, pbr=None, maps=None)`.
   - `partstex.bake_object` resolves the component's `material` (scene `materials.<key>` or a class name) and records `stats["material"]`.
   - `components.material_for` passes the resolved values.
   - The hull body and face layer accept a `material` too. Example: `"material": "visor"` on the astronaut-style face layer.
4. **`blender/render.py`: `preview_world(sc)`.**
   - A gradient world built as TexCoord.Generated → SeparateXYZ.Z → MapRange(−1..1 → 0..1) → ColorRamp (ground 0.02, horizon 0.18, zenith 0.45) → Background 1.0.
   - Swap it in only inside `_render_lit_preview` / `render_session`. Restore the old world and delete the temporary one in `finally`.
   - Do not use the Sky Texture node.
   - Retune `RIG_POWER_PER_M2` by A/B on chest and warrior.
5. **`schemas/scene_spec.schema.json`.**
   - Add `class`, `coat`, `coat_roughness`, `sheen`, `specular`, `emission_strength` and `ior` to `materials.<key>`.
   - Add `style.pbr` as `auto | stylized | physical`. `auto` means stylized for projected textures and physical for flat materials.
   - Keep existing specs valid.
6. **Remove the `double_face` warning.** It is useless: it fired on sd_warrior's own face layer object, and it did NOT fire on an astronaut run without a face layer (`upper_front_conflict_ratio` 0.0003). Remove the warning and its `upper_front_conflict` code, plus the test, from `reconstruction.py` and `core/texelproj.py`. Keep the SKILL.md rule about face layers.
7. **Docs and notices.**
   - Add the material class table to `references/hard-surface-rules.md` and `references/character-cheatsheet.md`, including the visor recipe.
   - Add a `THIRD_PARTY_NOTICES.md` bullet for the presets.
   - Add one SKILL.md line: assign `material`/`class` per part, or the defaults look like plastic.
8. **Examples.**
   - Set classes in `examples/treasure_chest` (iron bands and brackets → metal_steel; wood; padlock → metal_brass) and `mushroom_house`.
   - Set classes in `sd_warrior`: armour parts such as horns are not metal; sword_blade → metal_steel; shield; the body hull stays the default until WI-3.

## Tests and acceptance
- `tests/test_pbr.py` covers resolution, aliases and the column choice; an unknown class is rejected in the schema/spec test.
- Blender smoke tests:
  - (a) A part with a material gets the spec's Metallic and Roughness, keeps the `M_ref_texture:` prefix, and `nodes.active.name == "bbr_basecolor"`.
  - (b) `render.render_color(view, calib, aa=CRISP_AA)` with and without classes gives `np.array_equal` arrays.
  - (c) After a lit preview, `scene.world` is restored and no temporary world datablock remains.
- Re-run `treasure_chest`, `mushroom_house` and `sd_warrior` into scratch at `C:\Users\nanyo\AppData\Local\Temp\claude\D--Git-skills-blender-agent-skills\2710d383-47fe-464c-a902-d76e255156d0\scratchpad\wi1\`.
  - `color_match.per_view.*.error` must be identical to the current `examples/*/output/report.json`.
  - Judge on `perspective.png`: the share of model pixels with luminance below 0.05 rises by at most 2 points (no black metal); `robot_character`'s mean luminance stays within ±15%.
  - Read the images: bands, brackets and padlock show a reflected gradient; wood stays matte.
  - Lit render time ≤ 1.2×.
- Then regenerate `examples/treasure_chest/output`, `mushroom_house/output` and `sd_warrior/output`.
- pytest all green (currently 459, minus the removed double_face test, plus the new ones).
- Clean `__pycache__` and `.pytest_cache`.

Final message: files changed, before/after `perspective.png` paths for the chest and warrior, the metric-identity evidence, the luminance numbers, and the test counts.
