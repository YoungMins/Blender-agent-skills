# Turnaround Prompt Template (gpt-image)

Reference consistency is the ceiling of reconstruction quality. Views generated **together in one image** agree with each other far better than views generated one by one. Measured on the shipped examples, even good sheets disagree by 5-7% between the front and the mirrored back (the validation error floor), and a separately generated second profile disagrees with the first by 15%.

Workflow: generate **one sheet**, split it with `prepare_references.py split-sheet`, draft the spec with `draft-view-spec` (it marks the views as one sheet, `calibration.mode: "sheet"`), then run `prepare_references.py check --view-spec ...` **before** building anything.

## Main sheet (carve + color + validate)

Fill the `<...>` slots. Keep the numbered rules verbatim; they keep the views consistent.

```
One single image: a character turnaround sheet of <CHARACTER: one sentence - proportions, outfit, colors>.
Layout, left to right on ONE row, equal-sized figures, wide empty gaps between them:
  1. FRONT view
  2. PROFILE view, the character faces the RIGHT edge of the image
  3. BACK view
  <optional 4. PROFILE view, the character faces the LEFT edge of the image>

Rules:
1. Strict orthographic projection: no perspective, no foreshortening, no 3/4 angles. Every view is exactly 0, 90 or 180 degrees from the front.
2. IDENTICAL scale in all views, as if rendered by the same orthographic camera. The soles of the feet touch ONE shared horizontal baseline. The top of the head is on ONE shared horizontal line in every view.
3. Neutral A-pose: standing, legs straight and slightly apart, arms held 30-40 degrees away from the torso, hands open. No walking, no leaning, no tilted head.
4. Accessories - every view must agree, state each one:
   <one line per accessory, for example>
   - The round shield is carried FLAT ON THE BACK: in the front view only its rim peeks out behind the shoulders; in the back view it is a full disc; in BOTH profiles it is a thin vertical disc just behind the back (about <N> cm thick, centered at hip height).
   - The sword hangs from the belt on the character's left hip: <say where hilt and blade appear in the front view, in each profile and in the back view>.
   - Horns, ears, tail, hair: same size and position in every view.
5. Flat, even, shadowless lighting. No cast shadow, no contact shadow on the floor, no reflections, no gradient on the background.
6. Background: solid chroma green #00B140, perfectly uniform. NO green anywhere on the character (no green clothing, gems or eyes). Nothing touches the image borders or another figure; keep at least 5% empty margin around each figure.
7. No text, no labels, no grid lines, no turntable, no props other than the listed accessories.
```

Notes
- State the position of **every** accessory in **every** view. A generator that is not told that the shield shows in the profiles usually omits it there (the `sd_warrior` side view lacks its back shield), and the mismatch cannot be repaired afterwards.
- Each one-sided accessory costs consistency. Prefer a design without large asymmetric props when you have the choice.
- Chroma green is for white, grey or colored characters. For a green character use magenta `#FF00FF` and say so in rule 6.
- If the generator returns a perspective or 3/4 image, discard it; do not use it as a front/side/back view.

## Held-out 3/4 sheet (validation only)

A second, independent image for judging the result from angles the model was **not** built from. Never carve or color from it; compare it by eye (or keep it as a `preview` view) next to `previews/perspective.png`.

```
Using the same character as in the attached turnaround sheet (attached: <sheet image>), generate ONE image with
two figures side by side: a FRONT-LEFT three-quarter view (45 degrees) and a BACK-RIGHT three-quarter view (45 degrees).
Same neutral A-pose, same proportions, same colors, same accessory placement as the sheet. Same scale in both
figures, soles on one shared baseline, flat shadowless lighting, solid chroma green #00B140 background, no green
on the character, nothing touching the borders, no text.
```

## Adding views later

**Never add a separately generated single view without passing `check`.** A view that was not generated in the same image as the others brings its own scale, baseline and proportions. Procedure:

1. Add it to the view spec with only the `color` role.
2. Run `python scripts/prepare_references.py check --view-spec view_spec.json --out-dir <dir>`.
3. Keep the roles the gate recommends (opposite-pair mirror IoU: >= 0.95 carve, >= 0.92 color/validate, otherwise drop), read the XOR overlays, and fix or drop the view if it raises `scale_drift`, `vertical_shift`, `missing_accessory` or `mirrored_view`.
4. For a missing second profile, regenerate it **together with** the other views (a new sheet) instead of on its own.
