# Review plan: sd_warrior

## Round 1 (total error 5.1%, 24 review issues)

| Issue | Seen in | Cause | Change |
|---|---|---|---|
| Upper arms: colour error 0.19-0.21, label IoU 0.49 / 0.64, occlusion 0.34 / 0.44, 70% inpainted | review, review_sheet | the pauldrons were part of the body hull and covered the upper-arm capsules; the side label of `arm_l_upper` also held the pauldron (37, 25) while the front label held only the chainmail | `pauldron_l` (front 38, side 37+25, back 33) and `pauldron_r` (front 36, back 35, `mirror_of`) as anchored `sphere` parts with the label extents as radii, excluded from the hull; `arm_l_upper` side label = the chainmail polygon only |

## Round 2 (5.3%)

| Issue | Seen in | Cause | Change |
|---|---|---|---|
| Pauldrons read as dark balls | front close-up, texture-only render | `metal_steel` on a convex sphere reflected the dark lower world next to non-metal hull armour; and the chainmail capsule poked through the plate's front | pauldrons keep the default material like the armour around them; `arm_l_upper` base moved down / back under the plate (front 152,466 -> 148,478; side u 212 -> 205) |

Result: upper-arm colour error 0.19-0.21 -> 0.087-0.13, occlusion 0.34-0.44 -> 0.18, label IoU 0.49 / 0.64 -> 0.61 / 0.75.
The shoulders now read as plate over chainmail, as in the reference.

## Left as is

- The pauldron is a dome, not the drawn angled plate (part fitting snaps the ellipsoid to the label; a plate shell
  part would be needed).
- `body slab` at x +0.10, z 0.08-0.13 (22 x 80 mm): a sliver of the scabbard tip left in the hull after the
  `sword_blade` exclusion; small.
- Sword pommel / blade colour errors and inpainting: small parts, half hidden by the body in every view.
- The `_l` / `_r` names of this example follow the image, not the character (the labels tool warns); geometry and
  pairs are consistent.
