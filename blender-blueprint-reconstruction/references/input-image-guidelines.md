# Input Image Guidelines

Reconstruction quality is bounded by reference quality. Check the references against this list **before** building, and record any problems in `view_spec.reference_quality.notes` and in each view's `confidence`.

## Ideal references

- Close to orthographic, or a long telephoto lens. No visible vanishing lines.
- Little lens distortion.
- The object is centered, and the **whole object is visible**, with no crop at any edge.
- **Same scale in every view**: feet or floor on the same baseline, and tops aligned.
- A plain, flat background that **contrasts with the object**. A chroma green such as `#00B140` is ideal for white or grey objects. Transparent PNGs (real alpha) are best.
- Front and side views at exactly 90° to each other.
- At least one **real dimension** (`known_dimensions.height` is the most robust choice).
- One object per image. In a turnaround sheet, the views are separated by background gaps and do not overlap.
- Flat, even lighting, and no cast shadow on the floor. A shadow becomes part of the silhouette.

## Problematic references

| Problem | Effect | What to do |
|---|---|---|
| Strong perspective or wide angle | Near parts look too big, so the silhouette is inconsistent across views | Set confidence to `inferred`. Use the view for color only and drop `carve` from its roles. |
| Object partially cropped | Hull or bbox is truncated | Do not carve with that view. Request a new image. |
| Several objects overlapping | The mask merges them | Crop with `crop_px` or split the sheet first. |
| Angle not exactly front or side | Width or depth comes out wrong | Mark the view `inferred`. Prefer the other views for carving. |
| Front and side at different scales | Height disagrees between views | Calibration normalizes per-view height. Still verify the result against `known_dimensions`. |
| White object on white background | The mask leaks through, leaving holes in the hull | Use the alpha or chroma background. The flood fill only removes background connected to the border, but outline gaps still leak. |
| "Top" view rendered at a high angle (common with image generators) | Not a true top projection | Use `roles: ["preview"]` and confidence `unknown`. Never carve with it. |
| Ground shadow | A flat blob at the feet | Regenerate or paint it out before use. |

## Image generators (gpt-image etc.)

- Ask for **one turnaround sheet** (front | side | back) in a single image rather than separate images. Views generated together stay far more consistent with each other.
- Use the ready prompt in [turnaround-prompt.md](turnaround-prompt.md): orthographic, shared baseline and head line, A-pose, accessory placement stated per view, chroma green, plus a held-out 3/4 sheet prompt for validation.
- A chroma or flat backdrop seen through gaps the subject encloses (a staff beside hair under a hat brim, a hand on a hip) is removed from the mask automatically (`background.remove_enclosed`, default on for chroma backdrops, off for white / grey / black; `false` keeps the gaps). The log prints how many holes and what share of the image were removed; check the mask PNGs.
- Run `scripts/prepare_references.py check --view-spec ...` on the result before building. It measures the opposite-pair mirror IoU, the validation error floor and the inconsistency flags (see the [consistency gate](validation-rules.md#reference-consistency-gate)). Never add a separately generated single view without passing it.
- Ask for: "strict orthographic, same scale, feet on the same baseline, generous gaps between views, flat solid chroma-green background, no text, no shadow".
- Split the sheet with `scripts/prepare_references.py split-sheet`. Crops keep the full sheet height, so vertical alignment survives. `draft-view-spec` then sets `calibration.mode: "sheet"` on those views: one shared scale and baseline, horizontal origin from the body core instead of the bbox centre.
- Generated top views are usually perspective. Treat them as preview only unless their silhouette agrees with the front width and side depth (see `top_consistency_tol`).
- Generators rarely produce exact mirror-consistent front and back views. Use `back` for color and validation, not for carving.

## Toward reference quality scoring (future)

The fields `reference_quality` and per-view `confidence` are where an automatic quality score will go later. Candidate metrics include perspective estimation from line convergence, cross-view height agreement, background uniformity, and the fraction of mask touching the image border, which indicates a crop.
