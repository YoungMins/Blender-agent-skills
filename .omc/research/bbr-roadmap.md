# BBR skill: research and roadmap (2026-10-22)

Goal (user, /goal): research how an agent can model well from front/side/back images and make the skill as good as it can be.

## Diagnosis (deep-reasoner, measured on sd_warrior and robot)
1. The agent does metric 3D work by hand (Euler angles, bend, guessed depths). This is the biggest reliability risk.
   Example: the shield ended up embedded in the hull, and a second shield was printed on the cape.
2. Validation is in-sample for the hull, because it carves from the same masks it scores on.
   - v3's fan-shaped horns beat v6's correct cones in the front horn band (0.28% vs 1.69%).
   - The back view is a reference-noise floor: front/back mirror IoU is 0.945, matching the back error of 5.49%.
3. Reference inconsistency is not measured.
   - side vs right mirror IoU is 0.847.
   - Adding `right.png` made colour error worse: front 3.07→4.69%, conflict 6.7→12.9%.
   - Per-view bbox calibration anchors on horn tips.
4. Texture work is at diminishing returns. MRF showed no visible gain, and the remaining defects are geometric (layering, concavity).
5. SKILL.md is too large: about 2000 words, with one bullet of about 500 words.

## Target architecture
The agent makes semantic decisions only (parts, region labels, relations). Tools do all metric work.

| Stage | What happens |
|---|---|
| A Acquire | Single-sheet turnaround, using the prompt template |
| B Check | Consistency gate: pair IoU floors, recommended roles, flags |
| C Decompose | Numbered region proposals, labelled by the agent, become per-part masks |
| D Geometry | Residual hull from the label masks; auto-fit primitives (contact and mirror constraints); layer parts for concavities |
| E Texture | Label-masked per-object projection; mirrored virtual view |
| F Validate | Floor-relative excess error; per-part IoU; misattribution; camera-facing occlusion; held-out views |
| G Output | Visual profile now; game profile later |

## Roadmap (order)
1. Consistency gate, sheet-mode calibration and turnaround prompt template (S)
2. Measurement and benchmark harness: held-out views, misattribution, occlusion, synthetic ground truth, blind-run protocol (M)
3. Delete MRF, the vertex path and test-only code; shrink SKILL.md to 800 words or fewer; make tuned values the defaults (S)
4. Anchor-based placement (`anchors_px`), `attach_to` contact constraint, `mirror_of`, and no Euler angles (S–M)
5. Set-of-marks region proposals: label masks drive hull exclusion and projection (M)
6. Automatic per-part fitting (Nelder–Mead or CMA-ES on silhouette loss) (M–L)
7. Symmetry: mirrored virtual colour view and paired fitting (S–M)
8. Layered parts for concavities (face in helmet) (M–L)

Rejected:
- more texture-selection work
- separately generated single views
- rigid registration as a consistency fix
- photo-consistency carving

## External research highlights
- **Loop:** LL3M, SceneCraft and BlenderAlchemy all use a render-compare-fix loop. Use numeric per-part IoU as the critic instead of a VLM judge.
- **Inflation:** Teddy and Monster Mash style distance-transform inflation for rounding outlines.
- **Texturing:** Waechter et al. 2014 view selection and seam leveling (already tried). Zhou & Koltun misregistration → per-part 2D shift.
- **References:** an A-pose single sheet, then an automatic consistency check before modelling.

## Blind evaluation inputs
- `generated-images/sd_mage_turnaround.png`: SD mage with wide hat brim, staff and braids.
