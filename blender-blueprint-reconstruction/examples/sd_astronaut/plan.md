# Review plan: sd_astronaut

The agent's review pass (SKILL.md step 10), kept as a worked example. Each round lists what the review and the
previews showed, the cause, and the spec change.

## Round 1: first passing run (total error 1.5%)

| Issue | Seen in | Cause | Change |
|---|---|---|---|
| Visor painted on the face, no glass | perspective | no glass object | `visor` layer with `shell: true`, `cover: ["face"]`, `glass_clear`; side label `visor` |
| Nose, mouth and fringe as ridges across the face | turntable, oblique face renders | visual hull extrudes the side profile across the width | face layer `relief: true` (widths from the front view) |
| Face hidden from the side (`face color_error (side) 0.28`) | review | helmet walls carved to the front of the face | tool fix: the side carve mask loses the layer and glass side labels |

## Round 2 (total error 1.6%; one review issue)

| Issue | Seen in | Cause | Change |
|---|---|---|---|
| Arms are flat boards from 40 / 140 degrees (`slab` x -0.154 and +0.138, 60 x 181 mm) | turntable, review | the AI sheet holds the arms out in front and hangs them over the torso in the side; the hull took the torso + backpack depth | `arm_r` capsule + `hand_r` sphere, `_l` by `mirror_of`, all four in `exclude_components` |

Judgement: `anchors` reported a 4 cm height disagreement for the hand and 111 mm for the shoulder between front and
side (the side arm is foreshortened and drawn lower). The front shows the pose clearly, so the heights come from it
and the side only gives the depth (u). The silhouette error rose from 1.6% to 3.0% (the capsules do not follow the
drawn outline exactly), well under the 8% threshold; the arms are now round from every angle, which is what the
reference character is.

## Round 3 (total error 2.9%)

| Issue | Seen in | Cause | Change |
|---|---|---|---|
| Hands are sphere mittens; the reference has chunky gloves with fingers and a thumb | hand close-up vs front reference | no hand shape | `hand_r` / `hand_l` as `capsule` with `"fingers": 4` (wrist -> fingertips from the front, depth from the side) |
| First glove too thin, fingers splayed like claws; width check failed (-13%) | close-up, dimension check | default palm toward the body showed the hand edge-on from the front; finger proportions too slim | `palm: "+Y"`, thumb toward the body (the reference shows the back of the glove); glove defaults made chunky (short thick fingers, 60 degree curl, thicker palm and thumb); front radius 24 -> 27 px |

## Left as is

- `face color_error (side)` near 0.09: painted shading of the profile seen through the glass.
- `arm_r color_error (side)` 0.12, `hand_r` / `hand_l color_error (front)` 0.10 / 0.09: the side view draws the arm
  foreshortened and lower than the front (the disagreement above); the capsule follows the front, so its side
  outline and the drawn one differ. Matching both would need a pose no single 3D arm has. 
