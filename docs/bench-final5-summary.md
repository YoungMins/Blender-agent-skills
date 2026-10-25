# BBR benchmark final5

17 cases, wall time 1478 s. All scores are soft; the hull-only baseline of the synthetic cases is a silhouette hull with no parts.

Columns: train err = weighted mean silhouette error of the validate views; excess = error minus the reference floor (per view, averaged); held-out IoU = 8 azimuths vs the ground-truth renders with the exact camera; F@1% / F@2% / Chamfer = recon.obj vs gt.obj (tolerance as a share of the height; Chamfer = mean of both directed mean distances / height); PSNR = colour of the held-out renders inside both silhouettes; misattr / occlusion = worst object (best-view occlusion share, warn > 30 %); ring % = ring energy of the silhouette hull (Z-high-passed outer radius in the azimuths no carve view sees, share of the mean radius; a spiky silhouette turned into stacked horizontal rings scores high). `occ:` / `mis:` / `ring:` list the flagged objects.

| case | status | train err % | train excess % | held-out IoU % (mean / min) | F@1% | F@2% | Chamfer % | PSNR dB | misattr max % | occlusion max % | ring % | flagged | run s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chibi | passed | 1.4 | 1.6 | 94.5 / 90.7 | 69 | 79 | 1.42 | 23.2 | 0.0 (body) | 0 (body) | 1.0 | - | 62 |
| chibi__noacc | passed | 3.2 | 1.6 | 86.3 / 81.8 | 45 | 65 | 2.18 | 19.7 | 0.0 (body) | 0 (body) | 0.7 | - | 66 |
| chibi__scale2 | passed | 1.6 | 1.6 | 94.3 / 90.5 | 69 | 78 | 1.49 | 23.3 | 0.0 (body) | 0 (body) | 1.1 | - | 74 |
| creature | passed | 4.5 | 5.1 | 90.1 / 84.7 | 62 | 77 | 1.41 | 28.9 | 0.0 (body) | 15 (body) | 1.3 | float:body#5, float:body#3, float:body#4, float:body#2 | 74 |
| creature__noacc | passed | 3.8 | 3.5 | 49.1 / 28.5 | 8 | 16 | 9.50 | 25.4 | 0.0 (body) | 18 (body) | 2.3 | - | 70 |
| creature__scale2 | passed | 3.9 | 4.1 | 88.1 / 82.5 | 59 | 77 | 1.48 | 28.2 | 0.0 (body) | 15 (body) | 1.4 | float:body#3, float:body#2, float:body#4, float:body#5 | 72 |
| hatted | passed | 2.8 | 3.2 | 87.5 / 78.1 | 80 | 86 | 0.97 | 29.2 | 0.0 (body) | 0 (body) | 1.2 | float:body#2, float:body#3 | 90 |
| hatted__noacc | passed | 3.3 | 3.9 | 86.7 / 77.7 | 79 | 86 | 0.99 | 31.7 | 0.0 (body) | 0 (body) | 1.2 | float:body#2, float:body#3 | 92 |
| hatted__scale2 | passed | 2.9 | 3.2 | 87.6 / 77.8 | 80 | 86 | 1.00 | 29.4 | 0.0 (body) | 0 (body) | 1.1 | float:body#2, float:body#3 | 86 |
| mushroom_house | passed | 3.1 | 1.0 | n/a | n/a | n/a | n/a | n/a | 60.9 (door_frame_lo) | 34 (win_f_barv) | n/a | occ:win_f_barv, mis:door_frame_lo, mis:win_f_barh, mis:win_s_barh, mis:win_b_barh, mis:door_frame_arch, mis:win_f_barv, mis:win_b_glass, mis:win_f_glass, mis:win_s_glass, mis:win_b_barv, mis:win_b_ring, mis:win_s_ring, mis:win_s_barv, mis:win_f_ring, mis:door_knob | 187 |
| robot_character | passed | 2.8 | -2.6 | n/a | n/a | n/a | n/a | n/a | 0.0 (body) | 0 (body) | 1.5 | - | 78 |
| sd_dragon | passed | 6.7 | -3.6 | n/a | n/a | n/a | n/a | n/a | 5.2 (wing_l) | 56 (wing_l) | 2.0 | occ:wing_l, occ:arm_l, occ:wing_r, mis:wing_l | 339 |
| sd_fox_blind | passed | 6.2 | -0.1 | n/a | n/a | n/a | n/a | n/a | 7.2 (ear_l) | 41 (ear_r) | 2.2 | occ:ear_r, occ:ear_l, mis:ear_l | 259 |
| sd_twintail | passed | 7.8 | 0.3 | n/a | n/a | n/a | n/a | n/a | 3.4 (bear_body) | 18 (bear_head) | 2.9 | ring:body | 193 |
| sd_warrior | passed | 5.0 | -0.5 | n/a | n/a | n/a | n/a | n/a | 8.5 (arm_r_upper) | 35 (arm_r_upper) | 2.1 | occ:arm_r_upper, occ:arm_l_upper, mis:arm_r_upper, mis:arm_l_upper | 397 |
| sd_warrior_v6 | passed | 5.3 | 0.7 | n/a | n/a | n/a | n/a | n/a | 26.7 (arm_l_fore) | 55 (arm_r_fore) | 2.2 | occ:arm_r_fore, occ:arm_l_fore, occ:shield, mis:arm_l_fore, mis:arm_r_fore, mis:hand_l, mis:hand_r, mis:shield, mis:sword_pommel | 251 |
| treasure_chest | passed | 2.6 | 1.8 | n/a | n/a | n/a | n/a | n/a | 49.8 (brk_low_front_f_h) | 68 (ring_plate) | n/a | occ:ring_plate, occ:padlock_shackle, occ:brk_low_front_s_h, occ:band_side, occ:band_centre, occ:brk_low_back_s_h, occ:ring, occ:hasp, mis:brk_low_front_f_h, mis:brk_low_back_f_h, mis:brk_up_back_s_v, mis:brk_low_back_s_v, mis:brk_up_front_s_v, mis:brk_low_back_s_h, mis:brk_low_front_s_h, mis:brk_low_front_s_v, mis:brk_low_front_f_v, mis:brk_low_back_f_v, mis:padlock_shackle | 124 |

## Notes
- mushroom_house: scene spec `examples/mushroom_house/scene_spec.json` sha1 b647146853 (modified 2026-10-23 08:18)
- robot_character: scene spec `examples/robot_character/scene_spec.json` sha1 4a3646fa9c (modified 2026-10-22 15:58)
- sd_dragon: scene spec `bench/cases/sd_dragon/scene_spec.json` sha1 7291c79686 (modified 2026-10-23 11:21)
- sd_fox_blind: scene spec `bench/cases/sd_fox_blind/scene_spec.json` sha1 8f65e91bcd (modified 2026-10-23 01:56)
- sd_twintail: scene spec `bench/cases/sd_twintail/scene_spec.json` sha1 f284124f9b (modified 2026-10-23 09:15)
- sd_warrior: scene spec `examples/sd_warrior/scene_spec.json` sha1 3ff524d4f9 (modified 2026-10-23 04:24)
- sd_warrior_v6: scene spec `bench/cases/sd_warrior_v6/scene_spec.json` sha1 4dab5f3a5c (modified 2026-10-22 22:47)
- treasure_chest: scene spec `examples/treasure_chest/scene_spec.json` sha1 9eed8968c7 (modified 2026-10-23 06:58)
- chibi__noacc: perturbation - horn_l, horn_r missing from the side view
- chibi__scale2: perturbation - side view 2 % too short (derived depth drifts)
- creature__noacc: perturbation - tail_1, tail_2, tail_3 missing from the side view
- creature__scale2: perturbation - side view 2 % too short (derived depth drifts)
- hatted__noacc: perturbation - staff, orb missing from the side view
- hatted__scale2: perturbation - side view 2 % too short (derived depth drifts)
