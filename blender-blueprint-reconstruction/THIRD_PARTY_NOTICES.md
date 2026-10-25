# Third-party notices

## EL4CTEO/blender-skills (MIT)

Parts of the silhouette-hull post-processing were adapted (re-written, not vendored) from
<https://github.com/EL4CTEO/blender-skills>.

```
MIT License

Copyright (c) 2026 EL4CTEO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Adapted ideas (module: source script):

- `scripts/bbr/blender/hullpost.py`, voxel cleanup: headless-safe voxel remesh through a collapsed Remesh modifier
  (`blender-sculpting/scripts/sculpt_ops.py`; no Sculpt Mode, which crashes with `-b`).
- `scripts/bbr/blender/hullpost.py`, retopo: quadriflow / decimate followed by a shrinkwrap back onto the source,
  with `temp_override` contexts (`blender-retopology-uv/scripts/auto_retopo.py`). Changed: nearest-surface shrinkwrap,
  pre-decimation and manifold repair before quadriflow, automatic fallback to decimate.
- `scripts/bbr/blender/hullpost.py`, UV unwrap: smart project + island packing with a pixel-based margin
  (`blender-retopology-uv/scripts/uv_pack.py`).
- `scripts/bbr/blender/render.py`, preview lighting: studio 3-point rig (key / fill / rim azimuth, elevation, power
  and size ratios) (`blender-lighting-lookdev/scripts/turntable.py`).
- `scripts/bbr/core/pbr.py`, material class presets: the physical column ports the measured roughness / metallic / coat /
  sheen values of `skills/blender-pbr-materials/references/pbr-values.md`. Changed: the stylized column, class names and
  aliases, glass without transmission.
- Floating-island detection idea (`blender-asset-qa/scripts/qa_check.py`) informed `scripts/bbr/core/islands.py`.

Everything else in the post-processing (UV-space rasterizer, depth-buffer visibility, texel color projection,
padding, color metric) is original numpy code of this skill.
