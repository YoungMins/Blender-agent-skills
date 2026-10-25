"""Render the reconstructed model from the held-out camera of its case and export its mesh.  Run inside Blender:

    blender -b --factory-startup -P bench/render_holdout.py -- --blend RUN/<name>.blend --case bench/cases/chibi --out RUN

Writes ``<out>/holdout_renders/az_NNN.png`` (same camera, resolution and flat colour rendering as the ground-truth
``holdout/`` renders) and ``<out>/recon.obj`` (evaluated AI_MODEL meshes, world space). The camera target is moved by the report's ``frame_offset_m``
(the model origin is the body axis when a prop drags the bbox centre; the ground truth is centred on its bbox). ``score.py`` compares both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SK / "scripts"))

import bpy  # noqa: E402

from bbr.blender import synthetic  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    case = json.loads((Path(args.case) / "case.json").read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    print("exported", synthetic.export_obj(out / "recon.obj"))
    if case.get("holdout_camera"):
        cam = dict(case["holdout_camera"])
        cam["centre"] = list(score.frame_shift(out, cam["centre"], sign=1.0))   # the model origin may be the body axis, not the bbox centre
        written = synthetic.render_azimuths(cam, out / "holdout_renders")
        print(f"rendered {len(written)} held-out views")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
