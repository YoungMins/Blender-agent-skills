"""Score one reconstruction against its case (numpy + PIL only, no Blender).

    python3 bench/score.py --case bench/cases/chibi --run bench/runs/<ts>/chibi

Reads ``<run>/report.json`` and, for ground-truth cases, ``<run>/recon.obj`` + ``<run>/holdout_renders/az_NNN.png``
(written by ``bench/render_holdout.py`` from the reconstructed .blend with the SAME camera as the ground-truth renders).
Writes ``<run>/score.json`` and prints a Markdown row.

Scores (all soft, never gates):
* training views: silhouette IoU, error and ``excess = error - floor`` per validate view (the floor is the reference
  inconsistency, see ``report.consistency``); the mean excess is what the reconstruction could still improve;
* held-out views (ground-truth cases): IoU over the 8 azimuths against the ground-truth renders, with the exact camera
  (no pose fitting), and colour PSNR inside both silhouettes;
* geometry (ground-truth cases): F-score at 1 % and 2 % of the height and normalised Chamfer distance of
  ``recon.obj`` against ``gt.obj``;
* rings: ``ring_pct`` of the silhouette hulls (``report.hull.<id>.rings``, see :mod:`bbr.core.ringmetric`): the Z-high-passed
  outer radius in the azimuths no carve view sees, as a share of the mean radius; a spiky silhouette (hair, fur) turned
  into a stack of horizontal rings ("lampshade") scores high; a hull above ``FLAG_PCT`` is flagged ``ring:<id>``;
* flags: misattribution / camera-facing occlusion maxima and the flagged objects of the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

SK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SK / "scripts"))

from bbr.core import geomscore, imageio  # noqa: E402
from bbr.core.imageops import erode  # noqa: E402
from bbr.core.metrics import iou  # noqa: E402
from bbr.core.ringmetric import FLAG_PCT  # noqa: E402

ALPHA_THRESHOLD = 0.5
PSNR_ERODE_PX = 2
HEADER = ("| case | status | train err % | train excess % | held-out IoU % (mean / min) | F@1% | F@2% | Chamfer % | "
          "PSNR dB | misattr max % | occlusion max % | ring % | flagged | run s |")
SEPARATOR = "|" + "---|" * 14


def load_case(case_dir: Path | str) -> dict[str, Any]:
    """``case.json`` with every path resolved against the case directory (``..`` allowed)."""
    d = Path(case_dir).resolve()
    case = json.loads((d / "case.json").read_text(encoding="utf-8"))
    case["dir"] = d
    for key in ("view_spec", "scene_spec", "gt_obj", "holdout_dir"):
        if case.get(key):
            case[key] = (d / case[key]).resolve()
    return case


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}"


def train_scores(report: dict[str, Any]) -> dict[str, Any]:
    """Training-view IoU / error / floor / excess per validate view and their means."""
    views = {}
    for view, info in (report.get("reference_match", {}).get("per_view") or {}).items():
        if "error" in info and "iou" in info:
            views[view] = {k: info.get(k) for k in ("iou", "error", "floor", "excess_error")}
    excess = [v["excess_error"] for v in views.values() if v["excess_error"] is not None]
    return {"views": views, "total_error": report.get("reference_match", {}).get("total_error"),
            "mean_iou": float(np.mean([v["iou"] for v in views.values()])) if views else None,
            "mean_excess_error": float(np.mean(excess)) if excess else None}


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """PSNR (dB, peak 1) of two RGB images inside ``mask`` (None for an empty mask)."""
    if not mask.any():
        return None
    mse = float(np.mean((a[..., :3][mask] - b[..., :3][mask]) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def heldout_scores(case: dict[str, Any], run_dir: Path) -> Optional[dict[str, Any]]:
    """IoU and colour PSNR of the reconstruction's held-out renders against the ground-truth ones (exact camera)."""
    cam, gt_dir, rec_dir = case.get("holdout_camera"), case.get("holdout_dir"), Path(run_dir) / "holdout_renders"
    if not cam or not gt_dir or not rec_dir.is_dir():
        return None
    per_az: dict[str, Any] = {}
    for az in cam["azimuths"]:
        name = f"az_{int(round(az)):03d}.png"
        gt_path, rec_path = Path(gt_dir) / name, rec_dir / name
        if not gt_path.is_file() or not rec_path.is_file():
            continue
        gt, rec = imageio.load_rgba(gt_path), imageio.load_rgba(rec_path)
        gm, rm = gt[..., 3] > ALPHA_THRESHOLD, rec[..., 3] > ALPHA_THRESHOLD
        both = erode(gm & rm, PSNR_ERODE_PX)
        per_az[str(int(round(az)))] = {"iou": float(iou(gm, rm)), "psnr": psnr(gt, rec, both if both.any() else gm & rm)}
    if not per_az:
        return None
    ious = [v["iou"] for v in per_az.values()]
    psnrs = [v["psnr"] for v in per_az.values() if v["psnr"] is not None]
    return {"per_azimuth": per_az, "mean_iou": float(np.mean(ious)), "min_iou": float(np.min(ious)),
            "mean_psnr": float(np.mean(psnrs)) if psnrs else None}


def frame_shift(run_dir: Path | str, xyz, sign: float = -1.0) -> np.ndarray:
    """``xyz`` (point or (N,3) array) moved by ``sign`` x the report's ``frame_offset_m`` (x, y).

    The ground truth is centred on its bbox; the pipeline's origin may be the body axis instead (a prop drags the bbox
    centre), which translates the whole model. sign -1 brings recon vertices into the ground-truth frame, +1 moves the
    held-out camera target onto the model. Reports without the field (older runs) are not shifted."""
    path = Path(run_dir) / "report.json"
    off = json.loads(path.read_text(encoding="utf-8")).get("frame_offset_m") or {} if path.is_file() else {}
    return np.asarray(xyz, dtype=float) + sign * np.array([off.get("x", 0.0), off.get("y", 0.0), 0.0])


def geometry_scores(case: dict[str, Any], run_dir: Path, n: int = geomscore.DEFAULT_SAMPLES) -> Optional[dict[str, Any]]:
    """F-score @ 1 % / 2 % of the height and normalised Chamfer of ``recon.obj`` against the ground-truth mesh."""
    gt_obj, rec_obj = case.get("gt_obj"), Path(run_dir) / "recon.obj"
    if not gt_obj or not Path(gt_obj).is_file() or not rec_obj.is_file():
        return None
    gv, gt = geomscore.read_obj(gt_obj)
    rv, rt = geomscore.read_obj(rec_obj)
    rv = frame_shift(run_dir, rv)
    return geomscore.mesh_scores(rv, rt, gv, gt, scale=case.get("height_m"), n=n)


def ring_scores(report: dict[str, Any]) -> dict[str, Any]:
    """``ring_pct`` per silhouette hull of the report, its maximum and the hulls above ``FLAG_PCT``."""
    per = {cid: float(h["rings"]["ring_pct"]) for cid, h in (report.get("hull") or {}).items()
           if isinstance(h, dict) and isinstance(h.get("rings"), dict) and h["rings"].get("ring_pct") is not None}
    return {"per_hull": per, "max_pct": max(per.values()) if per else None,
            "flagged": sorted(cid for cid, v in per.items() if v > FLAG_PCT)}


def flags(report: dict[str, Any]) -> dict[str, Any]:
    """Misattribution / occlusion maxima and flagged objects, consistency flags and run status from a report."""
    attr = report.get("attribution") or {}
    cons = report.get("consistency") or {}
    hold = report.get("holdout") or {}
    return {"misattribution_max": attr.get("misattribution_max"), "occlusion_max": attr.get("occlusion_max"),
            "occluded_objects": attr.get("occluded_objects", []), "misattributed_objects": attr.get("misattributed_objects", []),
            "consistency_passed": cons.get("passed"), "consistency_flags": cons.get("flags"),
            "pipeline_holdout_mean_iou": hold.get("mean_iou"), "warnings": report.get("warnings", []),
            "attribution_error": attr.get("error"), "floating_parts": [g["part"] for g in report.get("floating_parts") or []]}


def score_case(case_dir: Path | str, run_dir: Path | str, n_samples: int = geomscore.DEFAULT_SAMPLES) -> dict[str, Any]:
    """All scores of one run (see the module doc); ``status`` is ``missing_report`` when the run produced nothing."""
    case, run = load_case(case_dir), Path(run_dir)
    out: dict[str, Any] = {"case": case["name"], "kind": case.get("kind"), "perturbation": case.get("perturbation")}
    report_path = run / "report.json"
    if not report_path.is_file():
        return {**out, "status": "missing_report"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out.update(status=report.get("status"), passed=report.get("passed"), elapsed_s=report.get("elapsed_s"),
               train=train_scores(report), heldout=heldout_scores(case, run), geometry=geometry_scores(case, run, n_samples),
               rings=ring_scores(report), flags=flags(report))
    out["row"] = markdown_row(out)
    return out


def markdown_row(s: dict[str, Any]) -> str:
    """One Markdown table row of a :func:`score_case` result (see ``HEADER``)."""
    train, held, geo, fl = s.get("train") or {}, s.get("heldout"), s.get("geometry"), s.get("flags") or {}
    rings = s.get("rings") or {}
    mis, occ = fl.get("misattribution_max"), fl.get("occlusion_max")
    flagged = [f"occ:{n}" for n in fl.get("occluded_objects", [])] + [f"mis:{n}" for n in fl.get("misattributed_objects", [])] + [f"float:{n}" for n in fl.get("floating_parts", [])]
    flagged += [f"ring:{n}" for n in rings.get("flagged", [])]
    held_txt = "n/a" if not held else f"{_pct(held['mean_iou'])} / {_pct(held['min_iou'])}"
    cells = [s["case"], str(s.get("status")), _pct(train.get("total_error")), _pct(train.get("mean_excess_error")), held_txt,
             "n/a" if not geo else _pct(geo["fscore@1%"], 0), "n/a" if not geo else _pct(geo["fscore@2%"], 0),
             "n/a" if not geo else _pct(geo["chamfer"], 2),
             "n/a" if not held or held["mean_psnr"] is None else f"{held['mean_psnr']:.1f}",
             "n/a" if not mis else f"{_pct(mis['share'])} ({mis['object']})",
             "n/a" if not occ else f"{_pct(occ['share'], 0)} ({occ['object']})",
             "n/a" if rings.get("max_pct") is None else f"{rings['max_pct']:.1f}",
             ", ".join(flagged) or "-", "n/a" if s.get("elapsed_s") is None else f"{s['elapsed_s']:.0f}"]
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="case directory (contains case.json)")
    ap.add_argument("--run", required=True, help="reconstruction output directory (report.json ...)")
    ap.add_argument("--samples", type=int, default=geomscore.DEFAULT_SAMPLES, help="surface samples per mesh")
    args = ap.parse_args(argv)
    result = score_case(args.case, args.run, args.samples)
    Path(args.run, "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(HEADER)
    print(SEPARATOR)
    print(result.get("row", f"| {result['case']} | {result['status']} |"))
    return 0 if result.get("status") != "missing_report" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
