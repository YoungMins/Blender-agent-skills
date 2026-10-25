"""Run the pipeline on every benchmark case, score the results and write ``summary.md``.

    python3 bench/run_bench.py [--cases chibi,sd_warrior] [--no-variants] [--jobs 2] [--blender PATH] [--out DIR]
    python3 bench/run_bench.py --score-only bench/runs/<timestamp>        # re-score an existing run

Cases are the directories of ``bench/cases`` (``case.json``): synthetic ground-truth cases built by ``make_synthetic.py``
(plus their ``__scale2`` / ``__noacc`` perturbed variants) and regression cases that point at ``examples/``
(``sd_warrior``, ``robot_character``, ``treasure_chest`` = the hard-surface case). Every case runs in its own Blender process into ``bench/runs/<timestamp>/<case>/``
(``run.log``, ``report.json``, previews, ``score.json``); ground-truth cases then get a second process
(``render_holdout.py``) for the held-out renders and the mesh export that ``score.py`` needs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SK = BENCH.parent
sys.path.insert(0, str(BENCH))

import score  # noqa: E402

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
PIPELINE_TIMEOUT_S = 1800
HOLDOUT_TIMEOUT_S = 600


def find_blender(explicit: str | None) -> str:
    for cand in (explicit, os.environ.get("BLENDER"), DEFAULT_BLENDER, shutil.which("blender")):
        if cand and Path(cand).exists() or cand and cand == shutil.which("blender"):
            return cand
    raise SystemExit("Blender not found: pass --blender or set BLENDER")


def discover(cases_dir: Path, names: list[str] | None, variants: bool) -> list[Path]:
    dirs = sorted(p.parent for p in cases_dir.glob("*/case.json"))
    if names:
        missing = sorted(set(names) - {d.name for d in dirs})
        if missing:
            raise SystemExit(f"unknown cases: {missing}; available: {[d.name for d in dirs]}")
        dirs = [d for d in dirs if d.name in names]
    return [d for d in dirs if variants or "__" not in d.name]


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def run_blender(blender: str, args: list[str], log: Path, timeout: int) -> int:
    with log.open("ab") as fh:
        fh.write(f"\n$ blender {' '.join(args)}\n".encode())
        fh.flush()
        try:
            return subprocess.run([blender, "-b", "--factory-startup", *args], stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            fh.write(b"\nTIMEOUT\n")
            return -1


def run_case(case_dir: Path, out_root: Path, blender: str) -> dict:
    """Pipeline + held-out renders + score of one case; never raises (a failure is a ``status`` in the result)."""
    case = score.load_case(case_dir)
    out = (out_root / case["name"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    t0 = time.perf_counter()
    code = run_blender(blender, ["-P", str(SK / "scripts" / "run_pipeline.py"), "--", "--view-spec", str(case["view_spec"]),
                                 "--scene-spec", str(case["scene_spec"]), "--out", str(out)], log, PIPELINE_TIMEOUT_S)
    blends = sorted(out.glob("*.blend"))
    if case.get("holdout_camera") and blends:
        run_blender(blender, ["-P", str(BENCH / "render_holdout.py"), "--", "--blend", str(blends[0]), "--case",
                              str(case["dir"]), "--out", str(out)], log, HOLDOUT_TIMEOUT_S)
    result = score.score_case(case_dir, out)
    result["pipeline_exit_code"] = code
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{case['name']}: {result.get('status')} in {result['wall_s']} s", flush=True)
    return result


def spec_fingerprint(case_dir: Path) -> str:
    case = score.load_case(case_dir)
    if case.get("kind") != "regression":
        return ""
    sp = Path(case["scene_spec"])
    stamp = dt.datetime.fromtimestamp(sp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"`{sp.relative_to(SK).as_posix()}` sha1 {sha1(sp)} (modified {stamp})"


def write_summary(run_dir: Path, results: list[dict], case_dirs: list[Path], elapsed: float) -> Path:
    lines = [f"# BBR benchmark {run_dir.name}", "",
             f"{len(results)} cases, wall time {elapsed:.0f} s. All scores are soft; the hull-only baseline of the synthetic "
             "cases is a silhouette hull with no parts.", "",
             "Columns: train err = weighted mean silhouette error of the validate views; excess = error minus the reference "
             "floor (per view, averaged); held-out IoU = 8 azimuths vs the ground-truth renders with the exact camera; F@1% / "
             "F@2% / Chamfer = recon.obj vs gt.obj (tolerance as a share of the height; Chamfer = mean of both directed "
             "mean distances / height); PSNR = colour of the held-out renders inside both silhouettes; misattr / occlusion = "
             "worst object (best-view occlusion share, warn > 30 %); ring % = ring energy of the silhouette hull (Z-high-passed outer radius "
             "in the azimuths no carve view sees, share of the mean radius; a spiky silhouette turned into stacked horizontal rings "
             "scores high). `occ:` / `mis:` / `ring:` list the flagged objects.", "",
             score.HEADER, score.SEPARATOR]
    lines += [r.get("row", f"| {r['case']} | {r.get('status')} | " + " | ".join(["n/a"] * 12) + " |") for r in results]
    notes = []
    for d in case_dirs:
        fp = spec_fingerprint(d)
        if fp:
            notes.append(f"- {d.name}: scene spec {fp}")
    for r in results:
        if r.get("perturbation"):
            notes.append(f"- {r['case']}: perturbation - {r['perturbation']}")
        if r.get("status") in ("missing_report", "error") or r.get("pipeline_exit_code") not in (0, 2):
            notes.append(f"- {r['case']}: status {r.get('status')} (exit {r.get('pipeline_exit_code')}), see {r['case']}/run.log")
    if notes:
        lines += ["", "## Notes", *notes]
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--cases-dir", default=str(BENCH / "cases"))
    ap.add_argument("--no-variants", action="store_true", help="skip the __scale2 / __noacc perturbed variants")
    ap.add_argument("--jobs", type=int, default=1, help="cases to run in parallel (each is a Blender process)")
    ap.add_argument("--blender")
    ap.add_argument("--out", help="run directory (default bench/runs/<timestamp>)")
    ap.add_argument("--score-only", metavar="RUN_DIR", help="re-score an existing run directory and rewrite its summary")
    args = ap.parse_args(argv)
    names = args.cases.split(",") if args.cases else None
    case_dirs = discover(Path(args.cases_dir), names, not args.no_variants)
    t0 = time.perf_counter()
    if args.score_only:
        run_dir = Path(args.score_only)
        results = []
        for d in case_dirs:
            if (run_dir / d.name).is_dir():
                res = score.score_case(d, run_dir / d.name)
                (run_dir / d.name / "score.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
                results.append(res)
    else:
        blender = find_blender(args.blender)
        run_dir = Path(args.out).resolve() if args.out else BENCH / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"run directory {run_dir}", flush=True)
        if args.jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
                results = list(pool.map(lambda d: run_case(d, run_dir, blender), case_dirs))
        else:
            results = [run_case(d, run_dir, blender) for d in case_dirs]
    path = write_summary(run_dir, results, case_dirs, time.perf_counter() - t0)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
