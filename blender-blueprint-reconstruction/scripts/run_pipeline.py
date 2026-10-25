"""CLI for Blender:  blender -b --factory-startup -P run_pipeline.py -- --view-spec V --scene-spec S --out DIR [opts]

Exit codes: 0 = validation passed (or an intentional --stop-after), 2 = ran but validation/blockout gate failed,
1 = error. ``--make-synthetic-references`` renders the SceneSpec through ortho cameras into the ViewSpec image paths.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbr import BBRError, get_logger  # noqa: E402

log = get_logger("run_pipeline")

EXIT_OK, EXIT_ERROR, EXIT_NOT_PASSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI."""
    p = argparse.ArgumentParser(prog="run_pipeline.py", description=__doc__)
    p.add_argument("--view-spec", required=True, help="ViewSpec JSON")
    p.add_argument("--scene-spec", required=True, help="SceneSpec JSON")
    p.add_argument("--out", help="output directory (required unless --make-synthetic-references)")
    p.add_argument("--max-iterations", type=int, help="override workflow.max_iterations")
    p.add_argument("--stop-after", choices=("blockout", "primary", "secondary", "final"),
                   help="write the stage preview and stop: primary = the hull body only, secondary = body plus all parts "
                        "(and the part-gap check), final = everything")
    p.add_argument("--verbose", action="store_true", help="also print Blender's own per-render 'Saved:' lines (quiet by default)")
    p.add_argument("--force", action="store_true", help="continue after a failed blockout / strict consistency gate")
    p.add_argument("--consistency-gate", choices=("warn", "strict"),
                   help="override workflow.consistency_gate: strict stops when the reference views disagree")
    p.add_argument("--ignore-flag", action="append", metavar="TYPE[:VIEW]",
                   help="consistency flag judged a false alarm (repeatable); also workflow.consistency.ignore_flags")
    p.add_argument("--render-scale", type=float, help="render masks at this multiple of the reference resolution")
    p.add_argument("--blend", help="open this .blend first (re-run on an existing scene) instead of an empty scene")
    p.add_argument("--make-synthetic-references", action="store_true",
                   help="render the SceneSpec into the ViewSpec image paths (test references) and exit")
    return p


def _blender_args(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


def make_synthetic(args: argparse.Namespace) -> int:
    """``--make-synthetic-references``: build the SceneSpec and render every view of the ViewSpec."""
    from bbr.blender import reconstruction as rc, synthetic
    from bbr.core.spec import load_scene_spec, load_view_spec
    spec = load_scene_spec(args.scene_spec)
    if any(c.type == "silhouette_hull" for c in spec.components):
        raise BBRError("synthetic references need a primitive-based SceneSpec (silhouette_hull needs references)")
    view_spec = load_view_spec(args.view_spec, check_files=False)
    inp = rc.Inputs(spec, spec.to_dict())
    rc.rebuild_all(inp, rc.resolve_config(None, spec.workflow))
    written = synthetic.make_synthetic_references(view_spec)
    for key, path in written.items():
        print(f"synthetic reference {key}: {path}")
    return EXIT_OK


def main(argv: list[str]) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    from bbr.blender import quiet
    quiet.ENABLED = not args.verbose
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
    try:
        import bpy
        if args.blend:
            bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        if args.make_synthetic_references:
            return make_synthetic(args)
        if not args.out:
            raise BBRError("--out is required")
        from bbr.blender import reconstruction as rc
        report = rc.run_reconstruction(args.view_spec, args.scene_spec, args.out,
                                       overrides={"max_iterations": args.max_iterations,
                                                  "render_scale": args.render_scale,
                                                  "consistency_gate": args.consistency_gate,
                                                  "consistency_ignore_flags": args.ignore_flag},
                                       stop_after=args.stop_after, force=args.force)
        print(rc.format_summary(report), flush=True)
        if report["status"].startswith("stopped"):
            return EXIT_OK
        return EXIT_OK if report["passed"] else EXIT_NOT_PASSED
    except Exception as exc:  # CLI boundary: report and map to exit code 1
        log.exception("pipeline failed")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main(_blender_args(sys.argv)))
