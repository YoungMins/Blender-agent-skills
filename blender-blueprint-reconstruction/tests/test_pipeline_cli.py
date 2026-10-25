"""run_pipeline CLI surface: stop stages, quiet native output, the delimited result block (no Blender needed)."""
import os
import sys

from bbr.blender import quiet
from bbr.core.summary import format_summary
from conftest import SK

sys.path.insert(0, str(SK / "scripts"))
import run_pipeline  # noqa: E402


def test_stop_after_secondary_and_verbose_are_accepted():
    args = run_pipeline.build_parser().parse_args(["--view-spec", "v", "--scene-spec", "s", "--out", "o", "--stop-after", "secondary"])
    assert args.stop_after == "secondary" and args.verbose is False
    assert run_pipeline.build_parser().parse_args(["--view-spec", "v", "--scene-spec", "s", "--verbose"]).verbose is True


def test_native_stdout_is_silenced_inside_the_block_and_restored_after(capfd):
    os.write(1, b"before\n")
    with quiet.native_stdout():
        os.write(1, b"Saved: 'x.png'\n")
    os.write(1, b"after\n")
    assert capfd.readouterr().out == "before\nafter\n"
    quiet.ENABLED = False
    try:
        with quiet.native_stdout():
            os.write(1, b"shown\n")
    finally:
        quiet.ENABLED = True
    assert capfd.readouterr().out == "shown\n"


def test_result_block_is_delimited_and_carries_the_part_gaps():
    report = {"name": "sd_penguin", "status": "stopped_after_secondary", "outputs": {"report": "r.json"}, "elapsed_s": 1.0,
              "part_gaps": [{"a": "handle", "b": "head", "ka": "base", "kb": "base", "gap_m": 0.031, "tol_m": 0.01,
                             "why": "anchored to the same pixel in the references"}]}
    lines = format_summary(report).splitlines()
    assert lines[0] == "===== BBR RESULT: sd_penguin =====" and lines[-1] == "===== END BBR RESULT: sd_penguin ====="
    assert any(l.startswith("WARNING part_gap: handle.base and head.base") and "31 mm" in l for l in lines)
