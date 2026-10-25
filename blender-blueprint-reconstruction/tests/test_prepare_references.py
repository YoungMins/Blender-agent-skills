"""prepare_references.py CLI on synthetic images."""
import json
import sys

import pytest

from bbr.core import imageio
from conftest import SK, ellipse_mask, paint, rect_mask

sys.path.insert(0, str(SK / "scripts"))
import prepare_references as cli  # noqa: E402


@pytest.fixture
def sheet(tmp_path):
    h, w = 200, 600
    m = ellipse_mask(h, w, 100, 100, 80, 40) | ellipse_mask(h, w, 100, 300, 80, 30) | rect_mask(h, w, 440, 20, 540, 180)
    path = tmp_path / "sheet.png"
    imageio.save_rgba(path, paint(m, (0.9, 0.9, 0.9)))
    return path


def test_split_draft_masks(sheet, tmp_path, capsys):
    out = tmp_path / "refs"
    assert cli.main(["split-sheet", "--image", str(sheet), "--out-dir", str(out), "--names", "front,side,back"]) == 0
    assert all((out / f"{n}.png").is_file() for n in ("front", "side", "back"))
    assert imageio.load_rgba(out / "front.png").shape[0] == 200               # full sheet height kept
    assert cli.main(["draft-view-spec", "--dir", str(out), "--height", "1.0", "--side-axis", "+X"]) == 0
    spec = json.loads((out / "view_spec.json").read_text())
    assert spec["views"]["side"]["camera_axis"] == "+X" and spec["known_dimensions"]["height"] == 1.0
    assert cli.main(["masks", "--view-spec", str(out / "view_spec.json"), "--out-dir", str(out / "masks")]) == 0
    text = capsys.readouterr().out
    assert "dimensions (m)" in text and (out / "masks" / "front_mask.png").is_file()


def test_errors_are_reported(tmp_path, capsys):
    assert cli.main(["draft-view-spec", "--dir", str(tmp_path), "--height", "1"]) == 1


def _views(tmp_path, front_mask, bg=(0.0, 0.8, 0.2), side_mask=None, name_dir="fox/references"):
    folder = tmp_path / name_dir
    folder.mkdir(parents=True)
    h, w = front_mask.shape
    side_mask = ellipse_mask(h, w, 100, 100, 80, 30) if side_mask is None else side_mask
    for key, m in (("front", front_mask), ("side", side_mask), ("back", front_mask)):
        imageio.save_rgba(folder / f"{key}.png", paint(m, (0.9, 0.5, 0.2), bg=bg))
    return folder


def test_draft_view_spec_is_complete_and_runnable(tmp_path, capsys):
    folder = _views(tmp_path, ellipse_mask(200, 200, 100, 100, 80, 40))
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "0.6", "--side-axis", "+X"]) == 0
    spec = json.loads((folder / "view_spec.json").read_text())
    assert spec["name"] == "fox"                                                    # the folder above the image folder
    views = spec["views"]
    assert (views["front"]["camera_axis"], views["back"]["camera_axis"], views["side"]["camera_axis"]) == ("-Y", "+Y", "+X")
    assert views["front"]["roles"] == ["carve", "color", "validate"] and views["side"]["roles"] == ["carve", "color", "validate"]
    assert views["back"]["roles"] == ["color", "validate"]
    assert views["front"]["background"] == {"mode": "auto", "tolerance": 0.12, "remove_enclosed": True}   # green backdrop: chroma
    assert spec["symmetry"]["x"] is True                                            # a centred ellipse: mirror IoU 1
    assert "chroma backdrop: front green" in capsys.readouterr().out
    from bbr.core.spec import load_view_spec
    assert load_view_spec(folder / "view_spec.json").view_axes() == {"front": "-Y", "side": "+X", "back": "+Y"}


def test_draft_view_spec_symmetry_from_the_mirror_iou_name_flag_and_grey_backdrop(tmp_path):
    lopsided = ellipse_mask(200, 200, 100, 100, 80, 40) | rect_mask(200, 200, 140, 20, 190, 70)      # a one-sided prop
    folder = _views(tmp_path, lopsided, bg=(0.5, 0.5, 0.5))
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "1", "--name", "mage"]) == 0
    spec = json.loads((folder / "view_spec.json").read_text())
    assert spec["name"] == "mage" and spec["symmetry"]["x"] is False
    assert spec["views"]["front"]["background"] == {"mode": "auto", "tolerance": 0.12}   # grey: no chroma handling
    assert spec["views"]["side"]["camera_axis"] == "-X"


def test_check_write_records_overrides_where_check_and_the_pipeline_read_them(tmp_path, capsys):
    folder = _views(tmp_path, ellipse_mask(200, 200, 100, 100, 80, 40) | rect_mask(200, 200, 150, 150, 190, 190))
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "1"]) == 0
    capsys.readouterr()
    spec_path = folder / "view_spec.json"
    args = ["check", "--view-spec", str(spec_path), "--out-dir", str(tmp_path / "chk"), "--json"]
    assert cli.main(args) == 0
    plain = json.loads(capsys.readouterr().out)
    assert plain["flags"] or plain["roles_to_change"]                               # something to judge
    flag = plain["flags"][0]["type"] if plain["flags"] else "scale_drift"
    changed = plain["roles_to_change"]
    only_cli = ["--ignore-flag", flag, *sum((["--roles-override", f"{v}={','.join(plain['views'][v]['current_roles'])}"] for v in changed), [])]
    assert cli.main(args + only_cli) == 0
    capsys.readouterr()
    assert "consistency" not in json.loads(spec_path.read_text())                   # without --write nothing is stored
    assert cli.main(args + only_cli + ["--write"]) == 0
    capsys.readouterr()
    stored = json.loads(spec_path.read_text())["consistency"]
    assert flag in stored["ignore_flags"] or not plain["flags"]
    from bbr.core.spec import load_view_spec
    assert load_view_spec(spec_path).consistency == stored                         # the pipeline's ViewSpec carries it
    assert cli.main(args) == 0                                                      # a later plain check applies it
    again = json.loads(capsys.readouterr().out)
    assert not again["roles_to_change"] and all(f["type"] != flag for f in again["flags"])


def test_pipeline_merge_adds_cli_flags_to_the_recorded_ones():
    from bbr.core.consistency import merge_overrides
    flags, roles = merge_overrides({"ignore_flags": ["scale_drift"], "roles_override": {"back": ["color"]}},
                                   {"ignore_flags": ["missing_accessory", "scale_drift"], "roles_override": {"back": ["color", "validate"]}},
                                   None, {"ignore_flags": ["vertical_shift:side"]})
    assert flags == ["scale_drift", "missing_accessory", "vertical_shift:side"]     # a union, never a replacement
    assert roles == {"back": ["color", "validate"]}                                 # the later source wins a view


def test_suggest_parts_cli_writes_json_and_numbered_overlays(tmp_path, capsys):
    from test_suggest import sphere_with_spikes
    mask, _ = sphere_with_spikes([-90, 20, 160])
    folder = _views(tmp_path, mask, side_mask=sphere_with_spikes([-90])[0])
    assert cli.main(["draft-view-spec", "--dir", str(folder), "--height", "1.0"]) == 0
    out = tmp_path / "suggest"
    assert cli.main(["suggest-parts", "--view-spec", str(folder / "view_spec.json"), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "suggestions" in text and "exclude_components" in text
    res = json.loads((out / "suggestions.json").read_text())
    assert len(res["suggestions"]) >= 3 and res["labels"]["front"] and res["components"][0]["params"]["anchors_px"]
    assert all((out / f"suggest_{v}.png").is_file() for v in ("front", "side", "back"))
    assert imageio.load_rgba(out / "suggest_front.png").shape[:2] == mask.shape
