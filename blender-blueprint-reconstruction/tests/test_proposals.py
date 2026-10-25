"""Region proposals (numbered, deterministic, <= 40 per view) and the overlay."""
import numpy as np

from bbr.core import proposals
from conftest import ellipse_mask, paint, rect_mask


def _figure():
    h, w = 200, 120
    body = ellipse_mask(h, w, 110, 60, 60, 35)
    horn = rect_mask(h, w, 22, 10, 30, 90)                                  # thin protrusion
    mask = body | horn
    img = paint(mask, (0.6, 0.6, 0.6))
    img[ellipse_mask(h, w, 90, 60, 20, 15)] = (0.9, 0.1, 0.1, 1.0)         # red chest patch
    img[rect_mask(h, w, 50, 150, 70, 168)] = (0.1, 0.1, 0.9, 1.0)          # blue belt
    return img, mask


def test_proposals_are_deterministic_and_stable():
    img, mask = _figure()
    a = proposals.propose_regions(img, mask)
    b = proposals.propose_regions(img.copy(), mask.copy())
    assert np.array_equal(a["id_map"], b["id_map"]) and a["regions"] == b["regions"]
    areas = [r["area"] for r in a["regions"]]
    assert areas == sorted(areas, reverse=True) and [r["id"] for r in a["regions"]] == list(range(1, len(areas) + 1))


def test_regions_partition_the_mask_and_respect_the_cap():
    img, mask = _figure()
    res = proposals.propose_regions(img, mask, max_regions=5)
    ids = res["id_map"]
    assert len(res["regions"]) <= 5 and ((ids > 0) == mask).all()
    for r in res["regions"]:
        x0, y0, x1, y1 = r["bbox"]
        assert r["area"] == int((ids == r["id"]).sum()) and mask[int(r["label_px"][1]), int(r["label_px"][0])]
        assert x0 <= r["centroid"][0] <= x1 and y0 <= r["centroid"][1] <= y1


def test_colour_patches_and_thin_protrusion_become_regions():
    img, mask = _figure()
    res = proposals.propose_regions(img, mask)
    ids = res["id_map"]
    assert ids[90, 60] != ids[110, 30] != ids[160, 60]                      # red patch / grey body / blue belt apart
    horn_id = ids[40, 26]
    assert horn_id != ids[110, 30] and next(r for r in res["regions"] if r["id"] == horn_id)["kind"] == "protrusion"


def test_overlay_draws_numbers_and_crops():
    img, mask = _figure()
    res = proposals.propose_regions(img, mask)
    ov = proposals.render_overlay(img, res["id_map"], res["regions"])
    assert ov.shape == (400, 240, 4) and ov.dtype == np.float32
    assert (ov[..., :3] == 1.0).all(axis=2).any()                           # white digit pixels exist
    crop = proposals.render_overlay(img, res["id_map"], res["regions"], 4, box=[10, 10, 70, 90])
    assert crop.shape[:2] == (80 * 4, 60 * 4)


def test_propose_cli_compact_writes_one_small_sheet(tmp_path, capsys):
    import json
    import sys
    from bbr.core import imageio
    from conftest import SK, paint
    sys.path.insert(0, str(SK / "scripts"))
    import prepare_references as cli
    mask = np.zeros((400, 300), bool)
    mask[60:340, 80:220] = True
    img = paint(mask, (0.8, 0.3, 0.2))
    img[60:150, 80:220, :3] = (0.2, 0.3, 0.9)
    imageio.save_rgba(tmp_path / "front.png", img)
    (tmp_path / "view_spec.json").write_text(json.dumps({"spec_version": "0.1", "name": "t", "views": {"front": {"path": "front.png"}},
                                                         "known_dimensions": {"height": 0.5}}), encoding="utf-8")
    out = tmp_path / "regions"
    assert cli.main(["propose", "--view-spec", str(tmp_path / "view_spec.json"), "--out-dir", str(out), "--compact",
                     "--max-regions", "6"]) == 0
    files = sorted(p.name for p in (out / "front").iterdir())
    assert files == ["compact.png", "id_map.npy", "regions.json"]            # no overlay.png / crops
    assert len(json.loads((out / "front" / "regions.json").read_text())["regions"]) <= 6
    sheet = imageio.load_rgba(out / "front" / "compact.png")
    assert sheet.shape[0] < 400 and sheet.shape[1] < 300 + 1                 # cropped to the subject at 1x
    assert cli.main(["propose", "--view-spec", str(tmp_path / "view_spec.json"), "--out-dir", str(out)]) == 0
    assert (out / "front" / "overlay.png").is_file()
