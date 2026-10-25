#!/usr/bin/env python3
"""Prepare reference images (plain python3 + PIL): split a turnaround sheet, extract masks, draft a ViewSpec.

  split-sheet --image IMG --out-dir DIR --names front,side,back [--bg-tol 0.12]
  masks       --view-spec P --out-dir DIR
  draft-view-spec --dir DIR (--height H | --width W | --depth D [--size-of body|silhouette]) [--side-axis -X] [--name N]
              (a runnable view_spec.json: camera axes, roles, background + chroma detection, symmetry.x from the front mirror IoU)
  measure     --view-spec P --view V [--px x,y ...] [--bbox x0,y0,x1,y1] [--json]
  measure     --view-spec P --points-file pts.json [--out metres.json] [--labels labels.json]
              (batch: pts.json = {view: {name: [x, y] | [x0, y0, x1, y1]}} in pixels -> metres JSON per name, plus the calibration and
               the pixel -> metre formula of each view; --labels = the scene's parts_labels, whose body label can move the axis)
  check       --view-spec P [--out-dir D] [--strict] [--json] [--ignore-flag TYPE[:view]] [--roles-override view=role,role] [--write]
              (do the views agree? mirror IoU, floors, XOR overlays; the two options record a flag / role change judged a false alarm,
               --write stores them in the view spec's "consistency" block so the pipeline applies them too)
  propose     --view-spec P --out-dir D [--views a,b] [--max-regions 40] [--compact]
              (numbered region proposals per view: D/<view>/overlay.png (+ crop_*.png), id_map.npy, regions.json; label them)
  labels      --view-spec P --labels labels.json --out-dir D [--scene-spec S] [--z-tol 0.04]
              (labels.json {view: {part: [regions]}} -> per-part masks, colour-coded previews, cross-view z-range check)
  (draft-view-spec and check also take --labels labels.json [--asymmetric NAME_OR_GLOB ...] [--regions-dir D]: the
   symmetry line then also says which labelled parts (a held prop, a posed arm) explain a low mirror IoU; labels prints it too)
  anchors     --from-px VIEW:name=x,y [...] [--id ID]   (typed pixel points -> the anchors_px snippet, no image needed)
  anchors     --view-spec P --view V [--view V2 ...] --bbox x0,y0,x1,y1 [--kind tube|capsule|blade|plate|disc|sphere|strand] [--points N] [--id ID] [--base-hint [VIEW:]x,y ...] [--overlay PNG]
              (propose anchors_px for one part in one view from a rough pixel box; prints a JSON snippet for the SceneSpec;
               with --labels labels.json --part ID the box is that part's label mask instead; several --view with --labels give
               one snippet for all views, a confidence and a cross-view base/tip warning; --kind auto (default) also picks sphere / capsule /
               cylinder / cone and prints suggested_type; the base is the end nearer the body label; warns when a _l part sits on the wrong side)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbr import BBRError, get_logger  # noqa: E402
from bbr.core import consistency, imageio, imageops, labels as labels_core, measure, partshape, proposals, suggest, symmetry  # noqa: E402
from bbr.core.despill import chroma_channel  # noqa: E402
from bbr.core.config import PipelineConfig  # noqa: E402
from bbr.core.calibration import calibrate_views, derive_height  # noqa: E402
from bbr.core.spec import load_view_spec, parse_view_spec  # noqa: E402

log = get_logger("prepare_references")

VIEW_FILE_KEYS = ("front", "back", "side", "left", "right", "top", "bottom")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SIDE_AXIS_CHOICES = ("-X", "+X")
DEFAULT_SIDE_AXIS = "-X"
SPEC_VERSION = "0.1"
DRAFT_ROLES = {"front": ("carve", "color", "validate"), "side": ("carve", "color", "validate"), "back": ("color", "validate")}
SHEET_FILE = "sheet_source.json"   # written by split-sheet; draft-view-spec then calibrates those views as one sheet


def cmd_split_sheet(args: argparse.Namespace) -> int:
    """Split a sheet into per-view crops (full sheet height) and print each crop's mask bbox."""
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    rgba = imageio.load_rgba(args.image)
    boxes = imageops.split_sheet(rgba, len(names), args.bg_tol)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SHEET_FILE).write_text(json.dumps(
        {"image": str(Path(args.image).resolve()), "names": names, "boxes": [list(b) for b in boxes]}, indent=2) + "\n",
        encoding="utf-8")
    for name, box in zip(names, boxes):
        crop = imageops.crop(rgba, box)
        path = out_dir / f"{name}.png"
        imageio.save_rgba(path, crop)
        mask = imageops.extract_mask(crop, "auto", tol=args.bg_tol)
        print(f"{name}: crop={box} size={crop.shape[1]}x{crop.shape[0]} mask_bbox={imageops.mask_bbox(mask)} -> {path}")
    return 0


def cmd_masks(args: argparse.Namespace) -> int:
    """Write mask PNGs for every enabled view and print the bbox / calibration summary."""
    spec = load_view_spec(args.view_spec)
    out_dir = Path(args.out_dir)
    masks: dict[str, np.ndarray] = {}
    for view in spec.enabled_views():
        rgba = imageio.load_view_image(view)
        bg = view.background
        masks[view.key] = imageops.extract_mask(rgba, bg.mode, bg.color, bg.tolerance, remove_enclosed=bg.remove_enclosed)
        imageio.save_mask(out_dir / f"{view.key}_mask.png", masks[view.key])
        x0, y0, x1, y1 = imageops.mask_bbox(masks[view.key])
        fill = float(masks[view.key][y0:y1, x0:x1].mean())
        print(f"{view.key}: image={rgba.shape[1]}x{rgba.shape[0]} bbox={(x0, y0, x1, y1)} fill={fill:.3f}")
    calibs, info = calibrate_views(spec, masks)
    print("dimensions (m):", {k: round(v, 4) for k, v in info["dimensions"].items()})
    print("provenance:", info["provenance"])
    for key, calib in calibs.items():
        print(f"{key}: axis={calib.camera_axis} m_per_px={calib.m_per_px:.6f} origin_px={calib.origin_px}")
    return 0


def _view_axis_roles(key: str, side_axis: str) -> dict:
    """camera_axis / roles written for a drafted view: the three standard views fully, the others from the schema defaults."""
    if key == "front":
        return {"camera_axis": "-Y", "roles": list(DRAFT_ROLES["front"])}
    if key == "back":
        return {"camera_axis": "+Y", "roles": list(DRAFT_ROLES["back"])}
    if key == "side":
        return {"camera_axis": side_axis, "roles": list(DRAFT_ROLES["side"])}
    return {}


def _draft_background(image_path: Path) -> tuple[dict, str | None]:
    """``background`` of a drafted view (auto mask, default tolerance) and the chroma backdrop colour name, if any."""
    rgb = imageio.load_rgba(image_path)[..., :3]
    channel = chroma_channel(imageops.border_background_color(rgb))
    background: dict = {"mode": "auto", "tolerance": imageops.DEFAULT_BG_TOL}
    if channel is not None:
        background["remove_enclosed"] = True     # backdrop enclosed by the subject (staff, hair, brim gaps) goes too
    return background, None if channel is None else ("red", "green", "blue")[channel]


def cmd_draft_view_spec(args: argparse.Namespace) -> int:
    """Write a runnable view_spec.json for the front/back/side/left/right/top/bottom images found in DIR."""
    directory = Path(args.dir)
    views: dict[str, dict] = {}
    chroma: dict[str, str] = {}
    for key in VIEW_FILE_KEYS:
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{key}{suffix}"
            if candidate.is_file():
                background, backdrop = _draft_background(candidate)
                views[key] = {"path": candidate.name, **_view_axis_roles(key, args.side_axis), "background": background,
                              "confidence": "observed"}
                if backdrop:
                    chroma[key] = backdrop
                break
    if not views:
        raise BBRError(f"no view images ({', '.join(VIEW_FILE_KEYS)}) found in {directory}")
    sheet_file = directory / SHEET_FILE
    sheet_names = json.loads(sheet_file.read_text(encoding="utf-8")).get("names", []) if sheet_file.is_file() else []
    for key in sheet_names:  # cut from one sheet: shared scale / baseline (see calibration.mode "sheet")
        if key in views and key not in ("top", "bottom"):
            views[key]["calibration"] = {"mode": "sheet"}
    notes = "drafted by prepare_references.py" + (f"; chroma backdrop: {', '.join(f'{k} {c}' for k, c in chroma.items())}" if chroma else "")
    if not (args.height or args.width or args.depth):
        raise BBRError("give the object size: --height, or --width / --depth (metres; the height is then derived from the views)")
    spec = {
        "spec_version": SPEC_VERSION, "name": args.name or directory.resolve().parent.name or directory.resolve().name,
        "views": views, "known_dimensions": {"width": None, "depth": None, "height": args.height},
        "symmetry": {"x": False, "y": False, "z": False},
        "reference_quality": {"orthographic": True, **({"ai_generated": True} if sheet_names else {}), "notes": notes},   # cut from one generated sheet: the dimension check allows 5 %
    }
    spec_obj = parse_view_spec(spec, directory)   # validate before anything is written: no half-written file on an error
    _, masks = measure.load_view_data(spec_obj, PipelineConfig())
    size_note = ""
    if not args.height:
        axes = spec_obj.view_axes()
        height, how, alt = derive_height({k: m for k, m in masks.items() if k in axes}, axes, args.width, args.depth, args.size_of)
        spec["known_dimensions"]["height"] = round(height, 4)
        spec["dimension_provenance"] = {"height": how}
        if args.size_of == "silhouette":   # the given extent IS the silhouette extent the pipeline calibrates with: record it as known
            spec["known_dimensions"].update(width=args.width, depth=args.depth)
        size_note = (f"height {height:.4f} m {how}. With --size-of {next(iter(alt))} the same number would give "
                     f"{next(iter(alt.values())):.4f} m: pick the one your {'width' if args.width else 'depth'} describes")
    spec_obj = parse_view_spec(spec, directory)
    sym_note = "symmetry.x false (no front / back view to measure)"
    sym_view = next((k for k in ("front", "back") if k in spec_obj.views and spec_obj.views[k].is_orthographic_usable), None)
    if sym_view:
        label_masks = labels_core.load_label_masks(args.labels, masks, args.regions_dir) if args.labels else {}
        calibs, _ = calibrate_views(spec_obj, masks, label_masks or None)
        sym = symmetry.front_symmetry(masks[sym_view], calibs[sym_view], label_masks.get(sym_view), args.asymmetric or [])
        spec["symmetry"]["x"] = sym["suggest_x"]
        sym_note = (f"symmetry.x {str(sym['suggest_x']).lower()} (left/right mirror IoU of {sym_view}: {sym['explanation']}; "
                    f"threshold {sym['threshold']:g}; set it by hand if the character carries one-sided props)")
    path = directory / "view_spec.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    load_view_spec(path)  # the written file loads
    sheet_views = [k for k, v in views.items() if "calibration" in v]
    print(f"wrote {path} (name {spec['name']!r}) with views: {', '.join(views)}"
          + (f" (sheet calibration: {', '.join(sheet_views)})" if sheet_views else ""))
    print("roles: " + "; ".join(f"{k} {','.join(v['roles'])}" for k, v in views.items() if "roles" in v)
          + f"; side camera_axis {args.side_axis}" + ("; chroma backdrop: " + ", ".join(f"{k} {c}" for k, c in chroma.items()) if chroma else ""))
    if size_note:
        print(size_note)
    print(sym_note)
    return 0


def _parse_numbers(text: str, count: int, what: str) -> list[float]:
    try:
        values = [float(t) for t in text.split(",")]
    except ValueError:
        values = []
    if len(values) != count:
        raise BBRError(f"{what} needs {count} comma-separated numbers, got {text!r}")
    return values


def cmd_measure(args: argparse.Namespace) -> int:
    """Print the world coordinates of image pixels / a pixel box on a view's plane (pipeline calibration)."""
    if args.points_file:
        try:
            named = json.loads(Path(args.points_file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BBRError(f"cannot read --points-file {args.points_file}: {exc}") from exc
        res = measure.measure_points(args.view_spec, named, labels_path=args.labels)
        text = json.dumps(res, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if not args.view:
        raise BBRError("measure needs --view with --px / --bbox, or --points-file")
    points = [_parse_numbers(p, 2, "--px") for p in args.px or []]
    bbox = _parse_numbers(args.bbox, 4, "--bbox") if args.bbox else None
    res = measure.measure_reference(args.view_spec, args.view, points, bbox, labels_path=args.labels)
    print(json.dumps(res, indent=2) if args.json else measure.format_measurement(res))
    return 0


def cmd_anchors(args: argparse.Namespace) -> int:
    """Propose ``anchors_px`` of a part (medial axis / principal axis + distance transform) as a JSON snippet.

    Several ``--view`` (with ``--labels``/``--part``) give one snippet for all views plus a cross-view check of
    base / tip and their heights.
    """
    if args.from_px:
        return _anchors_from_px(args)
    if not args.view_spec or not args.view:
        raise BBRError("anchors needs --view-spec and --view (or --from-px VIEW:name=x,y ...)")
    spec = load_view_spec(args.view_spec)
    views = list(args.view)
    for view in views:
        if view not in spec.views:
            raise BBRError(f"view {view!r} is not in the ViewSpec; views: {sorted(spec.views)}")
    if len(views) > 1 and not args.labels:
        raise BBRError("several --view need --labels labels.json --part ID (a pixel box is specific to one view)")
    rgba, masks = measure.load_view_data(spec, PipelineConfig())
    label_masks = labels_core.load_label_masks(args.labels, masks, args.regions_dir) if args.labels else {}
    calibs, info = calibrate_views(spec, masks, label_masks or None)
    hints = _parse_base_hints(args.base_hint, views)
    if args.labels and not args.part:
        raise BBRError("--labels needs --part ID (the part whose label mask is used)")
    found: dict[str, dict] = {}
    boxes: dict[str, list[float]] = {}
    rests: dict[str, np.ndarray | None] = {}
    part_masks: dict[str, np.ndarray] = {}
    for view in views:
        if view not in masks:
            raise BBRError(f"view {view!r} has no usable mask (preview-only or disabled)")
        part_masks[view] = masks[view]
        if args.labels:
            part_masks[view] = label_masks.get(view, {}).get(args.part)
            if part_masks[view] is None or not part_masks[view].any():
                print(f"# NOTE view {view!r} skipped: part {args.part!r} has no label pixels there (hidden or not labelled)", file=sys.stderr)
                part_masks.pop(view)
                continue
        if args.bbox:
            boxes[view] = _parse_numbers(args.bbox, 4, "--bbox")
        elif args.labels:   # a box 4 px wider than the label: the part does not "continue" past its own mask
            x0, y0, x1, y1 = imageops.mask_bbox(part_masks[view])
            boxes[view] = [x0 - 4, y0 - 4, x1 + 4, y1 + 4]
        else:
            raise BBRError("--bbox is required (or --labels labels.json --part ID)")
        rests[view] = _attachment_mask(masks[view], part_masks[view], boxes[view], label_masks.get(view, {}).get(labels_core.BODY))
        found[view] = measure.propose_anchors(part_masks[view], boxes[view], args.kind, hints.get(view), rests[view],
                                              n_points=args.points)
    views = [v for v in views if v in found]
    if not views:
        raise BBRError(f"part {args.part!r} has no label pixels in any of the views {list(args.view)}")
    if args.kind == "auto" and len(views) > 1:   # one primitive for all views: mixed anchor keys cannot be triangulated
        types = {v: f["suggested_type"] for v, f in found.items()}
        if len(set(types.values())) > 1:
            score: dict[str, float] = {}   # confidence-weighted vote; a curled view decides alone (seen end-on, a curled tail looks like a ball)
            for f in found.values():
                score[f["suggested_type"]] = score.get(f["suggested_type"], 0.0) + f["confidence"]
            best = partshape.CONE if any(f.get("curled") for f in found.values()) else max(score, key=lambda t: (score[t], partshape.TYPE_RANK[t]))
            for v in views:
                if types[v] != best:
                    found[v] = measure.propose_anchors(part_masks[v], boxes[v], "auto", hints.get(v), rests[v], force_type=best)
                    found[v].setdefault("warnings", []).append(f"the views disagree on the type ({types}); {best} used in every view")
    strand = args.kind == "strand"
    ends = {v: _strand_ends(f) for v, f in found.items()} if strand else found      # base / tip = the end points of a strand
    flipped = measure.harmonize_base_tip(ends, calibs) if len(views) > 1 else []
    for v in flipped if strand else []:
        measure.flip_strand(found[v])
    check = measure.check_anchor_views(ends, calibs)
    if flipped and measure._elongation_of(ends) >= measure.SYMMETRIC_ELONGATION:   # a stubby shape has no real base end: no news
        check["warnings"].insert(0, f"base / tip swapped in {', '.join(flipped)} to run the same way (higher / lower end) as in the other views; "
                                    "check the overlays, and pass --base-hint VIEW:X,Y if the base is the other end")
    kind = None
    lengths = {}
    suggested = {}
    for view, f in found.items():
        kind = f.pop("kind")
        suggested[view] = f.pop("suggested_type", None)
        f.pop("base_by", None)
        f.pop("curled", None)
        lengths[view] = f.pop("length_px", None)
        f.pop("candidates", None)
        f.pop("confidence", None)
        f.pop("warnings", None)
    snippet = {"anchors_px": found}
    part_id = args.id or (args.part if args.labels else None)
    types = sorted({t for t in suggested.values() if t})
    wrapped = {"suggested_type": types[0], "params": snippet} if len(types) == 1 else {"params": snippet}
    text = json.dumps({part_id: wrapped} if part_id else snippet, indent=2)
    print(re.sub(r"\[\s+([^\[\]]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text))   # numbers on one line
    print(f"# {kind} in view(s) {', '.join(views)}: axis length px {lengths}; merge the 'anchors_px' entries into the part's params"
          + ("" if len(views) > 1 else " and add the same physical points from the other views"), file=sys.stderr)
    if types:
        print(f"# suggested_type {', '.join(types)}: the component's \"type\" (--kind tube|capsule|sphere forces a primitive)", file=sys.stderr)
    print(f"# confidence {check['confidence']:.2f} (0-1; below 0.6 check the overlay and pass --base-hint)", file=sys.stderr)
    if args.labels:
        for view in views:
            w = labels_core.side_warning(args.part, view, part_masks[view], calibs[view], _object_height(info))
            if w:
                print(f"# WARNING {w}", file=sys.stderr)
    for w in check["warnings"]:
        print(f"# WARNING {w}", file=sys.stderr)
    if args.overlay:
        base = Path(args.overlay)
        for view in views:
            path = base if len(views) == 1 else base.with_name(f"{base.stem}_{view}{base.suffix}")
            _draw_anchors(rgba[view], boxes[view], found[view], path)
            print(f"# overlay: {path}", file=sys.stderr)
    return 0


SCALAR_ANCHORS = ("radius_px", "width_px")
POINT_ANCHORS = ("base", "mid", "tip", "center")


def _tidy(v: float) -> float | int:
    return int(v) if v == int(v) else v


def anchors_from_px(items: list[str]) -> dict[str, dict]:
    """``["front:base=118,543", "front:radius_px=8", ...]`` -> ``{view: {name: [x, y] | number | [a, b]}}`` (``anchors_px``)."""
    out: dict[str, dict] = {}
    for item in items:
        head, _, value = item.partition("=")
        view, _, name = head.partition(":")
        if not (view and name and value):
            raise BBRError(f"--from-px needs VIEW:name=x,y (or VIEW:radius_px=r), got {item!r}")
        if name in POINT_ANCHORS or re.fullmatch(r"p\d+", name):
            out.setdefault(view, {})[name] = [_tidy(v) for v in _parse_numbers(value, 2, f"--from-px {head}")]
        elif name in SCALAR_ANCHORS:
            nums = _parse_numbers(value, 2 if "," in value else 1, f"--from-px {head}")
            out.setdefault(view, {})[name] = [_tidy(v) for v in nums] if len(nums) == 2 else _tidy(nums[0])
        else:
            raise BBRError(f"--from-px: {name!r} is not an anchor; use {', '.join(POINT_ANCHORS + SCALAR_ANCHORS)}")
    return out


def _anchors_from_px(args: argparse.Namespace) -> int:
    """``anchors --from-px``: print the ``params.anchors_px`` snippet of typed pixel points (view pixels, as ``measure --px`` takes)."""
    found = anchors_from_px(args.from_px)
    if args.view_spec:
        spec = load_view_spec(args.view_spec)
        unknown = [v for v in found if v not in spec.views]
        if unknown:
            raise BBRError(f"views {unknown} are not in the ViewSpec; views: {sorted(spec.views)}")
    snippet = {"params": {"anchors_px": found}}
    text = json.dumps({args.id: snippet} if args.id else snippet, indent=2)
    print(re.sub(r"\[\s+([^\[\]]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text))
    print("# merge params.anchors_px into the part's component (set its \"type\"); omit a point in a view where it is hidden", file=sys.stderr)
    return 0


def _object_height(info: dict) -> float:
    dims = info["dimensions"]
    return float(dims.get("height") or 0.0) or max(dims.values())


def _attachment_mask(silhouette: np.ndarray, part: np.ndarray, box: list[float], body: np.ndarray | None) -> np.ndarray | None:
    """What a part is attached to, to orient its base: the body label when there is one, else the rest of the silhouette
    (outside the part's label, or outside the box when no label is used: the box edge sits at the junction)."""
    if body is not None and body.any():
        return body
    if part is not silhouette:
        return silhouette & ~part
    outside = np.ones(silhouette.shape, dtype=bool)
    outside[max(int(box[1]), 0):int(box[3]), max(int(box[0]), 0):int(box[2])] = False
    return silhouette & outside


def _parse_base_hints(items: list[str] | None, views: list[str]) -> dict[str, list[float]]:
    """``--base-hint`` values -> ``{view: [x, y]}``: ``VIEW:X,Y`` names its view; plain ``X,Y`` values are taken in order by the
    views without a named hint (exactly one per such view)."""
    named: dict[str, list[float]] = {}
    plain: list[list[float]] = []
    for item in items or []:
        head, sep, rest = item.partition(":")
        if sep:
            if head not in views:
                raise BBRError(f"--base-hint {item!r}: view {head!r} is not one of --view {views}")
            named[head] = _parse_numbers(rest, 2, "--base-hint")
        else:
            plain.append(_parse_numbers(item, 2, "--base-hint"))
    free = [v for v in views if v not in named]
    if plain and len(plain) != len(free):
        raise BBRError(f"--base-hint: {len(plain)} plain X,Y value(s) for {len(free)} view(s) {free}: give one per view in order, "
                       f"or name the view (--base-hint front:X,Y --base-hint side:X,Y); a pixel is specific to its view")
    return {**dict(zip(free, plain)), **named}


def _strand_ends(f: dict) -> dict:
    """Base / tip end points of a strand anchor dict (for the cross-view base / tip harmonisation and check)."""
    names = measure.strand_points(f)
    return {"base": f[names[0]], "tip": f[names[-1]], "base_by": f.get("base_by"), "confidence": f.get("confidence", 1.0),
            "warnings": f.get("warnings", [])}


def _draw_anchors(rgba: np.ndarray, bbox: list[float], anchors: dict, path: Path, margin: int = 24, zoom: int = 3) -> None:
    """Save the box region (zoomed) with the box and the proposed anchors drawn on it, to check them by eye."""
    from PIL import Image, ImageDraw
    h, w = rgba.shape[:2]
    x0, y0 = max(int(bbox[0]) - margin, 0), max(int(bbox[1]) - margin, 0)
    x1, y1 = min(int(bbox[2]) + margin, w), min(int(bbox[3]) + margin, h)
    img = Image.fromarray((np.clip(rgba[y0:y1, x0:x1, :3], 0, 1) * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    def at(p: list[float]) -> tuple[float, float]:
        return (p[0] - x0) * zoom, (p[1] - y0) * zoom

    draw.rectangle([*at(bbox[:2]), *at(bbox[2:])], outline=(255, 255, 0))
    radius = anchors.get("radius_px")
    pair = radius if isinstance(radius, list) else [radius, radius]
    pts = measure.strand_points(anchors)
    marks = [("base", (255, 0, 0), pair[0]), ("mid", (255, 160, 0), None), ("tip", (0, 0, 255), pair[1]), ("center", (255, 0, 255), radius)]
    if pts:       # a strand: one dot per point, red at the base to blue at the tip, each with its own radius
        rad = radius if isinstance(radius, list) and len(radius) == len(pts) else [pair[0] or 0] * len(pts)
        marks = [(n, (int(255 * (1 - i / max(len(pts) - 1, 1))), 0, int(255 * i / max(len(pts) - 1, 1))), rad[i]) for i, n in enumerate(pts)]
    for name, color, r in marks:
        if name in anchors:
            cx, cy = at(anchors[name])
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
            if r:
                draw.ellipse([cx - r * zoom, cy - r * zoom, cx + r * zoom, cy + r * zoom], outline=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    big = proposals.add_rulers(np.asarray(img).astype(np.float32) / 255.0, zoom, x0, y0, grid=True)
    imageio.save_rgba(path, _caption(big, _view_caption(zoom)))


def cmd_propose(args: argparse.Namespace) -> int:
    """Numbered region proposals per view (colour regions + thin protrusions) for the agent to label."""
    spec = load_view_spec(args.view_spec)
    rgba, masks = measure.load_view_data(spec, PipelineConfig())
    views = [v.strip() for v in args.views.split(",")] if args.views else list(masks)
    unknown = [v for v in views if v not in masks]
    if unknown:
        raise BBRError(f"views {unknown} have no usable mask; available: {sorted(masks)}")
    for view in views:
        res = proposals.propose_regions(rgba[view], masks[view], args.max_regions, args.k)
        out = Path(args.out_dir) / view
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "id_map.npy", res["id_map"])
        (out / "regions.json").write_text(json.dumps(
            {"view": view, "image_size": [masks[view].shape[1], masks[view].shape[0]], "params": res["params"],
             "regions": res["regions"]}, indent=1) + "\n", encoding="utf-8")
        crops: dict[str, np.ndarray] = {}
        if args.compact:   # one small sheet (view scale, cropped to the subject): cheap to read; --zoom / a rerun without --compact for detail
            x0, y0, x1, y1 = imageops.mask_bbox(masks[view])
            m = proposals.TICK_PX // 2
            imageio.save_rgba(out / "compact.png", _caption(proposals.render_overlay(rgba[view], res["id_map"], res["regions"], 1,
                                                                                       box=[x0 - m, y0 - m, x1 + m, y1 + m]), _view_caption(1)))
            print(f"{view}: {len(res['regions'])} regions -> {out / 'compact.png'}")
        else:
            imageio.save_rgba(out / "overlay.png", _caption(proposals.render_overlay(rgba[view], res["id_map"], res["regions"]),
                                                            _view_caption(proposals.OVERLAY_SCALE)))
            crops = proposals.overlay_crops(rgba[view], res["id_map"], res["regions"])
            for name, img in crops.items():
                imageio.save_rgba(out / f"{name}.png", _caption(img, _view_caption(proposals.CROP_SCALE)))
            print(f"{view}: {len(res['regions'])} regions -> {out / 'overlay.png'}" + (f" + {len(crops)} crops ({', '.join(crops)})" if crops else ""))
        if args.list:
            print(proposals.region_summary(res["regions"]))
        for zoom in args.zoom or []:   # view:x0,y0,x1,y1 -> a x4 crop with rulers in view pixels
            zview, _, zbox = zoom.partition(":")
            if zview == view:
                box = [int(v) for v in _parse_numbers(zbox, 4, "--zoom")]
                path = out / f"zoom_{box[0]}_{box[1]}_{box[2]}_{box[3]}.png"
                imageio.save_rgba(path, _caption(proposals.render_overlay(rgba[view], res["id_map"], res["regions"], proposals.CROP_SCALE,
                                                                          digit_scale=0.75, box=box), _view_caption(proposals.CROP_SCALE)))
                print(f"  zoom -> {path}")
    print(f"# {'compact.png is 1x the view' if args.compact else f'overlay.png is {proposals.OVERLAY_SCALE}x the view'}, rulers every {proposals.TICK_PX} view px (top / left); "
          f"coordinates in labels.json edits are view px. regions: {args.out_dir}/<view>/id_map.npy", file=sys.stderr)
    return 0


def cmd_suggest_parts(args: argparse.Namespace) -> int:
    """Part proposals from thin protrusions: labels / components / exclude_components to paste, plus numbered overlays."""
    out = Path(args.out_dir) if args.out_dir else Path(args.view_spec).resolve().parent / "suggest"
    res = suggest.suggest_from_files(args.view_spec, args.labels, args.regions_dir, args.open_frac, out)
    print(f"{len(res['suggestions'])} suggestions (views {', '.join(res['views'])}; labels as {res['label_style']}):")
    for s in res["suggestions"]:
        kind = "curved cone" if s["curved"] else s["type"]
        extra = f" hair lock (flatten {suggest.LOCK_FLATTEN:g})" if s["hair_lock"] else ""
        extra += f" mirror_of {s['mirror_of']}" if s["mirror_of"] else ""
        print(f"  {s['number']:>2} {s['id']:<8} {kind:<11} conf {s['confidence']:.2f}  views {','.join(s['views']):<16} z {s['z_range_m'][0]:.3f}-{s['z_range_m'][1]:.3f} m{extra}")
        for w in s["warnings"]:
            print(f"        WARNING {w}")
    for a in res["already_labelled"]:
        print(f"  skipped (already labelled {a['part']}): {a['view']} {a['bbox']}")
    print(f"exclude_components: {json.dumps(res['exclude_components'])}")
    print("files: " + ", ".join(res["files"]))
    print("# overlays: numbers = the suggestion numbers, rulers in view pixels. Paste `labels` (per view) into labels.json and `components` into "
          "scene_spec.json (rename the ids, drop what is not a part), add the kept ids to the hull's exclude_components, then run `labels` and the pipeline. "
          "Confidence is the shape fit (x0.7 when one view only); check every overlay.", file=sys.stderr)
    return 0


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    """RGBA float copy of ``img`` with ``text`` in its bottom-left corner (the scale of the image against the view)."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray((np.clip(img[..., :3], 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=max(12, img.shape[0] // 60))
    except TypeError:
        font = ImageFont.load_default()
    draw.text((6, img.shape[0] - 6), text, fill=(255, 255, 255), font=font, anchor="ls", stroke_width=2, stroke_fill=(0, 0, 0))
    return np.concatenate([np.asarray(pil).astype(np.float32) / 255.0, np.ones((*img.shape[:2], 1), np.float32)], axis=2)


def _view_caption(scale: int) -> str:
    return f"{scale}x view; rulers and coordinates = view px"


def _label_preview(rgba: np.ndarray, parts: dict[str, np.ndarray], scale: int = 1) -> np.ndarray:
    """Colour-coded label masks over the dimmed view (body untinted), part names drawn on them, view-pixel rulers and a
    scale caption (coordinates everywhere are VIEW pixels; ``scale`` only magnifies the picture)."""
    from PIL import Image, ImageDraw, ImageFont
    base = np.repeat(np.repeat(rgba[..., :3].astype(np.float64), scale, axis=0), scale, axis=1) * 0.55 + 0.1
    names = [p for p in parts if p != labels_core.BODY and parts[p].any()]
    for i, name in enumerate(names, start=1):
        big = np.repeat(np.repeat(parts[name], scale, axis=0), scale, axis=1)
        color = np.array(proposals.region_color(i))
        base[big] = base[big] * 0.4 + color * 0.6
        base[imageops.dilate(imageops.mask_edges(big), 1) & big] = color
    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(14, rgba.shape[0] // 45))
    except TypeError:   # Pillow < 10.1: fixed-size default font
        font = ImageFont.load_default()
    for name in names:
        x, y = proposals.inner_point(parts[name], imageops.mask_bbox(parts[name]))
        draw.text((x * scale, y * scale), name, fill=(255, 255, 255), font=font, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    return _caption(proposals.add_rulers(np.asarray(img).astype(np.float32) / 255.0, scale), _view_caption(scale))


def cmd_labels(args: argparse.Namespace) -> int:
    """labels.json -> per-part per-view masks, colour-coded previews and the cross-view z-range check."""
    spec = load_view_spec(args.view_spec)
    rgba, masks = measure.load_view_data(spec, PipelineConfig())
    label_masks = labels_core.load_label_masks(args.labels, masks, args.regions_dir)
    calibs, info = calibrate_views(spec, masks, label_masks or None)
    if not label_masks:
        raise BBRError(f"{args.labels} labels none of the views {sorted(masks)}")
    if args.scene_spec:
        from bbr.core.spec import load_scene_spec
        ids = {c.id for c in load_scene_spec(args.scene_spec).components}
        for view, parts in label_masks.items():
            for part in parts:
                if part != labels_core.BODY and not part.startswith(labels_core.IGNORE_PREFIX) and part not in ids:
                    print(f"warning: view {view}: label {part!r} is not a component of {args.scene_spec}", file=sys.stderr)
    out_dir = Path(args.out_dir)
    for view, parts in label_masks.items():
        for part, mask in parts.items():
            if mask.any():
                imageio.save_mask(out_dir / view / f"{part}.png", mask)
        imageio.save_rgba(out_dir / f"preview_{view}.png", _label_preview(rgba[view], parts, args.scale))
        counts = ", ".join(f"{p} {int(m.sum())}" for p, m in parts.items() if m.any())
        print(f"{view}: {counts} -> {out_dir / f'preview_{view}.png'} ({args.scale}x the view, rulers in view px)")
    for line in _symmetry_lines(spec, masks, calibs, label_masks, _scene_asymmetric(args.scene_spec) + (args.asymmetric or [])):
        print(line)
    height = _object_height(info)
    check = labels_core.check_z_ranges(labels_core.part_z_ranges(label_masks, calibs), height, args.z_tol)
    check["side_warnings"] = labels_core.side_warnings(label_masks, calibs, height)
    (out_dir / "labels_report.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check, indent=2) if args.json else labels_core.format_z_check(check))
    for w in check["side_warnings"]:
        print(f"WARNING {w}", file=sys.stderr)
    return 2 if args.strict and check["flagged"] else 0


def _scene_asymmetric(scene_spec: str | None) -> list[str]:
    """``asymmetric_parts`` of the SceneSpec's silhouette hull (empty without a SceneSpec)."""
    if not scene_spec:
        return []
    from bbr.core.spec import load_scene_spec
    hull = next((c for c in load_scene_spec(scene_spec).components if c.type == "silhouette_hull"), None)
    return [str(a) for a in (hull.params.get("asymmetric_parts") or [])] if hull else []


def _symmetry_lines(spec, masks: dict, calibs: dict, label_masks: dict, asymmetric: list[str]) -> list[str]:
    """``symmetry <view>: ...`` explanation lines for the upright front / back views (labels + asymmetric parts blanked)."""
    return [f"symmetry {k}: left/right mirror IoU " + symmetry.front_symmetry(masks[k], calibs[k], label_masks.get(k), asymmetric)["explanation"]
            for k in masks if k in calibs and spec.views[k].camera_axis in ("-Y", "+Y")]


def cmd_check(args: argparse.Namespace) -> int:
    """Reference consistency gate: print the pair table / flags and write one XOR overlay PNG per opposite pair.

    ``--ignore-flag`` / ``--roles-override`` add to what the ViewSpec's ``consistency`` block already records; ``--write``
    stores them there, so later ``check`` runs and the pipeline (which reads the same block) apply them."""
    spec = load_view_spec(args.view_spec)
    _, masks = measure.load_view_data(spec, PipelineConfig())
    label_masks = labels_core.load_label_masks(args.labels, masks, args.regions_dir) if args.labels else None
    calibs, _ = calibrate_views(spec, masks, label_masks)
    override = {}
    for item in args.roles_override or []:
        view, _, roles = item.partition("=")
        if view not in spec.views or not _:
            raise BBRError(f"--roles-override needs VIEW=role,role with a view of the ViewSpec, got {item!r}")
        override[view] = [r for r in roles.split(",") if r]
    ignore, roles_override = consistency.merge_overrides(spec.consistency, {"ignore_flags": args.ignore_flag or [], "roles_override": override})
    if args.write:
        _write_overrides(Path(args.view_spec), ignore, roles_override)
    result = consistency.analyze(masks, calibs, spec.view_axes(), {k: v.roles for k, v in spec.views.items()},
                                 labels=label_masks, asymmetric_parts=args.asymmetric or [],
                                 ignore_flags=ignore, roles_override=roles_override)
    out_dir = Path(args.out_dir) if args.out_dir else Path(spec.source_path).parent / "consistency"
    for pair in result["pairs"]:
        path = out_dir / f"xor_{pair['a']}_{pair['b']}.png"
        imageio.save_rgba(path, consistency.pair_overlay(masks, calibs, pair["a"], pair["b"]))
        pair["overlay"] = str(path)
    result["calibration"] = measure.calibration_summary(calibs)
    print(json.dumps(result, indent=2) if args.json else consistency.format_consistency(result))
    if not args.json:
        for key, c in result["calibration"].items():
            print(f"calibration {key}: camera {c['camera_axis']}, {c['m_per_px']:.7f} m/px, origin_px {tuple(c['origin_px'])}; "
                  + ", ".join(f"{a} = {f}" for a, f in c["formula"].items()))
        for pair in result["pairs"]:
            print(f"overlay {pair['a']}<->{pair['b']}: {pair['overlay']}  (green both, red {pair['a']} only, blue mirrored {pair['b']} only)")
    return 2 if args.strict and not result["passed"] else 0


def _write_overrides(path: Path, ignore_flags: list[str], roles_override: dict[str, list[str]]) -> None:
    """Store the judged-false-alarm flags / roles in the ViewSpec file's ``consistency`` block (validated before it is kept)."""
    from bbr.core.spec import validate_view_spec
    data = json.loads(path.read_text(encoding="utf-8"))
    block = {k: v for k, v in (("ignore_flags", ignore_flags), ("roles_override", roles_override)) if v}
    if block:
        data["consistency"] = block
    else:
        data.pop("consistency", None)
    errors = validate_view_spec(data)
    if errors:
        raise BBRError(f"{path} would be invalid, not written: {'; '.join(errors)}")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"recorded in {path}: consistency {json.dumps(block)} (the pipeline reads it from there)")


def _add_symmetry_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels", metavar="labels.json", help="part labels: the symmetry line then blanks one-sided parts (--asymmetric) and suggests more")
    p.add_argument("--regions-dir", metavar="DIR", help="where <view>/id_map.npy are (default: the labels file's _regions, else regions/ beside it)")
    p.add_argument("--asymmetric", action="append", metavar="NAME_OR_GLOB", help="label name or glob (spatula_*) blanked in the symmetry measure (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split-sheet", help="split a turnaround sheet into per-view PNG crops")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--names", required=True, help="comma-separated view names, left to right")
    p.add_argument("--bg-tol", type=float, default=imageops.DEFAULT_BG_TOL)
    p.set_defaults(func=cmd_split_sheet)
    p = sub.add_parser("masks", help="write mask PNGs and print bbox / calibration summary")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_masks)
    p = sub.add_parser("draft-view-spec", help="write a view_spec.json for the images found in a directory")
    p.add_argument("--dir", required=True)
    p.add_argument("--height", type=float, help="object height in meters (or give --width / --depth: the height is derived from the views)")
    p.add_argument("--width", type=float, help="object width (X) in meters; derives the height from the front view (known_dimensions.height, provenance recorded)")
    p.add_argument("--depth", type=float, help="object depth (Y) in meters; derives the height from the side view (used when --width is not given)")
    p.add_argument("--size-of", choices=("silhouette", "body"), default="silhouette",
                   help="what --width / --depth measure: the whole outline incl. handles, locks, props (silhouette, default; recorded as known width/depth) or the body without handles, locks and props (body; only the derived height is recorded)")
    p.add_argument("--side-axis", choices=SIDE_AXIS_CHOICES, default=DEFAULT_SIDE_AXIS,
                   help="camera_axis of the side view: -X if the object faces image-right, +X if image-left")
    p.add_argument("--name", help="object name (default: the name of the folder that contains the image folder)")
    _add_symmetry_args(p)
    p.set_defaults(func=cmd_draft_view_spec)
    p = sub.add_parser("measure", help="world coordinates of image pixels (place parts from the images)")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--view", help="view key, e.g. front (with --px / --bbox)")
    p.add_argument("--px", action="append", metavar="X,Y", help="image pixel (repeatable)")
    p.add_argument("--bbox", metavar="X0,Y0,X1,Y1", help="pixel box -> world extents")
    p.add_argument("--points-file", metavar="pts.json", help='batch: {"front": {"door_top": [340, 378], "door": [270, 400, 410, 585]}, "side": {...}} (two numbers a pixel, four a box) -> metres JSON')
    p.add_argument("--out", metavar="metres.json", help="with --points-file: also write the result here")
    p.add_argument("--labels", metavar="labels.json", help="the scene's parts_labels: calibrate like the pipeline does with labels (its body label can move the body axis)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.set_defaults(func=cmd_measure)
    p = sub.add_parser("anchors", help="propose anchors_px (base / mid / tip / radius ...) of a part from a rough pixel box")
    p.add_argument("--view-spec")
    p.add_argument("--from-px", action="extend", nargs="+", metavar="VIEW:NAME=X,Y", help="no image analysis: emit the anchors_px snippet of typed view-pixel points, e.g. front:base=118,543 front:tip=147,443 front:radius_px=8 side:base=280,560 (repeatable; names base mid tip center radius_px width_px)")
    p.add_argument("--view", action="append", help="view key, e.g. front; repeat (with --labels) for several views: one snippet, cross-view base/tip check")
    p.add_argument("--bbox", metavar="X0,Y0,X1,Y1", help="rough pixel box around the part; where the part merges into the body put the box edge at the junction (optional with --labels: crops the part's label mask, e.g. to split an arm from its hand)")
    p.add_argument("--labels", metavar="labels.json", help="use the label mask of --part in this view instead of the silhouette (separates parts inside the body outline)")
    p.add_argument("--part", help="component id in labels.json (with --labels)")
    p.add_argument("--regions-dir", metavar="DIR", help="where <view>/id_map.npy are (default: the labels file's _regions, else regions/ beside it)")
    p.add_argument("--kind", choices=measure.ANCHOR_KINDS, default="auto", help="auto (default): pick sphere / capsule / cone by how well each fits the mask and print suggested_type; tube: horn / tapered limb; capsule; blade; plate: root edge + tip of a wing / ear fin / leaf (type plate; thickness and face_view in params, the outline is the label); disc / sphere (largest inscribed circle); strand: N points along a curved tube (twin tail, braid, long hair lock, cable) with a radius each, type strand")
    p.add_argument("--id", help="component id to wrap the snippet with")
    p.add_argument("--base-hint", action="append", metavar="[VIEW:]X,Y", help="pixel near the end that is the base; repeat for several --view: VIEW:X,Y names the view, plain X,Y values go to the views in order, one each (default: the end that touches the most other foreground, else the end nearer the body, else the lower end of a vertical-ish rod / the wider end)")
    p.add_argument("--points", type=int, default=measure.STRAND_POINTS, metavar="N", help="with --kind strand: number of anchor points p0..pN-1 along the part (>= 3, default %(default)s)")
    p.add_argument("--overlay", metavar="PNG", help="write the view with the box and anchors drawn on it")
    p.set_defaults(func=cmd_anchors)
    p = sub.add_parser("propose", help="numbered region proposals per view (colour regions + thin protrusions) to label")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--out-dir", required=True, help="writes <out-dir>/<view>/overlay.png, id_map.npy, regions.json (+ crop_*.png)")
    p.add_argument("--views", help="comma-separated view keys (default: every usable view)")
    p.add_argument("--max-regions", type=int, default=proposals.MAX_REGIONS, help="fewer regions = a smaller overlay to read (try 20-25 first; raise it where a part is merged into a neighbour)")
    p.add_argument("--compact", action="store_true", help="write one small <view>/compact.png (1x, cropped to the subject) instead of overlay.png + crop_*.png")
    p.add_argument("--k", type=int, default=proposals.DEFAULT_K, help="k-means colour clusters before merging")
    p.add_argument("--list", action="store_true", help="also print every region (area, bbox, kind, colour)")
    p.add_argument("--zoom", action="append", metavar="VIEW:X0,Y0,X1,Y1", help="also write a x4 crop of this view-pixel box (repeatable)")
    p.set_defaults(func=cmd_propose)
    p = sub.add_parser("suggest-parts", help="part proposals from thin protrusions (ears, horns, tails, hair locks): labels, components, exclude list, overlays")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--labels", metavar="labels.json", help="existing labels: the body label is what parts attach to and already labelled parts are skipped")
    p.add_argument("--regions-dir", metavar="DIR", help="where <view>/id_map.npy are: labels then use region ids instead of polygons (default: the labels file's _regions, else regions/ beside references/)")
    p.add_argument("--out-dir", help="writes suggestions.json and suggest_<view>.png (default: <view spec dir>/suggest)")
    p.add_argument("--open-frac", type=float, default=suggest.OPEN_FRAC, help="opening radius as a share of the height (0.02-0.04; smaller finds thinner parts)")
    p.set_defaults(func=cmd_suggest_parts)

    p = sub.add_parser("labels", help="labels.json -> per-part masks, previews and the cross-view z-range check")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--labels", required=True, metavar="labels.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regions-dir", metavar="DIR", help="where <view>/id_map.npy are (default: the labels file's _regions, else regions/ beside it)")
    p.add_argument("--scene-spec", help="warn about labels that are not component ids")
    p.add_argument("--z-tol", type=float, default=labels_core.DEFAULT_Z_TOL_FRAC, help="z-range disagreement (fraction of the height) that is flagged")
    p.add_argument("--asymmetric", action="append", metavar="NAME_OR_GLOB", help="parts blanked in the symmetry line (default: the SceneSpec hull's asymmetric_parts)")
    p.add_argument("--scale", type=int, default=1, help="magnify the preview PNG (default 1; coordinates stay view pixels, rulers show them)")
    p.add_argument("--strict", action="store_true", help="exit code 2 when a part's z-range disagrees between views")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_labels)
    p = sub.add_parser("check", help="reference consistency gate: do the views describe the same object?")
    p.add_argument("--view-spec", required=True)
    p.add_argument("--out-dir", help="where the XOR overlay PNGs go (default: <spec dir>/consistency)")
    p.add_argument("--strict", action="store_true", help="exit code 2 when the references are inconsistent")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    _add_symmetry_args(p)
    p.add_argument("--ignore-flag", action="append", metavar="TYPE[:VIEW]", help="record a flag judged a false alarm after the XOR overlay shows the views agree (repeatable), e.g. scale_drift or vertical_shift:side")
    p.add_argument("--roles-override", action="append", metavar="VIEW=ROLE,ROLE", help="keep these roles for a view instead of the recommendation, judged a false alarm (repeatable)")
    p.add_argument("--write", action="store_true", help="record the --ignore-flag / --roles-override values (with those already recorded) in the ViewSpec's consistency block, where the pipeline reads them; without it they affect this command only")
    p.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    # argparse treats '-X' as an option; glue it to --side-axis
    for i, token in enumerate(argv[:-1]):
        if token == "--side-axis" and argv[i + 1] in SIDE_AXIS_CHOICES:
            argv[i:i + 2] = [f"--side-axis={argv[i + 1]}"]
            break
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BBRError as exc:
        print(f"error: {exc or type(exc).__name__}", file=sys.stderr)
        return 1
    except Exception as exc:   # never exit 1 silently: say what failed (the traceback goes to stderr too)
        import traceback
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
