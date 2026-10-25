"""Human-readable result block of a report (pure python: unit-testable without Blender)."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .colormetric import format_color_error_line
from .config import DIMENSION_TOLERANCE
from .consistency import format_consistency
from .debris import format_debris_report
from .dimcheck import format_check as format_dimension_check
from .gapcheck import format_floating, format_gap
from .metrics import format_error_line


def failed_checks(hard: Mapping[str, Mapping[str, Any]], match: Mapping[str, Any]) -> list[str]:
    """One phrase per failed validation check, naming the number that failed: ``dimensions (width -9.7% vs tolerance
    +-3%)``, ``reference_match (total error 9.1% > threshold 8.0%; side 12.0%)``, ``scale (non-unit: leg)`` ...
    ``hard`` = the geometry checks (``dimensions``, ``scale``, ``modifiers``, ``materials``, ``scene_structure``), ``match`` =
    the reference-match block. Empty when everything passed."""
    out: list[str] = []
    for name, check in hard.items():
        if check.get("passed", True):
            continue
        if name == "dimensions" and check.get("relative_error"):
            tol = float(check.get("tolerance", DIMENSION_TOLERANCE))
            attributed = {a["dimension"]: a["text"] for a in check.get("limiting") or []}
            bad = [attributed.get(k) or f"{k} {v * 100:+.1f}% (limit {tol * 100:g}%)"
                   for k, v in check["relative_error"].items() if abs(v) > tol]
            out.append(f"dimensions ({'; '.join(bad)})")
        elif name == "dimensions":
            out.append(f"dimensions ({check.get('reason', 'not measurable')})")
        else:
            detail = {"scale": ("non-unit scale", check.get("non_unit_scale")), "modifiers": ("wrong modifier order", check.get("wrong_order")),
                      "materials": ("no material", check.get("without_material")),
                      "scene_structure": ("missing", check.get("missing_collections") or check.get("without_agent_id")
                                          or check.get("duplicate_agent_ids") or ([] if check.get("origin_empty", True) else ["origin empty"]))}.get(name)
            what = f" ({detail[0]}: {', '.join(detail[1])})" if detail and detail[1] else ""
            out.append(f"{name}{what}")
    if not match.get("passed", True):
        if "total_error" in match:
            worst = [f"{v} {match[f'{v}_error'] * 100:.1f}%" for v in (match.get("per_view") or {}) if match.get(f"{v}_error", 0) > match["threshold"]]
            out.append(f"reference_match (total error {match['total_error'] * 100:.1f}% > threshold {match['threshold'] * 100:.1f}%"
                       + (f"; {', '.join(worst)}" if worst else "") + ")")
        else:
            out.append(f"reference_match ({match.get('reason', 'no validation views were compared')})")
    return out


def merge_label_iou(part_iou: dict[str, Any], partfit_report: Mapping[str, Any]) -> None:
    """Add the partfit label IoU to ``report.part_iou.parts.<id>``: ``label_iou`` = ``{view: iou_vis}`` (the part as built
    against its OWN label mask, ignoring pixels over other parts' labels) and ``mean_label_iou``. This is the number to trust
    for a part; ``mean_iou`` compares the part with the WHOLE reference in its bbox and is low for any thin part."""
    for pid, info in (part_iou.get("parts") or {}).items():
        per_view = {v: d["iou_vis"] for v, d in ((partfit_report.get(pid) or {}).get("iou") or {}).items() if "iou_vis" in d}
        if per_view:
            info["label_iou"] = per_view
            info["mean_label_iou"] = sum(per_view.values()) / len(per_view)


def format_measurements(report: Mapping[str, Any]) -> list[str]:
    """Lines for the held-out, attribution and per-part blocks of a report (empty when none ran)."""
    lines: list[str] = []
    hold = report.get("holdout") or {}
    if hold.get("error"):
        lines.append(f"Holdout : error {hold['error']}")
    for view, h in (hold.get("views") or {}).items():
        color = "" if h.get("color_error") is None else f", color {h['color_error'] * 100:.1f}%"
        angle = (f", azimuth {h['camera']['azimuth']:.0f} -> {h['best_azimuth']:.0f} (step {h['pose_search']['azimuth_step_deg']:g})"
                 if h.get("azimuth_searched") else "")
        lines.append(f"Holdout {view} IoU : {h['iou'] * 100:.1f}% (nominal {h['iou_nominal'] * 100:.1f}%{angle}{color})")
    attr = report.get("attribution") or {}
    if attr.get("error"):
        lines.append(f"Attribution : error {attr['error']}")
    elif attr.get("skipped"):
        lines.append(f"Attribution : skipped ({attr['skipped']})")
    elif "objects" in attr:
        mis, occ = attr.get("misattribution_max"), attr.get("occlusion_max")
        if mis:
            lines.append(f"Misattribution max : {mis['share'] * 100:.1f}% ({mis['object']}), "
                         f"{len(attr.get('misattributed_objects', []))} flagged")
        if occ:
            lines.append(f"Occlusion share max : {occ['share'] * 100:.0f}% ({occ['object']}); flagged "
                         f"{', '.join(attr.get('occluded_objects', [])) or 'none'} (> {attr['occlusion_warn'] * 100:.0f}% in the best view)")
    parts = report.get("part_iou") or {}
    if parts.get("error"):
        lines.append(f"Part IoU : error {parts['error']}")
    for pid, info in (parts.get("parts") or {}).items():
        if info.get("mean_iou") is None:
            continue
        label = (f"label IoU {info['mean_label_iou'] * 100:.0f}% ("
                 + " ".join(f"{v} {x * 100:.0f}" for v, x in info["label_iou"].items()) + "), "
                 if info.get("mean_label_iou") is not None else "")
        lines.append(f"Part {pid} : {label}on-reference {info['min_precision'] * 100:.0f}% (min), "
                     f"region IoU {info['mean_iou'] * 100:.0f}% (part vs the whole reference in its box; low for thin parts, not a defect)")
    return lines


def next_action(report: Mapping[str, Any], failed: list[str]) -> str:
    """Last line of the result block: for a FAILED run the instruction to fix and re-run (never stop on it while budget
    remains); for a passed run with floating parts the fix to apply first. Empty otherwise."""
    if report.get("passed") is False:
        what = failed[0] if failed else report.get("status", "the failing check")
        return (f"next: FAILED - not done. Read the recommendation, fix the biggest error source ({what}) in the specs and re-run; "
                f"repeat until it passes or the budget is spent. Do not stop on a failed run while budget remains.")
    if str(report.get("status", "")).startswith("stopped_after_") and report.get("dimension_check") is not None:
        dim = report["dimension_check"]       # parts-only run (--stop-after secondary): what to fix before the textured run
        todo = []
        if not dim.get("passed", True):
            todo.append("dimensions (" + ("; ".join(a["text"] for a in dim.get("limiting") or []) or "see Dimensions above") + ")")
        todo += [f"floating_part {g['part']}" for g in report.get("floating_parts") or []]
        todo += [f"part_gap {g['a']}~{g['b']}" for g in report.get("part_gaps") or []]
        return ("next: parts-only checks found: " + ", ".join(todo) + "; fix them before the full run.") if todo else             "next: parts-only checks passed (dimensions, floating_part, part_gap); run without --stop-after."
    floating = [g["part"] for g in report.get("floating_parts") or []]
    if floating:
        return f"next: passed, but {', '.join(floating)} float (WARNING floating_part): attach them and re-run before delivering."
    return ""


def format_face_detail(block: Optional[Mapping[str, Any]]) -> list[str]:
    """``face detail <id>: ...`` lines of a ``face_detail`` report block (see :mod:`bbr.core.facedetail`)."""
    out = []
    for lid, f in ((block or {}).get("layers") or {}).items():
        if "error" in f:
            out.append(f"face detail {lid}: error {f['error']}")
            continue
        relief = f"profile relief {f.get('profile_relief_m', 0) * 1000:.0f} mm " + ("on the midline" if f.get("relief_applied") else "NOT applied")
        parts = [relief, f"{f['texels_per_px']:.1f} texels/px" if "texels_per_px" in f else None,
                 f"colour error {f['color_error'] * 100:.1f}%" if "color_error" in f else None,
                 f"glass {', '.join(f['shells'])}" if f.get("shells") else None]
        out.append(f"face detail {lid}: " + ", ".join(p for p in parts if p))
    return out


def format_summary(report: Mapping[str, Any]) -> str:
    """Human-readable result block with per-view error percentages."""
    lines = [f"===== BBR RESULT: {report.get('name')} =====", f"status: {report.get('status')}"]
    match = report.get("reference_match") or {}
    for view, info in (match.get("per_view") or {}).items():
        if "error" in info:  # color-only views carry just color_error
            line = format_error_line(view, info["error"])
            if info.get("floor") is not None:
                line += f"  (floor {info['floor'] * 100:.1f}%, excess {info['excess_error'] * 100:+.1f}%)"
            lines.append(line)
    for view, info in (match.get("per_view") or {}).items():
        if "color_error" in info:
            lines.append(format_color_error_line(view, info["color_error"]))
    if "total_error" in match:
        lines.append(f"Total Error : {match['total_error'] * 100:.1f}% (threshold {match['threshold'] * 100:.1f}%)")
    if report.get("consistency"):
        lines.append(format_consistency(report["consistency"]))
    lines.extend(format_measurements(report))
    if report.get("dimension_check") and report.get("validation") is None:     # --stop-after secondary: the parts-only checks
        lines.extend(format_dimension_check(report["dimension_check"]))
    lines.extend(f"WARNING part_gap: {format_gap(g)}" for g in report.get("part_gaps") or [])
    lines.extend(f"WARNING floating_part: {format_floating(g)}" for g in report.get("floating_parts") or [])
    for hid, st in (report.get("hull") or {}).items():
        if isinstance(st, dict):
            lines.extend(f"WARNING hull_debris: {t}" for t in format_debris_report(st.get("debris") or [], hid))
    iterations = report.get("iterations") or []
    if len(iterations) > 1:
        lines.append("iterations: " + ", ".join(f"{it['iteration']}:{it['total_error']:.4f}" for it in iterations))
    if report.get("passed") is not None:
        validation = report.get("validation") or {}
        failed = failed_checks(((validation.get("geometry") or {}).get("checks")) or {}, validation.get("reference_match") or {})             if not report["passed"] else []
        lines.append("validation: " + ("PASSED" if report["passed"] else "FAILED" + (": " + "; ".join(failed) if failed else "")))
    if report.get("recommendation"):
        lines.append(f"recommendation: {report['recommendation']}")
    if report.get("error"):
        lines.append(f"error: {report['error']}")
    for key in ("blend", "report", "perspective_preview", "perspective_back_preview", "front_lit_preview", "side_lit_preview",
                "top_preview", "turntable_preview", "comparison_sheet", "face_closeup_preview"):
        if key in (report.get("outputs") or {}):
            lines.append(f"{key}: {report['outputs'][key]}")
    for cid, st in (report.get("hull") or {}).items():
        tex = st.get("texture") if isinstance(st, dict) else None
        if tex and "views" in tex:
            views = " ".join(f"{k} {100 * v:.0f}%" for k, v in tex["views"].items())
            tail = f"conflict {100 * tex['conflict_ratio']:.0f}%"
            lines.append(f"texture {cid}: {views} blend {100 * tex['blend_share']:.0f}% "
                         f"inpainted {100 * tex['inpainted_share']:.0f}% {tail} "
                         f"bg-leak {100 * tex.get('background_leak_ratio', 0.0):.2f}%")
    lines.extend(format_face_detail(report.get("face_detail")))
    lines.append(f"elapsed: {report.get('elapsed_s')} s")
    nxt = next_action(report, failed if report.get("passed") is False else [])
    if nxt:
        lines.append(nxt)
    lines.append(f"===== END BBR RESULT: {report.get('name')} =====")
    return "\n".join(lines)
