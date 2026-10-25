"""Symmetry about X = 0 (numpy only): a mirrored virtual colour view, the L/R symmetry measure behind the
``symmetry.x`` suggestion, and the ``symmetric_pairs`` hint of the part fitting.

* **Mirrored virtual view.** A view that looks along X (``side``, camera ``-X``) reflected through X = 0 is the view
  from the opposite axis (``+X``) of the mirrored object: its image is the horizontally flipped image, its mask the
  flipped mask, its labels the flipped labels with the left / right part ids swapped (``horn_l`` <-> ``horn_r``). It
  is registered with the model by construction (no second generated image to disagree with the first), so it can
  colour the flank the real views do not see, for objects that are symmetric. Labelled parts that are NOT symmetric
  (``asymmetric_parts``, e.g. ``sword_*``) are cut out of its mask, so they are never mirrored onto the other flank.
* **Symmetry measure.** The mirror IoU of an upright front / back silhouette about the world X = 0 plane, optionally
  with the labelled asymmetric parts blanked out (they would otherwise count as asymmetry of the body).
* **Pair hints.** ``symmetric_pairs`` (``[a, b]``, ``[a, b, tol_m]`` or ``{"a", "b", "tol_m"}``) pull part ``b``'s fit
  toward the mirror image of part ``a`` with a dead zone of ``tol_m`` instead of an exact mirror.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .. import SpecError
from .calibration import ViewCalibration
from .conventions import view_frame
from .imageops import dilate

SUGGEST_MIN_IOU = 0.95        # front-view L/R mirror IoU at which symmetry.x is suggested
ASYMMETRIC_DILATE_PX = 2      # growth of the asymmetric parts' labels before they are cut out of the virtual mask
AXIS_SEARCH_FRAC = 0.2        # the mirror axis is searched this far (share of the image width) either side of the origin
MIN_AXIS_SHIFT_PX = 3         # a better axis closer than this to the calibrated origin is noise
MAX_SUGGESTED_PARTS = 6       # the greedy search adds at most this many labelled parts to the blanked set
NOT_A_PART = ("body",)        # label names that are never blanked (the rest of the silhouette; "_x" labels are ignored too)
MIRROR_KEY_PREFIX = "mirror_"
_SIDE_TOKENS = {"l": "r", "r": "l", "left": "right", "right": "left"}
_TOKEN_SPLIT = re.compile(r"([_\W]+)")


# --------------------------------------------------------------------------- part names
def swap_side(name: str) -> Optional[str]:
    """``horn_l`` -> ``horn_r``, ``arm_l_upper`` -> ``arm_r_upper``, ``Left_hand`` -> ``Right_hand``; None when the
    name has no left / right token (``l`` / ``r`` / ``left`` / ``right``, delimited by ``_`` or other punctuation)."""
    parts = _TOKEN_SPLIT.split(name)
    swapped = False
    for i, tok in enumerate(parts):
        other = _SIDE_TOKENS.get(tok.lower())
        if other is not None and i % 2 == 0:
            parts[i] = other.upper() if tok.isupper() else other.capitalize() if tok[:1].isupper() else other
            swapped = True
    return "".join(parts) if swapped else None


def character_side(name: str) -> Optional[str]:
    """``"l"`` / ``"r"`` for a part named after the CHARACTER's left / right (``ear_l``, ``arm_left_upper``), else None."""
    for tok in _TOKEN_SPLIT.split(name)[::2]:
        if tok.lower() in ("l", "left"):
            return "l"
        if tok.lower() in ("r", "right"):
            return "r"
    return None


def mirror_pairs(names: Sequence[str], explicit: Optional[Mapping[str, str]] = None,
                 mirror_of: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """``{part: its mirror partner}`` (both directions) for ``names``: from the left / right naming convention
    (:func:`swap_side`), overridden by ``explicit`` ``{a: b}`` and by ``mirror_of`` ``{part: sibling}``."""
    pairs = {n: s for n in names if (s := swap_side(n)) is not None}
    for src in (explicit or {}, mirror_of or {}):
        for a, b in src.items():
            pairs[str(a)], pairs[str(b)] = str(b), str(a)
    return pairs


def is_asymmetric(name: str, patterns: Sequence[str]) -> bool:
    """True when ``name`` matches one of the ``asymmetric_parts`` (exact ids or ``fnmatch`` globs such as ``sword_*``)."""
    return any(fnmatch.fnmatchcase(name, str(p)) for p in patterns)


# --------------------------------------------------------------------------- mirrored virtual view
@dataclass
class MirrorView:
    """A real view reflected through X = 0: ``key`` (``mirror_<source>``), flipped ``rgba`` / ``mask``, the opposite-axis
    calibration and the flipped ``labels`` (``{part: mask}``, ids swapped through the pair map, asymmetric parts dropped)."""

    key: str
    source: str
    rgba: np.ndarray
    mask: np.ndarray
    calib: ViewCalibration
    labels: dict[str, np.ndarray]


def _require_x_view(calib: ViewCalibration) -> None:
    if calib.camera_axis not in ("+X", "-X"):
        raise SpecError(f"view {calib.view!r} looks along {calib.camera_axis}: only a view along X (side / right / left) "
                        f"can be mirrored through X = 0 into a virtual view")


def mirror_calibration(calib: ViewCalibration, key: Optional[str] = None) -> ViewCalibration:
    """Calibration of the camera on the opposite X axis that sees the horizontally flipped image: same scale, origin
    column ``W - ox`` (image right is -Y for ``-X`` and +Y for ``+X``, so reflecting through X = 0 only flips the image)."""
    _require_x_view(calib)
    w, _ = calib.image_size
    opposite = "+X" if calib.camera_axis == "-X" else "-X"
    return ViewCalibration(key or MIRROR_KEY_PREFIX + calib.view, opposite, calib.image_size, calib.m_per_px,
                           (w - calib.origin_px[0], calib.origin_px[1]))


def mirror_view(rgba: np.ndarray, mask: np.ndarray, calib: ViewCalibration,
                labels: Optional[Mapping[str, np.ndarray]] = None, pairs: Optional[Mapping[str, str]] = None,
                asymmetric: Sequence[str] = (), dilate_px: int = ASYMMETRIC_DILATE_PX) -> MirrorView:
    """The virtual view of ``(rgba, mask, calib)`` reflected through X = 0 (see the module doc).

    ``labels``: ``{part: bool mask}`` of the source view; each is flipped and renamed through ``pairs`` (parts without a
    partner keep their name: a centred part is its own mirror). Parts matching ``asymmetric`` are removed: their pixels,
    grown by ``dilate_px``, are cut from the virtual mask (and from every label), so nothing of them is ever sampled."""
    calib_m = mirror_calibration(calib)
    flip = lambda a: np.asarray(a)[:, ::-1]                      # noqa: E731
    mask_f = flip(np.asarray(mask, dtype=bool))
    cut = np.zeros_like(mask_f)
    out: dict[str, np.ndarray] = {}
    for part, m in (labels or {}).items():
        if is_asymmetric(part, asymmetric):
            cut |= flip(np.asarray(m, dtype=bool))
            continue
        name = (pairs or {}).get(part, part)
        out[name] = out.get(name, False) | flip(np.asarray(m, dtype=bool))
    if cut.any():
        cut = dilate(cut, dilate_px) if dilate_px > 0 else cut
        mask_f = mask_f & ~cut
    return MirrorView(calib_m.view, calib.view, np.ascontiguousarray(flip(rgba)), np.ascontiguousarray(mask_f), calib_m,
                      {k: np.ascontiguousarray(v & mask_f) for k, v in out.items()})


# --------------------------------------------------------------------------- symmetry measure
def _mirror_columns(mask: np.ndarray, calib: ViewCalibration) -> np.ndarray:
    """``mask`` reflected through the world plane X = 0 (an upright view whose image horizontal is world X)."""
    if view_frame(calib.camera_axis).right_axis[1] != "X":
        raise SpecError(f"view {calib.view!r} looks along {calib.camera_axis}: its image horizontal is not world X")
    h, w = mask.shape
    src = np.floor(2.0 * calib.origin_px[0] - (np.arange(w) + 0.5)).astype(int)
    ok = (src >= 0) & (src < w)
    out = np.zeros_like(mask)
    out[:, ok] = mask[:, src[ok]]
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def mirror_iou(mask: np.ndarray, calib: ViewCalibration, blank: Optional[np.ndarray] = None) -> float:
    """IoU of an upright front / back silhouette with its reflection through X = 0. ``blank`` (bool image) removes
    those pixels and their reflection from both sides first."""
    m = np.asarray(mask, dtype=bool)
    if blank is not None and blank.any():
        b = np.asarray(blank, dtype=bool)
        m = m & ~b & ~_mirror_columns(b, calib)
    return _iou(m, _mirror_columns(m, calib))


def _suggest_blank(mask: np.ndarray, calib: ViewCalibration, labels: Mapping[str, np.ndarray],
                   asymmetric: Sequence[str], blank: Optional[np.ndarray], current: float, threshold: float
                   ) -> tuple[list[str], Optional[float]]:
    """Greedy search: the labelled parts (not yet declared asymmetric) whose blanking, added one by one, lifts the mirror IoU
    to ``threshold``. ``([], None)`` when no set of at most ``MAX_SUGGESTED_PARTS`` parts gets there."""
    cand = {p: np.asarray(m, dtype=bool) for p, m in labels.items()
            if p not in NOT_A_PART and not p.startswith("_") and not is_asymmetric(p, asymmetric) and np.asarray(m).any()}
    chosen: list[str] = []
    for _ in range(MAX_SUGGESTED_PARTS):
        scored = {p: mirror_iou(mask, calib, m if blank is None else blank | m) for p, m in cand.items() if p not in chosen}
        if not scored:
            break
        best = max(scored, key=scored.get)                       # type: ignore[arg-type]
        if scored[best] <= current + 1e-4:
            break
        chosen.append(best)
        blank, current = (cand[best] if blank is None else blank | cand[best]), scored[best]
        if current >= threshold:
            return chosen, round(current, 4)
    return [], None


def _axis_shift(mask: np.ndarray, calib: ViewCalibration, labels: Mapping[str, np.ndarray], threshold: float
                ) -> tuple[int, ViewCalibration]:
    """Pixels the mirror axis must move (right positive) for the mask with EVERY labelled part blanked to reach ``threshold``,
    and the calibration with the moved origin; ``(0, calib)`` when no axis does. A prop drags the silhouette bbox centre,
    which is the default origin, off the body axis."""
    blank = np.zeros_like(mask)
    for part, m in labels.items():
        if part not in NOT_A_PART and not part.startswith("_"):
            blank |= np.asarray(m, dtype=bool)
    reach = int(AXIS_SEARCH_FRAC * mask.shape[1])
    best = (0, mirror_iou(mask, calib, blank))
    for d in range(-reach, reach + 1, 2):
        i = mirror_iou(mask, replace(calib, origin_px=(calib.origin_px[0] + d, calib.origin_px[1])), blank)
        if i > best[1]:
            best = (d, i)
    if abs(best[0]) < MIN_AXIS_SHIFT_PX or best[1] < threshold:
        return 0, calib
    return best[0], replace(calib, origin_px=(calib.origin_px[0] + best[0], calib.origin_px[1]))


def front_symmetry(mask: np.ndarray, calib: ViewCalibration, labels: Optional[Mapping[str, np.ndarray]] = None,
                   asymmetric: Sequence[str] = (), threshold: float = SUGGEST_MIN_IOU) -> dict[str, Any]:
    """Left / right symmetry of one front / back view and the ``symmetry.x`` suggestion.

    ``iou``: plain mirror IoU; ``iou_blanked``: with the labelled ``asymmetric`` parts blanked (None without any);
    ``suggest_x`` when the deciding value (the blanked one when present) reaches ``threshold``. With ``labels`` and no
    such value yet, a greedy search finds the other labelled parts (a posed arm, a held prop) whose blanking reaches it:
    ``suggested_blank`` / ``iou_suggested`` (``suggest_x`` is then true: declare them in the hull's ``asymmetric_parts``).
    ``explanation`` is the one-line version of all of this."""
    blank = None
    for part, m in (labels or {}).items():
        if is_asymmetric(part, asymmetric):
            blank = np.asarray(m, dtype=bool) if blank is None else blank | np.asarray(m, dtype=bool)
    iou = mirror_iou(mask, calib)
    blanked = mirror_iou(mask, calib, blank) if blank is not None and blank.any() else None
    decide = blanked if blanked is not None else iou
    suggested, iou_suggested, shift = [], None, 0
    if labels and decide < threshold:
        m = np.asarray(mask, dtype=bool)
        suggested, iou_suggested = _suggest_blank(m, calib, labels, asymmetric, blank, decide, threshold)
        if not suggested:                       # the labelled props are not all: the origin itself may sit off the body axis
            shift, shifted = _axis_shift(m, calib, labels, threshold)
            if shift:
                suggested, iou_suggested = _suggest_blank(m, shifted, labels, asymmetric, blank, decide, threshold)
    out = {"view": calib.view, "iou": round(iou, 4), "iou_blanked": None if blanked is None else round(blanked, 4),
           "threshold": threshold, "suggest_x": bool(decide >= threshold or (suggested and not shift)),
           "blanked_frac": 0.0 if blank is None else round(float(blank.sum()) / max(int(np.asarray(mask).sum()), 1), 4),
           "asymmetric": [str(a) for a in asymmetric], "suggested_blank": suggested, "iou_suggested": iou_suggested,
           "labelled": bool(labels), "axis_shift_px": shift}
    out["explanation"] = explain_symmetry(out)
    return out


def explain_symmetry(sym: Mapping[str, Any]) -> str:
    """One line for a :func:`front_symmetry` result, e.g. ``0.553 raw; 0.577 with spatula_* blanked; 0.986 with
    spatula_*, flipper_r blanked -> suggest symmetry.x = true (declare them in the hull's asymmetric_parts)``."""
    text = f"{sym['iou']:.3f} raw"
    declared = ", ".join(sym.get("asymmetric") or [])
    if sym["iou_blanked"] is not None:
        text += f"; {sym['iou_blanked']:.3f} with {declared} blanked"
    both = ", ".join((sym.get("asymmetric") or []) + (sym.get("suggested_blank") or []))
    if sym.get("suggested_blank") and sym.get("axis_shift_px"):
        shift = sym["axis_shift_px"]
        text += (f"; {sym['iou_suggested']:.3f} with {both} blanked AND the mirror axis moved "
                 f"{abs(shift)} px {'right' if shift > 0 else 'left'}: the body is symmetric but the calibrated origin sits off its axis "
                 f"(a prop drags the silhouette bbox centre; set the origin from the body core); keep symmetry.x false until then")
    elif sym.get("suggested_blank"):
        text += (f"; {sym['iou_suggested']:.3f} with {both} blanked "
                 f"-> suggest symmetry.x = true (declare them in the hull's asymmetric_parts)")
    elif sym["suggest_x"]:
        text += " -> suggest symmetry.x = true"
    elif sym.get("labelled"):
        text += f"; blanking labelled parts does not reach {sym['threshold']:g}: the body itself is asymmetric, keep symmetry.x false"
    else:
        text += "; pass labels to see whether one-sided parts (a prop, a posed arm) explain it"
    return text


# --------------------------------------------------------------------------- pair hints
def pair_hints(raw: Any) -> dict[str, tuple[str, float]]:
    """``symmetric_pairs`` -> ``{partner b: (primary a, tol_m)}``; entries are ``[a, b]``, ``[a, b, tol_m]`` or
    ``{"a": .., "b": .., "tol_m": ..}`` (default tolerance 0 = pull toward the exact mirror)."""
    out: dict[str, tuple[str, float]] = {}
    for item in raw or []:
        if isinstance(item, Mapping):
            a, b, tol = item.get("a"), item.get("b"), item.get("tol_m", 0.0)
        elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
            a, b, tol = item[0], item[1], (item[2] if len(item) == 3 else 0.0)
        else:
            raise SpecError(f"symmetric_pairs entry {item!r}: use [a, b], [a, b, tol_m] or {{'a': .., 'b': .., 'tol_m': ..}}")
        if not a or not b or a == b or float(tol) < 0:
            raise SpecError(f"symmetric_pairs entry {item!r}: needs two different part ids and tol_m >= 0")
        out[str(b)] = (str(a), float(tol))
    return out
