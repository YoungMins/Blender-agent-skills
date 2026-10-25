"""Axis conventions per view: the single source of truth for orientation.

Blender world, meters, Z up. The object's FRONT faces -Y; its own right side is at -X.
A view is defined by ``camera_axis``: the world direction from the origin TO the camera.
Image coordinates: u to the right, v DOWN (row 0 = top).

    camera_axis | looks along | image_right | image_up
    -Y (front)  | +Y         | +X          | +Z
    +Y (back)   | -Y         | -X          | +Z
    +X (right)  | -X         | +Y          | +Z
    -X (left)   | +X         | -Y          | +Z
    +Z (top)    | -Z         | +X          | +Y
    -Z (bottom) | +Z         | +X          | -Y
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import SpecError

AXES: dict[str, np.ndarray] = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

DEFAULT_CAMERA_AXIS: dict[str, str] = {
    "front": "-Y",
    "back": "+Y",
    "right": "+X",
    "left": "-X",
    "top": "+Z",
    "bottom": "-Z",
}

# camera_axis -> (image_right axis, image_up axis)
_FRAME_TABLE: dict[str, tuple[str, str]] = {
    "-Y": ("+X", "+Z"),
    "+Y": ("-X", "+Z"),
    "+X": ("+Y", "+Z"),
    "-X": ("-Y", "+Z"),
    "+Z": ("+X", "+Y"),
    "-Z": ("+X", "-Y"),
}

_AXIS_LETTERS = "XYZ"


@dataclass(frozen=True, eq=False)
class ViewFrame:
    """Orthonormal frame of a view. ``forward`` = camera look direction = -camera_axis."""

    camera_axis: str
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    right_axis: str
    up_axis: str


def normalize_axis(axis: str) -> str:
    """Normalize ``'-y'`` / ``' -Y '`` to ``'-Y'``; raise SpecError if invalid."""
    key = str(axis).strip().upper()
    if key not in AXES:
        raise SpecError(f"invalid axis {axis!r}; expected one of {sorted(AXES)}")
    return key


def view_frame(camera_axis: str) -> ViewFrame:
    """Return the :class:`ViewFrame` of a camera axis such as ``'-Y'``."""
    key = normalize_axis(camera_axis)
    right_axis, up_axis = _FRAME_TABLE[key]
    return ViewFrame(key, AXES[right_axis], AXES[up_axis], -AXES[key], right_axis, up_axis)


def axis_index(axis: str) -> tuple[int, int]:
    """``'-Y'`` -> ``(1, -1)``: (world component index, sign)."""
    key = normalize_axis(axis)
    return _AXIS_LETTERS.index(key[1]), (1 if key[0] == "+" else -1)
