"""blender-blueprint-reconstruction (bbr): multi-view blueprint -> 3D reconstruction toolkit."""
from __future__ import annotations

import logging

__version__ = "0.1.0"

LOGGER_ROOT = "bbr"


class BBRError(Exception):
    """Base class of every error raised by bbr."""


class SpecError(BBRError):
    """A ViewSpec / SceneSpec is invalid or cannot be loaded."""


class CalibrationError(BBRError):
    """Scale / origin calibration of a view failed."""


class ImageError(BBRError):
    """An image cannot be loaded, saved or processed."""


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``bbr`` namespace (``bbr.<name>``)."""
    if name == LOGGER_ROOT or name.startswith(LOGGER_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")
