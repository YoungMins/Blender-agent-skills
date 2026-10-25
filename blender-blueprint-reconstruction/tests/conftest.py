"""pytest setup: make ``bbr`` importable and provide synthetic-image helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

SK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SK / "scripts"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def ellipse_mask(h, w, cy, cx, ry, rx):
    """Bool mask of an axis-aligned ellipse (pixel centres)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy + 0.5 - cy) / ry) ** 2 + ((xx + 0.5 - cx) / rx) ** 2 <= 1.0


def rect_mask(h, w, x0, y0, x1, y1):
    """Bool mask of a rectangle (exclusive end)."""
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def paint(mask, color, bg=(0.0, 0.8, 0.2)):
    """RGBA float image: ``color`` where mask else ``bg``."""
    img = np.empty(mask.shape + (4,), np.float32)
    img[...] = (*bg, 1.0)
    img[mask] = (*color, 1.0)
    return img

# Blender-side tests run inside Blender (tests/blender/), not under pytest
collect_ignore = ["blender"]
