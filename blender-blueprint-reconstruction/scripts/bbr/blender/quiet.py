"""Silence Blender's own per-render ``Saved: '...'`` lines (written by native code, so Python logging cannot filter them)."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator

ENABLED = True   # ``run_pipeline.py --verbose`` turns it off


@contextlib.contextmanager
def native_stdout() -> Iterator[None]:
    """Send file descriptor 1 to the null device inside the block (no-op when disabled or when fd 1 is unavailable)."""
    if not ENABLED:
        yield
        return
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError):
        yield
        return
    os.dup2(null, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)
