"""Checkpoints: save/restore the .blend plus a JSON sidecar describing the state."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import bpy

from .. import BBRError, get_logger
from . import scene

log = get_logger(__name__)

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAMES = ("checkpoint_00_initial", "checkpoint_01_blockout", "checkpoint_02_primary",
                    "checkpoint_03_secondary", "checkpoint_04_final")


def _dir(out_dir: Path | str) -> Path:
    return Path(out_dir) / CHECKPOINT_DIR


def save_checkpoint(name: str, out_dir: Path | str, payload: Optional[dict[str, Any]] = None) -> Path:
    """Save ``<out_dir>/checkpoints/<name>.blend`` (copy; the session file is unchanged) and ``<name>.json``."""
    folder = _dir(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    try:
        known = json.loads(sc.get(scene.PROP_CHECKPOINTS, "[]"))
    except ValueError:
        known = []
    if name not in known:
        known.append(name)
    sc[scene.PROP_CHECKPOINTS] = json.dumps(known)
    path = folder / f"{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True, compress=True)
    sidecar = {"name": name, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "scene": scene.inspect_scene(),
               "payload": payload or {}}
    (folder / f"{name}.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    log.info("checkpoint saved: %s", path)
    return path


def list_checkpoints(out_dir: Path | str) -> list[str]:
    """Names of the checkpoints present in ``out_dir``."""
    return sorted(p.stem for p in _dir(out_dir).glob("*.blend"))


def restore_checkpoint(name: str, out_dir: Path | str) -> Path:
    """Open ``<out_dir>/checkpoints/<name>.blend`` (replaces the current session data)."""
    path = _dir(out_dir) / f"{name}.blend"
    if not path.is_file():
        raise BBRError(f"checkpoint not found: {path}; available: {list_checkpoints(out_dir)}")
    bpy.ops.wm.open_mainfile(filepath=str(path))
    log.info("checkpoint restored: %s", path)
    return path


def save_blend(path: Path | str) -> Path:
    """Save the session as ``path`` (the file becomes the current main file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
    return path
