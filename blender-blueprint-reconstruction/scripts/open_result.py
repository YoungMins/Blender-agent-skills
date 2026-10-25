"""Open a reconstruction result in the Blender UI for review.

Usage:
    blender <output>/<name>.blend --python scripts/open_result.py

Shows the AI_MODEL together with the REF_BLUEPRINT reference planes in Material Preview,
so the projected colors and the reference images are both visible. View-only: nothing is modified or saved.
"""
from __future__ import annotations

import bpy

REF_COLLECTION = "REF_BLUEPRINT"
SHADING_TYPE = "MATERIAL"
FIRST_DELAY_S = 0.5


def _configure_viewports() -> None:
    """Material Preview + frame everything in every 3D viewport."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.shading.type = SHADING_TYPE
            space.overlay.show_floor = True
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.view3d.view_all(center=False)


def _show_reference_planes() -> None:
    """Make the reference planes visible in the viewport (they stay unselectable and unrendered)."""
    coll = bpy.data.collections.get(REF_COLLECTION)
    if coll is None:
        return
    coll.hide_viewport = False
    for obj in coll.objects:
        obj.hide_viewport = False
        obj.hide_set(False)


def _setup() -> None:
    _show_reference_planes()
    _configure_viewports()
    return None  # one-shot timer


bpy.app.timers.register(_setup, first_interval=FIRST_DELAY_S)
