"""PBR material classes: named presets of Principled parameters in a physical and a stylized column.

The physical column is measured-value PBR; the stylized column suits a projected texture, whose base colour already
carries painted highlights (metal is only partly metallic, so it does not turn black under a small world).
Resolution order: class preset first, then the explicit keys of the material entry win.

Physical values are ported from ``skills/blender-pbr-materials/references/pbr-values.md`` of
EL4CTEO/blender-skills (MIT, Copyright (c) 2026 EL4CTEO; see THIRD_PARTY_NOTICES.md).
Pure Python: no bpy.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from .. import SpecError

PBR_KEYS = ("metallic", "roughness", "coat", "coat_roughness", "sheen", "specular", "emission_strength", "ior", "transmission")
PBR_MODES = ("auto", "stylized", "physical")
DEFAULT_CLASS = "default"
TRANSPARENT_TRANSMISSION = 0.5   # a material at or above this transmission is see-through (see is_transparent)

_METAL = {"metallic": 1.0}
_COAT = {"coat": 1.0, "coat_roughness": 0.03}

# class -> (physical, stylized)
CLASSES: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "metal_steel": ({**_METAL, "roughness": 0.4}, {"metallic": 0.6, "roughness": 0.45}),
    "metal_gold": ({**_METAL, "roughness": 0.25}, {"metallic": 0.65, "roughness": 0.35}),
    "metal_brass": ({**_METAL, "roughness": 0.3}, {"metallic": 0.6, "roughness": 0.4}),
    "cloth": ({"roughness": 0.9, "sheen": 0.5}, {"roughness": 0.9, "sheen": 0.3, "specular": 0.3}),
    "skin": ({"roughness": 0.5, "ior": 1.4}, {"roughness": 0.55}),
    "hair": ({"roughness": 0.45}, {"roughness": 0.5}),
    "leather": ({"roughness": 0.55}, {"roughness": 0.6}),
    "wood": ({"roughness": 0.7}, {"roughness": 0.7}),
    "stone": ({"roughness": 0.88}, {"roughness": 0.9}),
    "plastic": ({"roughness": 0.4}, {"roughness": 0.45}),
    "painted": ({"roughness": 0.4, **_COAT}, {"roughness": 0.45, "coat": 0.3, "coat_roughness": 0.03}),
    # opaque glossy glass: no transmission, so it keeps the projected colour
    "glass": ({"roughness": 0.05, "coat": 1.0, "coat_roughness": 0.02},) * 2,
    # clear glass (helmet bubble, separate visor shell): see-through, so the colour metric and the texture projection look
    # through it (is_transparent); only the lit preview shows it
    "glass_clear": ({"transmission": 1.0, "roughness": 0.03, "ior": 1.45, "specular": 0.5},) * 2,
    # visor / lens: a coat over a rougher layer
    "visor": ({"roughness": 0.45, **_COAT},) * 2,
    "emissive": ({"emission_strength": 1.0}, {"emission_strength": 0.6}),
    "rubber": ({"roughness": 0.88, "specular": 0.3}, {"roughness": 0.9, "specular": 0.3}),
    DEFAULT_CLASS: ({"roughness": 0.6},) * 2,
}
ALIASES = {"iron": "metal_steel", "steel": "metal_steel", "gold": "metal_gold", "brass": "metal_brass",
           "fabric": "cloth", "paint": "painted"}


def class_names() -> list[str]:
    """Every accepted ``class`` value (classes and aliases), sorted."""
    return sorted(set(CLASSES) | set(ALIASES))


def is_class(name: Optional[str]) -> bool:
    """True when ``name`` is a class or an alias."""
    return name in CLASSES or name in ALIASES


def canonical(name: str) -> str:
    """Class name behind ``name`` (alias resolved); raise SpecError for an unknown class."""
    key = ALIASES.get(name, name)
    if key not in CLASSES:
        raise SpecError(f"unknown material class {name!r}; known: {', '.join(class_names())}")
    return key


def column_for(projected: bool, style: Optional[Union[Mapping[str, Any], str]] = None) -> str:
    """``"stylized"`` or ``"physical"``. ``style`` is the SceneSpec style dict (or its ``pbr`` value): ``auto`` (default)
    = stylized for projected textures, physical for flat materials."""
    mode = style.get("pbr", "auto") if isinstance(style, Mapping) else (style or "auto")
    if mode not in PBR_MODES:
        raise SpecError(f"style.pbr must be one of {PBR_MODES}, got {mode!r}")
    return mode if mode != "auto" else ("stylized" if projected else "physical")


def resolve(entry: Optional[Union[Mapping[str, Any], str]], column: str) -> dict[str, float]:
    """PBR parameters of a material entry (``{"class": ..., "roughness": ...}`` or a bare class name): the class preset of
    ``column`` first, then the explicit keys of ``entry`` win. Only keys that are set are returned ({} for no class and
    no explicit key); ``base_color`` is not part of it."""
    if column not in ("physical", "stylized"):
        raise SpecError(f"column must be 'physical' or 'stylized', got {column!r}")
    if isinstance(entry, str):
        entry = {"class": entry}
    entry = entry or {}
    out: dict[str, float] = {}
    if entry.get("class"):
        out.update(CLASSES[canonical(entry["class"])][column == "stylized"])
    out.update({k: float(entry[k]) for k in PBR_KEYS if entry.get(k) is not None})
    return out


def is_transparent(values: Mapping[str, float]) -> bool:
    """True when resolved PBR ``values`` (:func:`resolve`) are see-through (transmission >= TRANSPARENT_TRANSMISSION)."""
    return float(values.get("transmission", 0.0)) >= TRANSPARENT_TRANSMISSION


def entry_for(key: Optional[str], materials: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Material entry of a component's ``material`` value: ``materials[key]``, else the class named ``key``, else {}."""
    if key is None:
        return {}
    if key in materials:
        return dict(materials[key])
    return {"class": key} if is_class(key) else {}
