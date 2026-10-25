"""Materials: principled color, reference image (projected texture)."""
from __future__ import annotations

from typing import Optional, Sequence

import bpy

from .. import get_logger

log = get_logger(__name__)

DEFAULT_COLOR = (0.8, 0.8, 0.8, 1.0)
DEFAULT_ROUGHNESS = 0.5
REF_COLOR_ROUGHNESS = 0.6
REF_TEXTURE_MATERIAL = "M_ref_texture"
REF_TEXTURE_NODE = "bbr_basecolor"
NODE_SPACING = 300
# pbr key (see bbr.core.pbr) -> Principled BSDF input (Blender 5.x names)
PBR_SOCKETS = {"metallic": "Metallic", "roughness": "Roughness", "coat": "Coat Weight", "coat_roughness": "Coat Roughness",
               "sheen": "Sheen Weight", "specular": "Specular IOR Level", "emission_strength": "Emission Strength",
               "ior": "IOR", "transmission": "Transmission Weight"}


def _rgba(color: Optional[Sequence[float]]) -> tuple[float, float, float, float]:
    c = list(color) if color is not None else list(DEFAULT_COLOR)
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)


def _fresh_tree(mat: bpy.types.Material) -> bpy.types.NodeTree:
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat.node_tree


def _apply_pbr(bsdf: bpy.types.Node, pbr: Optional[dict[str, float]]) -> None:
    """Set the Principled inputs named by ``pbr`` (keys of :data:`PBR_SOCKETS`); a socket this Blender lacks is skipped."""
    for key, value in (pbr or {}).items():
        socket = bsdf.inputs.get(PBR_SOCKETS.get(key, ""))
        if socket is None:
            log.warning("Principled BSDF has no input for pbr key %r", key)
        else:
            socket.default_value = value


def ensure_material(name: str, base_color: Optional[Sequence[float]] = None, roughness: float = DEFAULT_ROUGHNESS,
                    metallic: float = 0.0, pbr: Optional[dict[str, float]] = None) -> bpy.types.Material:
    """Principled material ``name`` (find -> update). Also sets the viewport color used by Workbench.

    ``pbr``: extra Principled values (:func:`bbr.core.pbr.resolve`: coat, sheen, specular, emission_strength, ior, and
    metallic / roughness, which then win); emission uses ``base_color``."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    color = _rgba(base_color)
    mat.diffuse_color = color
    mat.roughness = roughness
    mat.metallic = metallic
    nt = _fresh_tree(mat)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (NODE_SPACING, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    _apply_pbr(bsdf, pbr)
    if (pbr or {}).get("emission_strength", 0.0) > 0:
        bsdf.inputs["Emission Color"].default_value = color
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def ensure_texture_material(image: bpy.types.Image, name: str = REF_TEXTURE_MATERIAL,
                            roughness: float = REF_COLOR_ROUGHNESS, pbr: Optional[dict[str, float]] = None,
                            maps: Optional[dict[str, bpy.types.Image]] = None) -> bpy.types.Material:
    """Principled material whose Base Color is ``image`` (Image Texture node ``bbr_basecolor``, UV map, linear filtering).

    ``pbr``: extra Principled values as in :func:`ensure_material` (None = the plain graph); emission reads the base image.
    ``maps``: ``{"roughness" | "metallic": image}`` driving that input (Non-Color). ``bbr_basecolor`` is always the
    active node: Workbench's TEXTURE display, the colour metric and the bench PSNR read the active image node."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    nt = _fresh_tree(mat)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (2 * NODE_SPACING, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (NODE_SPACING, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = REF_TEXTURE_NODE
    tex.image = image
    tex.interpolation = "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _apply_pbr(bsdf, pbr)
    if (pbr or {}).get("emission_strength", 0.0) > 0:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    for i, (key, map_image) in enumerate((maps or {}).items(), start=1):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.name = node.label = f"bbr_{key}"
        node.image = map_image
        node.image.colorspace_settings.name = "Non-Color"
        node.location = (0, -i * NODE_SPACING)
        nt.links.new(node.outputs["Color"], bsdf.inputs[PBR_SOCKETS[key]])
    nt.nodes.active = tex   # last: adding nodes must not move the active one
    return mat


def ensure_reference_material(name: str, image: bpy.types.Image, opacity: float) -> bpy.types.Material:
    """Unlit (emission) image material blended with transparency at ``opacity``."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = _fresh_tree(mat)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    emission = nt.nodes.new("ShaderNodeEmission")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = opacity
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    for i, node in enumerate((tex, emission, clear, mix, out)):
        node.location = (i * NODE_SPACING, 0)
    nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    try:
        mat.surface_render_method = "BLENDED"
    except (AttributeError, TypeError):  # older Blender
        mat.blend_method = "BLEND"
    return mat


def assign_material(obj: bpy.types.Object, mat: Optional[bpy.types.Material]) -> None:
    """Make ``mat`` the only material of a mesh object (None clears)."""
    obj.data.materials.clear()
    if mat is not None:
        obj.data.materials.append(mat)
