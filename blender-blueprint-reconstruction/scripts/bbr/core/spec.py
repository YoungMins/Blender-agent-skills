"""Load / validate ViewSpec and SceneSpec (JSON files described by ``schemas/*.schema.json``).

Validation uses ``jsonschema`` when importable, else a small built-in validator that supports the
subset of JSON Schema used by our schemas. Semantic rules are checked in both cases.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .. import SpecError, get_logger
from . import pbr
from .conventions import AXES, DEFAULT_CAMERA_AXIS, normalize_axis
from .triangulate import ANCHOR_TYPES

log = get_logger(__name__)

SPEC_VERSION = "0.1"
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"
VIEW_SCHEMA_NAME = "view_spec.schema.json"
SCENE_SCHEMA_NAME = "scene_spec.schema.json"

ROLES = ("carve", "color", "validate", "preview", "holdout")
DEFAULT_ROLES: dict[str, list[str]] = {
    "front": ["carve", "color", "validate"],
    "side": ["carve", "color", "validate"],
    "left": ["carve", "color", "validate"],
    "right": ["carve", "color", "validate"],
    "back": ["color", "validate"],
    "top": ["validate"],
    "bottom": ["validate"],
}
DEFAULT_BACKGROUND_TOLERANCE = 0.12
NON_REFERENCE_ROLES = ("preview", "holdout")
STAGE_ORDER = ("blockout", "primary", "secondary", "detail")
DEFAULT_STYLE: dict[str, Any] = {
    "hard_surface": True, "bevel": True, "bevel_width": 0.003, "bevel_segments": 2, "weighted_normals": True,
    "texture": "auto", "pbr": "auto",
}
TEXTURE_MODES = ("auto", "project", "flat")
DEFAULT_WORKFLOW: dict[str, Any] = {
    "max_iterations": 3, "error_threshold": 0.05, "blockout_threshold": 0.12, "auto_correct": True,
}
HULL_TYPE = "silhouette_hull"
TRUE_SIZE_PART_TYPES = ("cone", "blade", "disc", "strand", "torus", "lathe", "layer_hull")  # sized by params / labels, not by ``size``


# --------------------------------------------------------------------------- schema validation
@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """Load a schema JSON from the skill's ``schemas`` directory."""
    path = SCHEMA_DIR / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"cannot load schema {path}: {exc}") from exc


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True


def _mini_validate(value: Any, schema: dict, root: dict, path: str, errors: list[str]) -> None:
    """Validate ``value`` against the JSON-Schema subset used by our schemas."""
    if "$ref" in schema:
        node: Any = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        _mini_validate(value, node, root, path, errors)
        return
    declared = schema.get("type")
    if declared is not None:
        types = [declared] if isinstance(declared, str) else list(declared)
        if not any(_type_ok(value, t) for t in types):
            errors.append(f"{path}: expected {'|'.join(types)}, got {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} must be < {schema['exclusiveMaximum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: string shorter than {schema['minLength']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if len(value) > schema.get("maxItems", len(value)):
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                _mini_validate(item, schema["items"], root, f"{path}[{i}]", errors)
    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{path}: fewer than {schema['minProperties']} properties")
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required key '{req}'")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            sub = f"{path}.{key}" if path != "$" else key
            if key in props:
                _mini_validate(item, props[key], root, sub, errors)
            elif extra is False:
                errors.append(f"{sub}: unknown key")
            elif isinstance(extra, dict):
                _mini_validate(item, extra, root, sub, errors)


def _schema_errors(data: Any, schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    try:
        import jsonschema  # type: ignore
    except ImportError:
        errors: list[str] = []
        _mini_validate(data, schema, schema, "$", errors)
        return errors
    validator = jsonschema.Draft202012Validator(schema)
    out = []
    for err in validator.iter_errors(data):
        loc = ".".join(str(p) for p in err.absolute_path) or "$"
        out.append(f"{loc}: {err.message}")
    return sorted(out)


def schema_errors_builtin(data: Any, schema_name: str) -> list[str]:
    """Schema errors using only the built-in validator (exposed for tests / jsonschema-free hosts)."""
    schema = load_schema(schema_name)
    errors: list[str] = []
    _mini_validate(data, schema, schema, "$", errors)
    return errors


# --------------------------------------------------------------------------- view spec
@dataclass
class BackgroundSpec:
    """How to separate object from background."""

    mode: str = "auto"
    color: Optional[tuple[float, ...]] = None
    tolerance: float = DEFAULT_BACKGROUND_TOLERANCE
    remove_enclosed: Optional[bool] = None   # drop enclosed backdrop-coloured holes; None = on for a chroma backdrop


@dataclass
class CalibrationSpec:
    """How the view's pixel scale / origin is obtained."""

    mode: str = "silhouette_bbox"
    px_per_meter: Optional[float] = None
    origin_px: Optional[tuple[float, float]] = None


@dataclass
class ViewEntry:
    """One reference view with defaults applied and an absolute image path."""

    key: str
    path: Path
    enabled: bool = True
    camera_axis: Optional[str] = None
    flip_horizontal: bool = False
    crop_px: Optional[tuple[int, int, int, int]] = None
    roles: list[str] = field(default_factory=list)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    camera_type: str = "orthographic"
    confidence: str = "observed"
    notes: str = ""
    camera_angles: Optional[tuple[float, float]] = None   # (azimuth, elevation) degrees of a non-axis holdout view

    def has_role(self, role: str) -> bool:
        """True if the view carries ``role``."""
        return role in self.roles

    @property
    def is_orthographic_usable(self) -> bool:
        """True when the view has an axis and a role other than preview / holdout (holdout views never feed
        carving, colour, calibration or correction; they are scored separately)."""
        return self.camera_axis is not None and any(r not in NON_REFERENCE_ROLES for r in self.roles)


@dataclass
class ViewSpec:
    """Parsed ViewSpec."""

    spec_version: str
    name: str
    base_dir: Path
    views: dict[str, ViewEntry]
    known_dimensions: dict[str, Optional[float]]
    symmetry: dict[str, bool]
    reference_quality: dict[str, Any]
    source_path: Optional[Path] = None
    consistency: dict[str, Any] = field(default_factory=dict)   # {ignore_flags, roles_override} recorded by ``check --write``
    dimension_provenance: dict[str, str] = field(default_factory=dict)   # {dimension: how a known value was derived} (draft-view-spec)

    def enabled_views(self, role: Optional[str] = None) -> list[ViewEntry]:
        """Enabled views (spec order), optionally only those with ``role``."""
        return [v for v in self.views.values() if v.enabled and (role is None or v.has_role(role))]

    def view_axes(self) -> dict[str, str]:
        """Enabled usable (non-preview) views -> camera_axis."""
        return {v.key: v.camera_axis for v in self.enabled_views() if v.is_orthographic_usable}  # type: ignore[misc]

    def holdout_views(self) -> list[ViewEntry]:
        """Enabled held-out views (role ``holdout``): scored in the report's ``holdout`` block, never used to build the model."""
        return self.enabled_views("holdout")


def _semantic_view_errors(data: dict) -> list[str]:
    errors: list[str] = []
    views = data.get("views", {})
    for key, v in views.items():
        cam = v.get("camera_axis")
        roles = v.get("roles")
        angled = isinstance(cam, dict)
        if angled:
            extra = sorted(set(cam) - {"azimuth", "elevation"})
            if extra:
                errors.append(f"views.{key}.camera_axis: unknown keys {extra}; use azimuth / elevation")
            if not isinstance(cam.get("azimuth"), (int, float)) or isinstance(cam.get("azimuth"), bool):
                errors.append(f"views.{key}.camera_axis: azimuth (degrees) is required")
            el = cam.get("elevation", 0)
            if not isinstance(el, (int, float)) or isinstance(el, bool) or not -90 <= el <= 90:
                errors.append(f"views.{key}.camera_axis: elevation must be a number in [-90, 90]")
            if key == "side" or any(r not in NON_REFERENCE_ROLES for r in (roles or ["holdout"])):
                errors.append(f"views.{key}: an azimuth/elevation camera_axis is only allowed for holdout / preview views")
            cam = None
        elif cam is not None and not (isinstance(cam, str) and cam.strip().upper() in AXES):
            errors.append(f"views.{key}.camera_axis: {cam!r} is not one of {sorted(AXES)} or an azimuth/elevation object")
            cam = None
        if roles and "holdout" in roles and set(roles) - set(NON_REFERENCE_ROLES):
            errors.append(f"views.{key}: role 'holdout' cannot be combined with {sorted(set(roles) - set(NON_REFERENCE_ROLES))}"
                          " (held-out views are never used to build or correct the model)")
        if key == "side":
            if cam is None:
                errors.append("views.side: 'side' requires an explicit camera_axis (+X or -X)")
            elif cam not in ("+X", "-X"):
                errors.append("views.side: camera_axis must be +X or -X")
        elif key not in DEFAULT_CAMERA_AXIS and cam is None and not angled:
            if roles is None or any(r != "preview" for r in roles):
                errors.append(f"views.{key}: unknown view key needs camera_axis or roles ['preview']")
        crop = v.get("crop_px")
        if crop and (crop[2] <= crop[0] or crop[3] <= crop[1]):
            errors.append(f"views.{key}.crop_px: x1/y1 must be greater than x0/y0")
        cal = v.get("calibration") or {}
        if cal.get("mode") == "manual" and (cal.get("px_per_meter") is None or cal.get("origin_px") is None):
            errors.append(f"views.{key}.calibration: manual mode requires px_per_meter and origin_px")
    if views and not any(v.get("enabled", True) for v in views.values()):
        errors.append("views: at least one enabled view is required")
    return errors


def validate_view_spec(data: Any) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errors = _schema_errors(data, VIEW_SCHEMA_NAME)
    if not errors and isinstance(data, dict):
        errors = _semantic_view_errors(data)
    return errors


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read spec {path}: {exc}") from exc
    except ValueError as exc:
        raise SpecError(f"invalid JSON in {path}: {exc}") from exc


def parse_view_spec(data: dict, base_dir: Path | str = ".", source_path: Optional[Path] = None) -> ViewSpec:
    """Validate ``data`` and build a ViewSpec; relative image paths resolve against ``base_dir``."""
    errors = validate_view_spec(data)
    if errors:
        raise SpecError("invalid ViewSpec:\n  " + "\n  ".join(errors))
    root = Path(base_dir)
    if data.get("base_dir"):
        override = Path(data["base_dir"])
        root = override if override.is_absolute() else root / override
    root = root.resolve()
    default_cal_mode = (data.get("calibration") or {}).get("mode", "silhouette_bbox")
    views: dict[str, ViewEntry] = {}
    for key, v in data["views"].items():
        raw_cam = v.get("camera_axis")
        angles = None
        if isinstance(raw_cam, dict):
            angles, cam = (float(raw_cam["azimuth"]), float(raw_cam.get("elevation", 0.0))), None
        else:
            cam = raw_cam or DEFAULT_CAMERA_AXIS.get(key)
            cam = normalize_axis(cam) if cam else None
        roles = v.get("roles")
        if roles is None:
            roles = list(DEFAULT_ROLES[key]) if key in DEFAULT_ROLES else (
                ["carve", "color", "validate"] if cam else (["holdout"] if angles else ["preview"]))
        if not cam or "preview" in roles and len(roles) == 1:
            log.info("view %s is preview-only; stored, not used for carving/validation", key)
        bg = v.get("background") or {}
        cal = v.get("calibration") or {}
        path = Path(v["path"])
        views[key] = ViewEntry(
            key=key,
            path=path if path.is_absolute() else root / path,
            enabled=v.get("enabled", True),
            camera_axis=cam,
            flip_horizontal=v.get("flip_horizontal", False),
            crop_px=tuple(v["crop_px"]) if v.get("crop_px") else None,
            roles=list(roles),
            background=BackgroundSpec(
                bg.get("mode", "auto"),
                tuple(bg["color"]) if bg.get("color") else None,
                bg.get("tolerance", DEFAULT_BACKGROUND_TOLERANCE),
                bg.get("remove_enclosed")),
            calibration=CalibrationSpec(
                cal.get("mode", default_cal_mode), cal.get("px_per_meter"),
                tuple(cal["origin_px"]) if cal.get("origin_px") else None),
            camera_type=v.get("camera_type", "orthographic"),
            confidence=v.get("confidence", "observed"),
            notes=v.get("notes", ""),
            camera_angles=angles,
        )
    known = {k: None for k in ("width", "depth", "height")}
    known.update(data.get("known_dimensions") or {})
    sym = {"x": False, "y": False, "z": False}
    sym.update(data.get("symmetry") or {})
    return ViewSpec(
        spec_version=data["spec_version"], name=data.get("name", "unnamed"), base_dir=root, views=views,
        known_dimensions=known, symmetry=sym,
        reference_quality=dict(data.get("reference_quality") or {}), source_path=source_path,
        consistency=dict(data.get("consistency") or {}),
        dimension_provenance=dict(data.get("dimension_provenance") or {}))


def load_view_spec(path: Path | str, check_files: bool = True) -> ViewSpec:
    """Load a ViewSpec JSON file; raise SpecError when invalid (or an enabled image is missing)."""
    path = Path(path).resolve()
    spec = parse_view_spec(_read_json(path), path.parent, path)
    if check_files:
        missing = [str(v.path) for v in spec.enabled_views() if not v.path.is_file()]
        if missing:
            raise SpecError(f"ViewSpec {path}: missing image files: {missing}")
    return spec


# --------------------------------------------------------------------------- scene spec
@dataclass
class Component:
    """One scene component with defaults applied."""

    id: str
    type: str
    stage: str = "primary"
    confidence: str = "observed"
    location: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: Optional[list[float]] = None
    params: dict[str, Any] = field(default_factory=dict)
    material: Optional[str] = None
    mirror: dict[str, bool] = field(default_factory=lambda: {"x": False, "y": False, "z": False})
    mirror_center: Optional[list[float]] = None   # world point the mirror planes pass through (default: the world origin)
    bevel: Any = None
    boolean: Optional[dict[str, Any]] = None
    notes: str = ""


@dataclass
class SceneSpec:
    """Parsed SceneSpec."""

    spec_version: str
    name: str
    category: str
    object_type: str
    units: str
    dimensions: dict[str, Optional[float]]
    symmetry: dict[str, bool]
    style: dict[str, Any]
    materials: dict[str, dict[str, Any]]
    components: list[Component]
    workflow: dict[str, Any]
    source_path: Optional[Path] = None
    parts_labels: Optional[str] = None   # labels.json (see core/labels.py), relative to the SceneSpec file

    def components_by_stage(self) -> list[Component]:
        """Components in build order (stage, then spec order)."""
        return sorted(self.components, key=lambda c: STAGE_ORDER.index(c.stage))

    def get_component(self, comp_id: str) -> Component:
        """Component by id; raise SpecError if absent."""
        for comp in self.components:
            if comp.id == comp_id:
                return comp
        raise SpecError(f"no component with id {comp_id!r}")

    def agent_id(self, comp_id: str) -> str:
        """Custom-property agent id of a component: ``<scene name>:<id>``."""
        return f"{self.name}:{comp_id}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict (round-trips through parse_scene_spec)."""
        out = asdict(self)
        out.pop("source_path", None)
        for comp in out["components"]:
            if comp.get("mirror_center") is None:
                comp.pop("mirror_center", None)
        if out.get("parts_labels") is None:
            out.pop("parts_labels", None)
        elif self.source_path is not None:   # the copy may be written elsewhere (corrected spec): keep the path valid
            out["parts_labels"] = str((Path(self.source_path).parent / out["parts_labels"]).resolve())
        return out


def _semantic_scene_errors(data: dict) -> list[str]:
    errors: list[str] = []
    comps = data.get("components", [])
    ids = [c["id"] for c in comps]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"components: duplicate id '{dup}'")
    materials = data.get("materials") or {}
    for i, c in enumerate(comps):
        where = f"components[{i}] ({c['id']})"
        params = c.get("params") or {}
        anchored = bool(params.get("anchors_px") or params.get("mirror_of"))   # pose from image anchors, no size / rotation typed
        if c.get("size") is None and c["type"] != HULL_TYPE and c["type"] not in TRUE_SIZE_PART_TYPES and not anchored                 and not (c["type"] == "plate" and params.get("outline")):
            errors.append(f"{where}: size is required for type '{c['type']}'")
        if anchored and c["type"] not in ANCHOR_TYPES:
            errors.append(f"{where}: anchors_px / mirror_of are supported for types {list(ANCHOR_TYPES)}, not '{c['type']}'")
        for ref_key in ("attach_to", "mirror_of"):
            ref = params.get(ref_key)
            if ref and ref not in ids:
                errors.append(f"{where}: params.{ref_key} '{ref}' is not a component id")
            if ref and ref == c["id"]:
                errors.append(f"{where}: params.{ref_key} cannot be the component itself")
        if c["type"] == "lathe" and not (params.get("profile") or params.get("profile_from")):
            errors.append(f"{where}: a lathe needs params.profile ([[z, r], ...] meters) or params.profile_from")
        at = params.get("attach_at")
        if at is not None:
            if not params.get("attach_to"):
                errors.append(f"{where}: params.attach_at needs params.attach_to (the component whose surface it names)")
            elif anchored:
                errors.append(f"{where}: params.attach_at places a part without anchors_px / mirror_of; use one or the other")
            if not isinstance(at, dict) or not ({"view", "px"} <= set(at) or {"azimuth_deg", "height"} <= set(at)):
                errors.append(f"{where}: params.attach_at must be {{\"view\": V, \"px\": [x, y]}} or {{\"azimuth_deg\": a, \"height\": z}}")
        joint = params.get("joint")
        if joint is not None:
            ref = joint.get("part") if isinstance(joint, dict) else None
            if ref not in ids or ref == c["id"]:
                errors.append(f"{where}: params.joint must be {{\"part\": <another component id>, \"at\": ..., \"self\": ...}}, got {joint!r}")
        for ref in params.get("adjacent_to") or []:
            if ref not in ids or ref == c["id"]:
                errors.append(f"{where}: params.adjacent_to '{ref}' is not another component id")
        if c.get("material") and c["material"] not in materials and not pbr.is_class(c["material"]):
            errors.append(f"{where}: material '{c['material']}' is neither defined in 'materials' nor a material class "
                          f"({', '.join(pbr.class_names())})")
        if c.get("mirror_center") and not any((c.get("mirror") or {}).values()):
            errors.append(f"{where}: mirror_center needs a mirror axis (mirror: {{\"x\": true, ...}})")
        boolean = c.get("boolean")
        if boolean and boolean["target"] not in ids:
            errors.append(f"{where}: boolean target '{boolean['target']}' is not a component id")
        if boolean and boolean["target"] == c["id"]:
            errors.append(f"{where}: boolean target cannot be the component itself")
    return errors


def validate_scene_spec(data: Any) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errors = _schema_errors(data, SCENE_SCHEMA_NAME)
    if not errors and isinstance(data, dict):
        errors = _semantic_scene_errors(data)
    return errors


def parse_scene_spec(data: dict, source_path: Optional[Path] = None) -> SceneSpec:
    """Validate ``data`` and build a SceneSpec with defaults applied."""
    errors = validate_scene_spec(data)
    if errors:
        raise SpecError("invalid SceneSpec:\n  " + "\n  ".join(errors))
    comps: list[Component] = []
    for c in data["components"]:
        mirror = {"x": False, "y": False, "z": False}
        mirror.update(c.get("mirror") or {})
        boolean = c.get("boolean")
        if boolean:
            boolean = {"target": boolean["target"], "operation": boolean.get("operation", "DIFFERENCE")}
        comps.append(Component(
            id=c["id"], type=c["type"], stage=c.get("stage", "primary"),
            confidence=c.get("confidence", "observed"),
            location=[float(x) for x in c.get("location", [0.0, 0.0, 0.0])],
            rotation_deg=[float(x) for x in c.get("rotation_deg", [0.0, 0.0, 0.0])],
            size=[float(x) for x in c["size"]] if c.get("size") else None,
            params=copy.deepcopy(c.get("params") or {}), material=c.get("material"), mirror=mirror,
            mirror_center=[float(x) for x in c["mirror_center"]] if c.get("mirror_center") else None,
            bevel=copy.deepcopy(c.get("bevel")), boolean=boolean, notes=c.get("notes", "")))
    dims = {"width": None, "depth": None, "height": None}
    dims.update(data.get("dimensions") or {})
    sym = {"x": False, "y": False, "z": False}
    sym.update(data.get("symmetry") or {})
    style = dict(DEFAULT_STYLE)
    style.update(data.get("style") or {})
    workflow = dict(DEFAULT_WORKFLOW)
    workflow.update(data.get("workflow") or {})
    return SceneSpec(
        spec_version=data["spec_version"], name=data["name"], category=data.get("category", "other"),
        object_type=data.get("object_type", ""), units=data.get("units", "meters"), dimensions=dims,
        symmetry=sym, style=style, materials=copy.deepcopy(data.get("materials") or {}),
        components=comps, workflow=workflow, source_path=source_path,
        parts_labels=data.get("parts_labels"))


PART_TEXTURE_TYPES = ("cone", "blade", "disc", "strand")   # thin-part primitives: projected texture by default


def project_textures(style: dict[str, Any], view_spec: Optional[ViewSpec]) -> bool:
    """True when hard-surface components get the projected reference colour (``style.texture``): ``project`` always (with a
    colour view), ``flat`` never, ``auto`` (default) when the ViewSpec has a ``color`` view and its references are not
    synthetic (``reference_quality.synthetic``: renders of the SceneSpec itself, where a flat material is exact)."""
    mode = style.get("texture", "auto")
    if mode == "flat" or view_spec is None or not view_spec.enabled_views("color"):
        return False
    return mode == "project" or not view_spec.reference_quality.get("synthetic", False)


def is_textured_component(comp: Component, character: bool = False, project: bool = False) -> bool:
    """True for a non-hull, non-cutter component that gets a projected texture of its own: ``params.color_projection`` when
    set, else every type in a character scene or when ``project`` (see :func:`project_textures`), and only ``cone`` /
    ``blade`` / ``disc`` elsewhere. Hidden boolean cutters are never textured."""
    if comp.type == HULL_TYPE or comp.boolean:
        return False
    contour_plate = comp.type == "plate" and bool(comp.params.get("anchors_px") or comp.params.get("mirror_of"))   # a hard-surface plate is a box
    return bool(comp.params.get("color_projection", character or project or contour_plate or comp.type in PART_TEXTURE_TYPES))


def load_scene_spec(path: Path | str) -> SceneSpec:
    """Load a SceneSpec JSON file; raise SpecError when invalid."""
    path = Path(path).resolve()
    return parse_scene_spec(_read_json(path), path)
