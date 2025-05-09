# =============================================================================
# ✅ Config Validator & Tool Schema Enforcer (utils/validate_config.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates the entire YAML configuration against expected structure, types, and tool-specific rules
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Performs comprehensive validation of the pipeline’s configuration, including schema version checks,
#   required section enforcement, and dynamic expression resolution. Registers tool-specific validators
#   for all supported workflow steps (e.g., enhance_images, copy_to_aws, deploy_lambda_monitor).
#   Raises `ConfigValidationError` for invalid structures, missing fields, or improper expressions.
#
# File Location:        /utils/validate_config.py
# Called By:            config_loader.py, orchestrator, CLI validation entrypoint
# Int. Dependencies:    arcpy_utils, expression_utils, path_resolver, config_loader, schema_paths
# Ext. Dependencies:    shutil, string, typing, pathlib
#
# Documentation:
#   See: docs/CONFIG_GUIDE.md and docs/config_schema_reference.md
#
# Notes:
#   - Uses per-tool validator dispatch via `TOOL_VALIDATORS` registry
#   - Supports `DISABLED` flags and external command validation (e.g., exiftool path)
#   - Designed to be run both via `__main__` and programmatically from tool execution
# =============================================================================

import shutil
import string
from typing import Union, Type, Tuple, Optional
from pathlib import Path
from utils.arcpy_utils import log_message
from utils.expression_utils import load_field_registry, resolve_expression
from utils.path_resolver import resolve_relative_to_pyt

SUPPORTED_SCHEMA_VERSIONS = {"1.0.1"}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def validate_config_section(cfg: dict, path: str, expected_type=dict, messages=None):
    """
    Validates that a nested section exists in the configuration and matches the expected type.
    
    Checks for the presence of a dot-separated key path within the config dictionary and verifies that the final value
    is of the specified type. Logs errors if the section is missing or of the wrong type.
    """
    if not path:
        log_message("Empty config path provided", messages, level="error", error_type=ConfigValidationError, config=cfg)
        return

    keys = path.split(".")
    current = cfg
    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            log_message(f"{'.'.join(keys[:i])} must be a dictionary", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
            return
        if key not in current:
            log_message(f"Missing required config section: '{path}'", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
            return
        current = current[key]

    if not isinstance(current, expected_type):
        log_message(f"Config section '{path}' must be of type {expected_type.__name__}", messages, level="error",
                    error_type=ConfigValidationError, config=cfg)


def validate_type(value, context: str, expected_type: Union[Type, Tuple[Type, ...]], messages=None):
    """
    Checks whether a value matches the expected Python type(s) and logs an error if not.
    
    Args:
        value: The value to validate.
        context: Identifier for the value's location in the config, used in error messages.
        expected_type: The required type or tuple of types for the value.
        messages: Optional message collector for logging errors.
    """
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__

        # Check if expected_type is a tuple, otherwise treat it as a single type
        if isinstance(expected_type, tuple):
            expected = ", ".join(t.__name__ for t in expected_type)
        else:
            expected = expected_type.__name__  # Just the name of the type if it's not a tuple

        log_message(f"{context} must be of type {expected}, but got {actual_type}", messages,
                    level="error", error_type=ConfigValidationError)


def validate_expression_block(block: dict, keys: list[str], cfg: dict, expected_type: Union[Type, Tuple[Type, ...]],
                              context: str, messages=None):
    """
    Validates that specified keys in a dictionary are either literals of the expected type or resolvable config
    expressions.

    For each key in `keys`, checks that the value in `block` is either an instance of `expected_type` or a string
    expression that resolves to `expected_type` using the provided config. Field expressions (strings starting with
    "field.") are ignored. Logs errors for missing keys, type mismatches, or failed expression resolution.
    """
    for key in keys:
        full_key = f"{context}.{key}"
        val = block.get(key)

        if val is None:
            log_message(f"{full_key} is required", messages, level="error", error_type=ConfigValidationError,
                        config=cfg)

        elif isinstance(val, expected_type):
            continue

        elif isinstance(val, str):
            try_resolve_config_expression(val, full_key, cfg, messages, expected_type=expected_type)

        else:
            log_message(f"{full_key} must be a {expected_type.__name__} or a resolvable config expression",
                        messages, level="error", error_type=ConfigValidationError, config=cfg)


def try_resolve_config_expression(expr: str, context: str, cfg: dict,  messages=None, expected_type=None):
    """
    Attempts to resolve a configuration expression and optionally checks its type.
    
    If the expression starts with "field.", resolution is skipped. If resolution fails or the resolved value does not
    match the expected type (if provided), an error is logged and None is returned.
    
    Args:
        expr: The configuration expression to resolve.
        context: Description of where the expression is located, used in error messages.
        cfg: The configuration dictionary used for resolution context.
        messages: Optional message collector for logging errors and warnings.
        expected_type: Optional type or tuple of types; if provided, the resolved value must match this type.
    
    Returns:
        The resolved value if successful and of the expected type, or None if resolution fails or is skipped.
    """
    if expr is None or not isinstance(expr, str):
        return None

    if expr.startswith("field."):
        return None  # Skip resolution of field expressions

    try:
        result = resolve_expression(expr, config=cfg)
        if expected_type and not isinstance(result, expected_type):
            log_message(
                f"{context}: resolved value must be of type {expected_type.__name__}, got {type(result).__name__}",
                messages, level="error", error_type=ConfigValidationError, config=cfg
            )
        return result
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        log_message(
            f"{context}: failed to resolve expression '{expr}': {e}",
            messages, level="error", error_type=ConfigValidationError, config=cfg
        )
        return None


def check_file_exists(path, context, messages):
    """
    Validates that a given executable or file path is either:
    - A valid file on disk (absolute or relative)
    - A valid command on the system PATH
    - Or marked 'DISABLED'
    """
    if path == "DISABLED":
        return

    # Check if it's an absolute or relative file path
    path_obj = Path(path)
    if path_obj.is_absolute() or path_obj.suffix:
        resolved = path_obj if path_obj.is_absolute() else resolve_relative_to_pyt(path)
        if resolved.is_file():
            return

    # Otherwise check if it's an available command
    if shutil.which(path):
        return

    # If neither valid file nor command, raise error
    log_message(
        f"{context} does not point to a valid file or executable (or use 'DISABLED'): {path}",
        messages, level="error", error_type=ConfigValidationError
    )



def check_required_keys(d, keys, context, messages):
    """
    Ensures all specified keys are present in a dictionary, logging an error for each missing key.
    
    Logs a detailed error message for each required key not found in the dictionary, using the provided context for
    clarity.
    """
    for key in keys:
        if key not in d:
            log_message(f"{context}.{key} is required", messages, level="error",
                        error_type=ConfigValidationError)


VALID_FIELD_TYPES = {
    "SHORT", "LONG", "BIGINTEGER", "FLOAT", "DOUBLE", "TEXT",
    "DATE", "DATEHIGHPRECISION", "DATEONLY", "TIMEONLY", "TIMESTAMPOFFSET",
    "BLOB", "GUID", "RASTER", "Integer", "SmallInteger", "String"
}


def validate_field_block(field_block: dict, context: str = "field", messages=None):
    """
    Validates a single field definition block for required keys and correct value types.
    
    Ensures that the field block contains valid 'name' and 'type' entries, and that optional keys such as 'length',
    'alias', 'expression', and 'oid_default' are present only with appropriate types and usage. Logs errors for
    missing or invalid entries.
    """
    for key in ["name", "type"]:
        if key not in field_block:
            log_message(f"{context}: Missing required key '{key}'", messages, level="error",
                        error_type=ConfigValidationError)

    name = field_block.get("name")
    ftype = field_block.get("type")

    if not isinstance(name, str):
        log_message(f"{context}: 'name' must be a string", messages, level="error",
                    error_type=ConfigValidationError)

    if ftype not in VALID_FIELD_TYPES:
        log_message(f"{context}: Invalid field type '{ftype}'. Must be one of: {', '.join(sorted(VALID_FIELD_TYPES))}",
                    messages, level="error", error_type=ConfigValidationError)

    if "length" in field_block:
        if ftype == "TEXT" and not isinstance(field_block["length"], int):
            log_message(f"{context}: 'length' must be an integer for TEXT fields", messages, level="error",
                        error_type=ConfigValidationError)
        elif ftype != "TEXT" and field_block["length"] is not None:
            log_message(f"{context}: 'length' should be omitted or null for non-TEXT fields", messages, level="error",
                        error_type=ConfigValidationError)

    if "alias" in field_block and not isinstance(field_block["alias"], str):
        log_message(f"{context}: 'alias' must be a string if provided", messages, level="error",
                    error_type=ConfigValidationError)

    if "expression" in field_block and not isinstance(field_block["expression"], str):
        log_message(f"{context}: 'expression' must be a string if provided", messages, level="error",
                    error_type=ConfigValidationError)

    if "oid_default" in field_block and not isinstance(field_block["oid_default"], (int, float, str)):
        log_message(f"{context}: 'oid_default' must be a string, int, or float if provided", messages, level="error",
                    error_type=ConfigValidationError)


def check_duplicate_field_names(cfg: dict, registry: dict, messages=None):
    """
    Checks for duplicate field names across the field registry and config-defined schema blocks.
    
    Ensures that all field names used in the Oriented Imagery Dataset schema are unique by checking
    the registry categories ('standard' and 'not_applicable' if enabled) and config-defined blocks
    ('mosaic_fields', 'linear_ref_fields', 'custom_fields'). Logs an error if any duplicates are found.
    """
    seen = set()
    duplicates = set()

    template = cfg.get("oid_schema_template", {})
    esri_defaults = template.get("esri_default", {})

    def add_and_check(name, source):
        """
        Adds a name to the set of seen names and records it as a duplicate if already present.
        
        Args:
            name: The name to check and add.
            source: The source associated with the name (unused in this function).
        """
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    # 1. Check ESRI registry (based on esri_default options)
    for key, field in registry.items():
        category = field.get("category")
        if category in {"standard", "not_applicable"} and esri_defaults.get(category, True):
            fname = field.get("name")
            if fname:
                add_and_check(fname, f"registry.{key}")

    # 2. Check config-defined schema blocks
    for block_key in ["mosaic_fields", "linear_ref_fields", "custom_fields"]:
        block = template.get(block_key, {})
        for key, field in block.items():
            fname = field.get("name")
            if fname:
                add_and_check(fname, f"{block_key}.{key}")

    # Report duplicates
    if duplicates:
        log_message(f"Found {len(duplicates)} duplicate field name(s): {sorted(duplicates)}",
                    messages, level="error", error_type=ConfigValidationError, config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL-SPECIFIC VALIDATORS (Modular)
# ─────────────────────────────────────────────────────────────────────────────


def validate_tool_mosaic_processor(cfg, messages=None):
    """
    Validates the configuration for the mosaic processor executable.
    
    Checks that the "executables.mosaic_processor" section exists and is a dictionary, and that the keys "exe_path",
    "grp_path", and "cfg_path" are present, non-empty strings, and point to existing files.
    """
    mp_cfg = cfg.get("executables", {}).get("mosaic_processor", {})
    validate_type(mp_cfg, "executables.mosaic_processor", dict, messages)

    for key in ["exe_path", "grp_path", "cfg_path"]:
        val = mp_cfg.get(key)
        validate_type(val, f"executables.mosaic_processor.{key}", str, messages)

        if not val.strip():
            log_message(f"executables.mosaic_processor.{key} must not be an empty string", messages,
                        level="error", error_type=ConfigValidationError, config=cfg)

        check_file_exists(val, f"executables.mosaic_processor.{key}", messages)


def validate_tool_build_oid_schema(cfg, messages=None):
    """
    Validates the configuration for building an OID schema template.
    
    Checks the presence and structure of the "oid_schema_template" section, including required keys in the "template"
    block and the existence of the field registry. Validates all fields in the registry and user-defined schema blocks
    for correctness and checks for duplicate field names.
    """
    schema_cfg = cfg.get("oid_schema_template", {})
    template_cfg = schema_cfg.get("template", {})
    esri_cfg = schema_cfg.get("esri_default", {})
    registry_path = esri_cfg.get("field_registry")

    if not registry_path:
        log_message("Missing required key: oid_schema_template.esri_default.field_registry", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    validate_type(template_cfg, "oid_schema_template.template", dict, messages)

    check_required_keys(
        template_cfg,
        ["auto_create_oid_template", "templates_dir", "gdb_path", "template_name"],
        "oid_schema_template.template",
        messages
    )

    validate_type(
        template_cfg.get("auto_create_oid_template"),
        "oid_schema_template.template.auto_create_oid_template",
        bool,
        messages
    )

    # Load and validate registry fields
    registry = load_field_registry(registry_path, config=cfg)
    for key, field in registry.items():
        validate_field_block(field, context=f"registry.{key}", messages=messages)

    # Validate user-defined schema blocks
    for block_key in ["mosaic_fields", "linear_ref_fields", "custom_fields"]:
        block = schema_cfg.get(block_key, {})
        validate_type(block, f"oid_schema_template.{block_key}", dict, messages)
        for key, field in block.items():
            validate_field_block(field, context=f"{block_key}.{key}", messages=messages)

    check_duplicate_field_names(cfg, registry, messages=messages)


def validate_tool_create_oriented_imagery_dataset(cfg, messages=None):
    """
    Validates the configuration for the 'create_oriented_imagery_dataset' tool.
    
    Checks that the 'spatial_ref' section exists and is a dictionary, and that the
    'gcs_horizontal_wkid' and 'vcs_vertical_wkid' keys are present and either integers
    or expressions resolvable to integers.
    """
    validate_config_section(cfg, "spatial_ref", expected_type=dict, messages=messages)

    sr = cfg.get("spatial_ref", {})
    validate_expression_block(
        sr,
        keys=["gcs_horizontal_wkid", "vcs_vertical_wkid"],
        cfg=cfg,
        expected_type=int,
        context="spatial_ref",
        messages=messages
    )


def validate_tool_add_images_to_oid(cfg, messages=None):
    """
    Validates the configuration for adding images to an Oriented Imagery Dataset.
    
    Checks for the presence and correctness of the 'OrientedImageryType' field in the field registry, ensuring its
    'oid_default' value is a valid string and one of the allowed types.
    """
    registry_path = cfg.get("oid_schema_template", {}).get("esri_default", {}).get("field_registry")
    if not registry_path:
        log_message("Missing required key: oid_schema_template.esri_default.field_registry", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    registry = load_field_registry(registry_path, config=cfg)
    field = registry.get("OrientedImageryType")

    if not field:
        log_message("Missing required field: OrientedImageryType", messages, level="error",
                    error_type=ConfigValidationError, config=cfg)

    # Optional: validate full structure
    validate_field_block(field, context="registry.OrientedImageryType", messages=messages)

    default = field.get("oid_default")
    validate_type(default, "OrientedImageryType.oid_default", str, messages)

    if default not in {"360", "Oblique", "Nadir", "Perspective", "Inspection"}:
        log_message("OrientedImageryType.oid_default must be one of: 360, Oblique, Nadir", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)


def validate_tool_assign_group_index(cfg, messages=None):
    """
    Validates the 'grp_idx_fields' section of the configuration for group index assignment.
    
    Checks that 'grp_idx_fields' exists under 'oid_schema_template' and is a dictionary. Validates each field block
    within for correct structure and types.
    """
    grp_idx = cfg.get("oid_schema_template", {}).get("grp_idx_fields", {})
    validate_type(grp_idx, "oid_schema_template.grp_idx_fields", dict, messages)

    for key, field in grp_idx.items():
        validate_field_block(field, context=f"grp_idx_fields.{key}", messages=messages)


def validate_tool_calculate_oid_attributes(cfg, messages=None):
    """
    Validates the configuration for the 'calculate_oid_attributes' tool.
    
    Checks the presence and structure of required OID fields in the field registry, enforces correct types and required
    defaults, and validates the orientation format. Ensures required mosaic and linear reference fields are present and
    valid. Validates the 'camera_offset' section and its sub-blocks for correct types and resolvable expressions.
    """
    template = cfg.get("oid_schema_template", {})
    esri_cfg = template.get("esri_default", {})
    registry_path = esri_cfg.get("field_registry")

    if not registry_path:
        log_message("Missing required key: oid_schema_template.esri_default.field_registry", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    registry = load_field_registry(registry_path, config=cfg)

    # Required OID fields (some must have oid_default)
    required_fields = [
        "CameraPitch", "CameraRoll", "NearDistance", "FarDistance", "CameraHeight", "SRS", "X", "Y", "Z"
    ]

    for key in required_fields:
        field = registry.get(key)
        if not field:
            log_message(f"Missing required field: {key}", messages, level="error", error_type=ConfigValidationError,
                        config=cfg)
        else:
            validate_field_block(field, context=f"registry.{key}", messages=messages)

            if key in ["CameraPitch", "CameraRoll", "NearDistance", "FarDistance"]:
                validate_type(field.get("oid_default"), f"{key}.oid_default", (int, float), messages)

    # Orientation format check
    orientation_type = registry.get("CameraOrientation", {}).get("orientation_format")
    validate_type(orientation_type, "CameraOrientation.orientation_format", str, messages)
    if orientation_type != "type1_short":
        log_message("CameraOrientation.orientation_format must be 'type1_short'", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    # Validate required mosaic_fields
    mosaic = template.get("mosaic_fields", {})
    for key in ["mosaic_reel", "mosaic_frame"]:
        if key not in mosaic:
            log_message(f"Missing required mosaic field: {key}", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        else:
            validate_field_block(mosaic[key], context=f"mosaic_fields.{key}", messages=messages)

    # Validate required linear_ref_fields
    linear = template.get("linear_ref_fields", {})
    for key in ["route_identifier", "route_measure"]:
        if key not in linear:
            log_message(f"Missing required linear_ref field: {key}", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        else:
            validate_field_block(linear[key], context=f"linear_ref_fields.{key}", messages=messages)

    # ✅ Validate camera_offset blocks
    offset_cfg = cfg.get("camera_offset", {})
    validate_type(offset_cfg, "camera_offset", dict, messages)

    for group in ["z", "camera_height"]:
        path = f"camera_offset.{group}"
        validate_config_section(cfg, path, expected_type=dict, messages=messages)

        block = offset_cfg.get(group, {})
        validate_expression_block(block, list(block.keys()), cfg, (int, float), path, messages)


def validate_tool_smooth_gps_noise(cfg, messages=None):
    """
    Validates the 'gps_smoothing' section of the configuration for required keys and value types.
    
    Checks that all required parameters are present in the 'gps_smoothing' section, verifies their types, and ensures
    'angle_bounds_deg' is a list of two numeric values. Logs errors for missing keys, type mismatches, or invalid list
    structure.
    """
    gps = cfg.get("gps_smoothing", {})
    validate_type(gps, "gps_smoothing", dict, messages)

    required_keys = {
        "capture_spacing_meters": (int, float),
        "deviation_threshold_m": (int, float),
        "angle_bounds_deg": list,
        "proximity_check_range_m": (int, float),
        "max_route_dist_deviation_m": (int, float),
        "smoothing_window": int,
        "outlier_reason_threshold": int
    }

    for key, expected_type in required_keys.items():
        val = gps.get(key)
        if val is None:
            log_message(f"gps_smoothing.{key} is required", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        else:
            validate_type(val, f"gps_smoothing.{key}", expected_type, messages)

    angle_bounds = gps.get("angle_bounds_deg")
    if angle_bounds is None:
        return
    if not (isinstance(angle_bounds, list) and len(angle_bounds) == 2):
        log_message("gps_smoothing.angle_bounds_deg must be a list of two values", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)
    else:
        for i, val in enumerate(angle_bounds):
            validate_type(val, f"gps_smoothing.angle_bounds_deg[{i}]", (int, float), messages)


def validate_tool_correct_gps_outliers(cfg, messages=None):
    """
    Validates the configuration for the "correct_gps_outliers" tool.
    
    Checks that the "spatial_ref" section exists and is a dictionary, and that the keys
    "gcs_horizontal_wkid" and "vcs_vertical_wkid" are present and are either integers or
    expressions resolving to integers.
    """
    validate_config_section(cfg, "spatial_ref", expected_type=dict, messages=messages)

    sr = cfg.get("spatial_ref", {})
    validate_expression_block(block=sr, keys=["gcs_horizontal_wkid", "vcs_vertical_wkid"], cfg=cfg,
                              expected_type=int, context="spatial_ref", messages=messages)


def validate_tool_update_linear_and_custom(cfg, messages=None):
    """
    Validates the 'linear_ref_fields' and 'custom_fields' sections of the OID schema template.
    
    Checks that both sections exist and are dictionaries. Validates each field block within these sections for correct
    structure and types. Ensures that the 'route_measure' field in 'linear_ref_fields' has type 'DOUBLE'.
    """
    validate_config_section(cfg, "oid_schema_template.linear_ref_fields", dict, messages)
    validate_config_section(cfg, "oid_schema_template.custom_fields", dict, messages)

    template = cfg.get("oid_schema_template", {})
    linear_ref = template.get("linear_ref_fields", {})
    custom_fields = template.get("custom_fields", {})

    for key, field in linear_ref.items():
        validate_field_block(field, context=f"linear_ref_fields.{key}", messages=messages)
        if key == "route_measure" and field.get("type") != "DOUBLE":
            log_message("oid_schema_template.linear_ref_fields.route_measure.type must be 'DOUBLE'", messages,
                        level="error", error_type=ConfigValidationError, config=cfg)

    for key, field in custom_fields.items():
        validate_field_block(field, context=f"custom_fields.{key}", messages=messages)


def validate_tool_enhance_images(cfg, messages=None):
    """
    Validates the 'image_enhancement' section of the configuration for image enhancement settings.
    
    Checks the presence and types of enhancement flags, output mode and suffix, and validates sub-blocks for white
    balance, contrast enhancement (CLAHE), and sharpening. Ensures required keys and value types are correct, and that
    kernel and method values are within allowed options. Logs errors for any violations.
    """
    section = cfg.get("image_enhancement", {})
    validate_type(section, "image_enhancement", dict, messages)

    # Basic flags
    for key in ["enabled", "adaptive", "apply_white_balance", "apply_contrast_enhancement", "apply_sharpening"]:
        if key in section:
            validate_type(section[key], f"image_enhancement.{key}", bool, messages)

    # Output section
    output = section.get("output", {})
    validate_type(output, "image_enhancement.output", dict, messages)

    mode = output.get("mode")
    if mode not in {"overwrite", "suffix", "directory"}:
        log_message("image_enhancement.output.mode must be 'overwrite', 'suffix', or 'directory'",
                    messages, level="error", error_type=ConfigValidationError, config=cfg)

    if mode == "suffix":
        validate_type(output.get("suffix"), "image_enhancement.output.suffix", str, messages)

    # White balance block
    if section.get("apply_white_balance", False):
        wb = section.get("white_balance", {})
        validate_type(wb, "image_enhancement.white_balance", dict, messages)
        method = wb.get("method")
        if method not in {"gray_world", "simple"}:
            log_message("image_enhancement.white_balance.method must be 'gray_world' or 'simple'",
                        messages, level="error", error_type=ConfigValidationError, config=cfg)

    # CLAHE block
    if section.get("apply_contrast_enhancement", False):
        clahe = section.get("clahe", {})
        validate_type(clahe, "image_enhancement.clahe", dict, messages)
        check_required_keys(clahe, ["clip_limit_low", "clip_limit_high", "contrast_thresholds", "tile_grid_size"],
                            "image_enhancement.clahe", messages)
        validate_type(clahe.get("clip_limit_low"), "clahe.clip_limit_low", (int, float), messages)
        validate_type(clahe.get("clip_limit_high"), "clahe.clip_limit_high", (int, float), messages)
        validate_type(clahe.get("contrast_thresholds"), "clahe.contrast_thresholds", list, messages)
        validate_type(clahe.get("tile_grid_size"), "clahe.tile_grid_size", list, messages)

    # Sharpening kernel
    if section.get("apply_sharpening", False):
        sharp = section.get("sharpen", {})
        validate_type(sharp, "image_enhancement.sharpen", dict, messages)
        kernel = sharp.get("kernel")
        if not (isinstance(kernel, list) and len(kernel) == 3 and all(len(row) == 3 for row in kernel)):
            log_message("image_enhancement.sharpen.kernel must be a 3x3 list", messages,
                        level="error", error_type=ConfigValidationError, config=cfg)


def validate_tool_rename_images(cfg, messages=None):
    """
    Validates the configuration for the image renaming tool's filename settings.
    
    Checks that the filename format string and parts dictionary are present and of correct types, ensures all
    placeholders in the format have corresponding part definitions, warns on unused parts, and validates that each part
    expression resolves to a string.
    """
    validate_config_section(cfg, "image_output.filename_settings", dict, messages)

    settings = cfg.get("image_output", {}).get("filename_settings", {})
    fmt = settings.get("format")
    validate_type(fmt, "image_output.filename_settings.format", str, messages)

    parts = settings.get("parts")
    validate_type(parts, "image_output.filename_settings.parts", dict, messages)

    # ✅ Parse placeholders in format string
    placeholders = {fname for _, fname, _, _ in string.Formatter().parse(fmt) if fname}
    part_keys = set(parts.keys())

    # ✅ Check that all placeholders have matching part definitions
    missing = placeholders - part_keys
    extra = part_keys - placeholders

    if missing:
        log_message(f"Filename format includes undefined placeholder(s): {sorted(missing)}", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    if extra:
        log_message(f"Parts contain unused definitions: {sorted(extra)}", messages,
                    level="warning", config=cfg)

    # ✅ Validate that each part expression (if string) resolves to a string (config or mixed)
    for part_name, expr in parts.items():
        if isinstance(expr, str):
            try_resolve_config_expression(expr, f"filename_settings.parts.{part_name}", cfg, messages,
                                          expected_type=str)


def validate_tool_apply_exif_metadata(cfg, messages=None):
    """
    Validates the configuration for applying EXIF metadata to images.
    
    Checks that all required metadata tags are present and correctly structured as strings or lists of strings, and
    that each expression resolves to a string. Also verifies the presence and existence of the exiftool executable path.
    """
    tags = cfg.get("image_output", {}).get("metadata_tags", {})
    validate_type(tags, "image_output.metadata_tags", dict, messages)

    # ✅ Ensure all required metadata fields are defined
    required_metadata_fields = [
        "Artist", "Copyright", "Software", "Make", "Model", "SerialNumber", "FirmwareVersion", "ImageDescription",
        "XPComment", "XPKeywords"
    ]
    missing_fields = [field for field in required_metadata_fields if field not in tags]
    if missing_fields:
        log_message(f"Missing required metadata_tags fields: {sorted(missing_fields)}", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    # ✅ Validate each field's structure and resolution
    for tag_name, expr in tags.items():
        if isinstance(expr, str):
            try_resolve_config_expression(expr, f"metadata_tags.{tag_name}", cfg, messages, expected_type=str)

        elif isinstance(expr, list):
            for i, item in enumerate(expr):
                validate_type(item, f"metadata_tags.{tag_name}[{i}]", str, messages)
                try_resolve_config_expression(item, f"metadata_tags.{tag_name}[{i}]", cfg, messages, expected_type=str)

        else:
            log_message(f"metadata_tags.{tag_name} must be a string or list of strings", messages,
                        level="error", error_type=ConfigValidationError, config=cfg)

    # ✅ Validate exiftool executable path
    exe_path = cfg.get("executables", {}).get("exiftool", {}).get("exe_path")
    if not exe_path:
        log_message("executables.exiftool.exe_path must be defined", messages, level="error",
                    error_type=ConfigValidationError, config=cfg)
    check_file_exists(exe_path, "executables.exiftool.exe_path", messages)


def validate_tool_geocode_images(cfg, messages=None):
    """
    Validates the geocoding configuration for the image geocoding tool.
    
    Checks that the geocoding method is set to "exiftool" and that the selected database is one of "default",
    "geolocation500", or "geocustom". For "geolocation500" and "geocustom" databases, ensures the corresponding config
    paths are provided and point to existing files. Also verifies that the ExifTool executable path is specified and
    exists.
    """
    geo_cfg = cfg.get("geocoding", {})
    validate_type(geo_cfg, "geocoding", dict, messages)

    method = geo_cfg.get("method", "").lower()
    db = geo_cfg.get("exiftool_geodb", "default").lower()

    # ✅ Validate method is explicitly "exiftool"
    if method != "exiftool":
        log_message("geocoding.method must be 'exiftool'", messages, level="error", error_type=ConfigValidationError,
                    config=cfg)

    # ✅ Validate database selection
    valid_dbs = {"default", "geolocation500", "geocustom"}
    if db not in valid_dbs:
        log_message(f"Unsupported geocoding.exiftool_geodb: {db}. Must be one of: {sorted(valid_dbs)}", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    # ✅ Validate external config paths for optional databases
    if db == "geolocation500":
        path_raw = geo_cfg.get("geoloc500_config_path")
        if not path_raw:
            log_message("Missing geolocation500 config path", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        path = resolve_relative_to_pyt(path_raw)
        check_file_exists(path, "geocoding.geoloc500_config_path", messages)

    if db == "geocustom":
        path_raw = geo_cfg.get("geocustom_config_path")
        if not path_raw:
            log_message("Missing geocustom config path", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        path = resolve_relative_to_pyt(path_raw)
        check_file_exists(path, "geocoding.geocustom_config_path", messages)

    # ✅ Check exiftool executable
    exe_path = cfg.get("executables", {}).get("exiftool", {}).get("exe_path")
    if not exe_path:
        log_message("executables.exiftool.exe_path must be defined", messages, level="error",
                    error_type=ConfigValidationError, config=cfg)
    check_file_exists(exe_path, "executables.exiftool.exe_path", messages)


def validate_tool_build_oid_footprints(cfg, messages=None):
    # ✅ Ensure spatial_ref exists and is a dictionary
    """
    Validates the configuration for the OID footprint building tool.
    
    Checks that the "spatial_ref" section exists and is a dictionary, ensures "pcs_horizontal_wkid" is present and
    resolves to a positive integer, and verifies that "transformation" is a string if defined. Logs errors for missing
    or invalid values.
    """
    validate_config_section(cfg, "spatial_ref", dict, messages)

    sr = cfg.get("spatial_ref", {})

    # ✅ Ensure pcs_horizontal_wkid resolves to a positive integer
    wkid_expr = sr.get("pcs_horizontal_wkid")
    if wkid_expr is None:
        log_message("Missing required key: spatial_ref.pcs_horizontal_wkid", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    resolved_wkid: Optional[int] = try_resolve_config_expression(
        wkid_expr, "spatial_ref.pcs_horizontal_wkid", cfg, messages=messages, expected_type=int)

    if resolved_wkid is not None and resolved_wkid <= 0:
        log_message("spatial_ref.pcs_horizontal_wkid must be a positive integer", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    # ✅ Validate transformation, if defined
    transform = sr.get("transformation")
    if transform is not None and not isinstance(transform, str):
        log_message("spatial_ref.transformation must be a string if defined", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)


def validate_tool_deploy_lambda_monitor(cfg, messages=None):
    # Validate aws.region and aws.lambda_role_arn
    """
    Validates the configuration for the Lambda monitor deployment tool.
    
    Checks that required AWS, image output, and project fields are present and of the correct types. Logs errors for
    missing or incorrectly typed values.
    """
    validate_config_section(cfg, "aws", dict, messages)
    validate_type(cfg["aws"].get("region"), "aws.region", str, messages)
    validate_type(cfg["aws"].get("lambda_role_arn"), "aws.lambda_role_arn", str, messages)

    # Validate image_output.folders.final
    validate_config_section(cfg, "image_output.folders", dict, messages)
    validate_type(cfg["image_output"]["folders"].get("renamed"), "image_output.folders.renamed", str, messages)

    # Validate project.slug and project.number
    validate_config_section(cfg, "project", dict, messages)
    validate_type(cfg["project"].get("slug"), "project.slug", str, messages)
    validate_type(cfg["project"].get("number"), "project.number", str, messages)


def validate_tool_copy_to_aws(cfg: dict, messages=None):
    """
    Validates the AWS configuration section for the copy-to-AWS tool.
    
    Checks for required AWS keys and their types, validates optional keys if present, ensures `max_workers` is an
    integer or a valid expression, and verifies that `s3_bucket_folder` resolves to a string.
    """
    aws = cfg.get("aws", {})

    required_keys = {
        "region": str,
        "s3_bucket": str,
        "s3_bucket_folder": str
    }

    for key, expected_type in required_keys.items():
        val = aws.get(key)
        if val is None:
            log_message(f"aws.{key} is required", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)
        else:
            validate_type(val, f"aws.{key}", expected_type, messages)

    # Optional keys
    optional_keys = {
        "skip_existing": bool,
        "retries": int,
        "keyring_aws": bool,
        "keyring_service_name": str,
        "access_key": str,
        "secret_key": str
    }

    for key, expected_type in optional_keys.items():
        if key in aws and aws[key] is not None:
            validate_type(aws[key], f"aws.{key}", expected_type, messages)

    # Special handling for max_workers to allow string expressions like "cpu*2"
    max_workers = aws.get("max_workers")
    if max_workers is not None:
        if isinstance(max_workers, int):
            pass  # valid
        elif isinstance(max_workers, str):
            if max_workers.lower().startswith("cpu*"):
                # Let copy_to_aws.py handle dynamic CPU resolution
                pass
            else:
                try:
                    result = try_resolve_config_expression(max_workers, "aws.max_workers", cfg, messages)
                    if not isinstance(result, int):
                        log_message("aws.max_workers must resolve to an int", messages, level="error",
                                    error_type=ConfigValidationError, config=cfg)
                except Exception as e:
                    log_message(f"aws.max_workers could not be resolved: {e}", messages, level="error",
                                error_type=ConfigValidationError, config=cfg)
    else:
        log_message("aws.max_workers must be an int or a resolvable string expression", messages,
                    level="error", error_type=ConfigValidationError, config=cfg)

    # Ensure s3_bucket_folder resolves to a string
    try_resolve_config_expression(aws.get("s3_bucket_folder"), "aws.s3_bucket_folder", cfg, messages, expected_type=str)


def validate_tool_generate_oid_service(cfg, messages=None):
    """
    Validates the configuration for the OID service generation tool.
    
    Checks the presence and types of required and optional keys in the "portal" and "aws" sections, including
    resolution of the "s3_bucket_folder" expression to a string.
    """
    validate_config_section(cfg, "portal", dict, messages)
    portal = cfg.get("portal", {})

    validate_type(portal.get("project_folder"), "portal.project_folder", str, messages)

    if "summary" in portal:
        validate_type(portal["summary"], "portal.summary", str, messages)

    if "portal_tags" in portal:
        validate_type(portal["portal_tags"], "portal.portal_tags", list, messages)

    validate_config_section(cfg, "aws", dict, messages)
    aws = cfg.get("aws", {})

    if "s3_bucket_folder" not in aws:
        log_message("Missing required key: aws.s3_bucket_folder", messages, level="error",
                    error_type=ConfigValidationError, config=cfg)

    try_resolve_config_expression(aws.get("s3_bucket_folder"), "aws.s3_bucket_folder", cfg, messages,
                                  expected_type=str)


TOOL_VALIDATORS = {
    "mosaic_processor": validate_tool_mosaic_processor,
    "build_oid_schema": validate_tool_build_oid_schema,
    "create_oriented_imagery_dataset": validate_tool_create_oriented_imagery_dataset,
    "add_images_to_oid": validate_tool_add_images_to_oid,
    "assign_group_index": validate_tool_assign_group_index,
    "calculate_oid_attributes": validate_tool_calculate_oid_attributes,
    "smooth_gps_noise": validate_tool_smooth_gps_noise,
    "correct_gps_outliers": validate_tool_correct_gps_outliers,
    "update_linear_and_custom": validate_tool_update_linear_and_custom,
    "enhance_images": validate_tool_enhance_images,
    "rename_images": validate_tool_rename_images,
    "apply_exif_metadata": validate_tool_apply_exif_metadata,
    "geocode_images": validate_tool_geocode_images,
    "build_oid_footprints": validate_tool_build_oid_footprints,
    "deploy_lambda_monitor": validate_tool_deploy_lambda_monitor,
    "copy_to_aws": validate_tool_copy_to_aws,
    "generate_oid_service": validate_tool_generate_oid_service
}

# ─────────────────────────────────────────────────────────────────────────────
# TOOL VALIDATOR DISPATCH WRAPPER (Optional Utility)
# ─────────────────────────────────────────────────────────────────────────────


def validate_tool_config(cfg: dict, tool: str, messages=None):
    """
    Validates the configuration for a specified tool using its registered validator.
    
    If the tool name is not recognized, logs an error and signals a configuration validation failure.
    
    Args:
        cfg: The full configuration dictionary to validate.
        tool: The name of the tool whose configuration should be validated.
        messages: Optional message collector for logging validation results.
    
    Raises:
        ConfigValidationError: If the tool name is not registered.
    """
    if tool in TOOL_VALIDATORS:
        TOOL_VALIDATORS[tool](cfg, messages)
    else:
        log_message(f"Unknown tool '{tool}'", messages, level="error", error_type=ConfigValidationError, config=cfg)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────


def validate_full_config(cfg: dict, messages=None):
    """
    Validates the entire configuration for the orchestrator workflow.
    
    Checks schema version support, required top-level sections, and key types. Validates spatial reference expressions
    and delegates to all tool-specific validators. Logs errors for any validation failures and reports success if all
    checks pass.
    """
    schema_version = cfg.get("schema_version")
    try:
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            log_message(f"Unsupported schema_version '{schema_version}'. Supported versions: "
                        f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}", messages, level="error",
                        error_type=ConfigValidationError, config=cfg)

        validate_type(cfg.get("debug_messages"), "debug_messages", bool, messages)

        required_sections = [
            "logs", "project", "camera", "camera_offset", "spatial_ref", "executables", "oid_schema_template",
            "gps_smoothing", "image_output.filename_settings", "image_output.metadata_tags", "aws", "portal",
            "geocoding"
        ]

        for section in required_sections:
            validate_config_section(cfg, section, messages=messages)

        validate_expression_block(cfg["spatial_ref"],
                                  ["gcs_horizontal_wkid", "vcs_vertical_wkid", "pcs_horizontal_wkid"],
                                  cfg, int, "spatial_ref", messages)

        # Tool-specific validation
        for tool in TOOL_VALIDATORS:
            TOOL_VALIDATORS[tool](cfg, messages)

    except ConfigValidationError as e:
        log_message(f"[Config Validation] {e}", messages, level="error", error_type=ConfigValidationError, config=cfg)

    log_message("\u2705 Full config validation passed.", messages, config=cfg)


if __name__ == "__main__":
    import sys
    from utils.config_loader import load_config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "../configs/config.yaml"
    config = load_config(config_path)
    validate_full_config(config)
