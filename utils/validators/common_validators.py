# =============================================================================
# 🛠️ Common Validators (utils/validators/common_validators.py)
# -----------------------------------------------------------------------------
# Purpose:             Provides reusable validation functions for configuration and schema checks
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Contains utility functions for type checking, config section validation, field validation, expression resolution,
#   duplicate detection, and file existence checks. Used by all validator scripts in the toolbox.
#
# File Location:        /utils/validators/common_validators.py
# Called By:            All validation scripts in utils/validators
# Notes:                Centralizes validation logic for consistency and maintainability.
# =============================================================================

from __future__ import annotations
import shutil
from typing import Union, Tuple, Type, Literal, get_args
from pathlib import Path

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.shared.expression_utils import resolve_expression


def validate_type(value, context: str, expected_type: Union[Type, Tuple[Type, ...]], cfg: ConfigManager):
    """
    Checks whether a value matches the expected Python type(s) and logs an error if not.

    Args:
        value: The value to validate.
        context: Identifier for the value's location in the config, used in error messages.
        expected_type: The required type or tuple of types for the value.
        cfg (ConfigManager): The configuration manager providing logger access.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()

    if not isinstance(value, expected_type):
        actual_type = type(value).__name__

        # Check if expected_type is a tuple, otherwise treat it as a single type
        if isinstance(expected_type, tuple):
            expected = ", ".join(t.__name__ for t in expected_type)
        else:
            expected = expected_type.__name__  # Just the name of the type if it's not a tuple

        logger.error(f"{context} must be of type {expected}, but got {actual_type}",
                     error_type=ConfigValidationError)

        return False
    return True


def validate_expression_block(block: dict, keys: list[str], cfg: ConfigManager,
                              expected_type: Union[Type, Tuple[Type, ...]], context: str):
    """
    Validates that specified keys in a dictionary are either literals of the expected type or resolvable config
    expressions.

    For each key in `keys`, checks that the value in `block` is either an instance of `expected_type` or a string
    expression that resolves to `expected_type` using the provided config. Field expressions (strings starting with
    "field.") are ignored. Logs errors for missing keys, type mismatches, or failed expression resolution.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger =cfg.get_logger()
    error_count = 0

    for key in keys:
        full_key = f"{context}.{key}"
        val = block.get(key)

        if val is None:
            logger.error(f"{full_key} is required", error_type=ConfigValidationError)
            error_count += 1

        elif isinstance(val, expected_type):
            continue

        elif isinstance(val, str):
            try_resolve_config_expression(val, full_key, cfg, expected_type=expected_type)

        else:
            logger.error(f"{full_key} must be a {expected_type.__name__} or a resolvable config expression",
                         error_type=ConfigValidationError)

    return error_count == 0


def check_required_keys(d, keys, context, cfg: ConfigManager):
    """
    Ensures all specified keys are present in a dictionary, logging an error for each missing key.

    Logs a detailed error message for each required key not found in the dictionary, using the provided context for
    clarity.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger =cfg.get_logger()

    for key in keys:
        if key not in d:
            logger.error(f"{context}.{key} is required", error_type=ConfigValidationError)
    return True


def validate_config_section(cfg: ConfigManager, path: str, expected_type=dict):
    """
    Validates that a nested section exists in the configuration and matches the expected type.

    Checks for the presence of a dot-separated key path within the config dictionary and verifies that the final value
    is of the specified type. Logs errors if the section is missing or of the wrong type.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()

    if not path:
        logger.error("Empty config path provided", error_type=ConfigValidationError)
        return False

    keys = path.split(".")
    current = cfg.raw
    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            logger.error(f"{'.'.join(keys[:i])} must be a dictionary", error_type=ConfigValidationError)
            return False
        if key not in current:
            logger.error(f"Missing required config section: '{path}'", error_type=ConfigValidationError)
            return False
        current = current[key]

    if not isinstance(current, expected_type):
        logger.error(f"Config section '{path}' must be of type {expected_type.__name__}",
                     error_type=ConfigValidationError)
        return False

    return True



def try_resolve_config_expression(expr: str, context: str, cfg: ConfigManager, expected_type=None):
    """
    Attempts to resolve a configuration expression and optionally checks its type.

    If the expression starts with "field.", resolution is skipped. If resolution fails or the resolved value does not
    match the expected type (if provided), an error is logged and None is returned.

    Args:
        expr: The configuration expression to resolve.
        context: Description of where the expression is located, used in error messages.
        cfg:
        expected_type: Optional type or tuple of types; if provided, the resolved value must match this type.

    Returns:
        The resolved value if successful and of the expected type, or None if resolution fails or is skipped.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()

    if expr is None or not isinstance(expr, str):
        return None

    if expr.startswith("field."):
        return None  # Skip resolution of field expressions

    try:
        result = resolve_expression(expr, cfg=cfg)
        if expected_type and not isinstance(result, expected_type):
            logger.error(f"{context}: resolved value must be of type {expected_type.__name__}, "
                         f"got {type(result).__name__}", error_type=ConfigValidationError)
        return result
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        logger.error(f"{context}: failed to resolve expression '{expr}': {e}", error_type=ConfigValidationError)
        return None


EsriFieldType = Literal[
    "SHORT", "LONG", "BIGINTEGER", "FLOAT", "DOUBLE", "TEXT",
    "DATE", "DATEHIGHPRECISION", "DATEONLY", "TIMEONLY", "TIMESTAMPOFFSET",
    "BLOB", "GUID", "RASTER"
]

VALID_FIELD_TYPES = set(get_args(EsriFieldType))  # auto-derived at runtime


def validate_field_block(field_block: dict, cfg: ConfigManager, context: str = "field"):
    """
    Validates a single field definition block for required keys and correct value types.

    Ensures that the field block contains valid 'name' and 'type' entries, and that optional keys such as 'length',
    'alias', 'expression', and 'oid_default' are present only with appropriate types and usage. Logs errors for
    missing or invalid entries.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()
    error_count = 0

    for key in ["name", "type"]:
        if key not in field_block:
            logger.error(f"{context}: Missing required key '{key}'", error_type=ConfigValidationError)
            error_count += 1

    name = field_block.get("name")
    ftype = field_block.get("type")

    if not isinstance(name, str):
        logger.error(f"{context}: 'name' must be a string", error_type=ConfigValidationError)
        error_count += 1

    if ftype not in VALID_FIELD_TYPES:
        logger.error(f"{context}: Invalid field type '{ftype}'. Must be one of: "
                     f"{', '.join(sorted(VALID_FIELD_TYPES))}", error_type=ConfigValidationError)
        error_count += 1

    if "length" in field_block:
        if ftype == "TEXT" and not isinstance(field_block["length"], int):
            logger.error(f"{context}: 'length' must be an integer for TEXT fields",
                         error_type=ConfigValidationError)
            error_count += 1
        elif ftype != "TEXT" and field_block["length"] is not None:
            logger.error(f"{context}: 'length' should be omitted or null for non-TEXT fields",
                         error_type=ConfigValidationError)
            error_count += 1

    if "alias" in field_block and not isinstance(field_block["alias"], str):
        logger.error(f"{context}: 'alias' must be a string if provided", error_type=ConfigValidationError)
        error_count += 1

    if "expression" in field_block and not isinstance(field_block["expression"], str):
        logger.error(f"{context}: 'expression' must be a string if provided", error_type=ConfigValidationError)
        error_count += 1

    if "oid_default" in field_block and not isinstance(field_block["oid_default"], (int, float, str)):
        logger.error(f"{context}: 'oid_default' must be a string, int, or float if provided",
                     error_type=ConfigValidationError)
        error_count += 1

    return error_count == 0


def check_file_exists(path, context, cfg: ConfigManager):
    """
    Validates that a given executable or file path is either:
    - A valid file on disk (absolute or relative)
    - A valid command on the system PATH
    - Or marked 'DISABLED'

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()

    if path == "DISABLED":
        return True

    path_obj = Path(path)

    # Check absolute or relative to script base
    candidates = [path_obj]
    if not path_obj.is_absolute():
        candidates.append(cfg.paths.script_base / path_obj)

    for candidate in candidates:
        if candidate.is_file():
            return True

    # Check if it's a valid command
    if shutil.which(path):
        return True

    # Otherwise raise error
    logger.error(f"{context} does not point to a valid file or executable (or use 'DISABLED'): {path}",
                 error_type=ConfigValidationError)
    return False


def check_duplicate_field_names(cfg: ConfigManager, registry: dict):
    """
    Checks for duplicate field names across the field registry and config-defined schema blocks.

    Ensures that all field names used in the Oriented Imagery Dataset schema are unique by checking
    the registry categories ('standard' and 'not_applicable' if enabled) and config-defined blocks
    ('mosaic_fields', 'linear_ref_fields', 'custom_fields'). Logs an error if any duplicates are found.

    Returns:
        bool: True if validation passed, False otherwise.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()

    seen = set()
    duplicates = set()

    esri_defaults = cfg.get("oid_schema_template.esri_default", {})

    def add_and_check(field_name, source):
        """
        Adds a name to the set of seen names and records it as a duplicate if already present.

        Args:
            field_name: The name to check and add.
            source: The source associated with the name (unused in this function).
        """
        if field_name in seen:
            duplicates.add(field_name)
        seen.add(field_name)

    # 1. Check ESRI registry (based on esri_default options)
    for key, field in registry.items():
        category = field.get("category")
        if category in {"standard", "not_applicable"} and esri_defaults.get(category, True):
            fname = field.get("name")
            if fname:
                add_and_check(fname, f"registry.{key}")

    # 2. Check config-defined schema blocks
    for block_key in ["mosaic_fields", "linear_ref_fields", "custom_fields"]:
        block = cfg.get(f"oid_schema_template.{block_key}", {})
        for key, field in block.items():
            fname = field.get("name")
            if fname:
                add_and_check(fname, f"{block_key}.{key}")

    # Report duplicates
    if duplicates:
        if cfg.get("debug_messages", False):
            for name in duplicates:
                logger.debug(f"Duplicate field: {name}")
        logger.error(f"Found {len(duplicates)} duplicate field name(s): {sorted(duplicates)}",
                     error_type=ConfigValidationError)

    return duplicates


def validate_keys_with_types(
    cfg: ConfigManager,
    section: dict,
    keymap: dict,
    context_prefix: str,
    required: bool = True,
) -> int:
    """
    Validates keys in a config section for type correctness, and optionally for presence.

    Args:
        cfg (ConfigManager): ConfigManager instance.
        section (dict): Config subsection to check (e.g. cfg.get(\"aws\")).
        keymap (dict): Mapping of keys to expected types.
        context_prefix (str): String prefix for logging context.
        required (bool): If True, missing keys are errors. If False, keys are only validated if present.

    Returns:
        int: Number of validation errors encountered.
    """
    from utils.manager.config_manager import ConfigManager
    logger = cfg.get_logger()
    error_count = 0

    for key, expected_type in keymap.items():
        context = f"{context_prefix}.{key}"
        val = section.get(key)

        if val is None:
            if required:
                logger.error(f"{context} is required", error_type=ConfigValidationError)
                error_count += 1
        else:
            if not validate_type(val, context, expected_type, cfg):
                error_count += 1

    return error_count

