# =============================================================================
# 🧠 Expression & Field Resolver (utils/expression_utils.py)
# -----------------------------------------------------------------------------
# Purpose:             Resolves dynamic expressions from config and row data into final values
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Provides utility functions to resolve expressions defined in YAML config or field registry.
#   Supports nested dot-path lookups, type conversion, formatting modifiers, concatenation,
#   and special keywords like `now.year`. Also loads and validates OID field registry schemas.
#
# File Location:        /utils/expression_utils.py
# Called By:            validate_full_config.py, most workflow steps
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    os, yaml, datetime, typing
#
# Documentation:
#   See: docs_legacy/UTILITIES.md and docs_legacy/config_schema_reference.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Supports modifiers: strip(), float(), int, date(), upper, lower
#   - Resolves both row-based and config-based expressions recursively
# =============================================================================

from __future__ import annotations
import os
import yaml
from datetime import datetime
from typing import Union, Optional, Any


REQUIRED_REGISTRY_KEYS = {"name", "type", "length", "alias", "category", "expr", "oid_default", "orientation_format"}


def load_field_registry(cfg: ConfigManager, category_filter: Optional[str] = None) -> dict:
    from utils.manager.config_manager import ConfigManager    
    """
    Loads and validates a field registry from a YAML file.

    The registry is checked for required keys in each field entry and optionally filtered by category. Raises an error
    if the file is missing, cannot be parsed as a dictionary, or if required keys are absent.

    Args:
        cfg: ConfigManager instance used to resolve config-based expressions.
        category_filter: If provided, only fields matching this category are included.

    Returns:
        A dictionary of validated field entries from the registry.

    Raises:
        FileNotFoundError: If the registry file does not exist.
        ValueError: If the file cannot be parsed as a dictionary or required keys are missing.
    """
    paths = cfg.paths
    logger = cfg.get_logger()

    registry_path = paths.oid_field_registry

    if not os.path.exists(registry_path):
        logger.error(f"Field registry file does not exist: {registry_path}", error_type=FileNotFoundError)

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    if not isinstance(registry, dict):
        logger.error(f"Field registry did not parse as a dictionary: {registry_path}", error_type=ValueError)

    validated = {}
    for key, field in registry.items():
        if not isinstance(field, dict):
            logger.error(f"Field '{key}' must be a dictionary", error_type=ValueError)

        missing = REQUIRED_REGISTRY_KEYS - set(field.keys())
        if missing:
            # expr and oid_default are optional, but we validate they exist for uniformity
            for opt in ["expr", "oid_default", "orientation_format"]:
                missing.discard(opt)
            if missing:
                logger.error(f"Field '{key}' missing required keys: {sorted(missing)}", error_type=ValueError)

        if category_filter and field.get("category") != category_filter:
            continue

        validated[key] = field

    return validated


def resolve_expression(expr: Union[str, float, int], cfg: ConfigManager, row: Optional[dict] = None) -> Any:
    from utils.manager.config_manager import ConfigManager    
    """
    Resolves an expression string using values from a configuration and optional data row.
    
    Supports field lookups, config value retrieval, concatenation of sub-expressions, quoted literals, and the special
    "now.year" expression for the current year. Returns the resolved value as a string or the original value if no
    resolution is performed.
    
    Args:
        expr: The expression to evaluate, which may be a string, float, or integer.
        cfg: ConfigManager instance used to resolve config-based expressions.
        row: Optional dictionary representing a data row, used to resolve field-based expressions (e.g., {FieldName}).
    
    Returns:
        The resolved value as a string, or the literal value if no resolution was required.
    """
    if not isinstance(expr, str):
        return str(expr)

    if " + " in expr:
        parts = [resolve_expression(p.strip(), row=row, cfg=cfg) for p in expr.split("+")]
        return "".join(str(p) for p in parts)

    if expr.startswith("field.") and row is not None:
        return _resolve_field_expr(expr[6:], row)

    if expr.startswith("config."):
        return _resolve_config_expr(expr[7:], cfg)

    if expr == "now.year":
        return str(datetime.now().year)

    if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
        return expr[1:-1]

    return expr


def _resolve_field_expr(expr: str, row: dict) -> str:
    """
    Resolves a field expression from a row dictionary, applying optional modifiers.
    
    Supports modifiers such as float formatting, integer conversion, date formatting, character stripping, and case
    transformations. Returns the final formatted string value.
    """
    base, *mods = expr.split(".")
    value = row.get(base)

    if value is None:
        return ""

    for mod in mods:
        if mod.startswith("float("):
            precision = int(mod[6:-1])
            try:
                value = float(value)
                value = f"{value:.{precision}f}"
            except (ValueError, TypeError):
                value = str(value)
        elif mod == "int":
            value = int(float(value))
        elif mod.startswith("date("):
            fmt = mod[5:-1]
            if isinstance(value, datetime):
                value = value.strftime(fmt)
        elif mod.startswith("strip("):
            char = mod[6:-1]
            value = str(value).replace(char, "")
        elif mod == "upper":
            value = str(value).upper()
        elif mod == "lower":
            value = str(value).lower()

    return value


def _resolve_config_expr(expr: str, cfg: ConfigManager) -> Any:
    from utils.manager.config_manager import ConfigManager    
    """
    Resolves a value from a ConfigManager using dot-separated key paths with optional modifiers.

    Supports modifiers such as stripping characters, date formatting, float precision, integer conversion,
    and case transformations.

    Args:
        expr: Dot-separated expression (e.g., "project.name.strip(-)").
        cfg: ConfigManager instance providing access to config, paths, and logger.

    Returns:
        The resolved configuration value, preserving its native type (e.g., int, float, str, bool, dict, etc.),
        unless a string modifier is applied, in which case a string is returned.

    Raises:
        KeyError: If the base config key cannot be resolved.
    """
    logger = cfg.get_logger()

    if expr == "now.year":
        return str(datetime.now().year)

    try:
        # Separate key path and modifiers
        parts = expr.split(".")
        base_parts = []
        mods = []
        for p in parts:
            if _is_modifier(p):
                mods.append(p)
            else:
                base_parts.append(p)

        # Resolve config value using ConfigManager
        key_path  = ".".join(base_parts)
        value = cfg.get(key_path)

        if value is None:
            raise KeyError(f"Config key '{key_path}' returned None")

    except (KeyError, AttributeError, TypeError) as e:
        logger.error(f"Error resolving config expression: '{expr}': {e}", error_type=KeyError)
        return ""


    # Apply modifiers
    for mod in mods:
        try:
            if mod.startswith("strip("):
                char = mod[6:-1]
                value = str(value).replace(char, "")
            elif mod.startswith("date("):
                fmt = mod[5:-1]
                if isinstance(value, datetime):
                    value = value.strftime(fmt)
            elif mod.startswith("float("):
                precision = int(mod[6:-1])
                value = round(float(value), precision)
            elif mod == "int":
                value = int(float(value))
            elif mod == "upper":
                value = str(value).upper()
            elif mod == "lower":
                value = str(value).lower()
        except ValueError:
            logger.error(f"Modifier '{mod}' failed on config value: {value}", error_type=ValueError)
            return ""

    return value


def _is_modifier(mod: str) -> bool:
    """
    Determines if a string represents a recognized expression modifier.
    
    Returns:
        True if the input string is a supported modifier (e.g., strip, date, float, int, upper, lower); otherwise, False.
    """
    return (
        mod.startswith("strip(") or
        mod.startswith("date(") or
        mod.startswith("float(") or
        mod in {"int", "upper", "lower"}
    )
