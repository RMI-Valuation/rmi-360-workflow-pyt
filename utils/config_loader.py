# =============================================================================
# ⚙️ Configuration Loader & Resolver (utils/config_loader.py)
# -----------------------------------------------------------------------------
# Purpose:             Loads, validates, and resolves dynamic configuration for the workflow
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Loads and validates the project configuration from YAML. Caches results and enforces
#   schema versioning. Includes expression resolution, project root inference, and
#   camera offset calculation from config fields.
#
# File Location:        /utils/config_loader.py
# Called By:            Nearly all tools and orchestration utilities
# Int. Dependencies:    validate_config, arcpy_utils
# Ext. Dependencies:    os, yaml, typing
#
# Documentation:
#   See: docs/UTILITIES.md and docs/config_schema_reference.md
#
# Notes:
#   - Automatically infers __project_root__ from OID if not explicitly set
#   - Enforces top-level schema version compatibility and deprecation warnings
# =============================================================================

__all__ = ["load_config", "prepare_config", "load_and_validate_config", "get_camera_offset_values",
           "get_default_config_path", "resolve_config"]

import os
import yaml
from typing import Any, Optional, NoReturn
from utils.validate_config import validate_full_config, validate_tool_config
from utils.arcpy_utils import log_message, infer_project_root_from_oid

EXPECTED_SCHEMA_VERSION = "1.0.1"  # Update this when schema changes significantly
CONFIG_CACHE: Optional[dict[str, Any]] = None


def _safe_float(value, default=0.0) -> float:
    """
    Safely converts a value to a float, returning a default on failure.
    
    Args:
        value: The value to convert to float.
        default: The value to return if conversion fails (defaults to 0.0).
    
    Returns:
        The converted float value, or the default if conversion is not possible.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_default_config_path(messages=None) -> str | NoReturn:
    """
    Returns the absolute path to the default configuration file.
    
    Checks for 'configs/config.yaml' first, then 'configs/config.sample.yaml'. If neither file exists, logs an error
    and raises FileNotFoundError.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    primary = os.path.join(base_dir, "config.yaml")
    fallback = os.path.join(base_dir, "config.sample.yaml")

    if os.path.isfile(primary):
        return primary
    elif os.path.isfile(fallback):
        return fallback
    else:
        msg = "No config.yaml or config.sample.yaml found in /configs"
        log_message(msg, messages, level="error", error_type=FileNotFoundError)
        raise FileNotFoundError(msg)


def load_config(config_path: Optional[str] = None, messages=None, debug: bool = False) -> dict[str, Any]:
    """
    Loads and parses the YAML configuration file, enforcing schema version and caching.
    
    If no path is provided, attempts to load the default config file. Validates that the loaded config is a dictionary,
    checks for schema version compatibility, and warns about deprecated keys. Returns an empty dictionary on failure.
    """
    global CONFIG_CACHE
    if CONFIG_CACHE is not None and config_path is None:
        return CONFIG_CACHE

    config_path = config_path or get_default_config_path(messages=messages)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG_CACHE = yaml.safe_load(f) or {}
            CONFIG_CACHE["__source__"] = os.path.abspath(config_path)

        if not isinstance(CONFIG_CACHE, dict):
            log_message("Config file did not parse into a dictionary structure.", messages, level="error",
                        error_type=ValueError, config=CONFIG_CACHE)
            raise ValueError("Config is not a dictionary")

        version = CONFIG_CACHE.get("schema_version")
        if version != EXPECTED_SCHEMA_VERSION:
            log_message(f"⚠️ Expected schema_version {EXPECTED_SCHEMA_VERSION}, got {version}", messages,
                        level="error",error_type=RuntimeError, config=CONFIG_CACHE)

        if "camera_calculations" in CONFIG_CACHE:
            log_message("⚠️ Warning: 'camera_calculations' is deprecated. Use 'camera_offset' structure instead.",
                        messages, level="warning", config=CONFIG_CACHE)

        if debug:
            log_message(f"[DEBUG] Loaded config from {CONFIG_CACHE['__source__']}", messages, level="debug",
                        config=CONFIG_CACHE)
            log_message(f"[DEBUG] Top-level config keys: {list(CONFIG_CACHE.keys())}", messages, level="debug",
                        config=CONFIG_CACHE)

        return CONFIG_CACHE

    except Exception as e:
        CONFIG_CACHE = None
        log_message(f"Failed to load config from {config_path}: {e}", messages, level="error", error_type=RuntimeError,
                    config={})
        raise


def prepare_config(
    config_file: Optional[str],
    project_folder: Optional[str] = None,
    oid_fc_path: Optional[str] = None,
    messages=None
) -> dict[str, Any]:
    """
    Loads the configuration file and sets the '__project_root__' key in the config dictionary.
    
    The project root is determined by the following priority: an explicit 'project_folder' argument, an existing
    '__project_root__' key in the config, or by inferring from 'oid_fc_path' if provided. Returns the config dictionary
    with '__project_root__' set accordingly.
    """
    config_file = config_file or get_default_config_path(messages)
    config = load_config(config_file, messages=messages)

    if project_folder:
        config["__project_root__"] = project_folder
        log_message(f"[DEBUG] Using explicit project_folder as __project_root__: {project_folder}", messages,
                    level="debug", config=config)
    elif "__project_root__" not in config and oid_fc_path:
        try:
            inferred_root = infer_project_root_from_oid(oid_fc_path, config, messages)
            config["__project_root__"] = inferred_root
            log_message(f"[DEBUG] Inferred __project_root__ from OID: {inferred_root}", messages, level="debug",
                        config=config)
        except Exception as e:
            log_message(f"⚠️ Could not infer project root from OID: {e}", messages, level="warning", config=config)

    return config


def load_and_validate_config(
    config: dict[str, Any],
    tool_name: str = None,
    messages=None,
) -> dict[str, Any]:
    """
    Validates a configuration dictionary, optionally for a specific tool.
    
    If a tool name is provided, performs tool-specific validation; otherwise, validates the entire configuration.
    Assumes the configuration has already been loaded and prepared, including resolution of the `__project_root__` key.
    
    Args:
        config: The configuration dictionary to validate.
        tool_name: Optional name of the tool for tool-specific validation.
        messages: Optional messages dictionary to use for validation messages.
    
    Returns:
        The validated configuration dictionary.
    """
    if tool_name:
        validate_tool_config(config, tool_name, messages=messages)
    else:
        validate_full_config(config, messages=messages)

    return config


def get_camera_offset_values(config: Optional[dict[str, Any]] = None, messages=None) -> tuple[float, float]:
    """
    Calculates the vertical offset and total camera height in meters from the config.
    
    If the config is not provided, it is loaded automatically. Returns zeros if the
    required fields are missing or invalid.
    
    Returns:
        A tuple containing (z_offset_meters, camera_height_meters).
    """
    if config is None:
        config = load_config()

    try:
        z_cm = sum(_safe_float(v) for v in config["camera_offset"]["z"].values())
        height_cm = sum(_safe_float(v) for v in config["camera_offset"]["camera_height"].values())
    except KeyError as e:
        log_message(f"Missing camera_offset key: {e}", messages, level="error", error_type=ValueError, config=config)
        raise
    except Exception as e:
        log_message(f"Failed to compute camera offset or height: {e}", level="error", error_type=ValueError,
                    config=config)
        raise

    return z_cm / 100.0, height_cm / 100.0


def resolve_config(
    config: Optional[dict] = None,
    config_file: Optional[str] = None,
    project_folder: Optional[str] = None,
    oid_fc_path: Optional[str] = None,
    messages=None,
    tool_name: Optional[str] = None,
) -> dict:
    """
    Resolves and validates a configuration dictionary for use in the RMI Mosaic 360 Tools.
    
    If a config dictionary is not provided, loads and prepares one using the specified file and project context, then
    validates it. Returns the fully prepared and validated configuration dictionary.
    """
    if config is None:
        config = prepare_config(
            config_file=config_file,
            project_folder=project_folder,
            oid_fc_path=oid_fc_path,
            messages=messages
        )
    return load_and_validate_config(config, tool_name=tool_name, messages=messages)
