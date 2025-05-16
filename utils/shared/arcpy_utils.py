# =============================================================================
# 📎 ArcPy Utilities (utils/arcpy_utils.py)
# -----------------------------------------------------------------------------
# Purpose:             Reusable helpers for field validation, value coercion, and OID backup.
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-11
# Last Updated:        2025-05-15
#
# Description:
#   Centralizes common ArcGIS Pro field existence validation, value coercion, and OID backup utilities.
#   Includes helpers for boolean conversion, type casting, and timestamped OID backup creation.
#
# File Location:        /utils/arcpy_utils.py
# Called By:            Multiple tools and utilities throughout the pipeline
# Int. Dependencies:    utils/manager/config_manager, utils/manager/log_manager
# Ext. Dependencies:    arcpy, datetime, pathlib, typing
#
# Documentation:
#   See: docs_legacy/UTILITIES.md and internal docstrings
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - OID backups automatically name and store snapshots per step
#   - Value/type coercion supports spatial reference and boolean handling
# =============================================================================

from __future__ import annotations
import arcpy
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Any


def validate_fields_exist(
    feature_class: str,
    required_fields: List[str],
    logger: Optional["LogManager"] = None,
    arcpy_mod: Any = None
) -> None:
    """
    Checks that all required fields are present in a feature class.

    Args:
        feature_class: The feature class path to validate.
        required_fields: List of field names that must exist.
        logger: Optional logger to use for reporting missing fields.
        arcpy_mod: Optional arcpy module for dependency injection/testing.
    Raises:
        ValueError – if any required field is missing (automatically raised via LogManager if used)
    """
    arcpy_mod = arcpy_mod or arcpy
    existing_fields = {f.name for f in arcpy_mod.ListFields(feature_class)}
    missing = [f for f in required_fields if f not in existing_fields]
    if missing:
        msg = f"Missing required field(s) in {feature_class}: {', '.join(missing)}"
        if logger:
            logger.error(msg, error_type=ValueError, indent=0)
        else:
            raise ValueError(msg)


def str_to_bool(val: Any) -> bool:
    """
    Converts a value to a native Python boolean.
    Accepts boolean values directly. For strings, returns True if the value is one of
    "true", "1", "yes", or "on" (case-insensitive); returns False otherwise.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return False


def str_to_value(
    value: Any,
    value_type: Union[str, type],
    logger: Optional["LogManager"] = None,
    arcpy_mod: Any = None,
    spatial_ref_type: Any = None
) -> Any:
    """
    Converts a value to a specified type, with special handling for spatial references.
    Args:
        value: The input value to convert.
        value_type: A type (e.g., int, float) or the string 'spatial_reference'.
        logger: Optional LogManager instance for debug/error output.
        arcpy_mod: Optional arcpy module for dependency injection/testing.
        spatial_ref_type:
    Returns:
        The converted value, or None if the input is invalid or conversion fails.
    """
    if value is None:
        return None
    arcpy_mod = arcpy_mod or arcpy
    if value_type == "spatial_reference":
        sr_type = spatial_ref_type or getattr(arcpy_mod, "SpatialReference", type(None))
        if isinstance(value, sr_type):
            return value
        # Safely determine which exceptions to catch
        execute_error = getattr(arcpy_mod, "ExecuteError", None)
        exception_types = (ValueError, TypeError)
        if isinstance(execute_error, type) and issubclass(execute_error, BaseException):
            exception_types += (execute_error,)
        try:
            return arcpy_mod.SpatialReference(value)
        except exception_types as e:
            if logger:
                logger.debug(f"Failed to convert '{value}' to spatial_reference: {e}", indent=1)
            return None
    try:
        return value_type(value)
    except (ValueError, TypeError) as e:
        if logger:
            logger.debug(f"Failed to convert '{value}' to {value_type}: {e}", indent=1)
        return None


def backup_oid(
    oid_fc: str,
    step_key: str,
    cfg: "ConfigManager",
    arcpy_mod: Any = None,
    path_mod: Any = None,
    datetime_mod: Any = None,
    logger: Optional[Any] = None
) -> None:
    """
    Creates a timestamped backup of the Oriented Imagery Dataset feature class before a processing step.
    Args:
        oid_fc: Input feature class path.
        step_key: Step name for labeling the snapshot.
        cfg: Loaded configuration manager.
        arcpy_mod: Optional arcpy module for dependency injection/testing.
        path_mod: Optional Path module for dependency injection/testing.
        datetime_mod: Optional datetime module for dependency injection/testing.
        logger: Optional logger for dependency injection/testing.
    """
    arcpy_mod = arcpy_mod or arcpy
    path_mod = path_mod or Path
    datetime_mod = datetime_mod or datetime
    logger = logger or cfg.get_logger()
    try:
        gdb_path = cfg.paths.backup_gdb
        gdb_path.parent.mkdir(parents=True, exist_ok=True)
        if not gdb_path.exists():
            arcpy_mod.management.CreateFileGDB(str(gdb_path.parent), gdb_path.name)
        oid_desc = arcpy_mod.Describe(oid_fc)
        oid_name = path_mod(oid_desc.name).stem
        timestamp = datetime_mod.now().strftime("%Y%m%d_%H%M")
        out_fc_name = f"{oid_name}_before_{step_key}_{timestamp}"
        out_fc_path = str(gdb_path / out_fc_name)
        logger.custom(f"Backing up OID before step '{step_key}' → {out_fc_name}", indent=1, emoji="💾")
        arcpy_mod.management.Copy(oid_fc, out_fc_path)
    except Exception as e:
        logger.warning(f"OID backup before step '{step_key}' failed: {e}", indent=1)
