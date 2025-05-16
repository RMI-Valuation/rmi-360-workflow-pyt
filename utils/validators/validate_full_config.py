# =============================================================================
# ✅ Config Validator & Tool Schema Enforcer (utils/validate_full_config.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates the entire YAML configuration against expected structure, types, and tool-specific rules
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Performs comprehensive validation of the pipeline’s configuration, including schema version checks,
#   required section enforcement, and dynamic expression resolution. Registers tool-specific validators
#   for all supported workflow steps (e.g., enhance_images, copy_to_aws, deploy_lambda_monitor).
#   Raises `ConfigValidationError` for invalid structures, missing fields, or improper expressions.
#
# File Location:        /utils/validate_full_config.py
# Called By:            orchestrator, CLI validation entrypoint
# Int. Dependencies:    arcpy_utils, expression_utils, path_resolver, config_loader, schema_paths
# Ext. Dependencies:    __future__,shutil, string, typing, pathlib
#
# Documentation:
#   See: docs_legacy/CONFIG_GUIDE.md and docs_legacy/config_schema_reference.md
#
# Notes:
#   - Uses per-tool validator dispatch via `TOOL_VALIDATORS` registry
#   - Supports `DISABLED` flags and external command validation (e.g., exiftool path)
#   - Designed to be run both via `__main__` and programmatically from tool execution
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.manager.config_manager import ConfigManager

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    validate_config_section,
    validate_expression_block
)


def validate_full_config(cfg: 'ConfigManager', logger=None) -> bool:
    from utils.manager.config_manager import SUPPORTED_SCHEMA_VERSIONS
    """
    Validates the entire configuration for the orchestrator workflow.

    Checks schema version support, required top-level sections, and key types. Validates spatial reference expressions
    and delegates to all tool-specific validators. Collects and logs all validation errors, reporting success only if
    all checks pass.

    Args:
        cfg (ConfigManager): The configuration manager instance.
        logger: Optional logger for testing; defaults to cfg.get_logger().

    Returns:
        True if validation passes, False otherwise.
    """
    logger = logger or cfg.get_logger()
    errors = []

    schema_version = cfg.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        msg = (f"Unsupported schema_version '{schema_version}'. Supported versions: "
               f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}")
        logger.error(msg, error_type=ConfigValidationError)
        errors.append(msg)

    try:
        validate_type(cfg.get("debug_messages"), "debug_messages", bool, cfg)
    except ConfigValidationError as e:
        logger.error(str(e), error_type=ConfigValidationError)
        errors.append(str(e))

    required_sections = [
        "logs", "project", "camera", "camera_offset", "spatial_ref", "executables", "oid_schema_template",
        "gps_smoothing", "image_output.filename_settings", "image_output.metadata_tags", "aws", "portal",
        "geocoding"
    ]
    for section in required_sections:
        try:
            validate_config_section(cfg, section, dict)
        except ConfigValidationError as e:
            logger.error(str(e), error_type=ConfigValidationError)
            errors.append(str(e))

    try:
        spatial_ref = cfg.get("spatial_ref")
        validate_expression_block(
            spatial_ref,
            ["gcs_horizontal_wkid", "vcs_vertical_wkid", "pcs_horizontal_wkid"],
            cfg, int, "spatial_ref"
        )
    except ConfigValidationError as e:
        logger.error(str(e), error_type=ConfigValidationError)
        errors.append(str(e))

    # Tool-specific validation
    for tool, validator in getattr(cfg, 'TOOL_VALIDATORS', {}).items():
        try:
            validator(cfg)
            logger.success(f"[{tool} validation] passed.", indent=1)
        except ConfigValidationError as e:
            logger.error(f"[{tool} validation] {e}", error_type=ConfigValidationError, indent=1)
            errors.append(f"{tool}: {e}")

    if not errors:
        logger.success("Full config validation passed.", indent=1)
        return True
    else:
        logger.error(f"Config validation failed with {len(errors)} error(s).", error_type=ConfigValidationError, indent=1)
        return False
