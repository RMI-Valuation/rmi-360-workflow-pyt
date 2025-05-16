# =============================================================================
# 🏷️ Rename Images Validator (utils/validators/rename_images_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for image renaming tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures filename format string and parts dictionary are present and correct, validates placeholders, and checks
#   that all expressions resolve to strings for renaming images.
#
# File Location:        /utils/validators/rename_images_validator.py
# Called By:            Image renaming workflows
# Notes:                Used for validation of filename formats and dynamic parts for output images.
# =============================================================================
import string

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_config_section,
    validate_type,
    try_resolve_config_expression
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the image renaming tool's filename settings.

    Checks that the filename format string and parts dictionary are present and of correct types, ensures all
    placeholders in the format have corresponding part definitions, warns on unused parts, and validates that each part
    expression resolves to a string.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    if not validate_config_section(cfg, "image_output.filename_settings", dict):
        error_count += 1

    fmt = cfg.get("image_output.filename_settings.format")
    fmt_no_lr = cfg.get("image_output.filename_settings.format_no_lr")
    parts = cfg.get("image_output.filename_settings.parts")

    # Validate both format and format_no_lr
    if fmt is None and fmt_no_lr is None:
        logger.error("At least one of 'format' or 'format_no_lr' must be defined in image_output.filename_settings.", error_type=ConfigValidationError)
        error_count += 1
    if fmt is not None and not validate_type(fmt, "image_output.filename_settings.format", str, cfg):
        error_count += 1
    if fmt_no_lr is not None and not validate_type(fmt_no_lr, "image_output.filename_settings.format_no_lr", str, cfg):
        error_count += 1
    if parts is None or not validate_type(parts, "image_output.filename_settings.parts", dict, cfg):
        error_count += 1
        parts = {}  # Prevent further errors

    # Validate placeholders for both formats
    for fmt_str, label in [(fmt, "format"), (fmt_no_lr, "format_no_lr")]:
        if fmt_str is None:
            logger.warning(f"'image_output.filename_settings.{label}' is not defined.")
            continue
        placeholders = {fname for _, fname, _, _ in string.Formatter().parse(fmt_str) if fname}
        part_keys = set(parts.keys())
        missing = placeholders - part_keys
        extra = part_keys - placeholders
        if missing:
            logger.error(f"Filename {label} includes undefined placeholder(s): {sorted(missing)}", error_type=ConfigValidationError)
            error_count += 1
        if extra and label == "format":
            logger.warning(f"Parts contain unused definitions: {sorted(extra)}")
        if not placeholders:
            logger.error(f"Filename {label} must include at least one placeholder (e.g. '{{segment_id}}')", error_type=ConfigValidationError)
            error_count += 1

    # ✅ Validate that each part expression (if string) resolves to a string (config or mixed)
    for part_name, expr in parts.items():
        if isinstance(expr, str):
            if not try_resolve_config_expression(expr, f"filename_settings.parts.{part_name}", cfg,
                                                 expected_type=str):
                error_count += 1

    return error_count == 0
