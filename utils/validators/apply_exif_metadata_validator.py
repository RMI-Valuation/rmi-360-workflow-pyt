# =============================================================================
# 🏷️ Apply EXIF Metadata Validator (utils/validators/apply_exif_metadata_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for applying EXIF metadata to images
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Checks required EXIF metadata tags, validates their structure and resolvable expressions, and ensures
#   the ExifTool executable path is present and valid for image processing.
#
# File Location:        /utils/validators/apply_exif_metadata_validator.py
# Called By:            EXIF metadata application workflows
# Notes:                Ensures all required EXIF tags and tool paths are set for downstream processing.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    try_resolve_config_expression
)

def validate(cfg: "ConfigManager") -> bool:
    """
    Validates the configuration for applying EXIF metadata to images.

    Checks that all required metadata tags are present and correctly structured as strings or lists of strings, and
    that each expression resolves to a string. Also verifies the presence and existence of the exiftool executable path.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    tags = cfg.get("image_output.metadata_tags", {})
    if not validate_type(tags, "image_output.metadata_tags", dict, cfg):
        error_count += 1

    # ✅ Ensure all required metadata fields are defined
    required_metadata_fields = [
        "Artist", "Copyright", "Software", "Make", "Model", "SerialNumber", "FirmwareVersion", "ImageDescription",
        "XPComment", "XPKeywords"
    ]
    missing_fields = [field for field in required_metadata_fields if field not in tags]
    if missing_fields:
        logger.error(f"Missing required metadata_tags fields: {sorted(missing_fields)}",
                     error_type=ConfigValidationError)
        error_count += 1

    extra_tags = set(tags.keys()) - set(required_metadata_fields)
    if extra_tags:
        logger.info(f"Found extra metadata_tags fields not in required list: {sorted(extra_tags)}")

    def validate_tag_block(tag_block, prefix="metadata_tags"):
        nonlocal error_count
        for tag_name, expr in tag_block.items():
            full_name = f"{prefix}.{tag_name}"
            if isinstance(expr, (str, int, float)):
                try:
                    resolved_value = try_resolve_config_expression(expr, full_name, cfg)
                    _ = str(resolved_value)
                except Exception as e:
                    logger.error(f"{full_name}: failed to resolve or stringify value: {e}", error_type=ConfigValidationError)
                    error_count += 1
            elif isinstance(expr, list):
                for i, item in enumerate(expr):
                    try:
                        resolved_value = try_resolve_config_expression(item, f"{full_name}[{i}]", cfg)
                        _ = str(resolved_value)
                    except Exception as e:
                        logger.error(f"{full_name}[{i}]: failed to resolve or stringify value: {e}", error_type=ConfigValidationError)
                        error_count += 1
            elif isinstance(expr, dict):
                # Nested dict (e.g., GPano block)
                validate_tag_block(expr, prefix=full_name)
            else:
                logger.error(f"{full_name} must be a string, number, list, or dict",
                             error_type=ConfigValidationError)
                error_count += 1

    # Validate all tags, including nested
    validate_tag_block(tags)

    # ✅ Validate exiftool executable path
    exe_path = cfg.paths.exiftool_exe
    if not cfg.paths.check_exiftool_available():
        logger.error(f"ExifTool not available or not found at path: {exe_path}", error_type=ConfigValidationError)
        error_count += 1

    return error_count == 0