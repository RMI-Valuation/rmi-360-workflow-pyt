# =============================================================================
# 🧮 Calculate OID Attributes Validator (utils/validators/calculate_oid_attributes_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for calculating OID attribute fields
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of required OID fields, validates types and defaults, and checks mosaic and
#   linear reference fields for attribute calculation tools.
#
# File Location:        /utils/validators/calculate_oid_attributes_validator.py
# Called By:            OID attribute calculation workflows
# Notes:                Used for validation of attribute calculation settings and field requirements.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.shared.expression_utils import load_field_registry
from utils.validators.common_validators import (
    validate_field_block,
    validate_type,
    validate_config_section,
    validate_expression_block
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the 'calculate_oid_attributes' tool.

    Checks the presence and structure of required OID fields in the field registry, enforces correct types and required
    defaults, and validates the orientation format. Ensures required mosaic and linear reference fields are present and
    valid. Validates the 'camera_offset' section and its sub-blocks for correct types and resolvable expressions.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    template = cfg.get("oid_schema_template", {})

    try:
        registry = load_field_registry(cfg)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load field registry: {e}", error_type=ConfigValidationError)
        return False

    # Required OID fields (some must have oid_default)
    required_fields = [
        "CameraPitch", "CameraRoll", "NearDistance", "FarDistance", "CameraHeight", "SRS", "X", "Y", "Z"
    ]

    for key in required_fields:
        field = registry.get(key)
        if not field:
            logger.error(f"Missing required field: {key}", error_type=ConfigValidationError)
            error_count += 1
        else:
            if not validate_field_block(field, cfg, context=f"registry.{key}"):
                error_count += 1

            if key in ["CameraPitch", "CameraRoll", "NearDistance", "FarDistance"]:
                if not validate_type(field.get("oid_default"), f"{key}.oid_default", (int, float), cfg):
                    error_count += 1

    # Orientation format check
    orientation_type = registry.get("CameraOrientation", {}).get("orientation_format")
    if validate_type(orientation_type, "CameraOrientation.orientation_format", str, cfg):
        if orientation_type != "type1_short":
            logger.error("CameraOrientation.orientation_format must be 'type1_short'", error_type=ConfigValidationError)
            error_count += 1
    else:
        error_count += 1

    # Validate required mosaic_fields
    mosaic = template.get("mosaic_fields", {})
    for key in ["mosaic_reel", "mosaic_frame"]:
        if key not in mosaic:
            logger.error(f"Missing required mosaic field: {key}", error_type=ConfigValidationError)
            error_count += 1
        else:
            if not validate_field_block(mosaic[key], cfg, context=f"mosaic_fields.{key}"):
                error_count += 1

    # Validate required linear_ref_fields
    linear = template.get("linear_ref_fields", {})
    for key in ["route_identifier", "route_measure"]:
        if key not in linear:
            logger.error(f"Missing required linear_ref field: {key}", error_type=ConfigValidationError)
            error_count += 1
        else:
            if not validate_field_block(linear[key], cfg, context=f"linear_ref_fields.{key}"):
                error_count += 1

    # ✅ Validate camera_offset blocks
    offset_cfg = cfg.get("camera_offset", {})
    if not validate_type(offset_cfg, "camera_offset", dict, cfg):
        error_count += 1

    for group in ["z", "camera_height"]:
        path = f"camera_offset.{group}"
        if not validate_config_section(cfg, path, expected_type=dict):
            error_count += 1
        else:
            block = offset_cfg.get(group, {})
            if not validate_expression_block(block, list(block.keys()), cfg, (int, float), path):
                error_count += 1

    return error_count == 0