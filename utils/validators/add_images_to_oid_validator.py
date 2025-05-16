# =============================================================================
# 🛂 Add Images to OID Validator (utils/validators/add_images_to_oid_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for adding images to an Oriented Imagery Dataset (OID)
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures the presence and correctness of the 'OrientedImageryType' field in the registry, validates field types,
#   and checks allowed image type values for adding images to an OID.
#
# File Location:        /utils/validators/add_images_to_oid_validator.py
# Called By:            OID creation and update workflows
# Notes:                Used for schema and value validation when ingesting new images into an OID.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.shared.expression_utils import load_field_registry
from utils.validators.common_validators import (
    validate_field_block,
    validate_type, check_file_exists
)


def validate(cfg: "ConfigManager") -> bool:
    """
    Validates the configuration for adding images to an Oriented Imagery Dataset.

    Checks for the presence and correctness of the 'OrientedImageryType' field in the field registry, ensuring its
    'oid_default' value is a valid string and one of the allowed types.

    Returns:
        bool: True if config is valid, False if issues are found.
    """
    logger = cfg.get_logger()
    error_count = 0

    registry_path = cfg.paths.oid_field_registry
    context = "oid_schema_template.esri_default.field_registry"
    if not check_file_exists(registry_path, context, cfg):
        logger.error("OID field registry .yaml file not found.")
        return False

    try:
        registry = load_field_registry(cfg)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load field registry: {e}", error_type=ConfigValidationError)
        return False

    field = registry.get("OrientedImageryType")
    if not field:
        logger.error("Missing required field: OrientedImageryType", error_type=ConfigValidationError)
        return False

    if not validate_field_block(field, cfg, context="registry.OrientedImageryType"):
        error_count += 1

    VALID_IMAGE_TYPES = {"360", "Oblique", "Nadir", "Perspective", "Inspection"}

    default = field.get("oid_default")
    if not validate_type(default, "OrientedImageryType.oid_default", str, cfg):
        error_count += 1
    elif default not in VALID_IMAGE_TYPES:
        logger.error(f"OrientedImageryType.oid_default must be one of: {', '.join(sorted(VALID_IMAGE_TYPES))}",
                     error_type=ConfigValidationError)
        error_count += 1

    return error_count == 0