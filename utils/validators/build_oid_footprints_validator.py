# =============================================================================
# 🗺️ Build OID Footprints Validator (utils/validators/build_oid_footprints_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for building OID footprints
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures the presence and correctness of the spatial reference block, validates WKID and transformation fields,
#   and checks that all required configuration is present for OID footprint generation.
#
# File Location:        /utils/validators/build_oid_footprints_validator.py
# Called By:            OID footprint building workflows
# Notes:                Used for validation of spatial reference and transformation settings in OID tools.
# =============================================================================

from typing import Optional

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_config_section,
    try_resolve_config_expression
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    # ✅ Ensure spatial_ref exists and is a dictionary
    """
    Validates the configuration for the OID footprint building tool.

    Checks that the "spatial_ref" section exists and is a dictionary, ensures "pcs_horizontal_wkid" is present and
    resolves to a positive integer, and verifies that "transformation" is a string if defined. Logs errors for missing
    or invalid values.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    if not validate_config_section(cfg, "spatial_ref", dict):
        error_count += 1

    # ✅ Ensure pcs_horizontal_wkid resolves to a positive integer
    wkid_expr = cfg.get("spatial_ref.pcs_horizontal_wkid")
    if wkid_expr is None:
        logger.error("Missing required key: spatial_ref.pcs_horizontal_wkid", error_type=ConfigValidationError)
        error_count += 1

    resolved_wkid: Optional[int] = try_resolve_config_expression(
        wkid_expr, "spatial_ref.pcs_horizontal_wkid", cfg, expected_type=int)

    if resolved_wkid is None:
        error_count += 1
    elif resolved_wkid <= 0:
        logger.error("spatial_ref.pcs_horizontal_wkid must be a positive integer", error_type=ConfigValidationError)
        error_count += 1

    # ✅ Validate transformation, if defined
    transform = cfg.get("spatial_ref.transformation")
    if transform is not None and not isinstance(transform, str):
        logger.error("spatial_ref.transformation must be a string if defined", error_type=ConfigValidationError)
        error_count += 1

    return error_count == 0
