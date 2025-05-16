# =============================================================================
# 📍 Correct GPS Outliers Validator (utils/validators/correct_gps_outliers_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for GPS outlier correction tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of spatial reference keys, validates WKID values and types for GPS outlier
#   correction workflows.
#
# File Location:        /utils/validators/correct_gps_outliers_validator.py
# Called By:            GPS outlier correction tools
# Notes:                Used for validation of spatial reference and WKID settings in GPS correction.
# =============================================================================

from utils.validators.common_validators import (
    validate_config_section,
    validate_expression_block
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the "correct_gps_outliers" tool.

    Checks that the "spatial_ref" section exists and is a dictionary, and that the keys
    "gcs_horizontal_wkid" and "vcs_vertical_wkid" are present and are either integers or
    expressions resolving to integers.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    error_count = 0

    if not validate_config_section(cfg, "spatial_ref", expected_type=dict):
        error_count += 1

    sr = cfg.get("spatial_ref", {})
    if not validate_expression_block(block=sr, keys=["gcs_horizontal_wkid", "vcs_vertical_wkid"], cfg=cfg,
                              expected_type=int, context="spatial_ref"):
        error_count += 1

    return error_count == 0