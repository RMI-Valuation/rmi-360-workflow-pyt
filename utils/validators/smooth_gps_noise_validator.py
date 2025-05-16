
# =============================================================================
# 🛰️ Smooth GPS Noise Validator (utils/validators/smooth_gps_noise_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for GPS noise smoothing tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of the 'gps_smoothing' configuration section, validates required keys and types,
#   and checks angle bounds structure for GPS smoothing tools.
#
# File Location:        /utils/validators/smooth_gps_noise_validator.py
# Called By:            GPS smoothing workflows
# Notes:                Used for validation of smoothing and outlier detection settings for GPS data.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    validate_keys_with_types
)

def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the 'gps_smoothing' section of the configuration for required keys and value types.

    Checks that all required parameters are present in the 'gps_smoothing' section, verifies their types, and ensures
    'angle_bounds_deg' is a list of two numeric values. Logs errors for missing keys, type mismatches, or invalid list
    structure.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    gps = cfg.get("gps_smoothing", {})
    if not validate_type(gps, "gps_smoothing", dict, cfg):
        error_count += 1

    required_keys = {
        "capture_spacing_meters": (int, float),
        "deviation_threshold_m": (int, float),
        "angle_bounds_deg": list,
        "proximity_check_range_m": (int, float),
        "max_route_dist_deviation_m": (int, float),
        "smoothing_window": int,
        "outlier_reason_threshold": int
    }

    error_count += validate_keys_with_types(cfg, gps, required_keys, "gps_smoothing", required=True)

    angle_bounds = gps.get("angle_bounds_deg")
    if angle_bounds is None:
        if not (isinstance(angle_bounds, list) and len(angle_bounds) == 2):
            logger.error("gps_smoothing.angle_bounds_deg must be a list of two values",
                         error_type=ConfigValidationError)
            error_count += 1
        else:
            for i, val in enumerate(angle_bounds):
                if not validate_type(val, f"gps_smoothing.angle_bounds_deg[{i}]", (int, float),
                                     cfg):
                    error_count += 1

    return error_count == 0