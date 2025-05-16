# =============================================================================
# 🌍 Geocode Images Validator (utils/validators/geocode_images_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for image geocoding tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures geocoding method and database selection are valid, checks for required config paths and ExifTool
#   executable, and validates all geocoding-related configuration for image processing.
#
# File Location:        /utils/validators/geocode_images_validator.py
# Called By:            Image geocoding workflows
# Notes:                Used for validation of geocoding method, database, and ExifTool path.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    check_file_exists
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the geocoding configuration for the image geocoding tool.

    Checks that the geocoding method is set to "exiftool" and that the selected database is one of "default",
    "geolocation500", or "geocustom". For "geolocation500" and "geocustom" databases, ensures the corresponding config
    paths are provided and point to existing files. Also verifies that the ExifTool executable path is specified and
    exists.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    geo_cfg = cfg.get("geocoding", {})
    if not validate_type(geo_cfg, "geocoding", dict, cfg):
        error_count += 1

    method = cfg.get("geocoding.method", "").lower()
    db = cfg.get("geocoding.exiftool_geodb", "default").lower()

    # ✅ Validate method is explicitly "exiftool"
    if method != "exiftool":
        logger.error("geocoding.method must be 'exiftool'", error_type=ConfigValidationError)
        error_count += 1

    # ✅ Validate database selection
    valid_dbs = {"default", "geolocation500", "geocustom"}
    if db not in valid_dbs:
        logger.error(f"Unsupported geocoding.exiftool_geodb: {db}. Must be one of: {sorted(valid_dbs)}",
                     error_type=ConfigValidationError)
        error_count += 1

    # ✅ Validate external config paths for optional databases
    if db == "geolocation500":
        path = cfg.paths.geoloc500_config_path
        if not path:
            logger.error("Missing geolocation500 config path", error_type=ConfigValidationError)
            error_count += 1
        if not check_file_exists(path, "geocoding.geoloc500_config_path", cfg):
            error_count += 1

    if db == "geocustom":
        path = cfg.paths.geocustom_config_path
        if not path:
            logger.error("Missing geocustom config path", error_type=ConfigValidationError)
            error_count += 1
        if not check_file_exists(path, "geocoding.geocustom_config_path", cfg):
            error_count += 1

    # ✅ Validate exiftool executable path
    exe_path = cfg.paths.exiftool_exe
    if not cfg.paths.check_exiftool_available():
        logger.error(f"ExifTool not available or not found at path: {exe_path}", error_type=ConfigValidationError)
        error_count += 1

    return error_count == 0