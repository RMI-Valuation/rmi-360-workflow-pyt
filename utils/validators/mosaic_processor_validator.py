# =============================================================================
# 🧩 Mosaic Processor Validator (utils/validators/mosaic_processor_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for the Mosaic Processor tool
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of mosaic processor executable, GRP paths, and config path for mosaic workflows.
#
# File Location:        /utils/validators/mosaic_processor_validator.py
# Called By:            Mosaic processor workflows
# Notes:                Used for validation of mosaic processor settings and executable paths.
# =============================================================================

from utils.validators.common_validators import validate_type
from utils.shared.rmi_exceptions import ConfigValidationError


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the Mosaic Processor tool.

    Uses PathManager to verify executable and GRP paths, and validates type and presence of cfg_path separately.
    """
    logger = cfg.get_logger()
    error_count = 0

    mp_cfg = cfg.get("executables.mosaic_processor", {})
    if not validate_type(mp_cfg, "executables.mosaic_processor", dict, cfg):
        error_count += 1

    # Validate cfg_path presence and type
    cfg_path = mp_cfg.get("cfg_path")
    if not validate_type(cfg_path, "executables.mosaic_processor.cfg_path", str, cfg):
        error_count += 1
    elif not cfg_path.strip():
        logger.error("executables.mosaic_processor.cfg_path must not be an empty string",
                     error_type=ConfigValidationError)
        error_count += 1

    # Use PathManager's built-in executable and GRP validation
    if not cfg.paths.validate_mosaic_config():
        logger.error("Mosaic Processor configuration is invalid.", error_type=ConfigValidationError)
        error_count += 1

    if not cfg.paths.check_mosaic_processor_available():
        error_count += 1

    return error_count == 0
