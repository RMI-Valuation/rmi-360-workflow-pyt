# =============================================================================
# 🖼️ Enhance Images Validator (utils/validators/enhance_images_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for image enhancement tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of the 'image_enhancement' configuration section and its required keys/types.
#
# File Location:        /utils/validators/enhance_images_validator.py
# Called By:            Image enhancement workflows
# Notes:                Used for validation of enhancement parameters for image processing.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    check_required_keys
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the 'image_enhancement' section of the configuration for image enhancement settings.

    Checks the presence and types of enhancement flags, output mode and suffix, and validates sub-blocks for white
    balance, contrast enhancement (CLAHE), and sharpening. Ensures required keys and value types are correct, and that
    kernel and method values are within allowed options. Logs errors for any violations.

    Returns:
        bool: True if no errors were logged, False if any validation failed.
    """
    logger = cfg.get_logger()
    error_count = 0

    section = cfg.get("image_enhancement", {})
    if not validate_type(section, "image_enhancement", dict, cfg):
        error_count += 1

    # Basic flags
    for key in ["enabled", "adaptive", "apply_white_balance", "apply_contrast_enhancement", "apply_sharpening"]:
        if key in section:
            if not validate_type(section[key], f"image_enhancement.{key}", bool, cfg):
                error_count += 1

    # Output section
    output = section.get("output", {})
    if not validate_type(output, "image_enhancement.output", dict, cfg):
        error_count += 1

    mode = output.get("mode")
    if mode not in {"overwrite", "suffix", "directory"}:
        logger.error("image_enhancement.output.mode must be 'overwrite', 'suffix', or 'directory'",
                     error_type=ConfigValidationError)
        error_count += 1

    if mode == "suffix":
        if not validate_type(output.get("suffix"), "image_enhancement.output.suffix", str, cfg):
            error_count += 1

    # White balance block
    if section.get("apply_white_balance", False):
        wb = section.get("white_balance", {})
        if not validate_type(wb, "image_enhancement.white_balance", dict, cfg):
            error_count += 1
        method = wb.get("method")
        if method not in {"gray_world", "simple"}:
            logger.error("image_enhancement.white_balance.method must be 'gray_world' or 'simple'",
                         error_type=ConfigValidationError)
            error_count += 1

    # CLAHE block
    if section.get("apply_contrast_enhancement", False):
        clahe = section.get("clahe", {})
        if not validate_type(clahe, "image_enhancement.clahe", dict, cfg):
            error_count += 1
        if not check_required_keys(clahe, ["clip_limit_low", "clip_limit_high", "contrast_thresholds",
                                           "tile_grid_size"], "image_enhancement.clahe", cfg):
            error_count += 1
        if not validate_type(clahe.get("clip_limit_low"), "clahe.clip_limit_low", (int, float), cfg):
            error_count += 1
        if not validate_type(clahe.get("clip_limit_high"), "clahe.clip_limit_high", (int, float), cfg):
            error_count += 1
        if not validate_type(clahe.get("contrast_thresholds"), "clahe.contrast_thresholds", list, cfg):
            error_count += 1
        if not validate_type(clahe.get("tile_grid_size"), "clahe.tile_grid_size", list, cfg):
            error_count += 1

    # Sharpening kernel
    if section.get("apply_sharpening", False):
        sharp = section.get("sharpen", {})
        if not validate_type(sharp, "image_enhancement.sharpen", dict, cfg):
            error_count += 1
        kernel = sharp.get("kernel")
        if not (isinstance(kernel, list) and len(kernel) == 3 and all(len(row) == 3 for row in kernel)):
            logger.error("image_enhancement.sharpen.kernel must be a 3x3 list", error_type=ConfigValidationError)
            error_count += 1

    return error_count == 0