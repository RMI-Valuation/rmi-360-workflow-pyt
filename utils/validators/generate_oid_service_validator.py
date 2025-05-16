
# =============================================================================
# 🆔 Generate OID Service Validator (utils/validators/generate_oid_service_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for generating OID services (portal and AWS)
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of required portal and AWS configuration sections and keys, including validation
#   of S3 bucket folder expressions and portal project folder settings.
#
# File Location:        /utils/validators/generate_oid_service_validator.py
# Called By:            OID service generation workflows
# Notes:                Used for validation of portal and AWS configuration for OID service deployment.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    validate_type,
    try_resolve_config_expression,
    validate_config_section
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the OID service generation tool.

    Checks the presence and types of required and optional keys in the "portal" and "aws" sections, including
    resolution of the "s3_bucket_folder" expression to a string.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    if not validate_config_section(cfg, "portal", dict):
        error_count += 1

    portal = cfg.get("portal", {})
    if not validate_type(portal.get("project_folder"), "portal.project_folder", str, cfg):
        error_count += 1

    if "summary" in portal:
        if not validate_type(portal["summary"], "portal.summary", str, cfg):
            error_count += 1

    if "portal_tags" in portal:
        if not validate_type(portal["portal_tags"], "portal.portal_tags", list, cfg):
            error_count += 1
        else:
            for i, tag in enumerate(portal["portal_tags"]):
                if not validate_type(tag, f"portal.portal_tags[{i}]", str, cfg):
                    error_count += 1

    if not validate_config_section(cfg, "aws", dict):
        error_count += 1

    aws = cfg.get("aws", {})
    if "s3_bucket_folder" not in aws:
        logger.error("Missing required key: aws.s3_bucket_folder", error_type=ConfigValidationError)
        error_count += 1

    if not try_resolve_config_expression(aws.get("s3_bucket_folder"), "aws.s3_bucket_folder", cfg,
                                         expected_type=str):
        error_count += 1

    return error_count == 0