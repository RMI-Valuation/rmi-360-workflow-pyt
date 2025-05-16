# =============================================================================
# 🛰️ Deploy Lambda Monitor Validator (utils/validators/deploy_lambda_monitor_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for deploying AWS Lambda monitoring tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of AWS, image output, and project keys for Lambda monitoring deployment.
#
# File Location:        /utils/validators/deploy_lambda_monitor_validator.py
# Called By:            Lambda monitor deployment workflows
# Notes:                Used for validation of AWS and project settings in Lambda monitoring tools.
# =============================================================================

from utils.validators.common_validators import (
    validate_config_section,
    validate_type
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for the Lambda monitor deployment tool.

    Checks that required AWS, image output, and project fields are present and of the correct types. Logs errors for
    missing or incorrectly typed values.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    error_count = 0

    if not validate_config_section(cfg, "aws", dict):
        error_count += 1
    if not validate_type(cfg.get("aws.region"), "aws.region", str, cfg):
        error_count += 1
    if not validate_type(cfg.get("aws.lambda_role_arn"), "aws.lambda_role_arn", str, cfg):
        error_count += 1

    # Validate image_output.folders.final
    if not validate_config_section(cfg, "image_output.folders", dict):
        error_count += 1
    if not validate_type(cfg.get("image_output.folders.renamed"), "image_output.folders.renamed", str, cfg):
        error_count += 1

    # Validate project.slug and project.number
    if not validate_config_section(cfg, "project", dict):
        error_count += 1
    if not validate_type(cfg.get("project.slug"), "project.slug", str, cfg):
        error_count += 1
    if not validate_type(cfg.get("project.number"), "project.number", str, cfg):
        error_count += 1

    return error_count == 0