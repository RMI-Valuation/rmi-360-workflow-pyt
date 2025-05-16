# =============================================================================
# ☁️ Copy to AWS Validator (utils/validators/copy_to_aws_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates AWS configuration for copying data to S3
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures presence and correctness of required AWS keys, validates types, and checks max_workers and S3 folder
#   configuration for compatibility with S3 upload workflows.
#
# File Location:        /utils/validators/copy_to_aws_validator.py
# Called By:            AWS S3 upload and sync tools
# Notes:                Used for validation of AWS credentials and S3 upload settings.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.validators.common_validators import (
    try_resolve_config_expression,
    validate_keys_with_types
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the AWS configuration section for the copy-to-AWS tool.

    Checks for required AWS keys and their types, validates optional keys if present, ensures `max_workers` is an
    integer or a valid expression, and verifies that `s3_bucket_folder` resolves to a string.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    aws = cfg.get("aws", {})

    # ✅ Required keys
    required_keys = {
        "region": str,
        "s3_bucket": str,
        "s3_bucket_folder": str
    }

    error_count += validate_keys_with_types(cfg, aws, required_keys, "aws", required=True)

    # ✅ Optional keys
    optional_keys = {
        "skip_existing": bool,
        "retries": int,
        "keyring_aws": bool,
        "keyring_service_name": str,
        "access_key": str,
        "secret_key": str
    }

    error_count += validate_keys_with_types(cfg, aws, optional_keys, "aws", required=False)

    # ✅ max_workers logic: allow int or resolvable expression
    max_workers = aws.get("max_workers")
    if max_workers is None:
        logger.error("aws.max_workers must be defined", error_type=ConfigValidationError)
        error_count += 1
    elif isinstance(max_workers, int):
        pass  # OK
    elif isinstance(max_workers, str) and max_workers.lower().startswith("cpu*"):
        pass  # OK
    else:
        resolved = try_resolve_config_expression(max_workers, "aws.max_workers", cfg, expected_type=int)
        if resolved is None:
            error_count += 1

    # ✅ Ensure s3_bucket_folder resolves to a string
    folder_expr = aws.get("s3_bucket_folder")
    if not try_resolve_config_expression(folder_expr, "aws.s3_bucket_folder", cfg, expected_type=str):
        error_count += 1

    return error_count == 0
