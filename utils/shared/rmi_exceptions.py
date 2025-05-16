# =============================================================================
# 🚨 Custom Exception Definitions (utils/rmi_exceptions.py)
# -----------------------------------------------------------------------------
# Purpose:             Defines custom exception classes for error handling in the RMI 360 workflow
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Contains project-specific exception classes, such as ConfigValidationError, to standardize error handling
#   and improve clarity in exception reporting across the codebase.
#
# File Location:        /utils/rmi_exceptions.py
# Called By:            utils/validate_full_config.py, other validation and config modules
# Notes:                Extend this file with additional exceptions as needed for future features.
# =============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass