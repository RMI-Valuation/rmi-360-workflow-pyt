# =============================================================================
# 🏷️ Assign Group Index Validator (utils/validators/assign_group_index_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for assigning group index fields in OID schema
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Ensures the 'grp_idx_fields' block exists and is valid, and that each group index field is correctly structured
#   for use in OID schema generation and workflows.
#
# File Location:        /utils/validators/assign_group_index_validator.py
# Called By:            OID schema and group assignment workflows
# Notes:                Used for validation of group index fields in OID creation and updates.
# =============================================================================

from utils.validators.common_validators import (
    validate_type,
    validate_field_block
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the 'grp_idx_fields' section of the configuration for group index assignment.

    Checks that 'grp_idx_fields' exists under 'oid_schema_template' and is a dictionary. Validates each field block
    within for correct structure and types.

    Returns:
        bool: True if all validations pass, False otherwise.
    """
    error_count = 0

    grp_idx = cfg.get("oid_schema_template.grp_idx_fields", {})
    if not validate_type(grp_idx, "oid_schema_template.grp_idx_fields", dict, cfg):
        error_count += 1

    if isinstance(grp_idx, dict):
        for key, field in grp_idx.items():
            if not validate_field_block(field, cfg, context=f"grp_idx_fields.{key}"):
                error_count += 1

    return error_count == 0