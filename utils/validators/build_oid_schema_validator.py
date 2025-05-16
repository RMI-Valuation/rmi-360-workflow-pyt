# =============================================================================
# 🏗️ Build OID Schema Validator (utils/validators/build_oid_schema_validator.py)
# -----------------------------------------------------------------------------
# Purpose:             Validates configuration for building OID schema templates
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Checks the presence and structure of the 'oid_schema_template' block, validates required keys, field registry,
#   and ensures all fields and blocks are correct for OID schema generation.
#
# File Location:        /utils/validators/build_oid_schema_validator.py
# Called By:            OID schema building workflows
# Notes:                Used for validation of schema blocks, templates, and field registries in OID tools.
# =============================================================================

from utils.shared.rmi_exceptions import ConfigValidationError
from utils.shared.expression_utils import load_field_registry
from utils.validators.common_validators import (
    validate_type,
    check_required_keys,
    validate_field_block,
    check_duplicate_field_names
)


def validate(cfg: "ConfigManager") -> bool:
    from utils.manager.config_manager import ConfigManager
    """
    Validates the configuration for building an OID schema template.

    Checks the presence and structure of the "oid_schema_template" section, including required keys in the "template"
    block and the existence of the field registry. Validates all fields in the registry and user-defined schema blocks
    for correctness and checks for duplicate field names.

    Returns:
        bool: True if config is valid, False otherwise.
    """
    logger = cfg.get_logger()
    error_count = 0

    schema_cfg = cfg.get("oid_schema_template", {})
    template_cfg = cfg.get("oid_schema_template.template", {})

    if not validate_type(template_cfg, "oid_schema_template.template", dict, cfg):
        error_count += 1

    if not check_required_keys(
        template_cfg,
        ["auto_create_oid_template", "templates_dir", "gdb_path", "template_name"],
        "oid_schema_template.template", cfg):
        error_count += 1

    if not validate_type(
        template_cfg.get("auto_create_oid_template"), "oid_schema_template.template.auto_create_oid_template",
        bool, cfg):
        error_count += 1

    esri_cfg = cfg.get("oid_schema_template.esri_default", {})
    active_categories = [cat for cat in ("standard", "not_applicable") if esri_cfg.get(cat, True)]

    # 1. Validate individual fields
    combined_registry = {}
    for category in active_categories:
        try:
            entries = load_field_registry(cfg, category_filter=category)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load field registry ({category}): {e}", error_type=ConfigValidationError)
            return False
        for key, field in entries.items():
            if not validate_field_block(field, cfg, context=f"registry.{key}"):
                error_count += 1
        combined_registry.update(entries)

    # Validate user-defined schema blocks
    for block_key in ["mosaic_fields", "grp_idx_fields", "linear_ref_fields", "custom_fields"]:
        block = schema_cfg.get(block_key, {})
        if not validate_type(block, f"oid_schema_template.{block_key}", dict, cfg):
            error_count += 1
        for key, field in block.items():
            if not validate_field_block(field, cfg, context=f"{block_key}.{key}"):
                error_count += 1

    duplicates = check_duplicate_field_names(cfg, combined_registry)
    if duplicates:
        error_count += 1

    return error_count == 0
