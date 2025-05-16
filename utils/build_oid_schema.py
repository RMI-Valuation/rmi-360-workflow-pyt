# =============================================================================
# 🧬 OID Schema Template Builder (utils/build_oid_schema.py)
# -----------------------------------------------------------------------------
# Purpose:             Creates a reusable schema template table for Oriented Imagery Datasets (OIDs)
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Builds a geodatabase table to serve as a schema template for OID creation. It integrates
#   fields from both a central field registry and project-level config blocks. Performs backup
#   of existing templates, schema validation, and writes fields using ArcPy management tools.
#
# File Location:        /utils/build_oid_schema.py
# Validator:            /utils/validators/build_oid_schema_validator.py
# Called By:            tools/create_oid_template_tool.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/expression_utils
# Ext. Dependencies:    arcpy, os, datetime, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/create_oid_and_schema.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Respects categories (standard, not_applicable) from registry
#   - Auto-backs up existing templates with timestamp before overwrite
# =============================================================================

__all__ = ["create_oid_schema_template"]

import arcpy
import os
from datetime import datetime
from typing import Optional, Callable, Any, cast

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import load_field_registry
from utils.validators.common_validators import EsriFieldType


def _field_tuple(f: dict) -> tuple[str, str, Optional[int], str]:
    return (
        f["name"],
        f["type"],
        f.get("length"),
        f.get("alias", f["name"])
    )

def create_oid_schema_template(
    cfg: ConfigManager,
    *,
    arcpy_mod=None,
    os_mod=None,
    registry_loader: Optional[Callable[..., dict]] = None,
    logger: Optional[Any] = None
) -> str:
    """
    Creates a schema template table for Oriented Imagery Datasets (OIDs) using configuration and field registry files.

    Args:
        cfg: Validated configuration manager.
        arcpy_mod: Optional arcpy module for test injection.
        os_mod: Optional os module for test injection.
        registry_loader: Optional loader for field registry.
        logger: Optional logger for test injection.
    Returns:
        The full path to the created schema template table.
    Raises:
        ValueError: If the required field registry path is missing in the configuration.
    """
    arcpy_mod = arcpy_mod or arcpy
    os_mod = os_mod or os
    registry_loader = registry_loader or load_field_registry
    logger = logger or cfg.get_logger()
    cfg.validate(tool="build_oid_schema")
    paths = cfg.paths
    esri_cfg = cfg.get("oid_schema_template.esri_default", {})
    try:
        if not os_mod.path.exists(paths.templates):
            os_mod.makedirs(paths.templates, exist_ok=True)
        if not arcpy_mod.Exists(paths.oid_schema_gdb):
            arcpy_mod.management.CreateFileGDB(paths.templates, os_mod.path.basename(paths.oid_schema_gdb))
        if arcpy_mod.Exists(paths.oid_schema_template_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{paths.oid_schema_template_name}_{timestamp}"
            arcpy_mod.management.Rename(paths.oid_schema_template_path, backup_name)
            logger.info(f"Existing schema template found and backed up as: {backup_name}", indent=0)
        arcpy_mod.management.CreateTable(paths.oid_schema_gdb, paths.oid_schema_template_path)
        fields: list[tuple[str, str, Optional[int], str]] = []

        # Load registry-defined fields (assumes prior validation)
        for category in ("standard", "not_applicable"):
            if esri_cfg.get(category, True):
                entries = registry_loader(cfg, category_filter=category)
                if not entries:
                    logger.debug(f"No fields loaded for category: {category}", indent=1)
                for f in entries.values():
                    fields.append(_field_tuple(f))

        # Add config-defined fields
        for group in ["mosaic_fields", "grp_idx_fields", "linear_ref_fields", "custom_fields"]:
            block = cfg.get(f"oid_schema_template.{group}", {})
            for f in block.values():
                fields.append(_field_tuple(f))
        added_fields = 0
        for name, ftype, length, alias in fields:
            field_type = cast(EsriFieldType, ftype)
            if not arcpy_mod.ListFields(paths.oid_schema_template_path, name):
                try:
                    arcpy_mod.management.AddField(
                        in_table=paths.oid_schema_template_path,
                        field_name=name,
                        field_type=field_type,
                        field_length=length,
                        field_alias=alias,
                        field_is_nullable="NULLABLE"
                    )
                    added_fields += 1
                except Exception as e:
                    logger.warning(f"Failed to add field '{name}' (type={ftype}, length={length}) to schema: {e}", indent=1)
        logger.success(f"OID schema template table created: {paths.oid_schema_template_path} with {added_fields} fields.", indent=0)
        return paths.oid_schema_template_path
    except Exception as e:
        logger.error(f"Failed to create OID schema template: {e}", indent=0)
        raise
