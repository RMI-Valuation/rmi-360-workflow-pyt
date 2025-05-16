# =============================================================================
# 🧱 OID Footprint Builder (utils/build_oid_footprints.py)
# -----------------------------------------------------------------------------
# Purpose:             Generates a footprint feature class for a given OID using ArcGIS tools
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-14
# Last Updated:        2025-05-15
#
# Description:
#   Builds a BUFFER-style footprint for an Oriented Imagery Dataset (OID) using ArcPy’s
#   BuildOrientedImageryFootprint. Resolves spatial reference and geographic transformation
#   from the config. Outputs a new feature class alongside the input OID.
#
# File Location:        /utils/build_oid_footprints.py
# Validator:            /utils/validators/build_oid_footprints_validator.py
# Called By:            tools/build_oid_footprints_tool.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/expression_utils
# Ext. Dependencies:    arcpy, os, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/build_oid_footprints.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Restores env settings after footprint creation
#   - Warns instead of failing if footprint creation is unsuccessful
# =============================================================================

import arcpy
import os
from typing import Optional

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import resolve_expression


def resolve_spatial_reference(cfg, logger):
    sr_expr = cfg.get("spatial_ref.pcs_horizontal_wkid")
    if sr_expr is None:
        logger.error("`spatial_ref.pcs_horizontal_wkid` missing from config.", error_type=KeyError, indent=2)
        return None
    try:
        wkid = int(resolve_expression(sr_expr, cfg))
        logger.custom(f"Using projected coordinate system: WKID {wkid}", indent=2, emoji="📐")
        return arcpy.SpatialReference(wkid)
    except Exception as e:
        logger.error(f"Failed to resolve spatial_ref.pcs_horizontal_wkid: {e}", error_type=ValueError, indent=2)
        return None


def resolve_geographic_transformation(cfg, logger):
    transform = cfg.get("spatial_ref.transformation") or None
    if transform:
        logger.custom(f"Applying geographic transformation: {transform}", indent=2, emoji="🌍")
    return transform


def get_output_path(oid_fc):
    desc = arcpy.Describe(oid_fc)
    out_dataset_path = desc.path
    out_dataset_name = f"{desc.baseName}_Footprint"
    return os.path.join(out_dataset_path, out_dataset_name), out_dataset_path, out_dataset_name


def build_footprint_with_env(oid_fc, output_sr, transform, out_dataset_path, out_dataset_name, output_path, logger):
    prev_sr = arcpy.env.outputCoordinateSystem
    prev_trans = arcpy.env.geographicTransformations
    try:
        arcpy.env.outputCoordinateSystem = output_sr
        if transform:
            arcpy.env.geographicTransformations = transform
        arcpy.oi.BuildOrientedImageryFootprint(
            in_oriented_imagery_dataset=oid_fc,
            out_dataset_path=out_dataset_path,
            out_dataset_name=out_dataset_name,
            footprint_option="BUFFER"
        )
        logger.success(f"OID footprint successfully created at: {output_path}", indent=1)
        return output_path
    except Exception as e:
        logger.warning(f"Failed to build OID footprints: {e}. Footprint creation can be done post-process.", indent=1)
        return None
    finally:
        arcpy.env.outputCoordinateSystem = prev_sr
        arcpy.env.geographicTransformations = prev_trans


def build_oid_footprints(cfg: ConfigManager, oid_fc: str) -> Optional[str]:
    """
    Generates a BUFFER-style footprint feature class for an Oriented Imagery Dataset (OID).

    Creates a new footprint feature class from the specified OID feature class using ArcPy's
    BuildOrientedImageryFootprint tool. The output spatial reference and optional geographic
    transformation are resolved from the provided configuration dictionary or YAML file.
    Returns the full path to the created footprint feature class, or None if spatial reference
    resolution fails.

    Args:
        cfg:
        oid_fc: Path to the input Oriented Imagery Dataset feature class.

    Returns:
        Optional[str]: Full path to the created footprint feature class, or None if failed.

    Raises:
        FileNotFoundError: If the input OID feature class does not exist.
        Exception: If footprint creation fails for any other reason.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="build_oid_footprints")

    logger.info("Starting OID footprint generation...", indent=1)

    if not arcpy.Exists(oid_fc):
        logger.error(f"Input OID does not exist: {oid_fc}", error_type=FileNotFoundError, indent=2)
        return None

    output_sr = resolve_spatial_reference(cfg, logger)
    if output_sr is None:
        return None
    transform = resolve_geographic_transformation(cfg, logger)
    output_path, out_dataset_path, out_dataset_name = get_output_path(oid_fc)
    return build_footprint_with_env(oid_fc, output_sr, transform, out_dataset_path, out_dataset_name, output_path, logger)
