# =============================================================================
# 🏗️ OID Feature Class Creator (utils/create_oid_feature_class.py)
# -----------------------------------------------------------------------------
# Purpose:             Creates a new Oriented Imagery Dataset (OID) feature class using a validated schema template
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Constructs a new OID feature class using ArcGIS Pro’s CreateOrientedImageryDataset tool. Validates schema
#   compatibility via registered field templates and ensures spatial reference compatibility. Supports config-driven
#   defaults for both horizontal and vertical coordinate systems and performs creation only if the OID does not exist.
#
# File Location:        /utils/create_oid_feature_class.py
# Validator:            /utils/validators/create_oid_validator.py
# Called By:            tools/create_oid_tool.py, orchestrator pipeline
# Int. Dependencies:    utils/manager/config_manager, utils/shared/schema_validator
# Ext. Dependencies:    arcpy, os, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/create_oid_and_schema.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Applies vertical WKID if defined in spatial_ref block of config
#   - Uses Terrain3D as default DEM for elevation reference
# =============================================================================

__all__ = ["create_oriented_imagery_dataset"]

import os
import arcpy
from typing import Optional, Union

from utils.manager.config_manager import ConfigManager
from utils.shared.schema_validator import ensure_valid_oid_schema_template


def create_oriented_imagery_dataset(
        cfg: ConfigManager,
        output_fc_path: str,
        spatial_reference: Optional[Union[int, arcpy.SpatialReference]] = None) -> str:
    """
    Creates an Oriented Imagery Dataset (OID) feature class at the specified path.

    This function generates a new OID feature class using a validated schema template, applying the provided or default
    spatial reference and configuration. It ensures the output does not already exist, validates the schema, and logs
    progress and errors through the configured messaging system.

    Args:
        cfg:
        output_fc_path: Full path where the new OID feature class will be created.
        spatial_reference: Optional. Horizontal spatial reference as a WKID (int), an arcpy.SpatialReference object,
        or None to use defaults from configuration.

    Returns:
        The path to the created OID feature class.

    Raises:
        ValueError: If the output feature class already exists or if the spatial reference is invalid.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="create_oriented_imagery_dataset")
    paths = cfg.paths

    output_gdb, oid_name = os.path.split(output_fc_path)

    if arcpy.Exists(output_fc_path):
        logger.error(f"Output feature class already exists: {output_fc_path}", error_type=FileExistsError, indent=1)
        return output_fc_path

    default_xy = cfg.get("spatial_ref.gcs_horizontal_wkid", 4326)
    default_z = cfg.get("spatial_ref.vcs_vertical_wkid", 5703)

    if isinstance(spatial_reference, arcpy.SpatialReference):
        sr = spatial_reference
    elif isinstance(spatial_reference, int):
        sr = arcpy.SpatialReference(spatial_reference, default_z)
    elif spatial_reference is None:
        sr = arcpy.SpatialReference(default_xy, default_z)
    else:
        logger.error(f"Invalid spatial_reference: {spatial_reference}. Must be None, WKID (int), or "
                     f"arcpy.SpatialReference.", error_type=ValueError, indent=1)
        raise

    with cfg.get_progressor(total=2, label="Creating OID...") as progressor:
        # ✅ Ensure schema template is valid (and rebuild it if needed)
        ensure_valid_oid_schema_template(cfg)
        progressor.update(1)

        logger.info(f"Creating Oriented Imagery Dataset at {output_fc_path}", indent=1)

        try:
            logger.debug("Calling CreateOrientedImageryDataset with parameters:", indent=2)
            logger.debug(f"out_dataset_path={output_gdb}", indent=3)
            logger.debug(f"out_dataset_name={oid_name}", indent=3)
            logger.debug(f"spatial_reference={sr}", indent=3)
            logger.debug(f"elevation_source='DEM'", indent=3)
            logger.debug(f"dem='https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer'", indent=3)
            logger.debug(f"lod='17'", indent=3)
            logger.debug(f"template={str(paths.oid_schema_template_path)}", indent=3)
            logger.debug(f"has_z='ENABLED'", indent=3)
            arcpy.oi.CreateOrientedImageryDataset(
                out_dataset_path=output_gdb,
                out_dataset_name=oid_name,
                spatial_reference=sr,
                elevation_source="DEM",
                dem="https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer",
                lod="17",
                template=[str(paths.oid_schema_template_path)],
                has_z="ENABLED"
            )
        except arcpy.ExecuteError as exc:
            logger.error(f"Arcpy failed while creating OID: {oid_name}: {exc}", error_type=RuntimeError, indent=1)

        progressor.update(2)

    logger.success(f"OID created successfully: {output_fc_path}", indent=1)

    return output_fc_path
