# =============================================================================
# 🧭 OID Image Import Logic (utils/add_images_to_oid_fc.py)
# -----------------------------------------------------------------------------
# Purpose:             Adds geotagged 360° images to an ArcGIS Oriented Imagery Dataset (OID)
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Scans and validates a folder of enhanced/final images, checks for reel_info
#   collisions, and appends entries to a target Oriented Imagery Dataset using
#   ArcPy’s Oriented Imagery tools. Includes schema validation and recursive support.
#
# File Location:        /utils/add_images_to_oid_fc.py
# Validator:            /utils/validators/add_images_to_oid_validator.py
# Called By:            tools/add_images_to_oid_tool.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/expression_utils
# Ext. Dependencies:    arcpy, pathlib
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/add_images_to_oid.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Supports recursive reel folder discovery and duplicate prevention
#   - Integrates schema validation and status/error logging
# =============================================================================

__all__ = ["add_images_to_oid"]

import arcpy
from pathlib import Path

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import load_field_registry


def warn_if_multiple_reel_info(image_folder, logger):
    """
    Checks for multiple 'reel_info.json' files within the specified image folder and its subfolders.
    
    If more than one 'reel_info.json' file is found, logs an error message listing all detected file paths.
    """
    reel_info_paths = list(image_folder.rglob("reel_info.json"))
    if len(reel_info_paths) > 1:
        logger.warning(
            f"Multiple reel_info.json files detected in image folder '{image_folder}':\n"
            + "\n".join(str(p) for p in reel_info_paths), indent=1
        )


def add_images_to_oid(cfg: ConfigManager, oid_fc_path: str) -> None:
    """
    Adds images from a project folder to an existing Oriented Imagery Dataset (OID).

    Resolves configuration to determine the image folder, validates the presence of required files and directories,
    and adds all JPEG images (including those in subfolders) to the specified OID feature class using ArcPy. Logs errors
    if the OID, image folder, or images are missing, and integrates with ArcGIS messaging for status updates.

    Args:
        cfg: Validated configuration manager.
        oid_fc_path: Path to the existing OID feature class.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="add_images_to_oid")

    image_folder = cfg.paths.original

    with cfg.get_progressor(total=2, label="Adding images to OID") as progressor:
        warn_if_multiple_reel_info(image_folder, logger)

        if not arcpy.Exists(oid_fc_path):
            logger.error(f"OID does not exist at path: {oid_fc_path}", error_type=FileNotFoundError, indent=1)

        if not image_folder.is_dir():
            logger.error(f"Image folder not found: {image_folder}", error_type=FileNotFoundError, indent=1)

        # Use pathlib to collect all .jpg files recursively
        jpg_files = list(Path(image_folder).rglob("*.jpg"))
        if not jpg_files:
            logger.error(f"No .jpg files found in image folder or its subfolders: {image_folder}", error_type=RuntimeError, indent=1)

        registry = load_field_registry(cfg)
        imagery_type = registry.get("OrientedImageryType", {}).get("oid_default", "360")

        logger.info(f"Adding images from '{image_folder}' (including subfolders) to OID: {oid_fc_path}", indent=1)
        progressor.update(1)

        try:
            arcpy.oi.AddImagesToOrientedImageryDataset(
                in_oriented_imagery_dataset=oid_fc_path,
                imagery_category=imagery_type,
                input_data=[str(image_folder)],
                include_sub_folders="SUBFOLDERS"
            )
            progressor.update(2)
        except arcpy.ExecuteError as exc:
            logger.error(f"Failed to add images to OID: {exc}", error_type=RuntimeError, indent=1)
            return

    logger.success("Images successfully added to OID.", indent=1)
