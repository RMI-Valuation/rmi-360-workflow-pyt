# =============================================================================
# 🧮 OID Attribute Enrichment Logic (utils/calculate_oid_attributes.py)
# -----------------------------------------------------------------------------
# Purpose:             Enriches Oriented Imagery Dataset features with camera orientation, reel, and Z attributes
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Applies default and derived values to fields in an OID feature class, including
#   orientation, SRS, reel, and frame info. Incorporates validation against a field
#   registry and adjusts Z-values using configured camera offsets. Validates field-of-view
#   defaults and integrates reel_info.json metadata if available.
#
# File Location:        /utils/calculate_oid_attributes.py
# Validator:            /utils/validators/calculate_oid_attributes_validator.py
# Called By:            tools/add_images_to_oid_tool.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/expression_utils
# Ext. Dependencies:    arcpy, os, json, re, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/add_images_to_oid.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Integrates reel_info.json if present to supplement reel assignment
#   - Skips processing if OID is empty or missing required fields
# =============================================================================

__all__ = ["enrich_oid_attributes"]

import arcpy
import re
import os
import json
from typing import Optional, Tuple

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import load_field_registry


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def check_oid_fov_defaults(oid_fc_path: str, registry: dict, logger):
    """
    Checks that HFOV and VFOV values in the OID feature class match expected defaults.
    
    Logs an error for each row where the HorizontalFieldOfView or VerticalFieldOfView
    differs from the default values specified in the registry.
    """
    expected_hfov = registry.get("HFOV", {}).get("oid_default")
    expected_vfov = registry.get("VFOV", {}).get("oid_default")

    with arcpy.da.SearchCursor(oid_fc_path, ["HorizontalFieldOfView", "VerticalFieldOfView"]) as cursor:
        for i, row in enumerate(cursor):
            hfov, vfov = row
            if hfov != expected_hfov or vfov != expected_vfov:
                logger.error(f"Row {i}: HFOV or VFOV does not match expected values from registry.\n "
                             f"Expected HFOV={expected_hfov}, VFOV={expected_vfov}, but got HFOV={hfov}, VFOV={vfov}",
                             error_type=ValueError, indent=1)


def load_reel_from_info_file(image_path: str, logger) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempts to find and read a reel_info.json file near the specified image path.
    
    Searches the parent and grandparent directories of the image for a reel_info.json file.
    If exactly one file is found, returns its "reel" value and the file path. If multiple files
    are found or an error occurs, logs a warning and returns (None, None).
    
    Args:
        image_path: Full path to the image file.
        logger: Logger instance.
    
    Returns:
        A tuple containing the reel value and the path to the reel_info.json file, or (None, None)
        if not found or if an error occurs.
    """
    try:
        dirs_to_check = [
            os.path.dirname(os.path.dirname(image_path)),   # grandparent
            os.path.dirname(image_path)                     # parent
        ]

        reel_info_paths = [os.path.join(d, "reel_info.json") for d in dirs_to_check if os.path.isfile(os.path.join(d, "reel_info.json"))]

        if len(reel_info_paths) == 1:
            path = reel_info_paths[0]
            with open(path, "r") as f:
                reel_data = json.load(f)
                return reel_data.get("reel"), path

        if len(reel_info_paths) > 1:
            logger.warning(f"Multiple reel_info.json files found near: {image_path}\nFiles:\n" + "\n".join(reel_info_paths), indent=1)
    except Exception as e:
        logger.error(f"Failed to load reel_info.json near: {image_path}\n{e}", indent=1)

    return None, None


def extract_reel_from_path(image_path: str) -> Optional[str]:
    """
    Extracts a 4-digit reel number from the grandparent directory of an image path.
    
    The function searches for a folder name matching the pattern 'reel_XXXX' (where XXXX is a 4-digit number) two
    levels above the provided image file. Returns the reel number if found, otherwise returns None.
    """
    reel_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    match = re.search(r"reel_(\d{4})", reel_folder, re.IGNORECASE)
    return match.group(1) if match else None


def extract_frame_from_filename(image_path: str) -> Optional[str]:
    """
    Extracts a 6-digit frame number from the image filename.
    
    The function searches for a pattern like '_000234.jpg' at the end of the filename and returns the frame number if
    found. Returns None if the pattern is not present or an error occurs.
    """

    image_file = os.path.basename(image_path)
    match = re.search(r"_(\d{6})\.jpg$", image_file, re.IGNORECASE)
    return match.group(1) if match else None


def enrich_oid_attributes(cfg: ConfigManager, oid_fc_path: str, adjust_z: bool = True) -> None:
    """
    Enriches an Oriented Imagery Dataset with derived and default attribute values.

    This function updates the specified OID feature class by setting default values for camera parameters, adjusting Z
    coordinates by camera offset if requested, and populating orientation, spatial reference, reel, and frame fields.
    Reel and frame numbers are extracted from image paths or associated metadata files. Field of view values are
    validated against expected defaults. The function is typically used after adding images to the dataset.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="calculate_oid_attributes")

    # Load registry and schema safely
    registry = load_field_registry(cfg)

    # Compute Z offset and camera height
    try:
        z_cm = sum(_safe_float(v) for v in cfg.get("camera_offset.z", {}).values())
        height_cm = sum(_safe_float(v) for v in cfg.get("camera_offset.camera_height").values())
        z_offset = z_cm / 100.0
        camera_height = height_cm / 100.0
    except Exception as e:
        logger.error(f"Failed to compute camera offset or height: {e}", error_type=ValueError, indent=1)
        return

    h_wkid = cfg.get("spatial_ref.gcs_horizontal_wkid", 4326)
    v_wkid = cfg.get("spatial_ref.vcs_vertical_wkid", 5703)


    pitch = registry.get("CameraPitch", {}).get("oid_default", 90)
    roll = registry.get("CameraRoll", {}).get("oid_default", 0)
    near = registry.get("NearDistance", {}).get("oid_default", 2)
    far = registry.get("FarDistance", {}).get("oid_default", 50)

    # Determine fields to fetch and update
    esri_cfg = cfg.get("oid_schema_template.esri_default", {})
    schema_cfg = cfg.get("oid_schema_template", {})
    fields = ["OID@", "SHAPE@X", "SHAPE@Y", "SHAPE@Z"]

    for field in registry.values():
        if field.get("category") == "standard" or (field.get("category") == "not_applicable" and
                                                   esri_cfg.get("not_applicable", False)):
            fields.append(field["name"])

    for fdef in schema_cfg.get("mosaic_fields", {}).values():
        fields.append(fdef["name"])

    # Deduplicate in case of overlap
    fields = list(dict.fromkeys(fields))
    field_to_index = {name: i for i, name in enumerate(fields)}

    check_oid_fov_defaults(oid_fc_path, registry, logger)

    # Read first image path to extract reel metadata
    with arcpy.da.SearchCursor(oid_fc_path, ["ImagePath"], where_clause="ImagePath IS NOT NULL") as cursor:
        first_image_path = next((row[0].strip() for row in cursor if row[0]), None)

    # Handle empty dataset scenario
    if not first_image_path:
        logger.error("No images found in the OID dataset. Skipping OID attribute calculation.", indent=1)
        return

    reel_from_info, reel_info_path_used = load_reel_from_info_file(first_image_path, logger)
    if reel_info_path_used:
        logger.info(f"📄 Using reel_info.json from: {reel_info_path_used}", indent=1)

    # Count rows to prepare progressor
    row_count = int(arcpy.management.GetCount(oid_fc_path)[0])

    updated = 0
    with cfg.get_progressor(total=row_count, label="Enriching OID attributes") as progressor:
        with arcpy.da.UpdateCursor(oid_fc_path, fields) as cursor:
            for i, row in enumerate(cursor, start=1):
                # Safely access required fields
                try:
                    x = row[field_to_index["SHAPE@X"]]
                    y = row[field_to_index["SHAPE@Y"]]
                    z = row[field_to_index["SHAPE@Z"]]
                except KeyError:
                    logger.warning(f"Missing SHAPE@X/Y/Z fields on row {i}, skipping row.", indent=2)
                    continue
                adjusted_z = z + z_offset if adjust_z else z

                heading = row[field_to_index["CameraHeading"]] if "CameraHeading" in field_to_index else None
                if heading is None:
                    logger.warning(f"Missing CameraHeading for row {i}, skipping row.", indent=2)
                    continue

                image_path = row[field_to_index["ImagePath"]].strip() if "ImagePath" in field_to_index and row[field_to_index["ImagePath"]] else None
                if not image_path:
                    logger.warning(f"Missing ImagePath for row {i}, skipping row.", indent=2)
                    continue

                orientation = f"1|{h_wkid}|{v_wkid}|{x:.6f}|{y:.6f}|{adjusted_z:.3f}|{heading:.1f}|{pitch:.1f}|{roll:.1f}"
                reel = extract_reel_from_path(image_path) or reel_from_info
                frame = extract_frame_from_filename(image_path)

                # Only assign to fields that exist in the schema
                if "CameraPitch" in field_to_index:
                    row[field_to_index["CameraPitch"]] = pitch
                if "CameraRoll" in field_to_index:
                    row[field_to_index["CameraRoll"]] = roll
                if "NearDistance" in field_to_index:
                    row[field_to_index["NearDistance"]] = near
                if "FarDistance" in field_to_index:
                    row[field_to_index["FarDistance"]] = far
                if "X" in field_to_index:
                    row[field_to_index["X"]] = x
                if "Y" in field_to_index:
                    row[field_to_index["Y"]] = y
                if "Z" in field_to_index:
                    row[field_to_index["Z"]] = adjusted_z
                if "SHAPE@Z" in field_to_index:
                    row[field_to_index["SHAPE@Z"]] = adjusted_z
                if "SRS" in field_to_index:
                    row[field_to_index["SRS"]] = f"{h_wkid},{v_wkid}"
                if "CameraHeight" in field_to_index:
                    row[field_to_index["CameraHeight"]] = camera_height
                if "CameraOrientation" in field_to_index:
                    row[field_to_index["CameraOrientation"]] = orientation
                if "Reel" in field_to_index:
                    row[field_to_index["Reel"]] = reel
                if "Frame" in field_to_index:
                    row[field_to_index["Frame"]] = frame

                cursor.updateRow(row)
                updated += 1
                progressor.update(i)

    logger.success(f"Updated {updated} image(s) with orientation, Z, and mosaic fields.", indent=1)
