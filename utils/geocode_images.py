# =============================================================================
# 🌍 Reverse Geocoding via ExifTool (utils/geocode_images.py)
# -----------------------------------------------------------------------------
# Purpose:             Adds XMP geolocation tags to images in an OID using ExifTool and GPSPosition metadata
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-14
# Last Updated:        2025-05-15
#
# Description:
#   Iterates through images referenced in an OID feature class and uses ExifTool to copy GPSPosition into the
#   XMP geolocate field. Supports alternate geolocation DBs by referencing external config files. Generates
#   argument and log files and performs in-place metadata updates using ExifTool’s batch mode.
#
# File Location:        /utils/geocode_images.py
# Validator:            /utils/validators/geocode_images_validator.py
# Called By:            tools/geocode_images_tool.py, process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/arcpy_utils
# Ext. Dependencies:    arcpy, subprocess, os
# External Tools:       ExifTool (must be installed and available via PATH or config path)
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/geocode_images.md
#
# Notes:
#   - Skips images without GPS or valid path
#   - Uses ExifTool’s -execute mode for efficient batch tagging
# =============================================================================

___all___ = ["geocode_images"]

import os
import subprocess
import arcpy

from utils.manager.config_manager import ConfigManager
from utils.shared.arcpy_utils import validate_fields_exist


def get_exiftool_cmd(cfg, logger):
    method = cfg.get("geocoding.method", "").lower()
    if method != "exiftool":
        logger.warning("Geocoding skipped (unsupported method)", indent=1)
        return None
    db_choice = cfg.get("geocoding.exiftool_geodb", "default").lower()
    logger.custom(f"Using geolocation DB: {db_choice}", indent=1, emoji="🌍")
    exiftool_cmd = ["exiftool"]
    if db_choice == "geolocation500":
        config_path = cfg.paths.geoloc500_config_path
    elif db_choice == "geocustom":
        config_path = cfg.paths.geocustom_config_path
    else:
        config_path = None
    if config_path:
        exiftool_cmd.extend(["-config", str(config_path.resolve())])
    return exiftool_cmd

def build_geocode_args_and_log(rows, logger):
    lines = []
    log_entries = []
    for oid, path, x, y in rows:
        if not os.path.exists(path):
            logger.warning(f"Image path does not exist: {path}", indent=1)
            continue
        lines.extend([
            "-overwrite_original_in_place",
            "-XMP:XMP-iptcExt:geolocate<gpsposition",
            path.replace('\\', '/'),
            "-execute",
            ""
        ])
        log_entries.append(f"📍 Geocoded OID {oid} → {os.path.basename(path)}")
    return lines, log_entries

def write_args_and_log_files(args_file, log_file, lines, log_entries):
    with open(args_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries))

def run_exiftool(cmd, args_file, logger):
    cmd = list(cmd) + ["-@", args_file]
    try:
        subprocess.run(cmd, check=True)
        logger.success("Reverse geocoding completed.", indent=1)
    except subprocess.CalledProcessError as e:
        logger.error(f"ExifTool geocoding failed: {e}", indent=1)

def geocode_images(cfg: ConfigManager, oid_fc: str) -> None:
    """
    Applies reverse geocoding tags to images in a feature class using ExifTool.

    This function processes each image referenced in the input feature class, copying GPSPosition metadata to the XMP
    geolocate tag using ExifTool. It supports different ExifTool geolocation databases as specified in the
    configuration. Images without valid paths are skipped, and all actions are logged. The function writes ExifTool
    argument and log files, then executes the ExifTool command to update image metadata in place.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="geocode_images")

    required_fields = ["OID@", "ImagePath", "X", "Y"]
    validate_fields_exist(oid_fc, ["ImagePath", "X", "Y"])

    with arcpy.da.SearchCursor(oid_fc, required_fields) as cursor:
        rows = [row for row in cursor]

    args_file = cfg.paths.get_log_file_path("geocode_args", cfg)
    log_file = cfg.paths.get_log_file_path("geocode_logs", cfg)
    exiftool_cmd = get_exiftool_cmd(cfg, logger)
    if exiftool_cmd is None:
        return
    lines, log_entries = build_geocode_args_and_log(rows, logger)
    write_args_and_log_files(args_file, log_file, lines, log_entries)
    run_exiftool(exiftool_cmd, args_file, logger)
