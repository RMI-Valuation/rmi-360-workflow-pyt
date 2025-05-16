# =============================================================================
# 🏷️ EXIF Metadata Tagging Logic (utils/apply_exif_metadata.py)
# -----------------------------------------------------------------------------
# Purpose:             Applies EXIF metadata to images using ExifTool, based on dynamic config expressions
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Resolves tag expressions from config for each image in an OID feature class, then generates
#   a batch ExifTool command file. Supports tagging standard and GPS outlier images in-place.
#   Logs resolved metadata and captures success/failure outcomes.
#
# File Location:        /utils/apply_exif_metadata.py
# Validator:            /utils/validators/apply_exif_metadata_validator.py
# Called By:            tools/rename_and_tag_tool.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/arcpy_utils, utils/shared/expression_utils
# Ext. Dependencies:    arcpy, os, subprocess
# External Tools:       ExifTool (must be installed and available via PATH or config path)
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/rename_and_tag.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Supports both string and list-based tag expressions
#   - Uses ExifTool's -@ arg file interface for efficient batch execution
#   - Logs all resolved metadata and batch execution results
# =============================================================================

__all__ = ["update_metadata_from_config"]

import os
import subprocess
import arcpy
from typing import Dict, Any

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import resolve_expression
from utils.shared.arcpy_utils import validate_fields_exist


def _extract_required_fields(tags, oid_fc=None):
    required_fields = set()
    def recurse_tags(tag_block):
        if isinstance(tag_block, dict):
            for v in tag_block.values():
                recurse_tags(v)
        elif isinstance(tag_block, list):
            for item in tag_block:
                recurse_tags(item)
        elif isinstance(tag_block, str):
            required_fields.update([part.split('.')[1] for part in tag_block.split() if part.startswith("field.")])
    recurse_tags(tags)
    required_fields.update(["X", "Y"])
    # Only require QCFlag if it exists in the feature class
    if oid_fc:
        oid_fields = {f.name for f in arcpy.ListFields(oid_fc)}
        if "QCFlag" in oid_fields:
            required_fields.add("QCFlag")
    return required_fields


def _flatten_tags(prefix: str, tags: dict, cfg: Any, row_dict: dict) -> Dict[str, str]:
    """
    Recursively flattens nested tag dictionaries for ExifTool, e.g.,
    {'GPano': {'PoseHeadingDegrees': '...'}} -> {'XMP-GPano:PoseHeadingDegrees': value}
    """
    flat: Dict[str, str] = {}
    for tag_name, value in tags.items():
        if isinstance(value, dict):
            # Nested dict (e.g., GPano)
            new_prefix = f"{prefix}{tag_name}:" if prefix else f"XMP-{tag_name}:"
            flat.update(_flatten_tags(new_prefix, value, cfg, row_dict))
        elif isinstance(value, list):
            # List of expressions (e.g., XPKeywords)
            keywords = []
            for item in value:
                try:
                    val = resolve_expression(item, cfg, row=row_dict)
                    if not isinstance(val, str):
                        val = str(val)
                    keywords.append(val)
                except Exception as e:
                    print(f"Failed to resolve tag {tag_name}: {e}")
            flat[f"{prefix}{tag_name}"] = ";".join(keywords)
        else:
            # String or numeric expression
            try:
                val = resolve_expression(value, cfg, row=row_dict)
                if not isinstance(val, str):
                    val = str(val)
                flat[f"{prefix}{tag_name}"] = val
            except Exception as e:
                print(f"Failed to resolve tag {tag_name}: {e}")
    return flat


def _resolve_tags(cfg: Any, tags: dict, row_dict: dict) -> Dict[str, str]:
    # Handles both flat and nested tags
    return _flatten_tags("", tags, cfg, row_dict)


def _write_exiftool_args(cfg, tags, rows, cursor_fields):
    args_file = cfg.paths.get_log_file_path("exiftool_args", cfg)
    log_file = cfg.paths.get_log_file_path("exiftool_logs", cfg)
    lines = []
    log_entries = []
    logger = cfg.get_logger()
    for i, row in enumerate(rows):
        # Build row_dict from actual cursor_fields
        row_dict = dict(zip(cursor_fields, row))
        path = row_dict["ImagePath"]
        if not os.path.exists(path):
            print(f"Image path does not exist: {path}")
            continue

        resolved_tags = _resolve_tags(cfg, tags, row_dict)
        if i == 0:
            logger.debug(f"row_dict for first image: {row_dict}")
            logger.debug(f"resolved_tags for first image: {resolved_tags}")
        for tag_name, value in resolved_tags.items():
            lines.append(f"-{tag_name}={value}")

        # Only add GPS_OUTLIER logic if QCFlag is present
        if "QCFlag" in row_dict and row_dict["QCFlag"] == "GPS_OUTLIER":
            lat, lon = row_dict["Y"], row_dict["X"]
            lat_ref = "North" if lat >= 0 else "South"
            lon_ref = "East" if lon >= 0 else "West"
            lines.extend([
                f"-GPSLatitude={abs(lat)}",
                f"-GPSLatitudeRef={lat_ref}",
                f"-GPSLongitude={abs(lon)}",
                f"-GPSLongitudeRef={lon_ref}"
            ])

        lines.extend([
            "-overwrite_original_in_place",
            path.replace('\\', '/'),
            "-execute",
            ""
        ])
        log_entries.append(f"✅ Tagged OID {row_dict['OID@']} → {os.path.basename(path)}")

    with open(args_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries))


def _run_exiftool(cfg, args_file):
    logger = cfg.get_logger()
    exe_path = cfg.paths.exiftool_exe
    if not cfg.paths.check_exiftool_available():
        logger.error(f"ExifTool not found or not working at: {exe_path}", error_type=RuntimeError, indent=1)
    try:
        subprocess.run([exe_path, "-@", args_file], check=True)
        logger.success("Metadata tagging completed.", indent=1)
    except subprocess.CalledProcessError as e:
        logger.error(f"ExifTool failed: {e}", error_type=RuntimeError, indent=1)


def update_metadata_from_config(cfg: ConfigManager, oid_fc: str):
    """
    Updates image metadata for images referenced in a feature class using configuration rules.

    Applies metadata tags to images by evaluating expressions defined in a configuration file or dictionary. Tagging
    is performed in batch using ExifTool, with tag values resolved from feature class fields. Handles GPS metadata
    updates for images flagged as outliers and logs all operations and errors.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="apply_exif_metadata")

    tags = cfg.get("image_output.metadata_tags", {})
    if not tags:
        logger.error("No metadata_tags defined in config.yaml.", error_type=ValueError, indent=1)

    required_fields = _extract_required_fields(tags, oid_fc=oid_fc)
    validate_fields_exist(oid_fc, list(required_fields))

    oid_fields = {f.name for f in arcpy.ListFields(oid_fc)}
    base_fields = ["OID@", "ImagePath", "X", "Y"]
    if "QCFlag" in oid_fields:
        base_fields.append("QCFlag")
    cursor_fields = base_fields + [f for f in required_fields if f not in base_fields]

    logger.debug(f"Fields used for SearchCursor: {cursor_fields}")
    with arcpy.da.SearchCursor(oid_fc, cursor_fields) as cursor:
        rows = [row for row in cursor]

    _write_exiftool_args(cfg, tags, rows, cursor_fields)
    _run_exiftool(cfg, cfg.paths.get_log_file_path("exiftool_args", cfg))
