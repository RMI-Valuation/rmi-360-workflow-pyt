# =============================================================================
# 🗂️ Image Renaming Utility (utils/rename_images.py)
# -----------------------------------------------------------------------------
# Purpose:             Renames and organizes OID images using config-based filename expressions
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Resolves dynamic filename parts from config expressions (e.g., reel, MP, timestamp),
#   renames/copies each image to a flat output folder, updates the OID’s ImagePath and Name
#   fields, and logs all rename operations. Optionally deletes originals after copy.
#
# File Location:        /utils/rename_images.py
# Validator:            /utils/validators/rename_images_validator.py
# Called By:            tools/rename_and_tag_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/arcpy_utils, utils/shared/expression_utils, utils/shared/check_disk_space
# Ext. Dependencies:    arcpy, os, shutil, csv, pathlib
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/rename_images.md
#
# Notes:
#   - Auto-detects duplicate filenames and applies numeric suffix (e.g., _v2)
#   - Ensures output directory is writable and validated before execution
# =============================================================================

__all__ = ["rename_images"]

import arcpy
import os
import shutil
import csv
from pathlib import Path

from utils.manager.config_manager import ConfigManager
from utils.shared.arcpy_utils import validate_fields_exist
from utils.shared.expression_utils import resolve_expression
from utils.shared.check_disk_space import check_sufficient_disk_space


def _resolve_fields(cfg, row_dict, parts):
    """
    Resolve dynamic filename parts from config expressions.
    """
    return {
        key: resolve_expression(expr, cfg, row=row_dict)
        for key, expr in parts.items()
    }


def _get_unique_filename(output_dir, filename):
    """
    Get a unique filename by appending a numeric suffix if the file already exists.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    while (output_dir / filename).is_file():
        filename = f"{base}_v{counter}{ext}"
        counter += 1
    return filename


def _copy_and_delete(old_path, new_path, delete_originals):
    """
    Copy the file and delete the original if specified.
    """
    try:
        shutil.copy2(old_path, new_path)
        if delete_originals:
            os.remove(old_path)
    except (PermissionError, OSError) as e:
        raise Exception(f"Failed to copy/delete file: {e}")


def _update_row(cursor, row_dict, fields, new_path, filename):
    """
    Update the row with the new image path and filename.
    """
    row_dict["ImagePath"] = str(new_path)
    row_dict["Name"] = filename
    cursor.updateRow([row_dict.get(f) for f in fields])


def _write_rename_log(rename_log_path, log_rows):
    """
    Write the rename log to a CSV file.
    """
    try:
        with open(rename_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["OID", "OriginalPath", "OriginalName", "NewPath", "NewName"])
            writer.writeheader()
            writer.writerows(log_rows)
    except PermissionError as e:
        raise Exception(f"Failed to write rename log: {e}")


def rename_images(cfg: ConfigManager, oid_fc: str, delete_originals: bool = False, enable_linear_ref: bool = True):
    """
    Renames and copies image files for an Oriented Imagery Dataset according to configuration rules.

    All renamed images are placed in a flat directory specified by the configuration, and the `Name` and `ImagePath`
    attributes in the dataset are updated in-place. Optionally deletes original images after copying. A CSV log of
    rename operations is generated. Returns a list of dictionaries with updated image metadata.

    The filename format is selected based on the enable_linear_ref argument (passed at runtime, e.g. from ArcGIS Pro):
    - If enable_linear_ref is False, uses image_output.filename_settings.format_no_lr
    - If enable_linear_ref is True, uses image_output.filename_settings.format

    Args:
        cfg:
        oid_fc: Path to the feature class containing image records.
        delete_originals: If True, deletes original image files after copying.
        enable_linear_ref: If False, use the no-linear-referencing filename format.

    Returns:
        A list of dictionaries representing updated image records, including OID, new path, filename, QC flag,
        coordinates, and resolved metadata.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="rename_images")

    # Use format_no_lr if linear referencing is not enabled
    if not enable_linear_ref:
        fmt = cfg.get("image_output.filename_settings.format_no_lr", "{Name}.jpg")
    else:
        fmt = cfg.get("image_output.filename_settings.format", "{Name}.jpg")
    parts = cfg.get("image_output.filename_settings.parts", {})

    # QCFlag is optional; check if it exists in the feature class

    oid_fields = {f.name for f in arcpy.ListFields(oid_fc)}
    base_fields = ["OID@", "ImagePath", "Name", "X", "Y"]
    required_fields = base_fields.copy()
    if "QCFlag" in oid_fields:
        required_fields.append("QCFlag")
    for expr in parts.values():
        if isinstance(expr, str) and expr.startswith("field."):
            field_name = expr.split(".", 1)[1].split(".", 1)[0]
            if field_name not in required_fields:
                required_fields.append(field_name)

    validate_fields_exist(oid_fc, [f for f in required_fields if f != "OID@"])
    fields = list(dict.fromkeys(required_fields))

    check_sufficient_disk_space(oid_fc, cfg)

    output_dir = cfg.paths.renamed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify write permissions
    try:
        test_file = output_dir / ".permission_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        logger.error(f"Output directory is not writable: {output_dir}. Error: {e}", error_type=PermissionError, indent=1)

    # Prepare CSV log
    rename_log_path = cfg.paths.get_log_file_path("rename_log", cfg)
    logger.debug(f"Writing rename log to: {rename_log_path}", indent=1)
    log_rows = []

    logger.info(f"Writing renamed images to: {output_dir}", indent=1)

    updated_images = []

    with arcpy.da.UpdateCursor(oid_fc, fields) as cursor:
        for row in cursor:
            row_dict = dict(zip(fields, row))
            oid = row_dict["OID@"]
            old_path = Path(row_dict["ImagePath"])

            if not old_path.is_file():
                logger.warning(f"Image path missing or invalid for OID {oid}: {old_path}", indent=1)
                continue

            try:
                resolved = _resolve_fields(cfg, row_dict, parts)
                filename = fmt.format(**resolved)
                new_path = output_dir / _get_unique_filename(output_dir, filename)

                _copy_and_delete(old_path, new_path, delete_originals)
                _update_row(cursor, row_dict, fields, new_path, filename)

                updated_images.append({
                    "oid": oid,
                    "path": str(new_path),
                    "filename": filename,
                    "qcflag": row_dict.get("QCFlag"),
                    "x": row_dict.get("X"),
                    "y": row_dict.get("Y"),
                    "metadata": resolved
                })

                # Add to CSV log
                log_rows.append({
                    "OID": oid,
                    "OriginalPath": str(old_path),
                    "OriginalName": old_path.name,
                    "NewPath": str(new_path),
                    "NewName": filename
                })

                logger.debug(f"Renamed OID {oid} ➜ {filename}", indent=2)
            except Exception as e:
                logger.warning(f"Failed to rename/copy for OID {oid}: {e}", indent=2)
                continue

    _write_rename_log(rename_log_path, log_rows)
    logger.custom(f"Rename log saved to: {rename_log_path}", indent=1, emoji="📝")

    logger.success(f"{len(updated_images)} image(s) renamed and attributes updated.", indent=1)
    return updated_images
