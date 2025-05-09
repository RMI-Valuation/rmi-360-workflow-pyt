# =============================================================================
# 🗂️ Image Renaming Utility (utils/rename_images.py)
# -----------------------------------------------------------------------------
# Purpose:             Renames and organizes OID images using config-based filename expressions
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Resolves dynamic filename parts from config expressions (e.g., reel, MP, timestamp),
#   renames/copies each image to a flat output folder, updates the OID’s ImagePath and Name
#   fields, and logs all rename operations. Optionally deletes originals after copy.
#
# File Location:        /utils/rename_images.py
# Called By:            tools/rename_and_tag_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    config_loader, arcpy_utils, expression_utils, check_disk_space, path_utils
# Ext. Dependencies:    arcpy, os, shutil, csv, pathlib, typing
#
# Documentation:
#   See: docs/TOOL_GUIDES.md and docs/tools/rename_images.md
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
from typing import Optional

from utils.config_loader import resolve_config
from utils.arcpy_utils import validate_fields_exist, log_message
from utils.expression_utils import resolve_expression
from utils.check_disk_space import check_sufficient_disk_space
from utils.path_utils import get_log_path


def rename_images(
        oid_fc: str,
        delete_originals: bool = False,
        config: Optional[dict] = None,
        config_file: Optional[str] = None,
        messages=None):
    """
    Renames and copies image files for an Oriented Imagery Dataset according to configuration rules.

    All renamed images are placed in a flat directory specified by the configuration, and the `Name` and `ImagePath`
    attributes in the dataset are updated in-place. Optionally deletes original images after copying. A CSV log of
    rename operations is generated. Returns a list of dictionaries with updated image metadata.

    Args:
        oid_fc: Path to the feature class containing image records.
        delete_originals: If True, deletes original image files after copying.
        config: Optional configuration dictionary; if not provided, loaded from file.
        config_file: Optional path to configuration file if `config` is not provided.
        messages: Optional messaging or logging interface.

    Returns:
        A list of dictionaries representing updated image records, including OID, new path, filename, QC flag,
        coordinates, and resolved metadata.
    """
    config = resolve_config(
        config=config,
        config_file=config_file,
        oid_fc_path=oid_fc,
        messages=messages,
        tool_name="rename_images")

    folders = config.get("image_output", {}).get("folders", {})
    fmt = config.get("image_output", {}).get("filename_settings", {}).get("format", "{Name}.jpg")
    parts = config.get("image_output", {}).get("filename_settings", {}).get("parts", {})

    required_fields = ["OID@", "ImagePath", "Name", "QCFlag", "X", "Y"]
    for expr in parts.values():
        if isinstance(expr, str) and expr.startswith("field."):
            field_name = expr.split(".", 1)[1].split(".", 1)[0]
            if field_name not in required_fields:
                required_fields.append(field_name)

    validate_fields_exist(oid_fc, [f for f in required_fields if f != "OID@"])
    fields = list(dict.fromkeys(required_fields))

    folder_key = folders.get("enhanced")
    check_sufficient_disk_space(oid_fc=oid_fc, folder_key=folder_key, config=config, buffer_ratio=1.1, verbose=True,
                                messages=messages)

    # Prepare output folder
    try:
        parent_dir = folders["parent"]
        renamed_dir = folders["renamed"]
    except KeyError as ke:
        log_message(f"Required image_output.folders key missing: {ke}", messages, level="error",
                    error_type=KeyError, config=config)
        raise
    output_dir = Path(config["__project_root__"]) / parent_dir / renamed_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify write permissions
    try:
        test_file = output_dir / ".permission_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        log_message(f"❌ Output directory is not writable: {output_dir}. Error: {e}", messages, level="error",
                    error_type=PermissionError, config=config)

    log_message(f"[DEBUG] Writing renamed images to: {output_dir}", messages, level="debug", config=config)

    # Prepare CSV log
    rename_log_path = get_log_path("rename_log", config)
    log_message(f"[DEBUG] Writing rename log to: {rename_log_path}", messages, level="debug", config=config)
    log_fields = ["OID", "OriginalPath", "OriginalName", "NewPath", "NewName"]
    log_rows = []

    updated_images = []

    with arcpy.da.UpdateCursor(oid_fc, fields) as cursor:
        for row in cursor:
            row_dict = dict(zip(fields, row))
            oid = row_dict["OID@"]
            old_path = Path(row_dict["ImagePath"])

            if not old_path.is_file():
                log_message(f"⚠️ Image path missing or invalid for OID {oid}: {old_path}", messages,
                            level="warning", config=config)
                continue

            try:
                resolved = {
                    key: resolve_expression(expr, row=row_dict, config=config)
                    for key, expr in parts.items()
                }

                filename = fmt.format(**resolved)
                new_path = output_dir / filename

                base, ext = os.path.splitext(filename)
                counter = 1
                while new_path.is_file():
                    filename = f"{base}_v{counter}{ext}"
                    new_path = output_dir / filename
                    counter += 1

                try:
                    shutil.copy2(old_path, new_path)
                    if delete_originals:
                        os.remove(old_path)
                except (PermissionError, OSError) as e:
                    log_message(f"⚠️ Failed to copy/delete file for OID {oid}: {e}", messages, level="warning",
                                config=config)
                    continue

                row_dict["ImagePath"] = str(new_path)
                row_dict["Name"] = filename
                cursor.updateRow([row_dict.get(f) for f in fields])

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

                log_message(f"✅ Renamed OID {oid} ➜ {filename}", messages, level="debug", config=config)
            except KeyError as ke:
                log_message(f"⚠️ Failed to resolve filename part for OID {oid}: Missing key {ke}", messages,
                            level="warning", config=config)
                continue
            except ValueError as ve:
                log_message(f"⚠️ Failed to resolve filename part for OID {oid}: Invalid value: {ve}", messages,
                            level="warning", config=config)
                continue
            except Exception as e:
                log_message(f"⚠️ Failed to rename/copy for OID {oid}: {e}", messages, level="warning", config=config)
                continue

    # Write rename log CSV
    try:
        with open(rename_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        log_message(f"📝 Rename log saved to: {rename_log_path}", messages, config=config)
    except PermissionError as e:
        log_message(f"❌ Failed to write rename log: {e}", messages, level="warning", config=config)

    log_message(f"✅ {len(updated_images)} image(s) renamed and attributes updated.", messages, config=config)
    return updated_images
