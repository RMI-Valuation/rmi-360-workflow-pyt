# =============================================================================
# 💾 Disk Space Checker (utils/check_disk_space.py)
# -----------------------------------------------------------------------------
# Purpose:             Verifies available disk space before performing image enhancement or export
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.1
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Estimates required disk space using the size of the base imagery folder (original or enhanced),
#   applies a configurable buffer ratio, and compares it against available space on the drive.
#   Prevents out-of-space failures during image-intensive steps in the pipeline.
#
# File Location:        /utils/check_disk_space.py
# Called By:            tools/enhance_images_tool.py, tools/rename_images_tool.py
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    arcpy, os, shutil, pathlib, typing
#
# Documentation:
#   See: docs_legacy/UTILITIES.md and docs_legacy/tools/enhance_images.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Automatically resolves the base folder from any image path in the OID
#   - Raises RuntimeError if insufficient space is detected
# =============================================================================

from __future__ import annotations
import arcpy
import os
import shutil
from pathlib import Path
from typing import Optional


def find_base_dir(dir_path: str, token: str) -> Optional[str]:
    """
    Finds the base directory in dir_path containing the token (case-insensitive).
    Returns the path up to and including the token, or None if not found.
    """
    idx = dir_path.lower().find(token.lower())
    return dir_path[: idx + len(token)] if idx != -1 else None


def get_folder_size(path: str, config: "ConfigManager") -> int:
    """
    Calculates the total size of all files within a directory, including subdirectories.

    Args:
        path: Path to the directory whose total file size will be computed.
        config (ConfigManager): ConfigManager instance.

    Returns:
        The cumulative size in bytes of all files contained in the directory and its subdirectories.
    """
    total = 0
    all_files = list(Path(path).rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    with config.get_progressor(total=len(files_only), label="Calculating folder size...") as prog:
        for i, file in enumerate(files_only, start=1):
            total += file.stat().st_size
            prog.update(i)
    return total


def check_sufficient_disk_space(
    oid_fc: str,
    cfg: "ConfigManager",
    cursor_factory=None,
    disk_usage_func=None,
    folder_size_func=None
) -> bool:
    """
    Checks if sufficient disk space is available for image operations on an Oriented Imagery Dataset.

    Estimates the required disk space by calculating the size of the relevant 'original' or 'enhanced' image directory
    and applying a safety buffer. Raises an error if the available space on the drive is less than the estimated
    requirement.

    Args:
        oid_fc: Path to the Oriented Imagery Dataset feature class.
        cfg (ConfigManager): ConfigManager instance.
        cursor_factory: Optional; factory for creating a SearchCursor for testing.
        disk_usage_func: Optional; function to get disk usage for testing.
        folder_size_func: Optional; function to calculate folder size for testing.

    Raises:
        ValueError: If no valid image path is found or if the image path does not include expected folder names.
        FileNotFoundError: If the base image directory does not exist.
        RuntimeError: If available disk space is insufficient.

    Returns:
        True if sufficient disk space is available.
    """
    logger = cfg.get_logger()

    if not cfg.get("disk_space.check_enabled", True):
        logger.info("Disk space check is disabled via config.")
        return True

    buffer_ratio: float = cfg.get("disk_space.min_buffer_ratio", 1.1)

    # Dependency injection for testability
    cursor_factory = cursor_factory or (lambda fc, fields: arcpy.da.SearchCursor(fc, fields))
    disk_usage_func = disk_usage_func or shutil.disk_usage
    folder_size_func = folder_size_func or get_folder_size

    # Get one valid ImagePath from the FC
    with cursor_factory(oid_fc, ["ImagePath"]) as cursor:
        image_path = next((row[0] for row in cursor if row[0]), None)

    if not image_path:
        logger.error(f"No valid ImagePath found in the OID feature class: {oid_fc}", error_type=ValueError, indent=1)

    # Determine target folder from ImagePath
    target_dir = os.path.dirname(image_path)
    drive_root = Path(target_dir).anchor

    original_folder = cfg.get("image_output.folders.original", "original").lower()
    enhanced_folder = cfg.get("image_output.folders.enhanced", "enhanced").lower()

    base_dir = find_base_dir(target_dir, original_folder) or find_base_dir(target_dir, enhanced_folder)

    if not base_dir:
        logger.error(
            f"ImagePath does not include '{original_folder}' or '{enhanced_folder}' folder. Path: {target_dir}",
            error_type=ValueError, indent=1)

    if not os.path.exists(base_dir):
        logger.error(f"Base folder not found: {base_dir}", error_type=FileNotFoundError, indent=1)

    folder_size = folder_size_func(base_dir, cfg)
    estimated_required = int(folder_size * buffer_ratio)
    free_space = disk_usage_func(drive_root).free

    logger.debug(f"Checking disk space on drive: {drive_root}", indent=1)
    logger.debug(f"Checking space in folder: {base_dir}", indent=1)
    logger.debug(f"Base folder size (used): {folder_size / 1e9:.2f} GB", indent=1)
    logger.debug(f"Estimated required: {estimated_required / 1e9:.2f} GB (with buffer)", indent=1)
    logger.debug(f"Available: {free_space / 1e9:.2f} GB", indent=1)

    if free_space < estimated_required:
        logger.error("Insufficient disk space.", indent=1)
        logger.error(f"Needed (with buffer): {estimated_required / 1e9:.2f} GB", indent=2)
        logger.error(f"Available: {free_space / 1e9:.2f} GB", indent=2)
        logger.error("Cannot continue.", indent=1, error_type=RuntimeError)

    return True
