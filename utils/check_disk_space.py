# =============================================================================
# 💾 Disk Space Checker (utils/check_disk_space.py)
# -----------------------------------------------------------------------------
# Purpose:             Verifies available disk space before performing image enhancement or export
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Estimates required disk space using the size of the base imagery folder (original or enhanced),
#   applies a configurable buffer ratio, and compares it against available space on the drive.
#   Prevents out-of-space failures during image-intensive steps in the pipeline.
#
# File Location:        /utils/check_disk_space.py
# Called By:            tools/enhance_images_tool.py, tools/rename_images_tool.py
# Int. Dependencies:    arcpy_utils, path_utils
# Ext. Dependencies:    arcpy, os, shutil
#
# Documentation:
#   See: docs/UTILITIES.md and docs/tools/enhance_images.md
#
# Notes:
#   - Automatically resolves the base folder from any image path in the OID
#   - Raises RuntimeError if insufficient space is detected
# =============================================================================

import arcpy
import os
import shutil
from utils.arcpy_utils import log_message
from utils.path_utils import get_image_folder_path


def check_sufficient_disk_space(oid_fc, folder_key, config=None, buffer_ratio=1.1, verbose=False, messages=None):
    """
    Checks if sufficient disk space is available for image processing using a representative folder
    (e.g., 'original', 'enhanced') derived from the OID ImagePath field.

    This function:
    - Retrieves a sample image path from the OID feature class
    - Resolves the configured image folder base (e.g., 'original') using the path and config
    - Estimates total folder size recursively and applies a safety buffer
    - Compares the required space against available disk space on the corresponding drive

    Args:
        oid_fc (str): Path to the Oriented Imagery Dataset feature class (must include 'ImagePath' field).
        folder_key (str): Key to the configured image folder in config['image_output']['folders'],
                          typically 'original' or 'enhanced'.
        config (dict, optional): Full resolved configuration dictionary.
        buffer_ratio (float, optional): Safety multiplier applied to estimated size. Default is 1.1.
        verbose (bool, optional): If True, logs folder and size details.
        messages (list, optional): ArcGIS message interface or CLI-compatible logger.

    Raises:
        ValueError: If no valid image path is found or the target folder cannot be resolved.
        FileNotFoundError: If the resolved folder path does not exist.
        RuntimeError: If available disk space is insufficient for the estimated need.

    Returns:
        bool: True if sufficient space is available; False if an early failure condition is met.
    """


    if config is None:
        config = {}

    # Read from config if available
    if config:
        disk_cfg = config.get("disk_space", {})
        if not disk_cfg.get("check_enabled", True):
            if verbose:
                log_message("Disk space check is disabled via config.", messages, config=config)
            return True
        buffer_ratio = disk_cfg.get("min_buffer_ratio", buffer_ratio)

    # Get one valid ImagePath from the FC
    with arcpy.da.SearchCursor(oid_fc, ["ImagePath"]) as cursor:
        image_path = next((row[0] for row in cursor if row[0]), None)

    if not image_path:
        log_message("No valid ImagePath found in the OID feature class.", messages, level="error",
                    error_type=ValueError, config=config)
        return False

    base_dir, drive_root = get_image_folder_path(sample_path=image_path, folder_key=folder_key, config=config, messages=messages)
    if not base_dir:
        log_message(f"❌ Could not locate folder '{folder_key}' in the image path.", messages,
                    level="error", error_type=ValueError, config=config)
        return False

    if not os.path.exists(base_dir):
        log_message(f"Base folder not found: {base_dir}", messages, level="error", error_type=FileNotFoundError,
                    config=config)

    # Calculate total existing size
    def get_folder_size(path):
        """
        Calculates the total size of all files within a directory, including subdirectories.
        
        Args:
            path: Path to the directory whose total file size will be computed.
        
        Returns:
            The cumulative size in bytes of all files contained in the directory and its subdirectories.
        """
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total

    folder_size = get_folder_size(base_dir)
    estimated_required = int(folder_size * buffer_ratio)

    # Get available space on drive
    free_space = shutil.disk_usage(drive_root).free

    if verbose:
        log_message(f"Checking disk space on drive: {drive_root}", messages, config=config)
        log_message(f"Checking space in folder: {base_dir}", messages, config=config)
        log_message(f"Base folder size (used): {folder_size / 1e9:.2f} GB", messages, config=config)
        log_message(f"Estimated required: {estimated_required / 1e9:.2f} GB (with buffer)", messages, config=config)
        log_message(f"Available: {free_space / 1e9:.2f} GB", messages, config=config)

    if free_space < estimated_required:
        log_message(f"❌ Insufficient disk space.\n"
                    f"Needed (with buffer): {estimated_required / 1e9:.2f} GB\n"
                    f"Available: {free_space / 1e9:.2f} GB",
                    messages, level="error", error_type=RuntimeError, config=config)

    return True
