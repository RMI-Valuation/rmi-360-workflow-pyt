# =============================================================================
# 🏷️ EXIF Metadata Utilities (exif_utils.py)
# -----------------------------------------------------------------------------
# Purpose:     Handles metadata copying between images using ExifTool subprocess calls.
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
#
# Description:
#   This utility wraps ExifTool to copy all EXIF tags from a source image to an enhanced
#   version after processing. It suppresses command line output and supports Windows subprocess hiding.
#
# Dependencies:
#   - subprocess, os, pathlib
# =============================================================================
import os
import subprocess
from pathlib import Path


def copy_exif_metadata(
    original: Path,
    enhanced: Path,
    exiftool_path: str = "exiftool"
) -> bool:
    """
    Copies all EXIF metadata from the original image to the enhanced image using ExifTool.

    Executes a subprocess call to transfer all tags from the source to the target image.
    Suppresses output and hides the subprocess window on Windows.

    Args:
        original (Path): Source image with the desired EXIF metadata.
        enhanced (Path): Target image to receive copied metadata.
        exiftool_path (str): Path to the ExifTool executable (default is "exiftool").

    Returns:
        bool: True if metadata copy succeeded, False if the subprocess failed.
    """
    try:
        # Suppress console window on Windows
        startupinfo = None
        if os.name == "nt":  # Windows only
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(
            [exiftool_path, "-TagsFromFile", str(original), "-overwrite_original", "-all:all", str(enhanced)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        return True
    except subprocess.CalledProcessError:
        return False