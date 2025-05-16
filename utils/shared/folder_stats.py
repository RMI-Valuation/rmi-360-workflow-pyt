# =============================================================================
# 📁 Folder Statistics Utility (utils/folder_stats.py)
# -----------------------------------------------------------------------------
# Purpose:             Calculates file counts and total size of image directories for reporting
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Recursively walks a given directory to count image files (default: .jpg) and compute their
#   cumulative size. Formats byte totals into human-readable strings using `humanize`. Used for
#   reporting logs and disk space estimation in enhancement and renaming steps.
#
# File Location:        /utils/folder_stats.py
# Called By:            utils/check_disk_space.py, reporting
# Int. Dependencies:    None
# Ext. Dependencies:    os, humanize, pathlib, typing
#
# Documentation:
#   See: docs_legacy/UTILITIES.md
#   (Ensure this doc is current; update if needed.)
#
# Notes:
#   - Recursively matches any extensions provided via argument (default: ['.jpg'])
#   - Skips invalid or missing folders gracefully (returns 0 count and "0 B")
# =============================================================================

import os
import humanize
from pathlib import Path
from typing import List, Optional, Tuple


def folder_stats(path: str, extensions: Optional[List[str]] = None) -> Tuple[int, str]:
    """
    Calculate statistics for image files in a directory (recursive, case-insensitive).

    Args:
        path: Path to the directory to analyze.
        extensions: List of file extensions to include (default: ['.jpg']). Extensions are matched case-insensitively.

    Returns:
        A tuple (number_of_files, human_readable_size):
            - number_of_files: Count of matching files.
            - human_readable_size: Total size in human-readable format.
    """
    if not path or not os.path.exists(path):
        return 0, "0 B"

    if extensions is None:
        extensions = ['.jpg']
    # Normalize extensions to lower-case
    norm_exts = {e.lower() for e in extensions}
    p = Path(path)
    jpg_files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in norm_exts]
    total_size = 0
    for f in jpg_files:
        try:
            total_size += f.stat().st_size
        except OSError:
            continue  # Skip files that can't be accessed
    return len(jpg_files), humanize.naturalsize(total_size, binary=True)
