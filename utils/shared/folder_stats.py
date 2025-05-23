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
#   cumulative size. Formats byte totals into human-readable strings. Used for
#   reporting logs and disk space estimation in enhancement and renaming steps.
#
# File Location:        /utils/folder_stats.py
# Called By:            utils/check_disk_space.py, reporting
# Int. Dependencies:    None
# Ext. Dependencies:    os, pathlib, typing
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
import math
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


def format_size(num_bytes: int) -> str:
    """Return a human-readable string for a file size (e.g., 1.5 GiB)."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if num_bytes < 1024:
        return f"{num_bytes:.1f} B"
    # Calculate the appropriate unit using logarithm base 1024
    exponent = int(math.log(num_bytes, 1024))
    suffixes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
    exponent = min(exponent, len(suffixes) - 1)
    quotient = num_bytes / (1024 ** exponent)
    return f"{quotient:.1f} {suffixes[exponent]}"


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

    def safe_stat(f):
        try:
            return f.stat().st_size
        except OSError:
            return 0

    with ThreadPoolExecutor() as executor:
        total_size_concurrent = sum(executor.map(safe_stat, jpg_files))
    return len(jpg_files), format_size(total_size_concurrent)
