# =============================================================================
# 🗂️ Log Path Builder (utils/path_utils.py)
# -----------------------------------------------------------------------------
# Purpose:             Resolves and constructs log file paths based on config expressions and filename keys
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Dynamically constructs full file paths for log files using `logs.path`, `logs.prefix`, and log keys
#   from the configuration. Supports expression-based prefixes, auto-directory creation, and error handling
#   for malformed log configuration or path resolution.
#
# File Location:        /utils/path_utils.py
# Called By:            multiple tools and log-writing utilities
# Int. Dependencies:    expression_utils
# Ext. Dependencies:    os, typing
#
# Documentation:
#   See: docs/UTILITIES.md
#
# Notes:
#   - Automatically inserts dynamic prefix (if defined) before filename
#   - Ensures log directory exists or raises descriptive error
# =============================================================================

import os
from typing import Optional
from os.path import splitdrive
from pathlib import Path
from utils.expression_utils import resolve_expression


def get_log_path(log_key: str, config: dict) -> str:
    """
    Constructs and returns the full file path for a log file based on the given log key and configuration.
    
    Retrieves the log directory and filename from the configuration, optionally prepending a dynamically resolved
    prefix to the filename. Ensures the log directory exists before returning the complete path.
    
    Args:
        log_key: The key identifying the log file in the configuration.
        config: The configuration dictionary containing log settings.
    
    Returns:
        The absolute path to the log file, with an optional prefix if specified in the configuration.
    
    Raises:
        ValueError: If the log filename is not a string, if the prefix resolves to an unsupported type, or if prefix
        resolution fails.
    """
    logs_cfg = config.get("logs", {})
    log_dir = os.path.join(config.get("__project_root__", "."), logs_cfg.get("path", "logs"))

    log_file = logs_cfg.get(log_key)
    if log_file is None:
        raise ValueError(f"logs.{log_key} is not defined in the configuration")
    if not isinstance(log_file, str):
        raise ValueError(f"logs.{log_key} must be a string filename, got {type(log_file).__name__}")

    # Get and resolve prefix (if any)
    prefix_expr = logs_cfg.get("prefix")
    prefix_str: Optional[str] = None
    if prefix_expr:
        try:
            prefix_str = resolve_expression(prefix_expr, config=config)
            if not isinstance(prefix_str, (str, int, float)):
                raise ValueError(f"logs.prefix resolved to unsupported type: {type(prefix_str).__name__}")
            prefix_str = str(prefix_str)
        except Exception as e:
            raise ValueError(f"Failed to resolve logs.prefix expression '{prefix_expr}': {str(e)}") from e

    # Insert prefix into filename if available
    if prefix_str and prefix_str.strip():
        base, ext = os.path.splitext(log_file)
        log_file = f"{prefix_str}_{base}{ext}"

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Failed to create log directory '{log_dir}'. {e}") from e

    return os.path.join(log_dir, log_file)


def get_image_folder_path(
        sample_path: str,
        folder_key: str,
        config: dict,
        messages=None) -> tuple[str, str] | tuple[None, None]:
    """
    Resolves the base path to the configured image folder (e.g., 'original', 'enhanced')
    by inspecting the provided image path.

    Args:
        sample_path (str): Path to an image (or directory containing images).
        folder_key (str): Key in config.image_output.folders (e.g., 'original', 'enhanced').
        config (dict): Full resolved config dictionary.
        messages (list): Optional logger for warnings or errors.

    Returns:
        (base_path, drive_root): Tuple of full folder path and its drive root.
    """
    def _find_base(path: str, token: str) -> Optional[str]:
        idx = path.lower().find(token.lower())
        return path[: idx + len(token)] if idx != -1 else None

    img_folders = config.get("image_output", {}).get("folders", {})
    target_token = img_folders.get(folder_key, folder_key)

    # Try direct path match first
    base_path = _find_base(sample_path, target_token)
    if base_path and os.path.exists(base_path):
        drive_root = splitdrive(base_path)[0] + os.sep
        return base_path, drive_root

    # Fallback: scan from drive root
    drive_root = splitdrive(sample_path)[0] + os.sep
    if os.path.exists(drive_root):
        for root, dirs, _ in os.walk(drive_root):
            for d in dirs:
                if d.lower() == target_token.lower():
                    fallback_path = os.path.join(root, d)
                    return fallback_path, drive_root

    if messages is not None:
        messages.append(f"❌ Could not resolve path to '{folder_key}' folder from {sample_path}")

    return None, None


def get_image_base_path(sample_path, folder_key, config, messages=None):
    base_path, _ = get_image_folder_path(sample_path, folder_key, config, messages)
    return base_path


def get_enhancement_profile_path(config: dict) -> Path:
    filename = config.get("image_enhancement", {}).get("profile_json", "enhancement_profile.json")
    return Path(config.get("__project_root__", ".")) / filename
