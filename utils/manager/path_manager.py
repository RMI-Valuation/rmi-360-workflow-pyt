# =============================================================================
# 📂 Path Manager Utility (utils/manager/path_manager.py)
# -----------------------------------------------------------------------------
# Purpose:             Resolves all script, config, image, log, and report paths for the RMI 360 Workflow Toolbox.
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-10
# Last Updated:        2025-05-15
#
# Description:
#   Centralizes and simplifies path resolution across the Python Toolbox project.
#   Supports both static paths (e.g. templates, lambdas) and dynamic project-specific
#   paths (e.g. logs, pano image folders). Fully supports config-based overrides and expression resolution.
#
# File Location:        /utils/manager/path_manager.py
# Called By:            ArcGIS tools, utility modules, CLI scripts
# Int. Dependencies:    utils/expression_utils
# Ext. Dependencies:    pathlib, yaml, os, subprocess
#
# Documentation:
#   See: docs/source/path_manager.rst
#   (Ensure this doc is current; update if needed.)
#
# Notes:
#   - Script base is resolved to the repo root unless explicitly passed.
#   - Verifies presence of rmi_360_workflow.pyt to validate repo structure.
#   - Integrates with ConfigManager and LogManager for coordinated path management.
# =============================================================================

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import yaml
from typing import Any, List, Optional, Union


__all__ = ["PathManager"]

class PathManager:
    """
    Central path manager for RMI 360 Workflow.

    Resolves all paths related to:
    - 🗂 Script structure (configs/, templates/, lambdas/)
    - 📁 Project outputs (logs/, backups/, panos/, renamed/)
    - 🧩 Schema components (GDBs, registries)
    - 🔧 Executables (ExifTool, Mosaic Processor)

    Supports config overrides, expression-based log prefixes, and runtime executable checks.
    """
    def __init__(self, project_base: Path, config: Union[dict, ConfigManager] = None, script_base: Path = None):
        """
        Initialize the PathManager.

        Args:
            project_base (Path): The base path of the active project.
            config (dict or ConfigManager, optional): The loaded configuration, either raw or wrapped.
            script_base (Path, optional): The root folder where rmi_360_workflow.pyt lives.
                                          Defaults to the parent of the calling file's parent.
        """
        self.script_base: Path = script_base or Path(__file__).resolve().parents[2]
        self.project_base: Path = Path(project_base).resolve()
        if config is not None and config.__class__.__name__ == "ConfigManager":
            self.cfg = config.raw
        else:
            self.cfg = config or {}

        if not (self.script_base / "rmi_360_workflow.pyt").exists():
            raise ValueError(f"Resolved script base {self.script_base} does not contain rmi_360_workflow.pyt")

    # --- Script-Level Paths ---
    @property
    def templates(self):
        """Path to template directory (configurable via oid_schema_template.template.templates_dir)."""
        folder = self._get_config_value("oid_schema_template.template.templates_dir", default="templates")
        return self.script_base / folder

    @property
    def configs(self):
        """Path to configs directory under script base."""
        return self.script_base / "configs"

    @property
    def lambdas(self):
        """Path to aws_lambdas directory under script base."""
        return self.script_base / "aws_lambdas"

    @property
    def primary_config_path(self):
        """Returns the path to config.yaml under script base."""
        return self.configs / "config.yaml"

    @property
    def fallback_config_path(self):
        """Returns the path to config.sample.yaml under script base."""
        return self.configs / "config.sample.yaml"

    # --- Project-Level Paths ---
    @property
    def backups(self):
        """Path to project backups folder (configurable via orchestrator.backup_folder)."""
        folder = self._get_config_value("orchestrator.backup_folder", default="backups")
        return self.project_base / folder

    @property
    def backup_gdb(self):
        """Path to project backup fgdb (configurable via orchestrator.oid_backup_fgdb)."""
        folder = self._get_config_value("orchestrator.oid_backup_fgdb")
        return self.backups / folder

    @property
    def logs(self):
        """Path to logs folder (configurable via logs.path)."""
        folder = self._get_config_value("logs.path", default="logs")
        return self.project_base / folder

    @property
    def report(self):
        """Path to reports folder (configurable via logs.report_path)."""
        folder = self._get_config_value("logs.report_path", default="report")
        return self.project_base / folder

    # --- Image Folders ---
    @property
    def panos(self):
        """Path to the base panos folder (configurable via image_output.folders.parent)."""
        folder = self._get_config_value("image_output.folders.parent", default="panos")
        return self.project_base / folder

    @property
    def original(self):
        """Path to the original images folder (configurable via image_output.folders.original)."""
        folder = self._get_config_value("image_output.folders.original", default="original")
        return self.panos / folder

    @property
    def enhanced(self):
        """Path to the enhanced images folder (configurable via image_output.folders.enhanced)."""
        folder = self._get_config_value("image_output.folders.enhanced", default="enhance")
        return self.panos / folder

    @property
    def renamed(self):
        """Path to the renamed/final images folder (configurable via image_output.folders.renamed)."""
        folder = self._get_config_value("image_output.folders.renamed", default="final")
        return self.panos / folder

    @property
    def oid_field_registry(self):
        """Path to the ESRI OID field registry YAML (from oid_schema_template.esri_default.field_registry)."""
        rel_path = self._get_config_value(
            "oid_schema_template.esri_default.field_registry", default="configs/esri_oid_fields_registry.yaml"
        )
        return (self.script_base / rel_path).resolve()

    @property
    def oid_schema_gdb(self):
        """Path to the OID schema GDB (inside templates dir, resolved from config)."""
        gdb_name = self._get_config_value("oid_schema_template.template.gdb_path", default="templates.gdb")
        return self.templates / gdb_name

    @property
    def oid_schema_template_name(self) -> str:
        """Returns the configured name of the OID schema template table/feature class."""
        return self._get_config_value("oid_schema_template.template.template_name", default="oid_schema_template")

    @property
    def oid_schema_template_path(self) -> str:
        """Returns the configured path of the OID schema template table/feature class."""
        return self.oid_schema_gdb / self.oid_schema_template_name

    @property
    def geoloc500_config_path(self):
        """Resolved geolocation500 config path from config (relative to script base)."""
        if self._get_config_value("geocoding.exiftool_geodb") == "geolocation500":
            rel_path = self._get_config_value("geocoding.geoloc500_config_path",
                                              default="templates/exiftool/geolocation500.config")
            return (self.script_base / rel_path).resolve()
        return None

    @property
    def geocustom_config_path(self):
        """Resolved geocustom config path from config (relative to script base)."""
        if self._get_config_value("geocoding.exiftool_geodb") == "geocustom":
            rel_path = self._get_config_value("geocoding.geocustom_config_path",
                                              default="templates/exiftool/geocustom.config")
            return (self.script_base / rel_path).resolve()
        return None

    # --- Executables ---
    @property
    def exiftool_exe(self):
        """Path to ExifTool executable (default: 'exiftool')."""
        return self._get_config_value("executables.exiftool.exe_path", default="exiftool")

    @property
    def mosaic_processor_exe(self):
        """Path to Mosaic Processor executable (no default)."""
        return self._get_config_value("executables.mosaic_processor.exe_path")

    @property
    def mosaic_processor_grp(self):
        """
        Returns the resolved path to the Mosaic GRP calibration file, if configured.
        Returns None if unset or misconfigured.
        """
        val = self._get_config_value("executables.mosaic_processor.grp_path")
        if isinstance(val, (str, Path)):
            return Path(val).resolve()
        return None

    @property
    def mosaic_processor_cfg(self):
        """Path to Mosaic Processor config file (no default)."""
        return self._get_config_value("executables.mosaic_processor.cfg_path")

    @property
    def lambda_pm_path(self):
        """Path to the lambda_progress_monitor.py."""
        progress_monitor_rel_path = "aws_lambdas/lambda_progress_monitor.py"
        return self.script_base / progress_monitor_rel_path

    @property
    def lambda_dr_path(self):
        """Path to the disable_rule.py."""
        deactivator_rel_path = "aws_lambdas/disable_rule.py"
        return self.script_base / deactivator_rel_path

    def get_log_file_path(self, log_key: str, cfg: Optional["ConfigManager"] = None) -> Path:
        from utils.shared.expression_utils import resolve_expression
        """
        Constructs the full path for a log file with optional prefix.

        Builds a log file path by combining:
        1. The base logs directory (from logs.path config or "logs" default)
        2. An optional prefix (from logs.prefix config, can be an expression)
        3. The specific log filename (from logs.<log_key> config)

        The prefix is resolved using expression_utils if it contains template syntax.

        Args:
            log_key (str): The name of the log config key (e.g., 'process_log', 'enhance_log').
            cfg:

        Returns:
            Path: A fully resolved Path object pointing to the log file.

        Raises:
            ValueError: If logs.<log_key> is not defined or not a string in the configuration.
            ValueError: If logs.prefix is defined but cannot be resolved to a string/int/float.

        Examples:
            pm.get_log_file_path("enhance_log")
                Path("/project/logs/enhance_log.txt")  # With no prefix

            # With prefix "20230510" from logs.prefix config
            pm.get_log_file_path("enhance_log")
                Path("/project/logs/20230510_enhance_log.txt")
        """
        logs_cfg = self.cfg.get("logs", {})
        log_dir = self.logs
        log_file = logs_cfg.get(log_key)

        if not isinstance(log_file, str):
            raise ValueError(f"logs.{log_key} must be a string filename defined in the configuration.")

        # Resolve optional prefix expression
        prefix_expr = logs_cfg.get("prefix")
        prefix = ""
        if prefix_expr:
            try:
                resolved = resolve_expression(prefix_expr, cfg=cfg)
                if not isinstance(resolved, (str, int, float)):
                    raise ValueError(f"logs.prefix must resolve to a string/int/float, got {type(resolved).__name__}")
                prefix = str(resolved).strip()
            except Exception as e:
                raise ValueError(f"Failed to resolve logs.prefix '{prefix_expr}': {str(e)}") from e

        # Inject prefix into filename
        if prefix:
            base, ext = os.path.splitext(log_file)
            log_file = f"{prefix}_{base}{ext}"

        return log_dir / log_file

    def validate_mosaic_config(self, messages: Optional[List] = None, config: Optional[dict] = None, 
                          log_func: Optional[callable] = None) -> bool:
        """
        Validates mosaic processor configuration and logs any errors found.

        Checks if the mosaic processor executable and GRP calibration file are properly 
        configured and accessible. Logs errors through the provided log_func if issues are found.

        Args:
            messages (List, optional): ArcGIS message object for logging. Defaults to None.
            config (dict, optional): Config dict for contextual logging. Defaults to None.
            log_func (callable, optional): Logging function that accepts (message, messages, level, config).
                                          Defaults to None.

        Returns:
            bool: True if configuration is valid, False if any issues were found
        """
        if log_func is None:
            # If no log function is provided, we can't log errors
            return bool(self.mosaic_processor_exe and 
                       self.mosaic_processor_grp and 
                       self.mosaic_processor_grp.exists())

        valid = True

        if not self.mosaic_processor_exe:
            log_func("❌ Mosaic Processor executable missing from config.", messages, level="error", config=config)
            valid = False

        if not self.mosaic_processor_grp or not self.mosaic_processor_grp.exists():
            log_func("❌ GRP file is missing or not found.", messages, level="error", config=config)
            valid = False

        return valid

    def check_exiftool_available(self) -> bool:
        """
        Checks if ExifTool is available and runnable.

        Tests if the ExifTool executable configured in executables.exiftool.exe_path
        can be run successfully. Uses _is_executable_available with default arguments.

        Returns:
            bool: True if ExifTool is available and runs without error, False otherwise
        """
        return self._is_executable_available(self.exiftool_exe)

    def check_mosaic_processor_available(self) -> bool:
        """
        Checks if Mosaic Processor is available and runnable.

        Tests if the Mosaic Processor executable configured in executables.mosaic_processor.exe_path
        can be run successfully. Uses _is_executable_available with default arguments.

        Returns:
            bool: True if Mosaic Processor is available and runs without error, False otherwise
        """
        return self._is_executable_available(self.mosaic_processor_exe)

    # --- Helpers ---
    def _get_config_value(self, dotted_key: str, default: Any = None) -> Any:
        """
        Resolve a nested key in the config dict using dot notation.

        Args:
            dotted_key (str): The key path in dot notation (e.g., "logs.path")
            default (any, optional): Value to return if the key is not found. Defaults to None.

        Returns:
            any: The value at the specified key path, or the default if not found

        Examples:
            pm._get_config_value("logs.path", "logs")
                "custom_logs"  # If config has {"logs": {"path": "custom_logs"}}
            pm._get_config_value("nonexistent.key", "default_value")
                "default_value"
        """
        keys = dotted_key.split(".")
        val = self.cfg
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    @staticmethod
    def _is_executable_available(exe_path: str, test_args: list[str] = None) -> bool:
        """
        Returns True if the given executable runs without error.

        Args:
            exe_path (str): Path to the executable to test
            test_args (list[str], optional): Arguments to pass to the executable. 
                                            Defaults to ['-ver'] which works for many CLI tools.

        Returns:
            bool: True if executable is available and runs without error, False otherwise
        """
        if test_args is None:
            test_args = ["-ver"]

        # Handle platform-specific issues
        startupinfo = None
        if os.name == 'nt':  # Windows
            # Prevent command window from showing
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            result = subprocess.run(
                [exe_path] + test_args, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False,
                startupinfo=startupinfo,
                timeout=5  # Add timeout to prevent hanging
            )
            return result.returncode == 0
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            return False

    @classmethod
    def from_config_file(cls, config_path: Path, project_base: Path, script_base: Path = None) -> "PathManager":
        """
        Load configuration from a YAML file and initialize a PathManager.

        Args:
            config_path (Path): Path to the YAML config file.
            project_base (Path): The base path of the project.
            script_base (Path, optional): Override script base.

        Returns:
            PathManager: Initialized instance with config loaded from the specified file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the config file contains invalid YAML.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(project_base=project_base, config=config, script_base=script_base)
