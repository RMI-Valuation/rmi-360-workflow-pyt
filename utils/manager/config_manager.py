# =============================================================================
# 🧠 Config Manager Utility (utils/manager/config_manager.py)
# -----------------------------------------------------------------------------
# Purpose:             Loads, validates, and manages access to YAML configuration and project settings for the RMI 360 Workflow Toolbox.
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-11
# Last Updated:        2025-05-15
#
# Description:
#   Centralized configuration manager for the toolbox. Wraps access to key config
#   values, handles expression resolution, schema validation, and integrates tightly
#   with LogManager and PathManager. Supports both CLI and ArcGIS Pro tools.
#
# File Location:        /utils/manager/config_manager.py
# Called By:            ArcGIS tools, orchestrators, log initializers
# Int. Dependencies:    utils/manager/path_manager, utils/manager/log_manager, utils/validate_full_config, utils/expression_utils, utils/exceptions, utils/validators
# Ext. Dependencies:    yaml, pathlib, typing, os
#
# Documentation:
#   See: docs_legacy/CONFIG_MANAGER.md
#   (Ensure this doc is current; update if needed.)
#
# Notes:
#   - Automatically populates __project_root__ if project_base is passed
#   - Supports .get(), .resolve(), and .validate() access patterns
#   - Integrates tool-specific validators for robust schema enforcement
# =============================================================================

import os
import yaml
from typing import Any, Optional, Union, Dict, List
from pathlib import Path

from utils.validators.validate_full_config import validate_full_config
from utils.shared.rmi_exceptions import ConfigValidationError
from utils.shared.expression_utils import resolve_expression

from utils.validators import (
    mosaic_processor_validator,
    build_oid_schema_validator,
    create_oid_validator,
    add_images_to_oid_validator,
    assign_group_index_validator,
    calculate_oid_attributes_validator,
    smooth_gps_noise_validator,
    correct_gps_outliers_validator,
    update_linear_and_custom_validator,
    enhance_images_validator,
    rename_images_validator,
    apply_exif_metadata_validator,
    geocode_images_validator,
    build_oid_footprints_validator,
    deploy_lambda_monitor_validator,
    copy_to_aws_validator,
    generate_oid_service_validator
)

SUPPORTED_SCHEMA_VERSIONS = {"1.1.0"}


class ConfigManager:
    """
    Manages configuration for the RMI 360 Imaging Workflow Toolbox.

    This class loads YAML configuration, validates its schema, and exposes helper
    methods for retrieving nested keys, resolving expressions, and integrating with
    logging and path management. Intended for consistent environment setup across tools.

    Attributes:
        _config (dict): The parsed configuration dictionary.
        _config_path (str): Path to the loaded config file.
        _project_base (Path): Project root directory for resolving outputs.
        _paths (PathManager): Path manager instance for resolving file paths.
        _lm (LogManager): Logger instance for logging messages.

    Typical usage:
        cfg = ConfigManager.from_file("path/to/config.yaml", project_base)
        cfg.validate(tool="enhance")
        log = cfg.get_logger()
        path = cfg.paths.logs
    """
    def __init__(self, config: Dict, config_path: Optional[str] = None, project_base: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager from a parsed config dictionary.

        Args:
            config (dict): Parsed YAML configuration.
            config_path (str, optional): Path to the loaded config file.
            project_base (str or Path, optional): Project root directory for resolving outputs.
        """
        self._config = config
        self._config_path = config_path or config.get("__source__")
        self._project_base = Path(project_base).resolve() if project_base else None
        if self._project_base:
            self._config["__project_root__"] = str(self._project_base)
        if self._project_base:
            from utils.manager.path_manager import PathManager
            from utils.manager.log_manager import LogManager
            self._paths = PathManager(project_base=self._project_base, config=self._config)
            self._lm = LogManager(messages=None, config=self._config, path_manager=self._paths)
        else:
            self._paths = None
            self._lm = None

    @classmethod
    def from_file(cls, path: Optional[str] = None, project_base: Optional[Union[str, Path]] = None, *,
                  messages: Optional[list] = None) -> "ConfigManager":
        from utils.manager.log_manager import LogManager
        """
        Class method to load a config file, validate schema version, and return a ConfigManager.

        Args:
            path (str, optional): Path to a config YAML. Uses default fallback if None.
            project_base (str or Path, optional): Path to the root project directory.
            messages (list, optional): Optional list for ArcPy-style messages.

        Returns:
            ConfigManager: Initialized configuration manager.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config file doesn't parse into a dictionary.
            RuntimeError: If the schema version doesn't match the expected version.
        """
        config_path = path or cls._get_default_config_path(messages)
        config = {}
        lm = LogManager(messages=messages, config=config)

        try:
            # Attempt to open and parse the config file
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    config["__source__"] = os.path.abspath(config_path)
                    lm.config = config
            except FileNotFoundError:
                lm.error(f"Config file not found: {config_path}", error_type=FileNotFoundError)
                raise
            except yaml.YAMLError as e:
                lm.error(f"Invalid YAML in config file: {e}", error_type=ValueError)
                raise ValueError(f"Invalid YAML in config file: {e}") from e

            # Validate the parsed config is a dictionary
            if not isinstance(config, dict):
                error_msg = f"Config file did not parse into a dictionary. Got {type(config).__name__} instead."
                lm.error(error_msg, error_type=ValueError)
                raise ValueError(error_msg)

            # Validate schema version
            version = config.get("schema_version")
            if version not in SUPPORTED_SCHEMA_VERSIONS:
                error_msg = f"⚠️ Expected schema_version {SUPPORTED_SCHEMA_VERSIONS}, got {version}"
                lm.error(error_msg, error_type=RuntimeError)
                raise RuntimeError(error_msg)

            if config.get("debug_messages", False):
                lm.debug(f"Loaded config from {config['__source__']}")

            return cls(config, config_path, project_base=project_base)

        except (FileNotFoundError, ValueError, RuntimeError):
            # These exceptions are already logged and formatted above
            raise
        except Exception as e:
            # Catch any other unexpected exceptions
            lm.error(f"Unexpected error loading config: {e}", error_type=RuntimeError)
            raise RuntimeError(f"Unexpected error loading config: {e}") from e

    @staticmethod
    def _get_default_config_path(messages: Optional[list] = None) -> str:
        from utils.manager.log_manager import LogManager
        from utils.manager.path_manager import PathManager
        """
        Resolve default config path using PathManager.

        Checks for 'config.yaml' or falls back to 'config.sample.yaml' in the config's directory.
        Uses a temporary PathManager to locate the config files.

        Args:
            messages (list, optional): Optional list for ArcPy-style messages.

        Raises:
            FileNotFoundError: If neither config.yaml nor config.sample.yaml exists.

        Returns:
            str: Absolute path to the selected config file.
        """
        lm = LogManager(messages=messages, config={})
        script_base = Path(__file__).resolve().parent.parent
        temp_paths = PathManager(project_base=Path.cwd(), config={}, script_base=script_base)
        primary = temp_paths.primary_config_path
        fallback = temp_paths.fallback_config_path

        if primary.is_file():
            return str(primary)
        elif fallback.is_file():
            return str(fallback)
        else:
            lm.error("No config.yaml or config.sample.yaml found in /configs", error_type=FileNotFoundError)
            raise FileNotFoundError("No config.yaml or config.sample.yaml found in /configs")

    def validate(self, tool: Optional[str] = None) -> None:
        """
        Run schema validation on the config, optionally scoped to a tool.

        Validates the configuration against the expected schema. If a tool name is provided,
        only the configuration for that specific tool is validated. Otherwise, the entire
        configuration is validated.

        Args:
            tool (str, optional): Name of the tool section to validate (e.g., "enhance_images").

        Raises:
            ValueError: If the configuration fails validation.
        """
        try:
            if tool:
                self.validate_tool_config(tool)
            else:
                result = validate_full_config(self)
                if result is False:
                    raise ValueError("Full config validation failed.")
        except ConfigValidationError as e:
            raise ValueError(f"Validation failed for tool '{tool}': {e}") from e

    def has_section(self, section: str) -> bool:
        """
        Check if a specific section exists in the configuration.

        Args:
            section (str): The top-level section name to check for.

        Returns:
            bool: True if the section exists, False otherwise.

        Examples:
            cfg.has_section("logs")
                True
            cfg.has_section("nonexistent_section")
                False
        """
        return section in self._config and isinstance(self._config[section], dict)

    def get_sections(self) -> List[str]:
        """
        Get a list of all top-level sections in the configuration.

        Returns:
            List[str]: List of section names that are dictionaries.

        Examples:
            cfg.get_sections()
                ['logs', 'project', 'camera', 'camera_offset', 'spatial_ref', 'executables', ...]
        """
        return [k for k, v in self._config.items() if isinstance(v, dict)]

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Access a nested config key using dot notation.

        Traverses the configuration dictionary using the dot-separated key path.
        Returns the default value if any part of the path is not found.

        Args:
            key_path (str): Dot-separated key path, e.g., 'logs.process_log'.
            default (Any): Value to return if key is not found.

        Returns:
            Any: Retrieved value or default if the key path doesn't exist.

        Examples:
            cfg.get("logs.process_log")
                "process_log.txt"
            cfg.get("nonexistent.key", "fallback")
                "fallback"
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def resolve(self, expr: Union[str, int, float], row: Optional[Dict] = None) -> Any:
        """
        Resolve an expression string using config and optional row context.

        Args:
            expr (str|int|float): Expression to resolve.
            row (dict, optional): Row context for field substitution.

        Returns:
            Any: Resolved expression value.
        """
        return resolve_expression(expr, cfg=self, row=row)

    @property
    def raw(self) -> dict:
        """
        Return the full raw config dictionary.

        Returns:
            dict: The raw parsed configuration.
        """
        return self._config

    @property
    def source_path(self) -> Optional[str]:
        """
        Returns the file path from which the config was loaded.

        Returns:
            str or None: Absolute path to config file or None if unknown.
        """
        return self._config_path

    @property
    def paths(self) -> "PathManager":
        from utils.manager.path_manager import PathManager
        """
        Access the initialized PathManager.

        Raises:
            RuntimeError: If PathManager was not initialized.

        Returns:
            PathManager: Resolved project/script-aware path helper.
        """
        if self._paths is None:
            raise RuntimeError("PathManager was not initialized — project_base is missing.")
        return self._paths

    def get_logger(self, messages: Optional[list] = None) -> "LogManager":
        from utils.manager.log_manager import LogManager
        """
        Return a LogManager instance, optionally updating its message sink.

        If a messages list is provided and a logger already exists, the logger's
        message sink is updated with the new list. If no logger exists yet, a
        RuntimeError is raised.

        Args:
            messages (list, optional): ArcPy-style message list for logging output.

        Returns:
            LogManager: Instance tied to the current config and path.

        Raises:
            RuntimeError: If LogManager was not initialized (project_base is missing).
        """
        if messages and self._lm:
            self._lm.messages = messages
        if not self._lm:
            raise RuntimeError("LogManager was not initialized — project_base is missing.")
        return self._lm

    def get_progressor(self, total: int, label: str = "Processing...", step: int = 1) -> "ProgressorManager":
        from utils.manager.progressor_manager import ProgressorManager
        """
        Returns a ProgressorManager initialized with the active LogManager.

        Args:
            total (int): Total steps for the progressor
            label (str): Label to show for progress bar
            step (int): Step increment

        Returns:
            ProgressorManager: Progress tracker instance
        """
        from utils.manager.progressor_manager import ProgressorManager
        return ProgressorManager(total=total, label=label, step=step, log_manager=self.get_logger())

    TOOL_VALIDATORS = {
        "mosaic_processor": mosaic_processor_validator.validate,
        "build_oid_schema": build_oid_schema_validator.validate,
        "create_oriented_imagery_dataset": create_oid_validator.validate,
        "add_images_to_oid": add_images_to_oid_validator.validate,
        "assign_group_index": assign_group_index_validator.validate,
        "calculate_oid_attributes": calculate_oid_attributes_validator.validate,
        "smooth_gps_noise": smooth_gps_noise_validator.validate,
        "correct_gps_outliers": correct_gps_outliers_validator.validate,
        "update_linear_and_custom": update_linear_and_custom_validator.validate,
        "enhance_images": enhance_images_validator.validate,
        "rename_images": rename_images_validator.validate,
        "apply_exif_metadata": apply_exif_metadata_validator.validate,
        "geocode_images": geocode_images_validator.validate,
        "build_oid_footprints": build_oid_footprints_validator.validate,
        "deploy_lambda_monitor": deploy_lambda_monitor_validator.validate,
        "copy_to_aws": copy_to_aws_validator.validate,
        "generate_oid_service": generate_oid_service_validator.validate
    }

    def validate_tool_config(self, tool: str):
        """
        Validates the configuration for a specified tool using its registered validator.

        If the tool name is not recognized, logs an error and signals a configuration validation failure.

        Args:
            tool: The name of the tool whose configuration should be validated.

        Raises:
            ConfigValidationError: If the tool name is not registered.
        """
        logger = self.get_logger()

        if tool in self.TOOL_VALIDATORS:
            self.TOOL_VALIDATORS[tool](self)
        else:
            logger.error(f"Unknown tool '{tool}'", error_type=ConfigValidationError)
