# 📦 CHANGELOG

*All notable changes to this project will be documented in this file.*

## [2025-05-23]
### 🧰 RMI 360 Environment Checker and other quality-of-life changes
#### Added
- Added `rmi_360_env_checker.pyt` to the repository to check for required libraries in the ArcGIS Pro Python environment. This utility does not affect workflow toolbox versioning.
- Added ArcGIS Pro license matrix (`docs_legacy/ARCGISPRO_LICENSE_REQUIREMENTS.md`) summarizing which ArcPy functions/classes require Basic, Standard, or Advanced licenses.
- Added note to README indicating that a Standard license or above is required for Oriented Imagery tools.

#### Removals/Deprecations
- Removed `humanize` package requirement. Replaced with a native Python function in `folder_stats.py` for human-readable file size formatting. The Toolbox now runs in the base ArcGIS Pro Python environment without extra dependencies.

---

## [v1.1.1] - 2025-05-22
### 🛠️ Bug Fix Update
#### Added
- Added `OrientationAccuracy` as a standard field in `esri_oid_fields_registry.yaml` (per ArcGIS Pro 3.5 OID attribute updates).
- Added `verify_aws_credentials()` function to `aws_utils.py` for AWS credential verification.
- Added additional validation to fail early if proper AWS credentials are not provided.

#### Fixed
- Fixed indentation of `project.local_proj_wkid`. It needs to be a child of `project`. This caused failure in validation.
- Fixed a regression in configuration management where the project root directory path key was changed from `__project_root__` to `__project_base__`. This update restores the correct key (`__project_root__`) for full compatibility with workflow tools and logging.
- Updated the sample configuration so `geocoding.exiftool_geodb` now defaults to `default`, avoiding errors when the specialized `geolocation500` GeoDB is not present in templates. This improves out-of-the-box compatibility for new projects.
- Resolved bugs in `build_oid_schema.py` affecting OID generation and schema consistency.
- Fixed AWS credential verification in `copy_to_aws.py` to match the logic in `deploy_lambda_monitor.py`, which is now handled from `aws_utils.py`.This ensures that invalid or placeholder credentials are detected before any operations are attempted, improving reliability and error reporting.
- Fixed a bug in `generate_oid_service.py` where the OID was not duplicated with AWS urls.

> **⚠️ Note: ⚠️**  
> Version **v1.1.0** contained critical bugs that may render the workflow unusable.  
> Please use version **v1.1.1** or later, which resolves these issues.

> **⚠️ Unit Tests Note:** Many unit tests may currently be broken due to some bug fixes in this release.

---

## [v1.1.0] - 2025-05-20
### 🚀 Public Release - RMI 360 Imaging Workflow Python Toolbox v1.1.0 (Major Refactor: Managers, Modularization, and Enhancements)
#### Added
- Manager Classes: Introduced ConfigManager, LogManager, PathManager, and ProgressorManager for centralized configuration, logging, path resolution, and progress tracking.
- Modular Validators: Added tool-specific validator modules under utils/validators/ for robust, modular config validation.
- Shared Utilities: New/refactored shared utilities for ArcPy, AWS, disk space, expression resolution, folder stats, metrics, report data, exceptions, and schema validation.
- Comprehensive Tests: Extensive new and updated unit/integration tests for all major modules, managers, validators, and utilities.
- Legacy Documentation: Added detailed legacy docs for all tools, utilities, and managers.
- Orchestrator Enhancements: Fine-grained control over optional steps, with improved configuration options, documentation, and error handling.
- Metadata Enhancements: Expanded metadata handling with new fields for richer context and propagation across workflow steps.

#### Changed
- Centralized Configuration & Logging: All tools and utilities now use ConfigManager and LogManager for consistent config access and structured logging.
- Unified Path Handling: All path resolution is now managed by PathManager for cross-platform consistency.
- Progress Tracking: Progress reporting is standardized via ProgressorManager.
- Refactored APIs: All major workflow functions and tools now accept manager objects instead of raw config dicts/files.
- Parameter Handling: Tools now require a project folder parameter and use config managers for all settings.
- Documentation Structure: Docs reorganized; legacy docs added, old docs/tools removed or moved.

#### Removed
- Legacy Utilities: Deleted or replaced modules: config_loader.py, executable_utils.py, pad_mosaic_frame_numbers.py, path_resolver.py, path_utils.py, schema_paths.py, validate_config.py.
- Old Documentation: Removed or replaced old docs/tools in docs/ and dev_docs/.

#### Fixed
- Static Analysis: Improved import/export patterns for better static analysis and namespace clarity.
- Error Handling: Unified and improved error handling and logging across all modules.

#### Docs
- Legacy Docs: Added/updated documentation for all tools, utilities, and managers under docs_legacy/.
- README: Updated to reflect new structure, features, and usage notes.

#### Tests
- Expanded Coverage: Added or updated tests for all managers, utilities, validators, and workflow modules.
- Test Data: Added comprehensive sample/test config and placeholder files.

#### Breaking Changes
- Manager APIs: All tools and utilities now require manager objects (ConfigManager, etc.) instead of raw config dicts/files.
- Logging Configuration: Logging is now handled exclusively via LogManager; old logging utilities are removed.
- Import Patterns: Utilities and shared modules now use explicit imports and modular APIs.
- Config Schema: Configuration files must be updated to match the new schema and structure.

#### Migration Notes
- Update all tool invocations to use the new manager-based APIs.
- Review and update configuration files to match the new schema (see sample/test configs).
- Review updated documentation for usage examples and migration guidance.
- Remove any dependencies on deleted legacy utility modules.

---

## [1.0.0] – 2025-05-08
### 🎉 Initial Stable Release – RMI 360 Imaging Workflow Python Toolbox v1.0.0
#### 🚀 Major Highlights
- Modular ArcGIS Toolbox structure with Setup, Individual Tools, Reporting, and Orchestrator categories.
- New master workflow runner (`Process360Workflow`) supports end-to-end execution, step control, backups, and metrics.
- Centralized and deeply nested `config.yaml` with schema validation, field registry, and dynamic expression support.
- Field-level schema enforcement powered by `esri_oid_fields_registry.yaml`.

#### 📸 Imagery & Metadata Processing
- OpenCV-based image enhancement pipeline with CLAHE, white balance, sharpening, and EXIF preservation.
- EXIF/XMP metadata tagging via `apply_exif_metadata.py` driven by `metadata_tags` config block.
- Standardizes frame numbering (6-digit) via `pad_mosaic_frame_numbers.py`.

#### 🛰️ GPS, Schema, and Geometry
- Advanced GPS outlier detection with route-deviation logic and optional automatic correction.
- Z-offset and camera height logic computed from configurable lever-arm offsets.
- Schema template creation and OID feature class population fully driven by field registry.

#### ☁️ Cloud & Web Integration
- S3 upload pipeline rewritten with `TransferManager`, resumable uploads, real-time progress logs, and cancel file support.
- Lambda-based upload dashboard writes `status.html` with live metrics.
- Generates hosted OID service using AWS-based image paths and AGOL integration.

#### 📊 Reporting & Logging
- Full project report generation with charts, metrics, reel statistics, and export logs.
- JSON-based intermediate report structure enables reruns and backup-safe reporting.

#### 🛠 Developer Tools
- `expression_utils.py`: powerful syntax engine supporting field/config/constant expressions.
- `validate_config.py`: extensible tool-by-tool config validator with field and type checking.
- `progressor_utils.py`: shared progress manager for both CLI and ArcGIS Pro tools.

---
