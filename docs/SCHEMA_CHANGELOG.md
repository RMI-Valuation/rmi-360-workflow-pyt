
# 📄 SCHEMA_CHANGELOG.md

This changelog tracks structural changes to the Oriented Imagery Dataset (OID) schema, the `config.yaml` layout, and the field registry.

---
## [1.0.1] - 2025-05-09
- Added support for `openai`
  - `api_key`
  - `ai_assist_img_enh.enabled`


## [1.0.0] – 2025-05-08

### ✅ Added
- Introduced structured `oid_schema_template` block in `config.yaml`.
- Added support for field groups:
  - `mosaic_fields` (e.g., Reel, Frame)
  - `grp_idx_fields` (e.g., GroupIndex)
  - `linear_ref_fields` (e.g., MP_Pre, MP_Num)
  - `custom_fields` (e.g., RR from project.rr_mark)
- Created `esri_oid_fields_registry.yaml` for required and optional ESRI fields with schema enforcement.
- Added support for
  - `oid_default` values in registry
  - `expression` logic in `custom_fields`
- All field creation now driven by config, not hardcoded logic.
- Schema template output can be reused across projects.

### 📁 New `config.yaml` Top-Level Keys (in order):
- `schema_version`
- `logs`
- `project`
- `camera`
- `camera_offset`
- `spatial_ref`
- `executables`
- `oid_schema_template`
- `gps_smoothing`
- `image_output`
- `aws`
- `portal`
- `geocoding`
- `image_enhancement`
- `orchestrator`

Each key corresponds to one or more steps in the workflow and is validated before execution.

### ✏️ Changed
- Previously hardcoded fields like `RR`, `MP_Num`, `GroupIndex` now fully config-defined.
- `CameraHeight` is now calculated from a detailed breakdown under `camera_offset`.
- Filename format and metadata tags now use `resolve_expression()` for dynamic control.

### ❌ Removed
- Hardcoded OID schema construction from tools.
- Deprecated legacy `camera_calculations` and `field_list` structures.

---

## 🗂 Notes

- Templates are built using `build_oid_schema.py`.
- Validation is performed by `validate_config.py` and `schema_validator.py`.
- Tools that consume schemas: `create_oid_feature_class.py`, `create_oid_template_tool.py`.

