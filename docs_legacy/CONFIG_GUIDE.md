
# ⚙️ Configuration Guide – RMI 360 Imaging Workflow Python Toolbox

This guide explains how to structure and validate the `config.yaml` file, which drives every step of the RMI 360 Imaging Workflow.

---

## 📄 What is `config.yaml`?

This is the central configuration file used by all tools in the RMI 360 Imaging Workflow Python Toolbox. It defines:
- Project information
- File and folder paths
- Filename formatting rules
- Schema definitions
- GPS smoothing parameters
- Upload settings
- Report output preferences

Every tool reads from this file either directly or through `prepare_config()`.

---

## 🧱 Structure Overview

```yaml
project:
  slug: ABC25110
  number: 25-110
  client: Test Client
  rr_mark: TC
  rr_name: Test Railroad
  description: Hi-Rail Test
  local_proj_wkid: 6492
```

### Top-Level Keys (commonly used)
- `project` — metadata about the current survey
- `camera_offset` — Z and camera height configuration
- `image_output` — controls folder structure and filename logic
- `gps_smoothing` — deviation and outlier detection settings
- `oid_schema_template` — defines what fields your OID should contain
- `logs` — filenames and prefix logic for log output
- `executables` — paths to tools like ExifTool and Mosaic Processor
- `aws`, `portal`, `geocoding`, `report` — optional integrations

---

## 🧮 Expression Syntax

Many fields support dynamic expressions using the `resolve_expression()` engine:

```yaml
filename_settings:
  format: "{project_slug}_{rr}_{mp_pre}{mp_num}_{capture_datetime}_RL{reel}_FR{frame}.jpg"
  parts:
    project_slug: config.project.slug.strip("-")
    mp_num: field.MP_Num.float(3)
    capture_datetime: field.AcquisitionDate.date(%Y%m%dT%H%M%SZ)
```

### Supported Modifiers
- `strip("-")`
- `float(n)`
- `upper`, `lower`
- `date(...)` for datetime fields
- Special: `now.year`, `' '` (space)

---

## 🧰 Schema Templates

Schemas are defined using a mix of:

```yaml
oid_schema_template:
  template:
    templates_dir: templates
    gdb_path: templates.gdb
    template_name: oid_schema_template
  esri_default:
    field_registry: configs/esri_oid_fields_registry.yaml
    standard: true
    not_applicable: false
  mosaic_fields:
    reel:
      name: Reel
      type: TEXT
    frame:
      name: Frame
      type: TEXT
  grp_idx_fields:
    group:
      name: GroupIndex
      type: LONG
  linear_ref_fields:
    route_identifier:
      name: MP_Pre
      type: TEXT
    route_measure:
      name: MP_Num
      type: DOUBLE
  custom_fields:
    rr:
      name: RR
      type: TEXT
      expression: config.project.rr_mark
```

These control what fields are created in the OID and how they are populated.

---

## ✅ Config Validation

Run this anytime to check your config:

```bash
python -m utils.validate_config --file configs/config.yaml
```

Validates:
- YAML structure and schema version
- Expressions
- Field naming
- Required blocks for each tool

---

## 🧪 Testing with `prepare_config()`

Each tool internally uses `prepare_config()` to:
- Load and resolve the config
- Set `__project_root__`
- Resolve expressions
- Apply default paths

You can use `prepare_config()` or `resolve_config()` in your own scripts to ensure tools behave the same way.

---

## 📘 See Also

- [Utilities Reference](./UTILITIES.md)
- [Sample Config](../configs/config.sample.yaml)
