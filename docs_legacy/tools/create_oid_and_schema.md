# 🛠️ Tools: Create OID Schema Template & Create Oriented Imagery Dataset

## 🧑‍💻 Tool Names
**CreateOIDTemplateTool** & **CreateOrientedImageryDatasetTool**

---

## 📝 Purpose

Defines and instantiates the schema for an Oriented Imagery Dataset (OID) used in the Mosaic 360 workflow. Ensures all required fields (standard, Mosaic-specific, linear referencing, custom) are created with consistent names, types, aliases, and defaults. Enables downstream tools to function with a standardized data structure.

---

## 🧰 Parameters

### Create OID Schema Template
| Parameter      | Required | Description                                                |
|---------------|----------|------------------------------------------------------------|
| Config File    | ✅       | Path to `config.yaml` with schema structure                |
| Project Folder | ✅       | Root folder for the project (for logs, assets)             |

### Create Oriented Imagery Dataset
| Parameter                   | Required | Description                                             |
|-----------------------------|----------|---------------------------------------------------------|
| Output OID Feature Class    | ✅       | Output feature class to create                          |
| Spatial Reference           | ⬜️      | Optional custom spatial reference                       |
| Config File                 | ⬜️      | Optional path to config with schema and project settings|
| Project Folder              | ✅       | Root for project outputs                                |

---

## 🗂️ Scripts & Components

| Script                                   | Role/Responsibility                                               |
|------------------------------------------|-------------------------------------------------------------------|
| `tools/create_oid_template_tool.py`      | ArcGIS Toolbox wrapper for schema template creation               |
| `tools/create_oid_tool.py`               | ArcGIS Toolbox wrapper for OID feature class creation             |
| `utils/build_oid_schema.py`              | Builds geodatabase table as schema template                       |
| `utils/create_oid_feature_class.py`      | Creates OID feature class using template and config               |
| `utils/manager/config_manager.py`        | Loads and validates configuration                                 |

---

## ⚙️ Behavior / Logic

**Schema Template Creation:**
1. Loads field definitions from ESRI registry YAML and config (`mosaic_fields`, `grp_idx_fields`, `linear_ref_fields`, `custom_fields`).
2. Builds a geodatabase table as a reusable schema template.
3. Backs up existing templates with timestamp.
4. Validates schema and field definitions.

**OID Feature Class Creation:**
1. Validates schema template and spatial reference (from config or parameter).
2. Creates new OID feature class in specified geodatabase.
3. Applies all field definitions and spatial reference.
4. Ensures output does not already exist.

---

## 🗃️ Inputs

- Project YAML config with schema and spatial reference blocks
- (For OID creation) Schema template table in geodatabase

---

## 📤 Outputs

- Schema template table (default: `oid_schema_template`) in configured GDB
- OID feature class with all required fields and spatial reference

---

## 🗝️ Configuration / Notes

- Schema and field definitions are in `config.yaml`:

```yaml
oid_schema_template:
  template:
    gdb_path: "templates/templates.gdb"
    template_name: "oid_schema_template"
field_registry: "esri_oid_fields_registry.yaml"
mosaic_fields: [...]
grp_idx_fields: [...]
linear_ref_fields: [...]
custom_fields: [...]
```

- Spatial reference (horizontal/vertical) can be set in config or as a parameter.
- Template is auto-backed up before overwrite.

---

## 🧩 Dependencies

- ArcGIS Pro
- Python with `arcpy`
- Project YAML config

---

## 🔗 Related Tools

- Add Images to OID
- Build OID Footprints
- Update Metadata
- Generate OID Service
- Confirms `spatial_ref.gcs_horizontal_wkid` and `vcs_vertical_wkid` are defined and resolvable
- Optionally resolves `pcs_horizontal_wkid` if used in other tools (like footprints)

---

## 📌 Purpose

These tools support the structured generation of OID datasets with project-specific fields, such as:
- Standard OID attributes (`CameraHeight`, `AcquisitionDate`, etc.)
- Mosaic-specific fields like `Reel`, `Frame`
- Linear referencing fields like `MP_Pre`, `MP_Num`
- Custom metadata such as `RR` (Railroad Code)

They ensure compatibility with downstream tools like `Add Images to OID`, `Update Metadata`, and `Generate OID Service`.

---

## 🛠 Create OID Schema Template

### ✔️ Parameters

| Parameter | Required | Description |
|----------|----------|-------------|
| `Config File` | ✅ | Path to `config.yaml` with schema structure |

### ⚙️ Output

- A template table in the GDB defined at:  
  `config.oid_schema_template.template.gdb_path`

- Template name (default):  
  `oid_schema_template`

- Saved inside `templates_dir`, e.g.:  
  `templates/templates.gdb/oid_schema_template`

### 🔍 Key Logic (in `build_oid_schema.py`)
- Loads field definitions from:
  - ESRI registry YAML (`field_registry`)
  - `mosaic_fields`, `grp_idx_fields`, `linear_ref_fields`, `custom_fields` in config
- Adds fields using `arcpy.management.AddField`
- Backs up any existing template with a timestamp
- Creates GDB if missing

### ✅ Validation
Validation is performed by `validate_tool_build_oid_schema()` in `utils/validators`:
- Checks required config keys (`field_registry`, `template_name`, etc.)
- Validates all field definitions
- Ensures no duplicates

---

## 🛠 Create Oriented Imagery Dataset

### ✔️ Parameters

| Parameter | Required | Description |
|----------|----------|-------------|
| `Output Oriented Imagery Dataset` | ✅ | Path to new feature class (e.g., `my.gdb/MyOID`) |
| `Spatial Reference` | ⬜️ | Overrides default GCS/VCS from config |
| `Config File` | ✅ | Path to `config.yaml` |
| `Project Folder` | ⬜️ | Optional override for `project_folder` context |

### ⚙️ Output
- Feature class with all OID fields, Z-values enabled
- Created using:  
  `arcpy.oi.CreateOrientedImageryDataset(...)`
- Uses online terrain service for elevation

### 🔍 Key Logic (in `create_oid_feature_class.py`)
- Loads config, resolves paths
- Resolves spatial reference:
  - Defaults to GCS 4326 (horizontal) + VCS 5703 (ellipsoidal height)
- Validates that schema template exists and is current
- Creates the feature class from template

### ✅ Validation
Validation is performed by `validate_tool_create_oriented_imagery_dataset()` in `utils/validators`:
- Confirms `spatial_ref.gcs_horizontal_wkid` and `vcs_vertical_wkid` are defined and resolvable
- Optionally resolves `pcs_horizontal_wkid` if used in other tools (like footprints)

---

## 🧩 Config Sections Used

```yaml
oid_schema_template:
  template:
    templates_dir: "../templates"
    gdb_path: "templates.gdb"
    template_name: "oid_schema_template"
  esri_default:
    field_registry: "../configs/esri_oid_fields_registry.yaml"
    standard: true
    not_applicable: false

mosaic_fields:
  mosaic_reel: ...
  mosaic_frame: ...

linear_ref_fields:
  route_identifier: ...
  route_measure: ...

custom_fields:
  custom1:
    name: "RR"
    expression: "config.project.rr_mark"
```

---

## 🧪 Example Usage

```python
from utils.build_oid_schema import create_oid_schema_template
from utils.create_oid_feature_class import create_oriented_imagery_dataset

# Step 1: Create schema template
create_oid_schema_template(config_file="config.yaml")

# Step 2: Create OID feature class from template
create_oriented_imagery_dataset(
    output_fc_path="C:/GIS/project.gdb/MyOID",
    config_file="config.yaml"
)
```

---

## 📝 Notes
- Schema templates are versionable and reusable
- Run the schema creation tool **once per config change**
- OID creation step is mandatory before using `Add Images to OID`
- Uses `DEM` from ESRI online service by default (can be extended)
