# 🛠️ Tool: Add Images to Oriented Imagery Dataset (OID)

## 🧑‍💻 Tool Name
**03 – Add Images to OID**

---

## 📝 Purpose

Adds rendered and renamed images to an existing Oriented Imagery Dataset (OID) using ArcGIS Pro's Oriented Imagery tools. This tool extracts and calculates required attributes (Reel, Frame, orientation, Z offset, etc.), applies group indices for ArcGIS Pro filtering, and ensures all schema requirements are met. This step is required before metadata tagging or uploading to AWS.

---

## 🧰 Parameters

| Parameter                   | Required | Description                                                         |
|-----------------------------|----------|---------------------------------------------------------------------|
| Project Folder              | ✅       | Folder for the current Mosaic 360 project (used to resolve paths)   |
| Oriented Imagery Dataset    | ✅       | Target OID feature class (must exist and use schema)                |
| Adjust Z (Apply Offset)     | ⬜️      | Toggles camera height/Z offset correction using values in config    |
| Config File                 | ✅       | Full path to `config.yaml`                                          |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                                                 |
|-------------------------------------|---------------------------------------------------------------------|
| `tools/add_images_to_oid_tool.py`   | ArcGIS Toolbox wrapper, parameter handling                          |
| `utils/add_images_to_oid_fc.py`     | Adds images to OID, schema validation                               |
| `utils/assign_group_index.py`       | Assigns group index based on `AcquisitionDate`                      |
| `utils/calculate_oid_attributes.py` | Computes and populates Z, SRS, CameraHeight, Reel, Frame, Orientation |
| `utils/manager/config_manager.py`   | Loads and validates configuration                                   |

---

## ⚙️ Behavior / Logic

1. Loads configuration (`config.yaml`) and resolves image source folder.
2. Validates OID feature class and schema.
3. Adds all JPEG images (including subfolders) to the OID using ArcPy tools.
4. Extracts and populates attributes: Reel, Frame, CameraPitch, CameraRoll, NearDistance, FarDistance, CameraHeight, CameraOrientation.
5. Optionally applies Z offset correction based on config.
6. Assigns cyclic `GroupIndex` for display filtering, based on `AcquisitionDate`.
7. Logs warnings for schema or metadata issues, supports robust error handling.

---

## 🗃️ Inputs

- Images in `panos/original` (resolved from config)
- OID feature class created via schema template
- Filename format and spatial reference from `config.yaml`

---

## 📤 Outputs

- OID feature class with new image features, including:
  - Linked file path and `Name`
  - X, Y, Z geometry (with optional Z adjustment)
  - CameraPitch (default: 90), CameraRoll (default: 0)
  - NearDistance (default: 2), FarDistance (default: 50)
  - CameraHeight, CameraOrientation (Type 1 string)
  - GroupIndex (for scalable display in ArcGIS Pro)

---

## 🗝️ Configuration / Notes

- Camera offset and orientation settings are drawn from `config.yaml`:

```
from utils.add_images_to_oid_fc import add_images_to_oid
from utils.assign_group_index import assign_group_index
from utils.calculate_oid_attributes import enrich_oid_attributes

add_images_to_oid(
    project_folder="D:/Projects/RMI25100",
    oid_fc_path="D:/GIS/RMI25100.gdb/OID",
    config_file="D:/Projects/RMI25100/config.yaml"
)

assign_group_index("D:/GIS/RMI25100.gdb/OID", config_file="D:/Projects/RMI25100/config.yaml")
enrich_oid_attributes("D:/GIS/RMI25100.gdb/OID", config_file="D:/Projects/RMI25100/config.yaml")
```

---

## 🔎 Config Sections Used

- `image_output.folders.original` – source image folder
- `oid_schema_template` – defines fields like `Reel`, `Frame`, `GroupIndex`
- `camera_offset` – used for Z and height calculations
- `spatial_ref` – defines horizontal and vertical WKIDs
- `esri_oid_fields_registry.yaml` – provides defaults like `HFOV`, `VFOV`

---

## ✅ Validation

From `validate_config.py`, the following validators are used:
- `validate_tool_add_images_to_oid`
- `validate_tool_assign_group_index`
- `validate_tool_calculate_oid_attributes`

These validate:
- Required fields in registry (e.g., `CameraPitch`, `CameraOrientation`)
- Properly configured `grp_idx_fields`, `mosaic_fields`, `linear_ref_fields`
- Resolvable WKIDs, Z offsets, and default values

---

## 📝 Notes

- Must be run **after** `Create OID` but **before** metadata tagging
- Ensures consistent filenames and GPS values for later tools
- Use `GroupIndex` to control display density in ArcGIS Pro:
  - 5m spacing → show all
  - 10m spacing → `GroupIndex IN (1,3)` or `(2,4)`
  - 20m spacing → `GroupIndex = 1` (or 2, 3, or 4)
