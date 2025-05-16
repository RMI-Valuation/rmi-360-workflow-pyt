# 🛠️ Tool: Build OID Footprints

## 🧑‍💻 Tool Name
**08 – Build OID Footprints**

---

## 📝 Purpose

Generates buffer-style multipoint footprints for each image in an Oriented Imagery Dataset (OID) using ArcGIS Pro's `BuildOrientedImageryFootprint` tool. These footprints are used to visualize camera fields of view, support spatial queries, and enable downstream OID service publishing.

---

## 🧰 Parameters

| Parameter                   | Required | Description                                                         |
|-----------------------------|----------|---------------------------------------------------------------------|
| Project Folder              | ✅       | Root folder for the project; used for resolving logs and asset paths |
| Oriented Imagery Dataset    | ✅       | Input OID feature class                                             |
| Config File                 | ⬜️      | Optional path to `config.yaml` for spatial reference/transformation  |

---

## 🗂️ Scripts & Components

| Script                                   | Role/Responsibility                                            |
|------------------------------------------|---------------------------------------------------------------|
| `tools/build_oid_footprints_tool.py`     | ArcGIS Toolbox wrapper, parameter handling                     |
| `utils/build_oid_footprints.py`          | Core logic using ArcPy's BuildOrientedImageryFootprint         |
| `utils/manager/config_manager.py`        | Loads and validates configuration                             |

---

## ⚙️ Behavior / Logic

1. Loads spatial reference (`pcs_horizontal_wkid`) and optional transformation from `config.yaml`.
2. Validates input OID feature class.
3. Runs ArcPy's `BuildOrientedImageryFootprint` to generate a footprint feature class alongside the OID.
4. Applies config overrides for spatial reference and transformation if provided.
5. Logs warnings for spatial reference issues or output failures.

---

## 🗃️ Inputs

- Oriented Imagery Dataset (OID)
- Project YAML config with spatial reference definitions

---

## 📤 Outputs

- Footprint feature class (e.g., `OID_Footprint`) in the same geodatabase or dataset as the input OID

---

## 🗝️ Configuration / Notes

- Spatial reference and transformation are defined in `config.yaml`:

```yaml
spatial_ref:
  pcs_horizontal_wkid: 6492
  transformation: null  # Optional; may be set for datum conversions
```

- The output feature class name is derived from the OID (e.g., `OID_Footprint`).
- Environment settings are restored after footprint creation.

---

## 🧩 Dependencies

- ArcGIS Pro
- Python with `arcpy`
- Project YAML config

---

## 🔗 Related Tools

- Add Images to OID
- Create OID Schema Template
- Generate OID Service

---

## 📤 Output

| Output | Description |
|--------|-------------|
| `OID_Footprint` | Feature class in same location as OID, using buffer footprint geometry |
| `arcpy.AddMessage` logs | Logged to ArcGIS Pro Messages tab or CLI output |

---

## ✅ Validation

Tool-level validator: `validate_tool_build_oid_footprints()` in `validate_config.py`:

- Ensures `spatial_ref` block exists
- Resolves `pcs_horizontal_wkid` to an integer
- Checks that WKID is positive
- Validates optional `transformation` (if provided) is a string

---

## 📝 Notes

- Uses `footprint_option = "BUFFER"` — suitable for panoramic 360 imagery
- OID must already contain fields such as `CameraHeading`, `HFOV`, and coordinates
- Best run after all geometry corrections and metadata tagging are complete
- Will overwrite existing footprint FC if one already exists with the same name
