# 🛤 Tool: Update Linear Referencing and Custom Attributes

## 🧰 Tool Name
**06 – Update Linear and Custom Attributes**

---

## 🧭 Purpose

This tool updates Oriented Imagery Dataset (OID) features by:

1. **Locating each image along a route centerline** using M-values (milepost data)
2. **Populating linear referencing fields** (e.g., `MP_Pre`, `MP_Num`)
3. **Populating custom project-specific attributes** (e.g., `RR`, `ClientCode`, `MP_Label`) based on expressions in `config.yaml`

This step ensures imagery records are **milepost-referenced and metadata-enriched**, allowing consistent filenames, metadata tags, and service attributes.

---

## 🧰 Parameters

| Parameter            | Required | Description                                      |
|----------------------|----------|--------------------------------------------------|
| OID Feature Class    | ✅       | Input OID to update                              |
| Config File          | ✅       | Path to `config.yaml` with field rules           |
| Project Folder       | ✅       | Project root for resolving outputs               |

---

## 🗂️ Scripts & Components

| Script                                  | Role/Responsibility                |
|-----------------------------------------|------------------------------------|
| `tools/update_linear_and_custom_tool.py`| ArcGIS Toolbox wrapper             |
| `utils/update_linear_and_custom.py`     | Core update logic                  |
| `utils/manager/config_manager.py`       | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads field calculation rules from config.
2. Calculates linear referencing fields (e.g., MP_Pre, MP_Num, Offset).
3. Applies custom project-specific field logic.
4. Updates OID attributes.
5. Logs changes and errors.

    alias: "MP Label"
    expression: "field.MP_Pre + '-' + field.MP_Num.float(3)"
```

---

## ⚙️ How It Works

```text
1. Load field definitions from config.yaml
2. (If enabled) Locate each point along the centerline using LocateFeaturesAlongRoutes
3. Update `MP_Pre` and `MP_Num` based on route ID and M-value
4. Evaluate expressions to update each custom field
5. Report results to ArcGIS Pro using arcpy.AddMessage
```

---

## 📤 Outputs

- **Linear Reference Fields**:
  - `MP_Pre` → Prefix/line identifier
  - `MP_Num` → Milepost decimal
- **Custom Fields**:
  - Populated using expressions (e.g., `RR`, `MP_Label`, `Client`)

---

## 🧪 Python Usage

```python
from utils.update_linear_and_custom import update_linear_and_custom

update_linear_and_custom(
    oid_fc="C:/GIS/ProjectOID.gdb/OID",
    centerline_fc="C:/GIS/Centerline.gdb/Routes",
    route_id_field="RouteID",
    enable_linear_ref=True,
    config_file="config.yaml"
)
```

---

## ✅ Validation

Validation logic ensures:

- `linear_ref_fields` and `custom_fields` are properly structured
- Types are compatible (`TEXT`, `DOUBLE`, etc.)
- Field names are unique
- Required expressions resolve correctly via `resolve_expression()`

Validators: `validate_tool_update_linear_and_custom()` in `validate_config.py`

---

## 📝 Notes

- Must be run **after geometry and attributes have been added to OID**
- If `enable_linear_ref` is unchecked, only custom fields will be updated
- Expressions can reference `config.project.*` or any other part of the config
- You can define as many custom fields as needed for metadata, tagging, or filenames
