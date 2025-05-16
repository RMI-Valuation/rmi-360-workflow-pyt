# 🌍 Tool: Geocode Images

## 🧰 Tool Name

---

## 📝 Purpose

Assigns spatial coordinates to images based on their EXIF GPS data or external CSV logs, updating the Oriented Imagery Dataset (OID) with accurate geometry. Supports batch geocoding, error reporting for missing/invalid location data, and prioritizes EXIF or CSV as configured.

---

## 🧰 Parameters

| Parameter            | Required | Description                                      |
|----------------------|----------|--------------------------------------------------|
| OID Feature Class    | ✅       | Input OID containing image references             |
| Config File          | ✅       | Path to `config.yaml` with geocoding options      |
| Project Folder       | ✅       | Project root for resolving outputs                |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                |
|-------------------------------------|------------------------------------|
| `tools/geocode_images_tool.py`      | ArcGIS Toolbox wrapper             |
| `utils/geocode_images.py`           | Core geocoding logic               |
| `utils/manager/config_manager.py`   | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads geocoding parameters from config.
2. Iterates over images in OID.
3. Extracts GPS data from EXIF or CSV (as prioritized in config).
4. Updates OID geometry fields.
5. Logs errors for missing or invalid data.

---

## 🗃️ Inputs

- OID feature class
- Project YAML config with geocoding options
- (Optional) CSV logs with GPS data

---

## 📤 Outputs

- OID feature class with updated geometry
- Error logs for images with missing coordinates

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
geocode_images:
  exif_priority: true
  csv_fallback: "gps_log.csv"
  geometry_fields: ["SHAPE@X", "SHAPE@Y"]
```

- If EXIF GPS is missing and `csv_fallback` is set, uses CSV for coordinates.
- Logs images where neither source is available.

---

## 📤 Output

| File | Description |
|------|-------------|
| `logs/geocode_args.txt` | ExifTool command batch file |
| `logs/geocode_logs.txt` | Human-readable log of geocoded images |

Metadata tags added (if available):
- `XMP:XMP-iptcExt:LocationShownCity`
- `XMP:XMP-iptcExt:ProvinceState`
- `XMP:XMP-iptcExt:CountryName`

---

## 🧪 Python Usage Example

```python
from utils.geocode_images import geocode_images

geocode_images(
    oid_fc="C:/GIS/OID.gdb/Images",
    config_file="C:/Projects/config.yaml",
    messages=None
)
```

---

## ✅ Validation

Validation is performed by `validate_tool_geocode_images()` in `utils/validators`:
- Checks that `method` is `"exiftool"`
- Validates that chosen DB is one of `default`, `geolocation500`, or `geocustom`
- Validates presence and path of `.config` file if using a custom or Geolocation500 DB
- Confirms ExifTool executable exists and is callable

---

## 📝 Notes

- Tool is safe to re-run; it overwrites existing tags without modifying image data
- ExifTool must be installed and accessible via system PATH or specified in `executables.exiftool.exe_path`
- Best run after GPS outlier correction and before final metadata tagging or AWS upload
