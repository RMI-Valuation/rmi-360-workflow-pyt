# 🏷 Tool: Rename and Tag Images

## 🧰 Tool Name
**05 – Rename and Tag Images**

---

## 📝 Purpose

Standardizes image filenames and applies metadata tags to images in preparation for upload and OID linkage. Supports customizable naming conventions, batch renaming, and writing EXIF/XMP tags based on project configuration.

---

## 🧰 Parameters

| Parameter         | Required | Description                                         |
|-------------------|----------|-----------------------------------------------------|
| Input Folder      | ✅       | Folder containing images to rename/tag               |
| Config File       | ✅       | Path to `config.yaml` with naming/tagging rules      |
| Project Folder    | ✅       | Project root for resolving outputs                   |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                |
|-------------------------------------|------------------------------------|
| `tools/rename_and_tag_tool.py`      | ArcGIS Toolbox wrapper             |
| `utils/rename_and_tag.py`           | Core renaming/tagging logic        |
| `utils/manager/config_manager.py`   | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads naming/tagging parameters from config.
2. Iterates over images in input folder.
3. Renames files according to convention.
4. Writes EXIF/XMP metadata tags.
5. Logs changes and errors.

---

## 🗃️ Inputs

- Folder of images
- Project YAML config with naming/tagging rules

---

## 📤 Outputs

- Renamed images in output folder
- Images with updated metadata tags
- Logs of changes/errors

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
rename_and_tag:
  naming_convention: "{project}_{date}_{seq}"
  tag_fields:
    - "ProjectName"
    - "AcquisitionDate"
  output_folder: "renamed_tagged"
```

- Naming convention supports placeholders for project, date, sequence, etc.
- Output folder is created if missing.

---

## 🧩 Dependencies

- Python with `exiftool`, `pandas`
- ArcGIS Pro
- Project YAML config

---

## ✅ Validation

Validation is performed by the appropriate validator in `utils/validators` (not `validate_config.py`).
- Checks that naming convention and tag fields are valid
- Ensures output folder is writable
- Validates EXIF tool path and config

---

## 🔗 Related Tools

- Enhance Images
- Add Images to OID
- Copy to AWS
- Generate OID Service

---

## 📝 Notes

- This is the final processing step before upload (`copy_to_aws`) or service generation
- Designed for batch‑safe execution (will not overwrite files silently)
- Can be run multiple times — images are renamed and tagged independently
- Best used after GPS correction and linear referencing are complete

---

## 📝 Example Usage (Python)

```python
from tools.rename_and_tag_tool import execute

execute(
    input_folder="path/to/images",
    config_file="path/to/config.yaml",
    project_folder="path/to/project",
    dry_run=False
)
```

---

## 🖇 Filename Structure

From `config.yaml → image_output.filename_settings`:

```yaml
format: "{project_slug}_{rr}_{mp_pre}{mp_num}_{capture_datetime}_RL{reel}_FR{frame}.jpg"

parts:
  project_slug: "config.project.slug"
  rr: "field.RR"
  mp_pre: "field.MP_Pre"
  mp_num: "field.MP_Num.float(3)"
  capture_datetime: "field.AcquisitionDate.date(%Y%m%dT%H%M%SZ)"
  reel: "field.Reel"
  frame: "field.Frame"
```

---

## 🏷 Metadata Tags

From `config.yaml → image_output.metadata_tags`:

```yaml
Artist: "config.project.company"
Copyright: "'© ' + now.year + ' ' + config.project.company"
Software: "config.camera.software"
Make: "config.camera.make"
Model: "config.camera.model"
SerialNumber: "config.camera.sn"
FirmwareVersion: "config.camera.firmware"
ImageDescription: "field.RR + ' MP ' + field.MP_Pre + '-' + field.MP_Num.float(3)"
XPComment: "config.project.number + ' ' + config.project.rr_name + ' - ' + config.project.description"
XPKeywords:
  - "config.project.company"
  - "360 Imagery"
  - "Panoramic"
  - "Oriented Imagery"
  - "Railroad"
  - "config.project.rr_name"
  - "config.project.rr_mark"
  - "config.camera.model"
  - "field.MP_Pre + '-' + field.MP_Num.float(3)"
  - "config.project.description"
  - "config.project.number"
```

---

## 🧠 Special Behaviors

- Filenames are checked for uniqueness — `_v1`, `_v2` suffixes are added if duplicates exist.
- `QCFlag == 'GPS_OUTLIER'` will trigger **GPSLatitude/Longitude overwrite** in metadata.
- ExifTool args and logs are written to:
  - `logs/exiftool_batch.args`
  - `logs/exiftool_log.txt`

---

## 📤 Outputs

| Output | Description |
|--------|-------------|
| Renamed Image Files | Saved to `/panos/final/` (or as defined in config) |
| Updated OID Fields | `ImagePath`, `Name` |
| EXIF/XMP Metadata | Embedded into each image using ExifTool |

---

## ✅ Validation

Validated via:

- `validate_tool_rename_images()` for filename generation
- `validate_tool_apply_exif_metadata()` for metadata tags and ExifTool path

Checks include:
- Filename placeholders match parts
- Expressions resolve to strings
- Required EXIF tags exist
- ExifTool path is valid

---

## 📝 Notes

- This is the final processing step before upload (`copy_to_aws`) or service generation
- Designed for batch‑safe execution (will not overwrite files silently)
- Can be run multiple times — images are renamed and tagged independently
- Best used after GPS correction and linear referencing are complete
