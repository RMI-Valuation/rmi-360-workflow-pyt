# 🛠️ Tool: Run Mosaic Processor

## 🧰 Tool Name
**01 – Run Mosaic Processor**

## 🧭 Purpose
This tool initiates the **first step** in the Mosaic 360 imagery pipeline. It runs the Mosaic Processor CLI with the Mistika VR backend to:
- Render stitched JPEG panoramas from raw `.mp4` reels
- Automatically fix reel issues
- Integrate GPS data from `.gpx` files into the resulting images
- Normalize filenames by zero-padding frame numbers to six digits

This step is **required before image enhancement, renaming, metadata tagging, or GIS ingestion**.

---

## 🧪 Under-the-Hood Workflow

1. **Render Phase** (skips GPX):
   - Runs Mosaic Processor CLI with rendering + reel fix
   - Output is saved under:  
     `project_folder / panos / original`

2. **Pad Frame Numbers**:
   - Renames image files to ensure frame numbers use six digits (e.g., `_FR000023.jpg`)

3. **GPX Integration Phase** (skips rendering):
   - Second Mosaic Processor CLI run with GPX embedding only

4. **Logging**:
   - Console output and errors are saved to a log file in the `logs/` directory

---

## 🧰 Parameters

| Parameter         | Required | Description                                         |
|-------------------|----------|-----------------------------------------------------|
| Project Folder    | ✅       | Root for project outputs and logs                   |
| Input Reels Folder | ✅       | Folder containing raw Mosaic `.mp4` reels with optional `.gpx` |
| Config File       | ⬜️       | Path to `config.yaml` with processing options        |
| Mosaic GRP Template Path | ⬜️       | Overrides `executables.mosaic_processor.grp_path` |
| Start Frame       | ⬜️       | First frame to process                               |
| End Frame         | ⬜️       | Last frame to process                                |

---

## 🗂️ Scripts & Components

| Script                                  | Role/Responsibility                |
|-----------------------------------------|------------------------------------|
| `tools/run_mosaic_processor_tool.py`    | ArcGIS Toolbox wrapper             |
| `utils/run_mosaic_processor.py`         | Core processing logic              |
| `utils/manager/config_manager.py`       | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads processing parameters from config.
2. Ingests and organizes raw images.
3. Performs QC checks and flags issues.
4. Extracts and standardizes metadata.
5. Outputs processed images and logs.

---

## 🗃️ Inputs

- Folder of raw images
- Project YAML config with processing options

---

## 📤 Outputs

- Processed images ready for OID creation
- QC and metadata logs

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
mosaic_processor:
  qc_checks: true
  organize_by_date: true
  extract_metadata: true
  output_folder: "processed_images"
```

- Output folder is created if missing.
- QC checks can be toggled on/off.

---

## 🧩 Dependencies

- Python with `opencv-python`, `pandas`
- ArcGIS Pro
- Project YAML config

---

## ✅ Validation

Validation is performed by the appropriate validator in `utils/validators`.
- Checks that input folder and output folder are valid
- Ensures config options are present and correct
- Validates image file types and required metadata fields

---

## 🔗 Related Tools

- Enhance Images
- Rename and Tag Images
- Add Images to OID
- Create OID Schema Template

---

## 📁 Output

- **Location**:  
  `project_folder / panos / original`
- **Format**:  
  JPEG images named like `..._FR000001.jpg`
- **Log File**:  
  `logs/mosaic_processor_output.log`
- **Optional JSON Metadata**:  
  `reel_info.json` if reel number is detected

---

## 🔧 Config Sections Used

| Path | Purpose |
|------|---------|
| `executables.mosaic_processor.exe_path` | Path to Mosaic Processor CLI |
| `executables.mosaic_processor.grp_path` | Default GRP file path |
| `image_output.folders.parent` and `.original` | Defines output folder hierarchy |
| `logs.mosaic_processor_log` | Log filename |

---

## 🛑 Validation Coverage

Tool-specific validator: `validate_tool_mosaic_processor(config)`

Checks:
- `exe_path`, `grp_path`, `cfg_path` are valid and not empty
- Files exist or are resolvable (unless `"DISABLED"`)
- Types are validated (`str`)
- Errors are logged using `log_message`

---

## 📝 Notes and Best Practices

- You can run this tool **multiple times per project**, once for each reel
- Use consistent naming for input reel folders: `reel_0001`, `reel_0002`, etc.
- The output is a dependency for the following tools:
  - `Enhance Images`
  - `Rename and Tag`
  - `Add Images to OID`

---

## 🧪 Example Usage in Python

```python
from utils.mosaic_processor import run_mosaic_processor

run_mosaic_processor(
    project_folder="D:/Projects/RMI25100",
    input_dir="D:/Projects/RMI25100/raw/reel_0001",
    grp_path="D:/Calibrations/Mosaic51.grp",
    config_file="D:/Projects/RMI25100/config.yaml"
)
```
