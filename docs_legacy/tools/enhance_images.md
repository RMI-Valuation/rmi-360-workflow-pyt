# 🖼 Tool: Enhance Images

## 🧰 Tool Name

---

## 📝 Purpose

Applies configurable image enhancement algorithms (white balance, contrast, denoising, sharpening, color correction, etc.) to all images referenced in an Oriented Imagery Dataset (OID) feature class. Designed for 360° panoramic images in Mosaic OID workflows.

---

## 🧰 Parameters

| Parameter            | Required | Description                                     |
|----------------------|----------|-------------------------------------------------|
| OID Feature Class    | ✅       | Input OID containing image paths                 |
| Config File          | ✅       | Path to `config.yaml` with enhancement options   |
| Project Folder       | ✅       | Project root for resolving outputs               |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                |
|-------------------------------------|------------------------------------|
| `tools/enhance_images_tool.py`      | ArcGIS Toolbox wrapper             |
| `utils/enhance_images.py`           | Core enhancement logic             |
| `utils/manager/config_manager.py`   | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads enhancement parameters from config.
2. Iterates over images referenced in OID.
3. Applies selected enhancements (white balance, contrast, denoise, sharpen, etc.).
4. Writes enhanced images to output folder.
5. Optionally updates OID with new image paths.

---

## 🗃️ Inputs

- OID feature class
- Project YAML config with enhancement options

---

## 📤 Outputs

- Enhanced images in output folder
- Updated OID with new image paths (if configured)

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
enhance_images:
  white_balance: true
  contrast: 1.2
  denoise: true
  sharpen: false
  output_folder: "enhanced_images"
```

- Output folder is created if missing.
- Enhancement steps are configurable and can be chained.

---

## 🧩 Dependencies

- Python with `opencv-python`, `numpy`
- ArcGIS Pro
- Project YAML config

---

## ✅ Validation

Validation is performed by the appropriate validator in `utils/validators` (not `validate_config.py`).
- Checks that enhancement config block exists and values are valid
- Ensures OID and output paths are writable

---

## 🔗 Related Tools

- Add Images to OID
- Rename and Tag Images
- Build OID Footprints
- Generate OID Service

---

## 📂 Output Options

Defined in `image_enhancement.output.mode`:

| Mode | Behavior |
|------|----------|
| `overwrite` | Replace original images |
| `suffix` | Append `_enh` or custom suffix |
| `directory` (default) | Save to alternate subfolder (e.g., `/panos/enhanced`) |

The `ImagePath` in the OID will be updated if mode ≠ `overwrite`.

---

## 📊 Logs

If configured:
- Writes `logs/enhance_log.csv` with:
  - Brightness / contrast before and after
  - Clip limit used
  - White balance method
  - RGB means before/after
  - Enhanced file path

---

## ⚙️ Key Config Block

```yaml
image_enhancement:
  enabled: true
  adaptive: true
  output:
    mode: "directory"
    suffix: "_enh"
  apply_white_balance: true
  white_balance:
    method: "gray_world"
  apply_contrast_enhancement: true
  clahe:
    tile_grid_size: [8, 8]
    contrast_thresholds: [60, 80]
    clip_limit_low: 1.5
    clip_limit_high: 2.5
  apply_saturation_boost: true
  saturation_boost:
    factor: 1.1
  apply_sharpening: true
  sharpen:
    kernel:
      - [0, -0.3, 0]
      - [-0.3, 2.6, -0.3]
      - [0, -0.3, 0]
  brightness_recovery: true
  brightness:
    threshold: 110
    factor: 1.1
```

---

## 🧪 Example Usage

```python
from utils.enhance_images import enhance_images_in_oid

enhance_images_in_oid(
    oid_fc_path="C:/GIS/Projects/OID.gdb/Imagery",
    config_file="C:/Projects/config.yaml"
)
```

---

## ✅ Validation

Validator: `validate_tool_enhance_images(config)`

Checks:
- Output mode is valid (`overwrite`, `suffix`, `directory`)
- CLAHE settings are correct types and format
- Sharpen kernel is 3×3
- White balance method is valid
- `enabled`, `adaptive`, and other flags are boolean

---

## 📝 Notes

- Designed for use **after Mosaic Processor** but **before renaming or tagging**
- Can be run incrementally or on subsets
- Designed specifically for outdoor panoramic imagery
- Visual improvement is generally subtle but effective for final output
