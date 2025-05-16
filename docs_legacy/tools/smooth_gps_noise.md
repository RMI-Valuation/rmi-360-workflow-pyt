# 🛠️ Tool: Smooth GPS Noise

## 🧑‍💻 Tool Name
**Smooth GPS Noise**

---

## 📝 Purpose

Corrects GPS outliers and smooths the location data for images in an Oriented Imagery Dataset (OID). Applies configurable filters to reduce jitter, interpolate missing points, and flag suspect positions for review.

---

## 🧰 Parameters

| Parameter            | Required | Description                                      |
|----------------------|----------|--------------------------------------------------|
| OID Feature Class    | ✅       | Input OID containing GPS data                     |
| Config File          | ✅       | Path to `config.yaml` with smoothing options      |
| Project Folder       | ✅       | Project root for resolving outputs                |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                |
|-------------------------------------|------------------------------------|
| `tools/smooth_gps_noise_tool.py`    | ArcGIS Toolbox wrapper             |
| `utils/smooth_gps_noise.py`         | Core smoothing logic               |
| `utils/manager/config_manager.py`   | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads smoothing parameters from config.
2. Identifies and flags GPS outliers.
3. Applies moving average, Kalman filter, or other smoothing method.
4. Interpolates missing or invalid points if enabled.
5. Updates OID geometry and logs changes.

---

## 🗃️ Inputs

- OID feature class
- Project YAML config with smoothing options

---

## 📤 Outputs

- OID feature class with smoothed geometry
- Logs of outliers and corrections

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
smooth_gps_noise:
  method: "moving_average"
  window_size: 5
  outlier_threshold: 10.0
  interpolate: true
```

- Method can be `moving_average`, `kalman`, etc.
- Outlier threshold is in meters.

---

## 🧩 Dependencies

- Python with `numpy`, `pandas`
- ArcGIS Pro
- Project YAML config

---

## ✅ Validation

Validation is performed by the appropriate validator in `utils/validators`.
- Checks that smoothing config block exists and values are valid
- Ensures OID and output paths are writable
- Validates method and threshold values

---

## 🔗 Related Tools

- Geocode Images
- Add Images to OID
- Enhance Images
- Build OID Footprints

## 📊 Outlier Detection Logic

Each point is evaluated for:

1. **Deviation from midpoint** – lateral distance from midpoint between neighbors
2. **Angle change** – deviation from straight line (acceptable range: `175–185°`)
3. **Step spacing** – deviation from expected GPS spacing (default: 5m ±0.75m)
4. **Route distance deviation** – optional; checks drift from reference line

Points failing ≥2 of the above checks are flagged as `"GPS_OUTLIER"` in `QCFlag`.

Outliers surrounded by other outliers are also flagged, even if they do not fail the threshold themselves.

---

## ⚙️ Config Settings (`config.yaml`)

```yaml
gps_smoothing:
  capture_spacing_meters: 5.0
  deviation_threshold_m: 0.5
  angle_bounds_deg: [175, 185]
  proximity_check_range_m: 0.75
  max_route_dist_deviation_m: 0.5
  smoothing_window: 2
  outlier_reason_threshold: 2
```

Optional debug CSV path:
```yaml
logs.gps_smooth_debug: "debuglog_outliers.csv"
```

---

## 🧪 Example Usage (Python)

```python
from utils.smooth_gps_noise import smooth_gps_noise
from utils.correct_gps_outliers import correct_gps_outliers

# Step 1: Flag suspect points
smooth_gps_noise(
    oid_fc="path/to/OID.gdb/OID",
    centerline_fc="path/to/centerline.gdb/Track",
    config_file="config.yaml"
)

# Step 2: Correct flagged points (optional)
correct_gps_outliers(
    oid_fc="path/to/OID.gdb/OID",
    config_file="config.yaml"
)
```

---

## 📤 Outputs

| Output | Description |
|--------|-------------|
| `QCFlag` (TEXT) | Added/updated for all flagged outliers (`"GPS_OUTLIER"`) |
| Geometry (XY) | Adjusted for flagged points (unless `Flag Only` is enabled) |
| `CameraOrientation` | Recomputed for corrected points |
| `logs/debuglog_outliers.csv` | Summary of flagged points and reasons |

---

## ✅ Validation

Validated in `validate_config.py` under:
- `validate_tool_smooth_gps_noise`
- `validate_tool_correct_gps_outliers`

Checks:
- Required keys in `gps_smoothing`
- Type validation (int/float/list)
- Ensures `angle_bounds_deg` is a 2-element list
- Ensures WKIDs are defined in `spatial_ref`

---

## 📝 Notes & Best Practices

- Use `Flag Only` for dry runs or QA before applying geometry updates
- Recommended before renaming/tagging/geocoding stages
- A well-aligned centerline improves detection near curves
- Debug CSV is useful for QA and validation of outlier detection
