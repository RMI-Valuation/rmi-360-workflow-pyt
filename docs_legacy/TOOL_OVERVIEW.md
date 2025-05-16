
# 🧰 RMI 360 Imaging Workflow Python Toolbox – Overview

The **RMI 360 Imaging Workflow Python Toolbox** is a modular, ArcGIS-integrated suite of tools designed to automate the 
full workflow of processing 360° imagery captured with the Mosaic 51 camera.

It transforms raw 360° images into a fully georeferenced Oriented Imagery Dataset (OID), enhanced with metadata, hosted 
on the cloud, and accompanied by a complete report.

---

## 📦 What This Toolbox Does

The toolbox allows you to:

- Run the full 360° imagery workflow with a single orchestrator tool, or one step at a time
- Enhance, rename, and tag images based on config-defined expressions
- Smooth GPS noise, correct outliers, and apply route-based attribution
- Upload imagery to AWS S3 and generate a hosted OID web service
- Produce a detailed report including step metrics, acquisition summaries, and reel stats

Every step is driven by `config.yaml` and supported by reusable utility scripts.

---

## 📂 Tool Categories

### 🛠 Setup Tools
- `SetAWSKeyringCredentialsTool`: Store AWS keys securely using keyring
- `CreateOIDTemplateTool`: Generate a reusable OID schema template

### ⚙️ Individual Tools
These tools represent the core pipeline steps and can be run independently:

- `RunMosaicProcessorTool`: Call the Mosaic CLI for rendering + GPX integration
- `CreateOIDTool`: Create a new OID based on the schema template
- `AddImagesToOIDTool`: Add images, assign GroupIndex, enrich fields
- `SmoothGPSNoiseTool`: Detect outliers based on deviation/angle/distance
- `UpdateLinearAndCustomTool`: Apply MP and custom attributes from config
- `EnhanceImagesTool`: Improve contrast, clarity, and metadata of 360 images
- `RenameAndTagImagesTool`: Rename files and apply EXIF/XMP metadata
- `GeocodeImagesTool`: Add place name metadata based on GPS coordinates
- `BuildOIDFootprintsTool`: Create footprint geometry for each image
- `CopyToAWSTool`: Upload renamed imagery to S3 with resumable support
- `GenerateOIDServiceTool`: Publish a hosted OID using updated AWS image paths

### 📈 Reporting Tools
- `GenerateReportFromJSONTool`: Convert saved report JSON to full HTML/PDF report

### 🚀 Orchestrator
- `Process360Workflow`: End-to-end runner with full config control, per-step execution, optional backups, wait points, and metrics logging.

---

## 🔄 How the Orchestrator Works

The `ProcessMosaic360Workflow` tool executes the full pipeline in configurable steps.

- You can choose the **start step**, and it will proceed sequentially.
- Certain steps can be **skipped** based on config toggles (e.g. `skip_enhance_images`).
- You can optionally enable:
  - 📦 OID backups between steps (`backup_oid_between_steps`)
  - ⏱ Wait points to pause before uploads (`wait_before_step`)

After each step, a report JSON is updated. At the end, a final HTML + PDF report is generated.

This orchestrator tool simplifies full workflow automation while remaining flexible for partial or custom runs.

---

## 📚 Related Documentation

- [Tool Guides](./TOOL_GUIDES.md)
- [Configuration Guide](./CONFIG_GUIDE.md)
- [Shared Utilities](./UTILITIES.md)

