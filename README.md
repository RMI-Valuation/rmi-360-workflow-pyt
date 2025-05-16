
# 🧰 RMI 360 Imaging Workflow Python Toolbox

![Version](https://img.shields.io/badge/version-v1.1.0-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![ArcGIS Pro](https://img.shields.io/badge/ArcGIS_Pro-3.4%2B-green)

A modular, camera-agnostic workflow built with ArcGIS Python Toolbox for processing and deploying 360° corridor imagery.

Optimized for Mosaic 51 cameras, with planned support for Insta360. Includes tools for enhancement, OID creation, AWS publishing, and detailed reporting.

---

## 📦 Overview

- 🖼️ Image enhancement (contrast, white balance, sharpening)
- 🏷️ EXIF metadata tagging
- 🗂️ File renaming and organization
- 🧭 ArcGIS Oriented Imagery Dataset (OID) creation and enrichment
- ☁️ AWS S3 upload with resumable transfer logic
- 📈 Lambda-based progress monitoring and status dashboard
- 🌍 Geocoding of image locations using spatial reference datasets

---

## 🧩 Key Features

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Toolbox Structure      | Built as an ArcGIS `.pyt` Toolbox + modular tool wrappers & utilities       |
| Config-Driven          | YAML-based config with expression resolution and field registries           |
| AWS Integration        | Upload to S3 with TransferManager + Lambda schedule tracking                |
| Resumable Transfers    | Upload interruption protection + log recovery                              |
| HTML & JSON Reporting  | Auto-generated step summaries and final status reports                      |
| Image Metadata Support | Auto tag EXIF metadata + rename by GPS, time, reel, frame, etc.             |

---

## 📁 Repository Structure

```
rmi-360-workflow-pyt/
├── rmi_360_workflow.pyt                # ArcGIS Python Toolbox
├── configs/
│   ├── config.sample.yaml              # Config template
│   └── esri_oid_fields_registry.yaml   # ESRI OID field definitions
├── tools/                              # ArcGIS tool wrappers
├── utils/                              # Reusable logic
│   ├── manager/                        # Managers (ConfigManager, LogManager, PathManager, ProgressorManager)
│   ├── shared/                         # Shared utilities
│   └── validators/                     # Validators
├── aws_lambdas/                        # Lambda upload status functions
├── templates/                          # HTML report templates
├── legacy_docs/                        # Full documentation set
├── docs/                               # Updated documentation set
├── dev_docs/                           # Future development documentation
```

---

## ✅ Quick Start (Dev Mode)

```bash
git clone https://github.com/RMI-Valuation/rmi-360-workflow-pyt.git
cd rmi-360-workflow-pyt

# Copy and edit config
cp configs/config.sample.yaml configs/config.yaml

# (Optional) Run config validation
python -m utils.validate_config --file configs/config.yaml
```

---

## 🧭 ArcGIS Pro Environment Setup

1. Open **ArcGIS Pro** and load a project (.aprx).
2. In the Catalog pane, right-click **Toolboxes** → **Add Toolbox**.
3. Browse to `rmi_360_workflow.pyt` and add it.
4. Access tools grouped under:
   - **Setup**
   - **Individual Tools**
   - **Reporting**
   - **Orchestrator**

5. Use individual tools or run the full pipeline with `ProcessMosaic360Workflow`.

---

## 🎞 Mosaic Processor Usage Notes

- The `RunMosaicProcessorTool` wraps the Mosaic CLI (requires Mosaic Processor + MistikaVR).
- It executes two passes:
  - Image rendering and reel fixing
  - GPX integration to embed GPS
- Requires `.grp` calibration file (provided by Mosaic) and raw `.mp4` input folders.

---

## 📖 Documentation Index

- 📘 [Toolbox Overview](docs_legacy/TOOL_OVERVIEW.md)
- 🔧 [Tool Guides](docs_legacy/TOOL_GUIDES.md)
- ⚙️ [Configuration Guide](docs_legacy/CONFIG_GUIDE.md)
- 🧰 [Shared Utilities](docs_legacy/UTILITIES.md)
- ☁️ [AWS Setup Guide](docs_legacy/AWS_SETUP_GUIDE.md)
- 📋 [Schema Changelog](docs_legacy/SCHEMA_CHANGELOG.md)
- 📄 [TODO (Developer Tasks)](./TODO.md)
- 📝 [Changelog](./CHANGELOG.md)
- 🛣 [Roadmap](docs_legacy/ROADMAP.md)

---

## 📝 License

Licensed under the [MIT License](./LICENSE).  
© 2025 RMI Valuation.

This project integrates with external proprietary software (e.g., ArcGIS Pro, Mosaic Processor, MistikaVR).  
Use of those tools is governed by their respective licenses and is not covered by this repository's license.
