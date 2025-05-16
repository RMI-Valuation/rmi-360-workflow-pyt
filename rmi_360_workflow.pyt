# =============================================================================
# 📦 RMI 360 Imaging Workflow Toolbox (rmi_360_workflow.pyt)
# -----------------------------------------------------------------------------
# Purpose:             ArcGIS Python Toolbox entry point for the RMI 360 Imaging Workflow
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   This file defines the ArcGIS Python Toolbox interface, registering all pipeline tools for the RMI 360 workflow.
#   It acts as a wrapper and dispatcher, exposing tools for imagery processing, OID management, reporting, and AWS integration.
#   No business logic is implemented here; all workflow logic resides in the /tools and /utils modules.
#
# File Location:        /rmi_360_workflow.pyt
# Called By:            ArcGIS Pro (Toolbox registration), ArcGIS Python Toolbox Loader
#
# Directory Layout:
#   tools/              → Tool classes (UI + execution)
#   utils/              → Core processing and validation modules
#   utils/manager/      → Configuration, Logging, Path and Progressor management modules
#   utils/validators/   → Configuration and schema validation modules
#   utils/shared/       → Reusable stateless utilities
#   docs/               → Markdown documentation
#   configs/            → YAML templates and schema registry
#   templates/          → HTML report and style templates
#
# Registered Tools:
#   - 🧰 Process360Workflow
#   - 🎞️ RunMosaicProcessorTool
#   - 🏗️ CreateOrientedImageryDatasetTool
#   - 🧬 CreateOIDTemplateTool
#   - 📐 BuildOIDFootprints
#   - 🧭 AddImagesToOIDTool
#   - 🧮 UpdateLinearAndCustomTool
#   - 🛰️ SmoothGPSNoiseTool
#   - 🏷️ RenameAndTagImagesTool
#   - ☁️ CopyToAwsTool
#   - 🌐 GenerateOIDService
#   - 🌍 GeocodeImagesTool
#   - 📝 GenerateReportFromJSONTool
#   - 🔐 SetAWSKeyringCredentialsTool
#
# Documentation:
#   See: docs/TOOL_OVERVIEW.md and docs/toolbox_reference.md
#
# Notes:
#   - Supports background execution where applicable
#   - Designed for ArcGIS Pro 3.4+ with Python 3.9+ environments
#   - All tool logic is modularized for maintainability and extensibility
# =============================================================================

# Import tools to run individually
from tools.process_360_orchestrator import Process360Workflow
from tools.run_mosaic_processor_tool import RunMosaicProcessorTool
from tools.create_oid_tool import CreateOrientedImageryDatasetTool
from tools.add_images_to_oid_tool import AddImagesToOIDTool
from tools.smooth_gps_noise_tool import SmoothGPSNoiseTool
from tools.update_linear_and_custom_tool import UpdateLinearAndCustomTool
from tools.rename_and_tag_tool import RenameAndTagImagesTool
from tools.geocode_images_tool import GeocodeImagesTool
from tools.build_oid_footprints_tool import BuildOIDFootprints
from tools.copy_to_aws_tool import CopyToAwsTool
from tools.generate_oid_service_tool import GenerateOIDService
from tools.generate_report_tool import GenerateReportFromJSONTool
from tools.create_oid_template_tool import CreateOIDTemplateTool
from tools.set_aws_keyring_tool import SetAWSKeyringCredentialsTool


class Toolbox(object):
    def __init__(self):
        self.label = "RMI 360 Imaging Workflow Python Toolbox"
        self.alias = "rmi360workflow"
        self.tools = [
            Process360Workflow,
            RunMosaicProcessorTool,
            CreateOrientedImageryDatasetTool,
            AddImagesToOIDTool,
            SmoothGPSNoiseTool,
            UpdateLinearAndCustomTool,
            RenameAndTagImagesTool,
            GeocodeImagesTool,
            BuildOIDFootprints,
            CopyToAwsTool,
            GenerateOIDService,
            GenerateReportFromJSONTool,
            CreateOIDTemplateTool,
            SetAWSKeyringCredentialsTool
        ]
