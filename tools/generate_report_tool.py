# =============================================================================
# 📝 Generate Report from JSON (tools/generate_report_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          GenerateReportFromJSONTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class for generating an HTML (and optionally PDF) report based on
#   a saved report JSON file. Optionally reattaches a configuration file for template path
#   resolution and branding. Output folder can be overridden or defaults to path from JSON.
#   Integrates with Core Utils for configuration, template management, and report generation.
#
# File Location:      /tools/generate_report_tool.py
# Core Utils:
#   - utils/generate_report.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/generate_report.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Report JSON File {report_json} (File): Path to the saved report_data_<slug>.json file containing report input and stats.
#   - Config File (Optional) {config_file} (File): Optional YAML config file to reattach template, logo, and slug context.
#
# Notes:
#   - Generates HTML report from Jinja2 templates; PDF export is optional.
#   - Can be re-run at any time using archived JSON data.
#   - Ensure config and template paths are correct for branding and output.
# =============================================================================

import arcpy
from utils.generate_report import generate_report_from_json
from utils.manager.config_manager import ConfigManager


class GenerateReportFromJSONTool(object):
    def __init__(self):
        """
        Initializes the GenerateReportFromJSONTool with label, description, and category metadata.
        """
        self.label = "15 - Generate Report from JSON"
        self.description = "Generates an HTML report from a saved report_data_<slug>.json file."
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the GenerateReportFromJSONTool ArcPy tool.
        
        Returns:
            A list of ArcPy Parameter objects specifying the required and optional inputs
            for generating reports from a JSON file.
        """
        params = []

        # Project Folder
        project_param = arcpy.Parameter(
            displayName="Project Folder",
            name="project_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        project_param.description = ("Root folder for this Mosaic 360 imagery project. All imagery and logs will be "
                                     "organized under this folder.")
        params.append(project_param)

        # Report JSON file
        json_param = arcpy.Parameter(
            displayName="Report JSON File",
            name="report_json",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        json_param.filter.list = ["json"]
        json_param.description = "Path to the saved report_data_<slug>.json file containing all report inputs."
        params.append(json_param)

        # Config file (optional reattachment)
        config_param = arcpy.Parameter(
            displayName="Config File (Optional)",
            name="config_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        config_param.description = (
            "Optional config.yaml file to reattach config context for paths, templates, and logos. "
            "Used only if not already included in the report JSON."
        )
        params.append(config_param)

        return params

    def execute(self, parameters, messages):
        """
        Executes the tool to generate a report from a JSON file using provided parameters.
        
        Extracts input values for the report JSON file, output directory, and configuration file, then calls the report
        generation function with these inputs.
        """
        project_folder = parameters[0].valueAsText
        json_path = parameters[1].valueAsText
        config_file = parameters[2].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )

        generate_report_from_json(
            cfg=cfg,
            json_path=json_path
        )
