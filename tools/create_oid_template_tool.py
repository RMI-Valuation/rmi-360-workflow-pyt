# =============================================================================
# 🧬 Create OID Schema Template (tools/create_oid_template_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          CreateOIDTemplateTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class that creates a reusable schema table for Oriented Imagery Datasets (OIDs),
#   based on the structure defined in the project configuration file. Ensures consistent field definitions
#   for OID feature classes and supports robust integration with RMI 360 workflow CORE utils.
#
# File Location:      /tools/create_oid_template_tool.py
# Core Utils:
#   - utils/build_oid_schema.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/create_oid_and_schema.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for the project; used for resolving logs and asset paths.
#   - Config File {config_file} (File): Path to the project-specific YAML configuration file defining schema and paths. If not provided, defaults to project config location.
#
# Notes:
#   - Generates a geodatabase table named in config["oid_schema_template"]["template"]["template_name"].
#   - Falls back to default config path if no parameter is provided.
#   - Ensure the configuration accurately reflects the required schema for downstream OID operations.
# =============================================================================

import arcpy
from utils.build_oid_schema import create_oid_schema_template
from utils.manager.config_manager import ConfigManager



class CreateOIDTemplateTool(object):
    def __init__(self):
        self.label = "Create OID Schema Template"
        self.description = ("Creates a reusable schema table for Oriented Imagery Datasets (OIDs) based on config.yaml "
                            "settings.")
        self.category = "Setup"

    def getParameterInfo(self):
        """
        Defines the input parameters required for the tool.
        
        Returns:
            A list containing a single required file parameter for specifying the config.yaml file.
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

        # Config file
        config_param = arcpy.Parameter(
            displayName="Config File",
            name="config_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        config_param.description = "Config.yaml file containing project-specific settings."
        params.append(config_param)

        return params

    def execute(self, parameters, messages):
        """
        Creates an OID schema template based on the provided configuration file.
        
        Uses the specified config file to generate a reusable schema table for Oriented Imagery Datasets (OIDs) and
        logs the location of the created template.
        """
        project_folder = parameters[0].valueAsText
        config_file = parameters[1].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )
        logger = cfg.get_logger()

        schema_path = create_oid_schema_template(cfg=cfg)
        logger.info(f"✅ Template created at: {schema_path}")
