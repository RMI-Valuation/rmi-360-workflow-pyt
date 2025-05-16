# =============================================================================
# 🧭 Add Images to OID (tools/add_images_to_oid_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          AddImagesToOIDTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class for adding rendered 360° images to an existing Oriented Imagery Dataset (OID).
#   Assigns group indices and enriches OID attributes after insertion. Uses project configuration
#   for source folders, schema enforcement, and vertical offset control. Ensures compatibility with
#   CORE utils and project schema templates. Designed for robust integration into the RMI 360 workflow.
#
# File Location:      /tools/add_images_to_oid_tool.py
# Core Utils:
#   - utils/add_images_to_oid_fc.py
#   - utils/assign_group_index.py
#   - utils/calculate_oid_attributes.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/add_images_to_oid.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for the Mosaic 360 imagery project.
#   - Oriented Imagery Dataset {oid_fc} (Feature Class): OID feature class to which images will be added.
#   - Adjust Z (Apply Offset) {adjust_z} (Boolean): If True, applies vertical offset to GPS elevation using config.
#   - Config File {config_file} (File): YAML config file. Optional; uses default if not provided.
#
# Notes:
#   - Applies GroupIndex and orientation enrichment if configured in project settings.
#   - Overwrites output feature class if it exists (arcpy.env.overwriteOutput = True).
#   - Designed for use with schema templates and robust error handling.
#   - Ensure documentation is kept current with tool updates.
# =============================================================================

import arcpy

from utils.add_images_to_oid_fc import add_images_to_oid
from utils.calculate_oid_attributes import enrich_oid_attributes
from utils.assign_group_index import assign_group_index
from utils.manager.config_manager import ConfigManager


class AddImagesToOIDTool(object):
    def __init__(self):
        self.label = "03 - Add Images to OID"
        self.description = ("Adds rendered 360° images to an existing Oriented Imagery Dataset (OID) using the ArcGIS "
                            "AddImagesToOrientedImageryDataset tool.")
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the AddImagesToOIDTool geoprocessing tool.
        
        Returns:
            A list of ArcPy Parameter objects specifying the required and optional inputs:
            - Project folder for organizing imagery and logs.
            - Existing Oriented Imagery Dataset feature class.
            - Optional flag to apply vertical offset to GPS elevation.
            - YAML configuration file with project settings.
        """
        params = []

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

        # Oriented Imagery Dataset (OID)
        oid_param = arcpy.Parameter(
            displayName="Oriented Imagery Dataset",
            name="oid_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )
        oid_param.description = "Path to an existing OID feature class created using the schema template."
        params.append(oid_param)

        # Adjust Z (optional toggle)
        adjust_z_param = arcpy.Parameter(
            displayName="Adjust Z (Apply Offset)",
            name="adjust_z",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        adjust_z_param.value = True  # default to True, since typically needed
        adjust_z_param.description = (
            "Whether to apply a vertical offset (Z) to GPS elevation using the formula defined in config.yaml. "
            "Disable if Mosaic Processor has already applied Z correction."
        )
        params.append(adjust_z_param)

        # Config file
        config_param = arcpy.Parameter(
            displayName="Config File",
            name="config_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        config_param.description = "Config.yaml file containing project-specific settings."
        params.append(config_param)

        return params

    def execute(self, parameters, messages):
        """
        Executes the tool to add 360° images to an Oriented Imagery Dataset and update its attributes.
        
        This method processes the provided parameters to add images from the specified project folder to the given OID
        feature class, assigns group indices, and enriches the OID records with additional attributes. The process uses
        project-specific settings from the configuration file and optionally applies a vertical offset to GPS elevation
        values.
        """
        project_folder = parameters[0].valueAsText
        oid_fc = parameters[1].valueAsText
        adjust_z = bool(parameters[2].value) if parameters[2].value is not None else True
        config_file = parameters[3].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,               # May be None
            project_base=project_folder,
            messages=messages
        )

        add_images_to_oid(
            cfg=cfg,
            oid_fc_path=oid_fc
        )

        assign_group_index(
            cfg=cfg,
            oid_fc_path=oid_fc
        )

        # Enrich added records with Z-adjusted, derived, and default fields
        enrich_oid_attributes(
            cfg=cfg,
            oid_fc_path=oid_fc,
            adjust_z=adjust_z
        )
