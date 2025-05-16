# =============================================================================
# 📐 Build OID Footprints (tools/build_oid_footprints_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          BuildOIDFootprints
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class for generating a BUFFER-style footprint feature class from an Oriented Imagery Dataset (OID).
#   Utilizes ArcGIS BuildOrientedImageryFootprint and supports optional configuration input for custom spatial reference
#   or transformation logic. Integrates with project config for workflow consistency and is designed for seamless use
#   within the RMI 360 workflow.
#
# File Location:      /tools/build_oid_footprints_tool.py
# Core Utils:
#   - utils/build_oid_footprints.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/build_oid_footprints.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for the project; used for resolving logs and asset paths.
#   - Oriented Imagery Dataset {oid_fc} (Feature Class): Path to an existing OID feature class to generate footprints from.
#   - Config File {config_file} (File): Path to the project config.yaml containing spatial reference and transformation settings (optional).
#
# Notes:
#   - Can run standalone or as part of the orchestrator workflow.
#   - Respects config overrides for spatial reference and geographic transformation.
#   - Sets canRunInBackground = False due to ArcPy environment handling.
#   - Ensure configuration is up-to-date for accurate spatial output.
# =============================================================================

import arcpy

from utils.build_oid_footprints import build_oid_footprints
from utils.manager.config_manager import ConfigManager


class BuildOIDFootprints(object):
    def __init__(self):
        self.label = "08 - Build OID Footprints"
        self.description = "Generates a BUFFER-style footprint feature class from an Oriented Imagery Dataset."
        self.canRunInBackground = False
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the BuildOIDFootprints geoprocessing tool.
        
        Returns:
            A list of ArcPy Parameter objects for the Oriented Imagery Dataset and optional config file.
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

        # Oriented Imagery Dataset (OID)
        oid_param = arcpy.Parameter(
            displayName="Oriented Imagery Dataset",
            name="oid_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )
        oid_param.description = "Path to an existing OID feature class."
        params.append(oid_param)

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
        Executes the tool to generate footprint features from an Oriented Imagery Dataset.
        
        Extracts input parameters and invokes the core footprint-building function, passing
        the dataset path, optional configuration file, and ArcGIS messaging object.
        """
        project_folder = parameters[0].valueAsText
        oid_fc = parameters[1].valueAsText
        config_file = parameters[2].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )

        # Call the core function with ArcGIS Pro messaging support
        build_oid_footprints(
            cfg=cfg,
            oid_fc=oid_fc
        )
