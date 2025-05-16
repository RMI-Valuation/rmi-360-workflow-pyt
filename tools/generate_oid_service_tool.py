# =============================================================================
# 🌐 Generate OID Service (tools/generate_oid_service_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          GenerateOIDService
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class that publishes an Oriented Imagery Dataset (OID) as a hosted
#   feature service on ArcGIS Online. Duplicates the OID with updated S3-based ImagePaths and
#   invokes ArcGIS's GenerateServiceFromOrientedImageryDataset tool. Optionally creates a portal
#   folder if it does not exist. Integrates with Core Utils for configuration and publishing logic.
#
# File Location:      /tools/generate_oid_service_tool.py
# Core Utils:
#   - utils/generate_oid_service.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/generate_oid_service.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for this Mosaic 360 imagery project. All imagery and logs will be organized under this folder.
#   - Oriented Imagery Dataset {oid_fc} (Feature Class): Existing OID feature class to be duplicated and published.
#   - Config File {config_file} (File): YAML config file containing AWS bucket, portal, and S3 path details.
#
# Notes:
#   - Automatically updates ImagePaths to public S3 URLs.
#   - Attempts to create portal folder if it does not exist.
#   - Requires authenticated ArcGIS Pro session with sharing privileges.
#   - Ensure config file is up-to-date and user has required portal privileges.
# =============================================================================

import arcpy
from utils.generate_oid_service import generate_oid_service
from utils.manager.config_manager import ConfigManager


class GenerateOIDService(object):
    def __init__(self):
        self.label = "10 - Generate OID Service"
        self.description = "Duplicates an OID with AWS paths and publishes it as a hosted Oriented Imagery service."
        self.canRunInBackground = False
        self.category = "Individual Tools"

    def getParameterInfo(self):
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
        project_folder = parameters[0].valueAsText
        oid_fc = parameters[1].valueAsText
        config_file = parameters[2].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )

        generate_oid_service(
            cfg=cfg,
            oid_fc=oid_fc
        )
