# =============================================================================
# 🌍 Geocode Images (tools/geocode_images_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          GeocodeImagesTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class that geotags images from an Oriented Imagery Dataset (OID)
#   using GPS metadata and ExifTool. Adds XMP geolocation data to image files in-place and
#   supports project-specific geolocation configuration files. Integrates with Core Utils for
#   configuration and batch geotagging logic.
#
# File Location:      /tools/geocode_images_tool.py
# Core Utils:
#   - utils/geocode_images.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/geocode_images.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   Oriented Imagery Feature Class {oid_fc} (Feature Class): The OID containing images to geotag.
#   Config File (optional) {config_file} (File): Optional config.yaml with geolocation DB and output settings.
#
# Notes:
#   - Requires ExifTool to be installed and available in PATH.
#   - Can be re-run safely; overwrites existing XMP location tags.
#   - Ensure config file and ExifTool installation are correct for successful geotagging.
# =============================================================================

import arcpy
from utils.geocode_images import geocode_images
from utils.manager.config_manager import ConfigManager


class GeocodeImagesTool:
    def __init__(self):
        self.label = "07 - Geocode Images"
        self.description = "Applies location tags to images using ExifTool and GPS metadata."
        self.canRunInBackground = True
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

        oid_param = arcpy.Parameter(
            displayName="Oriented Imagery Feature Class",
            name="oid_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )
        oid_param.description = "The Oriented Imagery Dataset to process."
        params.append(oid_param)

        config_param = arcpy.Parameter(
            displayName="Config File (optional)",
            name="config_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        config_param.description = "Config.yaml file containing project-specific settings."
        config_param.filter.list = ["yaml", "yml"]
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

        geocode_images(
            cfg=cfg,
            oid_fc=oid_fc
        )
