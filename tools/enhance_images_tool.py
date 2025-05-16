# =============================================================================
# 🖼️ Enhance Images (tools/enhance_images_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          EnhanceImagesTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class for enhancing 360° images referenced in an OID feature class.
#   Applies image processing operations including white balance, contrast, saturation boost,
#   and sharpening. Supports batch multiprocessing, configurable output modes, and robust logging
#   for integration with the RMI 360 workflow.
#
# File Location:      /tools/enhance_images_tool.py
# Core Utils:
#   - utils/enhance_images.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/enhance_images.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for this Mosaic 360 imagery project. All imagery and logs will be organized under this folder.
#   - Oriented Imagery Dataset {oid_fc} (Feature Class): Input OID feature class containing images to enhance.
#   - Config File (optional) {config_file} (File): Path to YAML config file. Defaults to /configs/config.yaml if omitted.
#
# Notes:
#   - Performs enhancement using OpenCV with multiprocessing support.
#   - Writes detailed logs and enhancement stats to project log folder.
#   - Ensure config file specifies output and enhancement options as needed for the workflow.
# =============================================================================

import arcpy
from utils.enhance_images import enhance_images_in_oid
from utils.manager.config_manager import ConfigManager


class EnhanceImagesTool:
    def __init__(self):
        """
        Initializes the EnhanceImagesTool with metadata for ArcPy geoprocessing.
        
        Sets the tool's label, description, background execution capability, and category.
        """
        self.label = "02 - Enhance Images"
        self.description = "Enhances 360 images using color correction, adaptive contrast, and sharpening."
        self.canRunInBackground = True
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the Enhance Images geoprocessing tool.
        
        Returns:
            A list of ArcPy Parameter objects specifying the required Oriented Imagery Dataset
            feature class and an optional YAML configuration file for project-specific settings.
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
        oid_param.description = "Path to an existing OID feature class created using the schema template."
        params.append(oid_param)

        config_param = arcpy.Parameter(
            displayName="Config File (optional)",
            name="config_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        config_param.description = "Path to config.yaml with project-specific settings. Defaults to config in /configs."
        config_param.filter.list = ["yaml", "yml"]
        params.append(config_param)

        return params

    def execute(self, parameters, messages):
        """
        Executes the image enhancement process for a specified Oriented Imagery Dataset.
        
        Extracts input parameters, determines the configuration file path, and invokes the image enhancement routine.
        Logs a success message upon completion or logs and raises a runtime error if an exception occurs.
        """
        project_folder = parameters[0].valueAsText
        oid_fc = parameters[1].valueAsText
        config_file = parameters[2].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )
        logger = cfg.get_logger()

        try:
            logger.info("Starting image enhancement...")
            enhance_images_in_oid(
                cfg=cfg,
                oid_fc_path=oid_fc
            )
            logger.info("✅ Image enhancement completed successfully.")

        except Exception as e:
            logger.error(f"Enhance Images Tool failed: {e}", error_type=RuntimeError)
