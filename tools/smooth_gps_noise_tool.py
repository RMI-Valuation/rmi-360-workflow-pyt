# =============================================================================
# 🛤️ Smooth GPS Noise (tools/smooth_gps_noise_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          SmoothGPSNoiseTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class that detects and flags suspect GPS points in an Oriented Imagery Dataset
#   (OID) using geometric deviation and optional centerline validation. Optionally applies corrections
#   based on detected outliers unless run in flag-only mode. Integrates with Core Utils for batch
#   GPS analysis, correction, and configuration management.
#
# File Location:      /tools/smooth_gps_noise_tool.py
# Core Utils:
#   - utils/smooth_gps_noise.py
#   - utils/correct_gps_outliers.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/smooth_gps_noise.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Oriented Imagery Dataset {oid_fc} (Feature Class): The OID feature class to analyze for GPS noise.
#   - Reference Centerline (optional) {centerline_fc} (Feature Class): Optional M-aware line to validate GPS position against.
#   - Flag Only (No Geometry Updates) {flag_only} (Boolean): If checked, flags outliers but does not correct geometry.
#   - Config File {config_file} (File): Path to the project configuration YAML file.
#
# Notes:
#   - Flag-only mode skips the coordinate update pass.
#   - Requires centerline to be in projected coordinate system if used.
#   - Updates QCFlag and optionally overwrites geometry of suspect features.
#   - Ensure config file and Core Utils are up-to-date for reliable GPS correction.
# =============================================================================

import arcpy
from utils.smooth_gps_noise import smooth_gps_noise
from utils.correct_gps_outliers import correct_gps_outliers
from utils.manager.config_manager import ConfigManager


class SmoothGPSNoiseTool(object):
    def __init__(self):
        self.label = "04 - Smooth GPS Noise"
        self.description = ("Detect and flag GPS outlier points in a 360° Oriented Imagery Dataset using configurable "
                            "geometric and route-based rules.")
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
            displayName="Oriented Imagery Dataset",
            name="oid_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )
        oid_param.filter.list = []
        oid_param.description = "The Oriented Imagery Dataset feature class to analyze."
        params.append(oid_param)

        route_param = arcpy.Parameter(
            displayName="Reference Centerline (optional)",
            name="centerline_fc",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Input"
        )
        route_param.filter.list = ["Polyline"]
        route_param.description = "Optional M-aware centerline feature to validate points against known track/route."
        params.append(route_param)

        flag_param = arcpy.Parameter(
            displayName="Flag Only (No Geometry Updates)",
            name="flag_only",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        flag_param.value = False
        flag_param.description = "Only flag GPS outliers in QCFlag field without modifying geometry."
        params.append(flag_param)

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
        Executes the GPS outlier detection and correction workflow for an Oriented Imagery Dataset.
        
        Extracts tool parameters, runs GPS outlier detection using geometric and route-based rules, and optionally
        corrects flagged GPS points based on configuration settings. The correction step is skipped if flag-only mode
        is enabled.
        """
        project_folder = parameters[0].valueAsText
        oid_fc = parameters[1].valueAsText
        centerline_fc = parameters[2].valueAsText if parameters[1].value else None
        flag_only = parameters[3].value if parameters[2].altered else False
        config_file = parameters[4].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )
        logger = cfg.get_logger()

        logger.info("\n--- Running GPS Outlier Detection ---")
        smooth_gps_noise(
            cfg=cfg,
            oid_fc=oid_fc,
            centerline_fc=centerline_fc
        )

        if not flag_only:
            logger.info("\n--- Correcting Flagged GPS Points ---")
            correct_gps_outliers(
                cfg=cfg,
                oid_fc=oid_fc
            )
        else:
            logger.info("\nSkipped correction step (flag-only mode or no outliers).")

        logger.info("\n--- SmoothGPSNoise Tool Complete ---")
