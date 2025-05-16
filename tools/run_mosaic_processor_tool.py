# =============================================================================
# 🎞️ Run Mosaic Processor (tools/run_mosaic_processor_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          RunMosaicProcessorTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class that wraps the Mosaic Processor command-line tool to render 360° video imagery
#   and apply GPX integration. Requires a GRP file from Mosaic and proper installation of Mosaic Stitcher
#   or MistikaVR. Can render full reels or specified frame ranges and supports config override. Integrates
#   with Core Utils for configuration, command execution, and logging.
#
# File Location:      /tools/run_mosaic_processor_tool.py
# Core Utils:
#   - utils/mosaic_processor.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/run_mosaic_processor.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Project Folder {project_folder} (Folder): Root folder for the Mosaic 360 project. Logs and imagery will be saved here.
#   - Input Reels Folder {input_dir} (Folder): Directory containing raw `.mp4` video folders from the Mosaic 360 camera.
#   - Config File {config_file} (File): Optional path to a YAML config file. If omitted, uses the default.
#   - Mosaic GRP Template Path (optional) {grp_path} (File): GRP calibration file for stitching, typically provided by Mosaic.
#   - Start Frame (optional) {start_frame} (Long): First frame to render. Leave blank to start from beginning.
#   - End Frame (optional) {end_frame} (Long): Last frame to render. Leave blank to render until end.
#
# Notes:
#   - Requires Mosaic Processor and GRP calibration files installed separately.
#   - Supports partial reel rendering via start/end frame range.
#   - Logs command and reel metadata to project output directory.
#   - Ensure Mosaic Processor, GRP, and config files are present and valid for successful processing.
# =============================================================================

import arcpy
from utils.mosaic_processor import run_mosaic_processor
from utils.manager.config_manager import ConfigManager


class RunMosaicProcessorTool(object):
    def __init__(self):
        self.label = "01 - Run Mosaic Processor"
        self.description = (
            "Runs Mosaic Processor to render 360° imagery and apply GPX integration. "
            "Requires a GRP template file provided by Mosaic and a properly licensed installation of MistikaVR or "
            "Mosaic Stitcher. Mosaic Processor must be installed and configured separately."
        )
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the Run Mosaic Processor ArcPy tool.
        
        Returns:
            A list of ArcPy Parameter objects specifying required and optional inputs for
            configuring a Mosaic 360 imagery processing project, including project folder,
            input reels folder, optional configuration and GRP files, and optional frame range.
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

        # Input folder for raw reel files
        input_param = arcpy.Parameter(
            displayName="Input Reels Folder",
            name="input_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        input_param.description = "Folder containing the raw video reels captured by the Mosaic 360 camera."
        params.append(input_param)

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

        # Optional start frame
        start_param = arcpy.Parameter(
            displayName="Start Frame (optional)",
            name="start_frame",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        start_param.description = "Start frame number for partial rendering. Leave blank to start from the beginning."
        params.append(start_param)

        # Optional end frame
        end_param = arcpy.Parameter(
            displayName="End Frame (optional)",
            name="end_frame",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        end_param.description = "End frame number for partial rendering. Leave blank to render through the last frame."
        params.append(end_param)

        return params

    def execute(self, parameters, messages):
        """
        Executes the Mosaic Processor tool using parameters provided by the ArcPy framework.
        
        Extracts input values for the Mosaic 360 imagery project, including project folder, input reels, configuration,
        GRP template, and optional frame range, then invokes the Mosaic Processor with these settings.
        """
        project_folder = parameters[0].valueAsText
        input_dir = parameters[1].valueAsText
        config_file = parameters[2].valueAsText
        start_frame = parameters[3].valueAsText
        end_frame = parameters[4].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )

        run_mosaic_processor(
            cfg=cfg,
            input_dir=input_dir,
            start_frame=start_frame or None,
            end_frame=end_frame or None
        )
