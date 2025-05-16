# =============================================================================
# 🏷️ Rename and Tag Images (tools/rename_and_tag_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          RenameAndTagImagesTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class to rename images based on project metadata and EXIF/XMP tagging rules
#   defined in a YAML configuration file. Updates both filenames and embedded metadata, and optionally
#   deletes original images post-renaming. Integrates with Core Utils for batch renaming, metadata updating,
#   and configuration management.
#
# File Location:      /tools/rename_and_tag_tool.py
# Core Utils:
#   - utils/rename_images.py
#   - utils/apply_exif_metadata.py
#   - utils/manager/config_manager.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/rename_and_tag.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Oriented Imagery Dataset (OID) {oid_fc} (Feature Class): The Oriented Imagery Dataset to process.
#   - Delete Original Files After Rename? {delete_originals} (Boolean): If checked, deletes the original image files after renaming.
#   - Config File {config_file} (File): Optional YAML config file defining metadata tags and naming convention.
#
# Notes:
#   - Renaming logic supports dynamic filename expression resolution.
#   - EXIF metadata updates are performed using ExifTool via batch mode.
#   - Ensure config file and ExifTool installation are correct for successful renaming and tagging.
# =============================================================================

import arcpy
from utils.rename_images import rename_images
from utils.apply_exif_metadata import update_metadata_from_config
from utils.manager.config_manager import ConfigManager


class RenameAndTagImagesTool(object):
    def __init__(self):
        self.label = "05 - Rename and Tag Images"
        self.description = (
            "Renames images based on project metadata and updates EXIF/XMP tags, "
            "using rules defined in the configuration."
        )
        self.category = "Individual Tools"

    def getParameterInfo(self):
        """
        Defines the input parameters for the Rename and Tag Images geoprocessing tool.
        
        Returns:
            A list of ArcPy Parameter objects for the oriented imagery dataset, deletion flag, and optional
            configuration file.
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

        oid_param = arcpy.Parameter(
            displayName="Oriented Imagery Dataset (OID)",
            name="oid_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )
        oid_param.description = "The Oriented Imagery Dataset to process."
        params.append(oid_param)

        delete_orig = arcpy.Parameter(
            displayName="Delete Original Files After Rename?",
            name="delete_originals",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        delete_orig.value = False
        delete_orig.description = "If checked, original image files will be deleted after renaming."
        params.append(delete_orig)

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
        Executes the image renaming and metadata tagging workflow for the specified dataset.
        
        This method renames image files and updates their EXIF/XMP metadata according to rules defined in a
        configuration file. It logs progress and results throughout the process.
        """
        project_folder = parameters[0].value
        oid_fc = parameters[1].valueAsText
        delete_originals = parameters[2].value
        config_file = parameters[3].valueAsText

        cfg = ConfigManager.from_file(
            path=config_file,  # may be None
            project_base=project_folder,
            messages=messages
        )
        logger = cfg.get_logger()

        logger.info("--- Starting Mosaic 360 Workflow ---")

        # Step 1: Rename Images
        logger.info("🔁 Step 1: Renaming images...")
        renamed_images = rename_images(
            cfg=cfg,
            oid_fc=oid_fc,
            delete_originals=delete_originals
        )
        logger.info(f"✅ {len(renamed_images)} images processed.")

        # Step 2: Update Metadata
        logger.info("📝 Step 2: Updating EXIF/XMP metadata...")
        update_metadata_from_config(
            cfg=cfg,
            oid_fc=oid_fc
        )

        logger.info("--- Mosaic 360 Workflow Complete ---")
