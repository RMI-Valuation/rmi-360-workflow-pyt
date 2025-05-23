# =============================================================================
# 🌐 OID Service Publisher (utils/generate_oid_service.py)
# -----------------------------------------------------------------------------
# Purpose:             Publishes an OID as a hosted Oriented Imagery Service on ArcGIS Online
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.1
# Author:              RMI Valuation, LLC
# Created:             2025-05-14
# Last Updated:        2025-05-22
#
# Description:
#   Duplicates an existing OID feature class and updates its ImagePath values to point to
#   published S3 URLs. Validates AWS configuration, generates service metadata from config
#   expressions, and creates a portal folder if needed before publishing the OID using
#   ArcGIS Pro’s `GenerateServiceFromOrientedImageryDataset` tool.
#
# File Location:        /utils/generate_oid_service.py
# Validator:            /utils/validators/generate_oid_service_validator.py
# Called By:            tools/generate_oid_service_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    arcpy, arcgis.gis, os, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/generate_oid_service.md
#
# Notes:
#   - Automatically checks/creates portal folder prior to publishing
#   - Logs full service parameters for debugging and transparency
# =============================================================================

__all__ = ["generate_oid_service"]

import arcpy
import os
from typing import Literal
from arcgis.gis import GIS
from urllib.parse import quote_plus

from utils.manager.config_manager import ConfigManager


def build_s3_url(bucket, region, bucket_folder, filename):
    return (
        f"https://{bucket}.s3.{region}.amazonaws.com/"
        f"{quote_plus(bucket_folder)}/{quote_plus(filename)}")


def update_oid_image_paths(oid_fc, bucket, region, bucket_folder, logger):
    updated_count = 0
    with arcpy.da.UpdateCursor(oid_fc, ["ImagePath"]) as cursor:
        for row in cursor:
            local_path = row[0]
            filename = os.path.basename(local_path)
            aws_url = build_s3_url(bucket, region, bucket_folder, filename)
            row[0] = aws_url
            cursor.updateRow(row)
            updated_count += 1
    logger.info(f"Updated {updated_count} image paths to AWS URLs.", indent=2)
    return updated_count


def assemble_service_metadata(cfg, oid_name):
    service_name = f"{oid_name}"
    portal_folder = cfg.resolve(cfg.get("portal.project_folder", ""))
    share_with: Literal["PRIVATE", "ORGANIZATION", "PUBLIC"] = cfg.get("portal.share_with", "PRIVATE")  # type: ignore
    add_footprint: Literal["FOOTPRINT", "NO_FOOTPRINT"] = cfg.get("portal.add_footprint", "FOOTPRINT")  # type: ignore
    tags_list = [cfg.resolve(t) for t in cfg.get("portal.portal_tags", [])]
    tags_str = ", ".join(tags_list)
    summary = cfg.resolve(cfg.get("portal.summary", ""))
    return service_name, portal_folder, share_with, add_footprint, tags_str, summary


def ensure_portal_folder(gis, portal_folder, logger):
    try:
        user = gis.users.me
        existing_folders = []

        try:
            folder_gen = gis.content.folders.list(owner=user)
        except Exception as e:
            logger.error(f"Failed to list folders for user '{user.username}': {e}", error_type=RuntimeError)
            return

        for f in folder_gen:
            folder_name = None

            # First try: preferred attribute access (Folder objects in API 2.4.0+)
            if hasattr(f, "name"):
                folder_name = f.name

            # Fallback: dictionary-style access if it's a dict or other edge type
            elif isinstance(f, dict):
                folder_name = f.get("title") or f.get("name")

            if folder_name:
                existing_folders.append(folder_name)
            else:
                logger.warning(f"⚠️ Could not extract folder name from object: {type(f)} → {f}", indent=2)

        if portal_folder not in existing_folders:
            logger.info(f"Portal folder '{portal_folder}' does not exist. Attempting to create it...", indent=2)
            try:
                gis.content.folders.create(folder=portal_folder, owner=user)
                logger.info(f"✅ Portal folder '{portal_folder}' created successfully.", indent=2)
            except Exception as e:
                logger.error(f"❌ Failed to create portal folder '{portal_folder}': {e}", error_type=RuntimeError)
        else:
            logger.info(f"📁 Portal folder found: {portal_folder}", indent=2)

    except Exception as e:
        logger.error(f"Portal folder check failed due to unexpected error: {e}", indent=2, error_type=RuntimeError)


def generate_oid_service(cfg: ConfigManager, oid_fc: str):
    """
    Duplicates an Oriented Imagery Dataset, updates image paths to AWS S3 URLs, and publishes it as a hosted Oriented
    Imagery Service on ArcGIS Online.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="generate_oid_service")
    logger.info("Starting OID Service Generation...", indent=1)

    # Required AWS details
    bucket = cfg.get("aws.s3_bucket")
    region = cfg.get("aws.region")
    bucket_folder = cfg.resolve(cfg.get("aws.s3_bucket_folder"))
    if not all([bucket, region, bucket_folder]):
        logger.error("Missing required AWS values in config.yaml", error_type=ValueError, indent=2)
        return None  # or `raise ValueError("AWS configuration incomplete")`

    # Derive output AWS OID path
    oid_gdb = os.path.dirname(oid_fc)
    oid_name = os.path.splitext(os.path.basename(oid_fc))[0]
    aws_oid_name = f"{oid_name}_aws"
    aws_oid_fc = os.path.join(oid_gdb, aws_oid_name)

    # Step 1: Duplicate the OID feature class
    if arcpy.Exists(aws_oid_fc):
        logger.info(f"Overwriting existing AWS OID: {aws_oid_fc}", indent=2)
        arcpy.management.Delete(str(aws_oid_fc))
    arcpy.management.Copy(str(oid_fc), str(aws_oid_fc))
    logger.info(f"Duplicated OID to: {aws_oid_fc}", indent=2)

    # Step 2: Update ImagePath values
    update_oid_image_paths(aws_oid_fc, bucket, region, bucket_folder, logger)

    # Step 3: Assemble service metadata
    service_name, portal_folder, share_with, add_footprint, tags_str, summary = assemble_service_metadata(cfg, oid_name)

    # Step 4: Check/create portal folder
    try:
        gis = GIS("pro")
        ensure_portal_folder(gis, portal_folder, logger)
    except Exception as e:
        logger.warning(f"Unable to check portal folders: {e}", indent=2)

    # Step 5: Publish using arcpy
    logger.custom("Service generation parameters:", indent=2, emoji="📦")
    logger.info(f"in_oriented_imagery_dataset: {aws_oid_fc}", indent=3)
    logger.info(f"service_name: {service_name}", indent=3)
    logger.info(f"portal_folder: {portal_folder}", indent=3)
    logger.info(f"share_with: {share_with}", indent=3)
    logger.info(f"add_footprint: {add_footprint}", indent=3)
    logger.info("attach_images: NO_ATTACH", indent=3)
    logger.info(f"tags: {tags_str}", indent=3)
    logger.info(f"summary: {summary}", indent=3)

    try:
        arcpy.oi.GenerateServiceFromOrientedImageryDataset(
            in_oriented_imagery_dataset=aws_oid_fc,
            service_name=service_name,
            portal_folder=portal_folder,
            share_with=share_with,
            add_footprint=add_footprint,
            attach_images="NO_ATTACH",
            tags=tags_str,
            summary=summary
        )
        logger.success(f"OID service '{service_name}' published successfully.", indent=1)
    except Exception as e:
        gp_messages = arcpy.GetMessages()
        logger.warning(f"ArcPy tool failed: {e}\n{gp_messages}", indent=1)
        raise
    return service_name
