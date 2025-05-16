# =============================================================================
# 🔢 Group Index Assignment Logic (utils/assign_group_index.py)
# -----------------------------------------------------------------------------
# Purpose:             Assigns cyclic GroupIndex values to OID features based on AcquisitionDate
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Loads features from an OID feature class, sorts by AcquisitionDate, and assigns a
#   repeating GroupIndex value (1 to N) to support scalable image display intervals
#   in ArcGIS Pro. Handles missing dates and schema validation.
#
# File Location:        /utils/assign_group_index.py
# Validator:            /utils/validators/assign_group_index_validator.py
# Called By:            tools/add_images_to_oid_tool.py
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    arcpy
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/assign_group_index.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Raises if GroupIndex field is missing or AcquisitionDate is null
#   - Supports configurable group size for flexible display
# =============================================================================

__all__ = ["assign_group_index"]

import arcpy

from utils.manager.config_manager import ConfigManager


def assign_group_index(
    cfg: ConfigManager,
    oid_fc_path: str,
    group_size: int = 4,
):
    """
    Assigns a cyclic group index to features in an Oriented Imagery Dataset based on acquisition date.
    
    Features are sorted by their AcquisitionDate attribute, and a repeating group index (from 1 to group_size) is
    assigned to each. This enables variable display intervals for imagery in ArcGIS Pro. Raises a RuntimeError if the
    group index field is missing, or a ValueError if any features have null AcquisitionDate values.
    
    Args:
        cfg: Validated configuration manager.
        oid_fc_path: Path to the Oriented Imagery Dataset feature class.
        group_size: Number of images per group cycle (default is 4).
    """
    logger = cfg.get_logger()
    cfg.validate(tool="assign_group_index")

    if not isinstance(group_size, int) or group_size <= 0:
        logger.error(f"Group size must be a positive integer, got {group_size}", error_type = ValueError, indent=1)
        return

    field_name = cfg.get("oid_schema_template.grp_idx_fields.group_index.name", "GroupIndex")
    logger.custom(f"Assigning {field_name} values to features, sorted by AcquisitionDate...", indent=1, emoji="🧭")

    # Ensure the field exists
    existing_fields = [f.name for f in arcpy.ListFields(oid_fc_path)]
    if field_name not in existing_fields:
        logger.error(f"Field '{field_name}' not found in feature class. Please ensure it is included in your schema.",
                     error_type=RuntimeError, indent=2)
        return

    # Step 1: Get all AcquisitionDates with OIDs
    rows = []
    with arcpy.da.SearchCursor(oid_fc_path, ["OID@", "AcquisitionDate"]) as cursor:
        for oid, acq_date in cursor:
            rows.append((oid, acq_date))

    # Step 2: Check for null AcquisitionDates (not allowed)
    null_oids = [oid for oid, acq_date in rows if acq_date is None]
    if null_oids:
        logger.error(f"{len(null_oids)} features have null AcquisitionDate values: {null_oids}", error_type=ValueError, indent=2)
        return

    # Step 3: Sort by AcquisitionDate
    rows.sort(key=lambda r: r[1])

    # Step 4: Compute group index (1-based)
    oid_to_index = {
        oid: (i % group_size) + 1 for i, (oid, _) in enumerate(rows)
    }

    # Step 5: Write GroupIndex values
    updated = 0
    with cfg.get_progressor(total=len(oid_to_index), label="Assigning GroupIndex") as progressor:
        with arcpy.da.UpdateCursor(oid_fc_path, ["OID@", field_name]) as cursor:
            for i, (oid, _current) in enumerate(cursor, start=1):
                new_val = oid_to_index.get(oid)
                if new_val is not None:
                    try:
                        cursor.updateRow((oid, new_val))
                        updated += 1
                    except Exception as e:
                        logger.warning(f"Failed to update GroupIndex for OID {oid}: {e}", indent=1)
                progressor.update(i)


    logger.success(f"Assigned GroupIndex values to {updated} feature(s).", indent=1)
    logger.custom("Tip: Use GroupIndex values to control image display intervals in ArcGIS Pro:", indent=1, emoji="🧠")
    logger.custom("5m = all images (no filter)", indent=2, emoji="🧠")
    logger.custom("10m = GroupIndex IN (1, 3) or (2, 4)", indent=2, emoji="🧠")
    logger.custom("20m = GroupIndex = 1 (or 2, 3, or 4)", indent=2, emoji="🧠")

