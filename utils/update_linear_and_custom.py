# =============================================================================
# 🧭 Linear Referencing + Custom Attribute Updater (utils/update_linear_and_custom.py)
# -----------------------------------------------------------------------------
# Purpose:             Assigns route ID, milepost, and custom fields to OID features using config-driven expressions
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Performs linear referencing of image points against an M-enabled centerline using LocateFeaturesAlongRoutes.
#   Updates route and MP fields as well as user-defined custom fields from config expressions.
#   Supports dynamic expression evaluation and field type coercion with warning messaging on failure.
#
# File Location:        /utils/update_linear_and_custom.py
# Validator:            /utils/validators/update_linear_and_custom_validator.py
# Called By:            tools/update_linear_and_custom_tool.py, orchestrator
# Int. Dependencies:    utils.manager.config_manager, utils.shared.expression_utils
# Ext. Dependencies:    arcpy, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/update_linear_and_custom.md
#
# Notes:
#   - Automatically projects OID to match centerline SR for referencing
#   - Supports optional disabling of linear referencing via config/tool parameter
# =============================================================================

__all__ = ["update_linear_and_custom"]

import arcpy
from typing import Optional

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import resolve_expression


def get_located_points(oid_fc: str, centerline_fc: str, route_id_field:str, logger) -> dict:
    """
    Finds the route identifier and milepost value for each point in the OID feature class by projecting it to the
    centerline's spatial reference and locating features along routes.
    
    Projects the input feature class to match the centerline's spatial reference, computes a suitable search tolerance
    based on the maximum distance from points to routes, and uses ArcPy's LocateFeaturesAlongRoutes to associate each
    point with its nearest route and milepost value.
    
    Returns:
        dict: A mapping from each object ID (OID) to a dictionary with 'route_id' and 'mp_value' keys.
    """
    try:
        # Get target spatial reference
        route_sr = arcpy.Describe(centerline_fc).spatialReference

        #Project OID feature class once
        projected_oid_fc = arcpy.management.Project(
            oid_fc,
            arcpy.CreateUniqueName("projected_oid_fc", arcpy.env.scratchGDB),
            route_sr
        )[0]

        # Load all route geometries
        routes = [row[0] for row in arcpy.da.SearchCursor(centerline_fc, ["SHAPE@"])]

        # Compute adaptive tolerance based on max distance from OID points to nearest route
        max_dist = 0
        with arcpy.da.SearchCursor(projected_oid_fc, ["SHAPE@"]) as cursor:
            for (point_geom,) in cursor:
                point = point_geom.centroid
                min_dist = min(route.queryPointAndDistance(point, use_percentage=False)[2] for route in routes)
                max_dist = max(max_dist, min_dist)
        tolerance = round(max_dist + 5, 2)

        logger.info(f"📏 Max distance to nearest route: {round(max_dist, 2)} → Using {round(tolerance, 2)} tolerance", indent=1)

        # Perform location along routes
        oid_table = arcpy.CreateUniqueName("oid_temp_table", arcpy.env.scratchGDB)
        arcpy.lr.LocateFeaturesAlongRoutes(
            in_features=projected_oid_fc,
            in_routes=centerline_fc,
            route_id_field=route_id_field,
            radius_or_tolerance=tolerance,
            out_table=oid_table,
            out_event_properties=f"{route_id_field} POINT MP",
            route_locations="FIRST",
            distance_field="NO_DISTANCE",
            in_fields="NO_FIELDS"
        )

        # Parse result table into a dict
        oid_to_loc = {}
        with arcpy.da.SearchCursor(oid_table, [route_id_field, "MP", "OID@"]) as cursor:
            for route_id_val, mp, oid in cursor:
                oid_to_loc[oid] = {"route_id": route_id_val, "mp_value": mp}
        return oid_to_loc

    except Exception as e:
        logger.warning(f"Linear referencing failed: {e}", indent=1)
        return {}


from typing import List, Dict, Any, Tuple

def compute_linear_and_custom_updates(
    cfg: ConfigManager,
    row: List[Any],
    update_fields: List[str],
    linear_fields: Dict,
    custom_field_defs: List[Tuple[str, str, str, str]],
    oid_to_loc: Dict,
    enable_linear_ref: bool,
    logger=None
) -> Tuple[List[Any], bool]:
    """
    Given a row and config, return the updated row and a boolean indicating if it was changed.
    """
    update = False
    context = dict(zip(update_fields, row))
    oid = row[0]
    # Linear reference updates
    if enable_linear_ref:
        loc = oid_to_loc.get(oid)
        route_id = mp_value = None
        if loc:
            route_id = loc.get("route_id")
            mp_value = loc.get("mp_value")
        for key, field_def in linear_fields.items():
            field_name = field_def.get("name")
            if key == "route_identifier":
                value = route_id
            elif key == "route_measure":
                value = mp_value
            else:
                continue
            idx = update_fields.index(field_name)
            if field_def["type"] == "DOUBLE":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    if logger:
                        logger.warning(f"Could not convert value for {field_name} to float.", indent=2)
                    value = None
            row[idx] = value
            update = True
    # Custom field updates
    for key, target_field, expression, field_type in custom_field_defs:
        try:
            # resolve_expression may raise
            value = resolve_expression(expression, cfg, row=context)
            if field_type == "DOUBLE":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    if logger:
                        logger.warning(f"Could not convert value for {target_field} to float.", indent=2)
                    continue
            row[update_fields.index(target_field)] = value
            update = True
        except Exception as e:
            if logger:
                logger.warning(f"Failed to resolve expression for {target_field}: {e}", indent=2)
    return row, update

def update_linear_and_custom(
        cfg: ConfigManager,
        oid_fc_path: str,
        centerline_fc: Optional[str] = None,
        route_id_field: Optional[str] = None,
        enable_linear_ref: bool = True):
    """
    Updates linear referencing and custom attribute fields for an Oriented Imagery Dataset feature class.

    If linear referencing is enabled, locates each image point along M-enabled centerlines and updates route identifier
    and milepost fields. Also evaluates and updates custom attribute fields based on configured expressions.

    Args:
        cfg: ConfigManager instance (must be validated).
        oid_fc_path: Path to the OID feature class.
        centerline_fc: Path to centerline routes (optional).
        route_id_field: Field name used for route matching (if linear ref is enabled).
        enable_linear_ref: Whether to compute linear route measures.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="update_linear_and_custom")

    # Load linear field definitions
    linear_fields = cfg.get("oid_schema_template.linear_ref_fields", {}) if enable_linear_ref else {}
    route_id_field_config = cfg.get("oid_schema_template.linear_ref_fields.route_identifier.name")
    route_meas_field_config = cfg.get("oid_schema_template.linear_ref_fields.route_measure.name")
    linear_field_names = [route_id_field_config, route_meas_field_config] if enable_linear_ref else []

    # Load custom field definitions
    custom_fields = cfg.get("oid_schema_template.custom_fields", {})
    custom_field_defs = [
        (key, field["name"], field.get("expression"), field.get("type"))
        for key, field in custom_fields.items()
        if "expression" in field
    ]
    custom_field_names = [name for _, name, _, _ in custom_field_defs]

    update_fields = ["OID@"] + linear_field_names + custom_field_names

    # 🔁 Only run linear referencing if requested
    oid_to_loc = {}
    if enable_linear_ref and centerline_fc and route_id_field:
        oid_to_loc = get_located_points(oid_fc_path, centerline_fc, route_id_field, logger)

    row_count = int(arcpy.management.GetCount(oid_fc_path)[0])

    # Update records
    updated_oids = set()
    failed_oids = set()
    with cfg.get_progressor(total=row_count, label="Updating linear/custom fields") as progressor:
        with arcpy.da.UpdateCursor(oid_fc_path, update_fields) as cursor:
            for i, row in enumerate(cursor, start=1):
                oid = row[0]
                try:
                    new_row, update = compute_linear_and_custom_updates(
                        cfg=cfg,
                        row=list(row),
                        update_fields=update_fields,
                        linear_fields=linear_fields,
                        custom_field_defs=custom_field_defs,
                        oid_to_loc=oid_to_loc,
                        enable_linear_ref=enable_linear_ref,
                        logger=logger
                    )
                    if update:
                        cursor.updateRow(new_row)
                        updated_oids.add(oid)
                except Exception as e:
                    failed_oids.add(oid)
                    logger.error(f"Failed to update OID {oid}: {e}", indent=2)
                progressor.update(i)
    logger.success(f"Updated {len(updated_oids)} feature(s) with linear and custom attributes." + (f" Failed to update {len(failed_oids)} OIDs." if failed_oids else ""), indent=1)
    if updated_oids:
        logger.debug(f"Updated OIDs: {sorted(updated_oids)}", indent=2)
    if failed_oids:
        logger.warning(f"Failed OIDs: {sorted(failed_oids)}", indent=2)

