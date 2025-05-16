# =============================================================================
# 🛰️ GPS Noise Smoothing Logic (utils/smooth_gps_noise.py)
# -----------------------------------------------------------------------------
# Purpose:             Detects and flags suspect GPS points in an OID using spatial + geometric checks
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Analyzes GPS tracks in an OID feature class to identify potential outlier points based on
#   spatial deviation, angle, spacing, and proximity to a centerline (if provided). Flags points
#   by updating a QCFlag field and optionally logs debug metrics to CSV for inspection.
#
# File Location:        /utils/smooth_gps_noise.py
# Validator:            /utils/validators/smooth_gps_noise_validator.py
# Called By:            tools/smooth_gps_noise_tool.py, orchestrator
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    arcpy, csv, math, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/smooth_gps_noise.md
#
# Notes:
#   - Flags GPS outliers using configurable multi-criteria scoring
#   - Optionally integrates centerline route distance checks if provided
# =============================================================================

__all__ = ["smooth_gps_noise"]

import arcpy
import csv
from typing import Optional, Dict, List
from math import radians, sin, cos, sqrt, atan2, degrees

from utils.manager.config_manager import ConfigManager


def haversine(x1, y1, x2, y2):
    """
    Calculates the great-circle distance in meters between two latitude/longitude points.
    
    Args:
        x1: Longitude of the first point in decimal degrees.
        y1: Latitude of the first point in decimal degrees.
        x2: Longitude of the second point in decimal degrees.
        y2: Latitude of the second point in decimal degrees.
    
    Returns:
        The distance between the two points in meters.
    """
    r = 6371000  # Earth radius in meters
    phi1, phi2 = radians(y1), radians(y2)
    d_phi = radians(y2 - y1)
    d_lambda = radians(x2 - x1)
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def angle_between(p1, p2, p3):
    """Calculate the angle at p2 formed by three points."""
    dx1, dy1 = p1[0] - p2[0], p1[1] - p2[1]
    dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
    angle1 = atan2(dy1, dx1)
    angle2 = atan2(dy2, dx2)
    return abs(degrees(angle2 - angle1)) % 360


def process_gps_metrics(
        points: List[dict],
        cfg: ConfigManager,
        routes,
        route_sr,
        reel: str,
        global_counter: List[int],
        total: int,
        logger):

    window = cfg.get("gps_smoothing.smoothing_window")


    with cfg.get_progressor(total=total, label=f"Smoothing GPS (Reel {reel})") as progressor:
        for i in range(len(points)):
            p = points[i]
            prev_pt = points[i - window] if i >= window else None
            next_pt = points[i + window] if i + window < len(points) else None

            if prev_pt and next_pt:
                mid_x = (prev_pt["x"] + next_pt["x"]) / 2
                mid_y = (prev_pt["y"] + next_pt["y"]) / 2
                p["deviation"] = haversine(p["x"], p["y"], mid_x, mid_y)
                p["angle"] = angle_between((prev_pt["x"], prev_pt["y"]),
                                           (p["x"], p["y"]),
                                           (next_pt["x"], next_pt["y"]))
            else:
                p["deviation"] = 0
                p["angle"] = 180

            p["step"] = haversine(points[i - 1]["x"], points[i - 1]["y"], p["x"], p["y"]) if i > 0 else 0
            p["route_dist"] = 0

            if route_sr:
                try:
                    pt_geom = arcpy.PointGeometry(arcpy.Point(p["x"], p["y"]), arcpy.SpatialReference(4326))
                    pt_proj = pt_geom.projectAs(route_sr)
                    min_dist = float("inf")
                    for route in routes:
                        dist = route.queryPointAndDistance(pt_proj.centroid, use_percentage=False)[2]
                        min_dist = min(min_dist, dist)
                    p["route_dist"] = min_dist
                except Exception as e:
                    logger.warning(f"Projection failed for OID {p['oid']}: {e}")
                    p["route_dist"] = 0

            global_counter[0] += 1
            progressor.update(global_counter[0])

        # Route distance smoothing
        for i in range(1, len(points) - 1):
            prev, curr, nxt = points[i - 1], points[i], points[i + 1]
            avg = (prev["route_dist"] + nxt["route_dist"]) / 2
            curr["route_dev"] = abs(curr["route_dist"] - avg)
            global_counter[0] += 1
            progressor.update(global_counter[0])


def smooth_gps_noise(cfg: ConfigManager, oid_fc: str, centerline_fc: Optional[str] = None) -> None:
    """
    Detects and flags outlier GPS points in a feature class based on spatial and geometric criteria.

    Analyzes each GPS point for deviation from the local path, angle with neighbors, step distance, and (optionally)
    distance from a provided centerline route. Flags points as outliers if they meet or exceed a configurable threshold
    of outlier criteria. Optionally outputs a debug CSV summarizing metrics and reasons for outlier status, and updates
    a "QCFlag" field in the input feature class for flagged points.

    Args:
        cfg:
        oid_fc: Path to the input feature class containing GPS points.
        centerline_fc: Optional path to a centerline feature class for route-based deviation checks.

    """
    logger = cfg.get_logger()
    cfg.validate(tool="smooth_gps_noise")

    spacing = cfg.get("gps_smoothing.capture_spacing_meters", 5.0)
    deviation_thresh = cfg.get("gps_smoothing.deviation_threshold_m", 0.5)
    angle_bounds = cfg.get("gps_smoothing.angle_bounds_deg", [175, 185])
    proximity_range = cfg.get("gps_smoothing.proximity_check_range_m", 0.75)
    max_route_dev = cfg.get("gps_smoothing.max_route_dist_deviation_m", 0.5)
    outlier_threshold = cfg.get("gps_smoothing.outlier_reason_threshold", 2)

    log_csv_path = cfg.paths.get_log_file_path("gps_smooth_debug", cfg)

    # Extract points by Reel
    points_by_reel: Dict[str, List[dict]] = {}
    with arcpy.da.SearchCursor(oid_fc, ["OID@", "SHAPE@", "AcquisitionDate", "Reel"]) as cursor:
        for oid, shape, ts, reel in cursor:
            if not reel:
                reel = "__UNREEL__"
            pt = shape.centroid
            points_by_reel.setdefault(reel, []).append({
                "oid": oid,
                "x": pt.X,
                "y": pt.Y,
                "z": pt.Z if pt.Z is not None else 0,
                "shape": pt,
                "ts": ts,
                "reel": reel
            })

    for pts in points_by_reel.values():
        pts.sort(key=lambda p: p["ts"])

    # Load centerline routes
    routes = []
    route_sr = None
    if centerline_fc:
        with arcpy.da.SearchCursor(centerline_fc, ["SHAPE@"]) as cursor:
            for row in cursor:
                routes.append(row[0])
        if routes:
            route_sr = routes[0].spatialReference

    # Count total for progress tracking
    total_points = sum(len(v) for v in points_by_reel.values()) * 2
    global_counter = [0]

    for reel, pts in points_by_reel.items():
        process_gps_metrics(pts, cfg, routes, route_sr, reel, global_counter, total_points, logger)


    # Flag outliers
    outlier_oids = set()
    csv_rows = []
    all_pts = [p for sub in points_by_reel.values() for p in sub]


    for i, p in enumerate(all_pts):
        # Initial exclusion: first point can never be flagged
        if i == 0:
            p["is_outlier"] = False
        else:
            reasons = {
                "Deviation": p["deviation"] > deviation_thresh,
                "Angle": not (angle_bounds[0] <= p["angle"] <= angle_bounds[1]),
                "Step": p["step"] < spacing - proximity_range or p["step"] > spacing + proximity_range,
                "RouteDev": p.get("route_dev", 0) > max_route_dev
            }
            count = sum(reasons.values())
            p["is_outlier"] = count >= outlier_threshold
            p["Reason_Count"] = count
            p.update({f"Reason_{k}": v for k, v in reasons.items()})

        if p["is_outlier"]:
            outlier_oids.add(p["oid"])

        # Append to CSV only if at least one reason is True
        if p.get("Reason_Count", 0):
            reasons_str = [k for k in ["Deviation", "Angle", "Step", "RouteDev"] if p.get(f"Reason_{k}")]
            csv_rows.append({
                "OID": p["oid"],
                "Deviation": round(p["deviation"], 3),
                "Angle": round(p["angle"], 2),
                "Step": round(p["step"], 3),
                "RouteDist": round(p["route_dist"], 3),
                "RouteDistDev": round(p.get("route_dev", 0), 3),
                "Reason_Deviation": p.get("Reason_Deviation", False),
                "Reason_Angle": p.get("Reason_Angle", False),
                "Reason_Step": p.get("Reason_Step", False),
                "Reason_RouteDev": p.get("Reason_RouteDev", False),
                "Reason_Count": p.get("Reason_Count", 0),
                "Is_Outlier": p["is_outlier"],
                "QCReason": ", ".join(reasons_str)
            })

    # Propagate flags for sequences (surrounded by outliers)
    for i in range(1, len(all_pts) - 1):
        if not all_pts[i]["is_outlier"] and all_pts[i - 1]["is_outlier"] and all_pts[i + 1]["is_outlier"]:
            all_pts[i]["is_outlier"] = True
            outlier_oids.add(all_pts[i]["oid"])
            for row in csv_rows:
                if row["OID"] == all_pts[i]["oid"]:
                    row["Is_Outlier"] = True
                    break

    # Write CSV debug log
    if csv_rows:
        try:
            with open(log_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            logger.info(f"📄 Debug CSV written to: {log_csv_path}", indent=1)
        except Exception as e:
            logger.error(f"Could not write debug CSV to {log_csv_path}: {e}", indent=1)

    # Add QCFlag if missing
    existing_fields = [f.name for f in arcpy.ListFields(oid_fc)]
    if "QCFlag" not in existing_fields:
        arcpy.management.AddField(oid_fc, "QCFlag", "TEXT", field_length=50)

    if outlier_oids:
        with cfg.get_progressor(total=len(outlier_oids), label="Updating QCFlag") as update_prog:
            with arcpy.da.UpdateCursor(oid_fc, ["OID@", "QCFlag"]) as cursor:
                for oid, flag in cursor:
                    if oid in outlier_oids:
                        cursor.updateRow((oid, "GPS_OUTLIER"))
                        update_prog.update(1)
    else:
        logger.info("No outlier flags to apply.", indent=1)

    logger.success(f"Detected and flagged {len(outlier_oids)} GPS outlier(s).", indent=1)
    logger.info(
        f"Processed {sum(len(v) for v in points_by_reel.values())} GPS points across {len(points_by_reel)} reel(s).", indent=1)

