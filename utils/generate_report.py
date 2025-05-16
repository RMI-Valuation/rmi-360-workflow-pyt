# =============================================================================
# 📝 HTML Report Generator (utils/generate_report.py)
# -----------------------------------------------------------------------------
# Purpose:             Generates an HTML report with charts and branding from pipeline run data
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Loads report data from JSON, attaches config if needed, and uses Jinja2 templates to render
#   a project summary report in HTML format. Also creates bar charts visualizing image counts
#   per reel and execution times per step. Designed for use after a full pipeline run.
#
# File Location:        /utils/generate_report.py
# Called By:            tools/generate_report_tool.py, process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    jinja2, matplotlib, json, os, re, pathlib, datetime, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/generate_report.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Automatically locates and injects logo and templates from the config directory
#   - Skips chart generation if no steps or reels data are present
# =============================================================================

import os
import re
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

from utils.manager.config_manager import ConfigManager


def generate_full_process_report(
    report_data: Dict[str, Any],
    cfg: ConfigManager,
    output_basename: str = "report"
):
    """
    Generates a process report in HTML format, including charts and branding.
    
    Creates the output directory if needed, adds a UTC timestamp to the report data, generates chart images for images
    per reel and step execution times, and renders an HTML report using a Jinja2 template. Returns the paths to the
    generated HTML file.
    
    Args:
        report_data: Dictionary containing report content, including reels, steps, and configuration.
        cfg (ConfigManager): Active configuration context, used for logging and path resolution.
        output_basename: Base filename for the output report files (default is "report").
    
    Returns:
        A dictionary with key "html_path" indicating the location of the generated file.
    """
    logger = cfg.get_logger()
    paths = cfg.paths
    output_dir = paths.report
    os.makedirs(output_dir, exist_ok=True)
    report_data["generated_on"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Generate charts
    try:
        chart_path_images = os.path.join(output_dir, "chart_images_per_reel.png")
        chart_path_steps = os.path.join(output_dir, "chart_step_times.png")

        # Verify output directory is writable
        test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (PermissionError, OSError) as e:
            logger.warning(f"Chart output directory not writable: {e}", indent=2)
            raise

        plot_images_per_reel(report_data.get("reels", []), chart_path_images)
        plot_time_per_step(report_data.get("steps", []), chart_path_steps, logger)
    except Exception as e:
        logger.warning(f"Failed to generate charts: {e}", indent=2)

    try:
        template_dir = paths.templates

        # Resolve absolute path to the logo file so it works in browser view
        logo = cfg.get("logs.logo_filename")
        logo_path = Path(template_dir) / "assets" / logo
        report_data["logo_path"] = logo_path.resolve().as_uri()

        logger.debug(f"Template dir: {template_dir}", indent=2)
        template_path = os.path.join(template_dir, "process_report_template.html")
        logger.debug(f"Checking for template file: {template_path}", indent=2)

        if not os.path.exists(template_path):
            logger.error("Template file not found at: {template_path}", indent=3)
            logger.error(f"Please ensure the template exists in the {template_dir} directory", indent=3)
            logger.error(f"or check the 'template.templates_dir' setting in your configuration.", indent=3)
            logger.error("Report generation failed.", indent=2, error_type=FileNotFoundError)

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html'])
        )

        template = env.get_template("process_report_template.html")
        html_out = template.render(**report_data)

        html_path = os.path.join(output_dir, f"{output_basename}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)

        logger.success(f"HTML report written to: {html_path}", indent=1)

        return {
            "html_path": str(html_path)
        }

    except Exception as e:
        logger.error(f"Report generation failed: {type(e).__name__}: {e}", indent=1)
        raise  # Re-raise if you want upstream logic to catch this


def plot_images_per_reel(reels, output_path):
    """
    Generates and saves a bar chart visualizing the number of images for each reel.
    
    Args:
        reels: List of dictionaries containing reel information, each with 'reel' and 'image_count' keys.
        output_path: Path where the generated PNG chart will be saved.
    
    If the reels list is empty or None, no chart is generated.
    """
    if not reels:
        return
    reels_sorted = sorted(reels, key=lambda r: r["reel"])
    reel_ids = [f"RL{r['reel']}" for r in reels_sorted]
    counts = [r.get("image_count", 0) for r in reels_sorted]

    plt.figure(figsize=(10, 4))
    plt.bar(reel_ids, counts, color="#337ab7")
    plt.xlabel("Reel")
    plt.ylabel("Image Count")
    plt.title("Image Count per Reel")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def extract_time_seconds(time_str: str) -> float:
    """Extracts the numeric value from a time string."""
    m = re.match(r"([0-9.]+)", str(time_str))
    return float(m.group(1)) if m else 0.0


def plot_time_per_step(steps, output_path, logger):
    """
    Generates a horizontal bar chart of execution times for completed workflow steps.
    
    Filters steps with a status of "✅", extracts their names and execution times, and saves
    the resulting chart as a PNG file to the specified output path.
    """
    steps_filtered = [s for s in steps if s["status"] == "✅"]
    step_names = [s["name"] for s in steps_filtered]
    times_sec = []

    # Process the execution times for each step
    for s in steps_filtered:
        try:
            times_sec.append(extract_time_seconds(s["time"]))  # Use the new time extraction function
        except Exception as e:
            # Log any errors, though the function itself ensures 0.0 is returned for invalid times
            logger.warning(f"Error processing time for step {s['name']}: {e}", indent=1)
            times_sec.append(0.0)

    # Generate the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(step_names, times_sec, color="#5cb85c")
    plt.xlabel("Time (seconds)")
    plt.title("Execution Time per Workflow Step")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report_from_json(cfg: ConfigManager, json_path: str):
    """
    Generates a process report from a JSON file.
    
    Loads report data from the specified JSON file, attaches configuration data if provided or available, determines
    the output directory, and generates an HTML report. Raises an exception if report generation fails.
    """
    logger = cfg.get_logger()

    logger.info("Starting report generation...", indent=1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        
        logger.custom(f"Loaded report data from: {json_path}", emoji="📄", indent=2)

        # Reattach config if missing or externally supplied
        if cfg:
            report_data["config"] = cfg
        else:
            logger.warning("Config not found in report JSON — some paths or logos may not resolve", indent=2)

        return generate_full_process_report(
            report_data=report_data,
            cfg=cfg
        )

    except Exception as e:
        logger.error(f"Failed to generate report from JSON: {e}", indent=1)
        raise
