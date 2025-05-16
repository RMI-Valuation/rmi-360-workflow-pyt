# =============================================================================
# 🖥️ Mosaic Processor Wrapper (utils/mosaic_processor.py)
# -----------------------------------------------------------------------------
# Purpose:             Executes Mosaic Processor CLI to render, fix reels, and integrate GPS metadata
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Runs the Mosaic Processor in three sequential stages:
#   1. Render + Reel Fix
#   2. Frame number zero-padding
#   3. GPX metadata integration
#   Logs progress, validates config, and writes a single reel_info.json (optional).
#
# File Location:        /utils/mosaic_processor.py
# Validator:            /utils/validators/mosaic_processor_validator.py
# Called By:            tools/run_mosaic_processor_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager
# Ext. Dependencies:    subprocess, json, re, os, typing, pathlib
# External Tools:       Mosaic Processor CLI, MistikaVR (external dependency)
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/run_mosaic_processor.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Uses cfg.validate(tool="mosaic_processor") for schema enforcement
#   - .cfg file support is reserved for future implementation
#   - Output is auto-padded to 6-digit frame numbers post-render
#
# =============================================================================
import os
import subprocess
import json
import re
from pathlib import Path
from typing import Optional

from utils.manager.config_manager import ConfigManager

__all__ = ["run_mosaic_processor"]


def build_mosaic_command(
    exe_path,
    input_dir,
    output_dir,
    grp_path,
    start_frame=None,
    end_frame=None,
    skip_gpx=False,
    skip_render=False,
    skip_reel_fix=False,
    wrap_in_shell: bool = True
):
    """
    Constructs a command to run the Mosaic Processor executable with specified options.
    
    Args:
        exe_path: Path to the Mosaic Processor executable.
        input_dir: Directory containing input image frames.
        output_dir: Directory where output will be saved.
        grp_path: Path to the calibration .grp file.
        start_frame: Optional starting frame number to process.
        end_frame: Optional ending frame number to process.
        skip_gpx: If True, disables GPX integration.
        skip_render: If True, disables rendering.
        skip_reel_fix: If True, disables reel fixing.
        wrap_in_shell: If True, returns the command as a shell-wrapped string suitable for Windows; otherwise returns a
        list of command arguments.
    
    Returns:
        A shell-wrapped command string or a list of command arguments for subprocess execution, depending on
        wrap_in_shell.
    """
    cmd = [
        exe_path,
        input_dir,
        "--output_dir", output_dir,
        "--grp_path", grp_path
    ]

    if start_frame is not None:
        cmd += ["--start_frame", str(start_frame)]
    if end_frame is not None:
        cmd += ["--end_frame", str(end_frame)]
    if skip_gpx:
        cmd.append("--no_gpx_integration")
    if skip_render:
        cmd.append("--no_render")
    if skip_reel_fix:
        cmd.append("--no_reel_fixing")

    cmd = [str(c) for c in cmd if c is not None]

    if wrap_in_shell:
        quoted_cmd = [f'"{cmd[0]}"'] + [f'"{c}"' if " " in c else c for c in cmd[1:]]
        return f'cmd /c "{" ".join(quoted_cmd)}"'
    else:
        return cmd


def run_processor_stage(
    cfg: ConfigManager,
    input_dir: str,
    output_dir: Path,
    start_frame,
    end_frame,
    log_f,
    log_path,
    stage_name: str,
    skip_render=False,
    skip_reel_fix=False,
    skip_gpx=False,
) -> bool:
    """
    Runs a single Mosaic Processor CLI stage (Render or GPX Integration).

    Args:
        cfg: ConfigManager instance.
        input_dir: Input directory containing raw frames.
        output_dir: Target output directory.
        start_frame: Start frame (optional).
        end_frame: End frame (optional).
        log_f: File handle for writing logs.
        log_path: Path to log file.
        stage_name: Label for the stage ("Render + Reel Fix" or "GPX Integration").
        skip_render: If True, disables rendering.
        skip_reel_fix: If True, disables reel fixing.
        skip_gpx: If True, disables GPX integration.

    Returns:
        True if stage succeeded, False otherwise.
    """
    logger = cfg.get_logger()
    exe_path = cfg.paths.mosaic_processor_exe
    grp_path = cfg.paths.mosaic_processor_grp
    cfg_path = cfg.paths.mosaic_processor_cfg
    exe_dir = os.path.dirname(exe_path)

    if cfg_path != "DISABLED":
        logger.warning(f"Note: cfg_path is not yet implemented. Provided value: {cfg_path}", indent=1)

    cmd = build_mosaic_command(
        exe_path=exe_path,
        input_dir=input_dir,
        output_dir=str(output_dir),
        grp_path=grp_path,
        start_frame=start_frame,
        end_frame=end_frame,
        skip_gpx=skip_gpx,
        skip_render=skip_render,
        skip_reel_fix=skip_reel_fix,
        wrap_in_shell=True,
    )

    logger.custom(f"Running command: {cmd}", indent=2, emoji="🧑🏻‍💻")
    log_f.write(f"COMMAND: {cmd}\n\n")

    result = subprocess.run(cmd, cwd=exe_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)

    logger.success(f"=== {stage_name} Complete ===", indent=1)
    log_f.write(f"=== {stage_name} ===\n")
    log_f.write(result.stdout or "")
    log_f.write("\n\n")

    if result.returncode != 0:
        logger.error(f"{stage_name} failed. See log for details.", error_type=RuntimeError, indent=0)
        logger.info(f"📄 Log written to: {log_path}", indent=1)
        return False

    return True


def pad_frame_numbers(output_dir: str, logger) -> int:
    """
    Renames JPG files in the output folder to use zero-padded 6-digit frame numbers.

    Args:
        output_dir: Directory containing frame image files.
        logger: ConfigManager logger instance.

    Returns:
        Number of files renamed.
    """
    renamed_count = 0
    logger.info(f"Checking for frame number padding in: {output_dir}", indent=2)

    padded = False  # TODO: Add support for checking if frame numbers are padded (Check if any file has a 6-digit frame number)
    if not padded:
        logger.info("Frame numbers are not padded, padding...", indent=3)
    else:
        logger.info("Frame numbers are padded. Skipping padding.", indent=3)

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                match = re.match(r"^(.*_)(\d+)\.jpg$", f)
                if match:
                    prefix, num = match.groups()
                    if len(num) < 6:
                        padded = num.zfill(6)
                        new_name = f"{prefix}{padded}.jpg"
                        old_path = os.path.join(root, f)
                        new_path = os.path.join(root, new_name)
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        logger.debug(f"Renamed: {f} → {new_name}", indent=4)

    logger.info(f"Total files padded: {renamed_count}", indent=3)
    logger.success("=== Pad frame numbers Complete ===", indent=1)
    return renamed_count


def run_mosaic_processor(
    cfg: ConfigManager,
    input_dir: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> None:
    """
    Runs the Mosaic Processor workflow for a project in three sequential stages.
    
    This function orchestrates the execution of the Mosaic Processor executable, first performing rendering and reel
    fixing, then running GPX integration. It handles configuration resolution, validates required paths, manages output
    directories, detects reel numbers, and logs all steps and subprocess outputs to a dedicated log file. If any
    critical step fails, the function logs an error and exits early.
    
    Args:
        cfg (ConfigManager): Active configuration manager with logging and paths.
        input_dir: Directory containing input image frames.
        start_frame: Optional starting frame number for processing.
        end_frame: Optional ending frame number for processing.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="mosaic_processor")

    output_dir = cfg.paths.original
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(input_dir):
        logger.error(f"Input folder does not exist: {input_dir}", error_type=FileNotFoundError, indent=1)

    # Optional: write reel_info.json if a reel folder name can be inferred
    reel_match = re.search(r"reel_(\d{4})", os.path.basename(input_dir), re.IGNORECASE)
    if reel_match:
        reel_number = reel_match.group(1)
        reel_info_path = output_dir / "reel_info.json"
        try:
            with open(reel_info_path, "w") as f:
                json.dump({"reel": reel_number}, f)
            logger.info(f"Saved reel number {reel_number} to {reel_info_path}", indent=1)
        except Exception as e:
            logger.warning(f"Failed to write reel_info.json to {reel_info_path}: {e}", indent=1)

    log_path = cfg.paths.get_log_file_path("mosaic_processor_log", cfg)

    try:
        with open(log_path, "w", encoding="utf-8") as log_f, \
                cfg.get_progressor(total=3, label="Mosaic Processor Workflow") as progressor:

            # === Step 1: Render + Reel Fix ===
            logger.info("=== Render + Reel Fix Started ===", indent=1)
            number_of_reels = len(os.listdir(input_dir))  # TODO: Is this the best way to get the number of reels? (just need to get number of folders in input_dir)
            logger.info(f"Found {number_of_reels} reel(s) in {input_dir}", indent=2)

            # TODO get folder names from input_dir and add logger.info for each folder
            for folder in os.listdir(input_dir):
                logger.info(f"🎞️ {folder}", indent=3)
            if not run_processor_stage(
                    cfg, input_dir, output_dir, start_frame, end_frame,
                    log_f, log_path,
                    stage_name="Render + Reel Fix",
                    skip_gpx=True
            ):
                return
            progressor.update(1)

            # === Step 2: Pad frame numbers ===
            logger.info("=== Pad frame numbers Started ===", indent=1)
            pad_frame_numbers(str(output_dir), logger)
            progressor.update(2)

            # === Step 3: GPX Integration ===
            logger.info("=== GPX Integration Started ===", indent=1)
            if not run_processor_stage(
                    cfg, input_dir, output_dir, start_frame, end_frame,
                    log_f, log_path,
                    stage_name="GPX Integration",
                    skip_render=True,
                    skip_reel_fix=True
            ):
                return
            progressor.update(3)

    except Exception as e:
        logger.error(f"Error during Mosaic Processor workflow: {e}", error_type=RuntimeError, indent=1)

    logger.custom(f"Mosaic Processor log saved to: {log_path}", indent=1, emoji="📄")
