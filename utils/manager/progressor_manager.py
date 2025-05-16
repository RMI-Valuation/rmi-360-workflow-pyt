# =============================================================================
# 📊 ArcGIS + CLI Progress Tracker (utils/manager/progressor_manager.py)
# -----------------------------------------------------------------------------
# Purpose:             Provides unified progress tracking for ArcGIS Pro and CLI contexts.
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
# Last Updated:        2025-05-15
#
# Description:
#   Defines a context-managed ProgressorManager class that attempts to initialize ArcGIS Pro's progressor
#   UI when available, falling back to console output otherwise. Supports updating the progress label
#   and position and gracefully degrades if ArcPy is unavailable or fails. Integrates with LogManager
#   for enhanced error reporting.
#
# File Location:        /utils/manager/progressor_manager.py
# Called By:            enhance_images.py, copy_to_aws.py, orchestrator tools
# Int. Dependencies:    utils/arcpy_utils, utils/manager/log_manager
# Ext. Dependencies:    arcpy, sys, typing
#
# Documentation:
#   See: docs_legacy/UTILITIES.md
#   (Ensure this doc is current; update if needed.)
#
# Notes:
#   - Resets ArcGIS progressor on exit if used.
#   - Compatible with both Toolbox and script execution modes.
#   - Logs progressor errors when integrated with LogManager.
# =============================================================================

import arcpy
import sys
from typing import Optional
from utils.manager.log_manager import LogManager


class ProgressorManager:
    def __init__(self, total: int, label: str = "Processing...", step: int = 1, log_manager: Optional[LogManager] = None):
        """
        Initializes a Progressor instance for tracking and reporting progress.
        
        Args:
            total: The total number of steps to track.
            label: The label to display for the progressor.
            step: The increment value for each progress update.
            log_manager: Optional LogManager instance for logging errors.
        """
        self.total = total
        self.label = label
        self.step = step
        self.log_manager = log_manager
        self.use_progressor = False
        self.completed = 0

    def __enter__(self):
        """
        Initializes the ArcGIS Pro progressor if possible when entering the context.
        
        If initialization fails or the total is not positive, disables ArcGIS progressor usage and falls back to CLI
        output.
        
        Returns:
            The Progressor instance.
        """
        if self.total > 0:
            try:
                arcpy.SetProgressor("step", self.label, 0, self.total, self.step)
                self.use_progressor = True
            except Exception as e:
                self.log_manager.warning(f"Could not initialize ArcGIS Pro progressor: {e}")
        return self

    def update(self, pos: int, label: Optional[str] = None):
        """
        Updates the progress display to the specified position, optionally updating the label.
        
        If the ArcGIS progressor is active, updates its label and position; otherwise, prints progress to the console
        as a percentage.
        """
        self.completed = pos
        if self.use_progressor:
            try:
                if label:
                    arcpy.SetProgressorLabel(label)
                arcpy.SetProgressorPosition(pos)
            except Exception as e:
                self.log_manager.warning(f"Progressor update failed: {e}")
                self.use_progressor = False

        if not self.use_progressor:
            percent = (pos / self.total) * 100 if self.total > 0 else 0
            label_str = label or f"Progress: {pos}/{self.total}"
            sys.stdout.write(f"\r{label_str} ({percent:.1f}%)")
            sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleans up progress reporting resources when exiting the context.
        
        If the ArcGIS progressor was used, attempts to reset it; otherwise, ensures CLI output formatting by printing
        a newline.
        """
        if self.use_progressor:
            try:
                arcpy.ResetProgressor()
            except Exception as e:
                if self.log_manager:
                    self.log_manager.warning(f"Could not reset progressor: {e}")
        else:
            print()  # Ensure newline after CLI progress
