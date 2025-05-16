# =============================================================================
# 🧰 Process Full Mosaic 360 Workflow (tools/process_360_orchestrator.py)
# -----------------------------------------------------------------------------
# Tool Name:          Process360Workflow
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   Orchestrates the full end-to-end Mosaic 360 image processing pipeline within ArcGIS Pro.
#   Executes a configurable series of steps, including image rendering, OID creation, image
#   enrichment, geolocation, cloud upload, and service publishing. Tracks progress and logs
#   results to a persistent report JSON for dashboard and reporting use. Integrates with Core Utils
#   for all workflow, configuration, and reporting steps.
#
# File Location:      /tools/process_360_orchestrator.py
# Core Utils:
#   - utils/build_step_funcs.py
#   - utils/step_runner.py
#   - utils/generate_report.py
#   - utils/manager/config_manager.py
#   - utils/shared/report_data_builder.py
#   - utils/shared/folder_stats.py
#   - utils/shared/gather_metrics.py
#   - utils/shared/arcpy_utils.py
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/process_360_orchestrator.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - Start From Step (Optional) {start_step} (String): Step label to start from. Skips earlier steps if selected.
#   - Project Folder {project_folder} (Folder): Root folder for the current 360 image processing project.
#   - Input Reels Folder {input_reels_folder} (Folder): Directory containing raw .mp4 reel folders from Mosaic camera.
#   - OID Dataset - Input {oid_fc_input} (Feature Class): Existing OID to update (used in most steps).
#   - OID Dataset - Output {oid_fc_output} (Feature Class): Output feature class (used only when creating a new OID).
#   - Centerline (M-enabled) {centerline_fc} (Feature Class): Required polyline for GPS smoothing and referencing.
#   - Route ID Field {route_id_field} (Field): Field in centerline used to identify routes for linear referencing.
#   - Enable Linear Referencing {enable_linear_ref} (Boolean): Whether to compute MP and Route ID values per image.
#   - Skip Enhance Images? {skip_enhance_images} (Boolean): If true, skips image enhancement even if config is enabled.
#   - Copy Processed Images to AWS S3? {copy_to_aws} (Boolean): Whether to trigger the AWS upload step.
#   - Custom Config File {config_file} (File): Optional override for default config.yaml path.
#   - Generate HTML Summary Report? {generate_report} (Boolean): Whether to generate a visual summary at the end.
#
# Notes:
#   - Uses a step dispatcher to execute ordered functions defined in build_step_funcs.
#   - Skipped steps are safely logged and retained in the report JSON.
#   - Can resume from an existing report if rerun on the same project.
#   - Ensure all Core Utils and config files are up-to-date for workflow success.
# =============================================================================

import arcpy
import time
import os
from typing import Optional, Any, Callable, Dict, List

from utils.manager.config_manager import ConfigManager
from utils.shared.arcpy_utils import str_to_bool
from utils.generate_report import generate_report_from_json
from utils.step_runner import run_steps
from utils.build_step_funcs import build_step_funcs, get_step_order
from utils.shared.gather_metrics import collect_oid_metrics, summarize_oid_metrics
from utils.shared.folder_stats import folder_stats
from utils.shared.report_data_builder import initialize_report_data, save_report_json, load_report_json_if_exists


class Process360Workflow(object):
    def __init__(
        self,
        arcpy_mod: Any = None,
        os_mod: Any = None,
        time_mod: Any = None,
        collect_oid_metrics_fn: Optional[Callable] = None,
        summarize_oid_metrics_fn: Optional[Callable] = None,
        folder_stats_fn: Optional[Callable] = None,
        build_step_funcs_fn: Optional[Callable] = None,
        get_step_order_fn: Optional[Callable] = None,
        run_steps_fn: Optional[Callable] = None,
        initialize_report_data_fn: Optional[Callable] = None,
        save_report_json_fn: Optional[Callable] = None,
        load_report_json_if_exists_fn: Optional[Callable] = None,
        generate_report_from_json_fn: Optional[Callable] = None
    ):
        """
        Allows injection of dependencies for testability.
        Defaults to real modules/functions if not provided.
        """
        self.arcpy_mod = arcpy_mod or arcpy
        self.os_mod = os_mod or os
        self.time_mod = time_mod or time
        self.collect_oid_metrics_fn = collect_oid_metrics_fn or collect_oid_metrics
        self.summarize_oid_metrics_fn = summarize_oid_metrics_fn or summarize_oid_metrics
        self.folder_stats_fn = folder_stats_fn or folder_stats
        self.build_step_funcs_fn = build_step_funcs_fn or build_step_funcs
        self.get_step_order_fn = get_step_order_fn or get_step_order
        self.run_steps_fn = run_steps_fn or run_steps
        self.initialize_report_data_fn = initialize_report_data_fn or initialize_report_data
        self.save_report_json_fn = save_report_json_fn or save_report_json
        self.load_report_json_if_exists_fn = load_report_json_if_exists_fn or load_report_json_if_exists
        self.generate_report_from_json_fn = generate_report_from_json_fn or generate_report_from_json

    @staticmethod
    def parameters_to_dict(parameters: List[Any]) -> Dict[str, Any]:
        """
        Converts ArcPy tool parameters to a dictionary keyed by parameter name.
        Ensures all GPBoolean and 'enable_*' parameters are Python bools for skip logic.
        """
        from utils.shared.arcpy_utils import str_to_bool
        result = {}
        for param in parameters:
            # Convert GPBoolean and all 'enable_*' params to bool
            if getattr(param, 'datatype', None) == 'GPBoolean' or param.name.startswith('enable_'):
                result[param.name] = str_to_bool(param.value)
            else:
                result[param.name] = param.valueAsText
        return result

    def _compute_and_store_folder_stats(self, labels: List[str], paths: Any, report_data: dict, logger: Any):
        """
        Helper to compute and store folder stats for the given labels.
        """
        for label in labels:
            try:
                folder_path = getattr(paths, label)
                count, size = self.folder_stats_fn(folder_path)
                report_data.setdefault("paths", {})[f"{label}_images"] = str(folder_path)
                report_data.setdefault("metrics", {})[f"{label}_count"] = count
                report_data.setdefault("metrics", {})[f"{label}_size"] = size
            except Exception as e:
                report_data.setdefault("metrics", {})[f"{label}_count"] = 0
                report_data.setdefault("metrics", {})[f"{label}_size"] = "0 B"
                logger.warning(f"Failed to compute folder stats for {label}: {e}")

    label = "Process Full Mosaic 360 Workflow"
    description = (
        "Runs the full Mosaic 360 image processing pipeline including:\n"
        "1) Run Mosaic Processor\n"
        "2) Create Oriented Imagery Dataset\n"
        "3) Add Images to OID\n"
        "4) Assign Group Index\n"
        "5) Calculate OID Attributes\n"
        "6) Smooth GPS Noise (optional)\n"
        "7) Correct Flagged GPS Points (optional)\n"
        "8) Update Linear and Custom Attributes (linear referencing optional)\n"
        "9) Enhance Images [EXPERIMENTAL] (optional)\n"
        "10) Rename Images\n"
        "11) Update EXIF Metadata\n"
        "12) Geocode Images (optional)\n"
        "13) Create OID Footprints\n"
        "14) Deploy Lambda Monitor (optional)\n"
        "15) Copy to AWS (optional)\n"
        "16) Generate OID Service (optional)"
    )
    canRunInBackground = False

    # Explicitly define the steps with a numeric index to ensure correct order
    STEP_FUNCTIONS = [
        (1, "run_mosaic_processor", "Run Mosaic Processor"),
        (2, "create_oid", "Create Oriented Imagery Dataset"),
        (3, "add_images", "Add Images to OID"),
        (4, "assign_group_index", "Assign Group Index"),
        (5, "enrich_oid", "Calculate OID Attributes"),
        (6, "smooth_gps", "Smooth GPS Noise"),
        (7, "correct_gps", "Correct Flagged GPS Points"),
        (8, "update_linear_custom", "Update Linear and Custom Attributes"),
        (9, "enhance_images", "Enhance Images [EXPERIMENTAL]"),
        (10, "rename_images", "Rename Images"),
        (11, "update_metadata", "Update EXIF Metadata"),
        (12, "geocode", "Geocode Images"),
        (13, "build_footprints", "Build OID Footprints"),
        (14, "deploy_lambda_monitor", "Deploy Lambda Monitor"),
        (15, "copy_to_aws", "Upload to AWS S3"),
        (16, "generate_service", "Generate OID Service")
    ]

    # Sort by numeric index and extract step names in the correct order
    STEP_ORDER = [step[1] for step in sorted(STEP_FUNCTIONS, key=lambda x: x[0])]

    def getParameterInfo(self):
        """
        Defines and returns the list of ArcPy parameters for the Mosaic 360 workflow tool.
        
        The parameters include options for selecting the workflow start step, specifying project and input folders,
        input and output Oriented Imagery Dataset (OID) feature classes, centerline and route ID fields, and toggles
        for linear referencing, image enhancement, AWS upload, custom configuration, and report generation. Parameter
        defaults, filters, and dependencies are set to guide user input in the ArcGIS interface.
        """
        params = []

        # 1. Project Folder [0]
        project_folder_param = arcpy.Parameter(
            displayName="Project Folder",
            name="project_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        params.append(project_folder_param)

        # 2. Input Reels Folder [1]
        input_reels_folder_param = arcpy.Parameter(
            displayName="Input Reels Folder",
            name="input_reels_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        params.append(input_reels_folder_param)

        # 3. Custom Config File [2]
        config_file_param = arcpy.Parameter(
            displayName="Custom Config File",
            name="config_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        config_file_param.filter.list = ["yaml", "yml"]
        params.append(config_file_param)

        # 4. Start From Step [3]
        step_label_map = {label: name for _, name, label in self.STEP_FUNCTIONS}
        label_to_name = {label: name for _, name, label in self.STEP_FUNCTIONS}
        name_to_label = {name: label for _, name, label in self.STEP_FUNCTIONS}
        step_labels = [label for _, _, label in self.STEP_FUNCTIONS]

        start_step_param = arcpy.Parameter(
            displayName="Start From Step (Optional)",
            name="start_step",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        start_step_param.filter.list = ["--SELECT STEP--"] + step_labels
        start_step_param.value = "--SELECT STEP--"
        params.append(start_step_param)

        # Save for use in updateParameters/updateMessages/execute
        self._step_label_to_name = label_to_name
        self._step_name_to_label = name_to_label

        # 5. OID Input [4]
        oid_fc_input_param = arcpy.Parameter(
            displayName="OID Dataset - Input (used in most steps)",
            name="oid_fc_input",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Input"
        )
        oid_fc_input_param.enabled = False
        params.append(oid_fc_input_param)

        # 6. OID Output [5]
        oid_fc_output_param = arcpy.Parameter(
            displayName="OID Dataset - Output (used only when creating a new OID)",
            name="oid_fc_output",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Output"
        )
        oid_fc_output_param.enabled = False
        params.append(oid_fc_output_param)

        # 7. Enable Smooth GPS (and Correct GPS) [6]
        enable_smooth_gps_param = arcpy.Parameter(
            displayName="Enable Smooth GPS (and Correct GPS)",
            name="enable_smooth_gps",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_smooth_gps_param.value = True
        params.append(enable_smooth_gps_param)

        # 8. Enable Linear Referencing [7]
        enable_linear_ref_param = arcpy.Parameter(
            displayName="Enable Linear Referencing",
            name="enable_linear_ref",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_linear_ref_param.value = True
        params.append(enable_linear_ref_param)

        # 9. Enable Enhance Images [8]
        enable_enhance_images_param = arcpy.Parameter(
            displayName="Enable Enhance Images [EXPERIMENTAL]",
            name="enable_enhance_images",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_enhance_images_param.value = False
        params.append(enable_enhance_images_param)

        # 10. Enable Geocode Images [9]
        enable_geocode_param = arcpy.Parameter(
            displayName="Enable Geocode Images",
            name="enable_geocode",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_geocode_param.value = True
        params.append(enable_geocode_param)

        # 11. Enable Copy to AWS [10]
        enable_copy_to_aws_param = arcpy.Parameter(
            displayName="Enable Copy to AWS",
            name="enable_copy_to_aws",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_copy_to_aws_param.value = True
        params.append(enable_copy_to_aws_param)

        # 12. Enable Deploy Lambda Monitor [11]
        enable_deploy_lambda_monitor_param = arcpy.Parameter(
            displayName="Enable Deploy Lambda Monitor",
            name="enable_deploy_lambda_monitor",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_deploy_lambda_monitor_param.value = True
        params.append(enable_deploy_lambda_monitor_param)

        # 13. Enable Generate OID Service [12]
        enable_generate_service_param = arcpy.Parameter(
            displayName="Enable Generate OID Service",
            name="enable_generate_service",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        enable_generate_service_param.value = True
        params.append(enable_generate_service_param)

        # 14. Centerline [13] (dynamic logic in updateParameters)
        centerline_param = arcpy.Parameter(
            displayName="Centerline (M-enabled, used for GPS smoothing and linear referencing)",
            name="centerline_fc",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Input"
        )
        centerline_param.filter.list = ["Polyline"]
        params.append(centerline_param)

        # 15. Route ID Field [14]
        route_id_param = arcpy.Parameter(
            displayName="Route ID Field",
            name="route_id_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        route_id_param.parameterDependencies = [centerline_param.name]
        route_id_param.filter.list = ["Short", "Long", "Text"]
        params.append(route_id_param)

        # 16. Generate HTML Summary Report [15]
        generate_report_param = arcpy.Parameter(
            displayName="Generate HTML Summary Report",
            name="generate_report",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        generate_report_param.value = True
        params.append(generate_report_param)

        return params

    def updateParameters(self, parameters):
        """
        Dynamically enables or disables OID input/output and Centerline/Route ID parameters based on selected step and toggles.
        """
        start_step_param = parameters[3]  # Start step param
        oid_in = parameters[4]  # oid_fc_input
        oid_out = parameters[5]  # oid_fc_output
        enable_smooth_gps = parameters[6]  # enable_smooth_gps
        enable_linear_ref = parameters[7]  # enable_linear_ref
        centerline_param = parameters[13]  # centerline_fc
        route_id_param = parameters[14]  # route_id_field

        # Defensive: ensure label-to-name mapping exists even if getParameterInfo not called
        if not hasattr(self, '_step_label_to_name') or not hasattr(self, '_step_name_to_label'):
            self._step_label_to_name = {label: name for _, name, label in self.STEP_FUNCTIONS}
            self._step_name_to_label = {name: label for _, name, label in self.STEP_FUNCTIONS}
        # OID enable/disable logic
        if start_step_param.altered:
            step_label = start_step_param.valueAsText
            step = self._step_label_to_name.get(step_label, "") if step_label and step_label != "--SELECT STEP--" else ""
            if step in ("run_mosaic_processor", "create_oid"):
                oid_in.enabled = False
                oid_out.enabled = True
            elif step:
                oid_in.enabled = True
                oid_out.enabled = False
            else:
                oid_in.enabled = False
                oid_out.enabled = False
            oid_in.value = None if not oid_in.enabled else oid_in.value
            oid_out.value = None if not oid_out.enabled else oid_out.value

        # Centerline and Route ID dynamic logic
        linear_ref_enabled = bool(enable_linear_ref.value)
        smooth_gps_enabled = bool(enable_smooth_gps.value)

        if linear_ref_enabled:
            centerline_param.enabled = True
            centerline_param.parameterType = "Required"
            route_id_param.enabled = True
            route_id_param.parameterType = "Required"
        elif smooth_gps_enabled:
            centerline_param.enabled = True
            centerline_param.parameterType = "Optional"
            route_id_param.enabled = False
            route_id_param.parameterType = "Optional"
            route_id_param.value = None
        else:
            centerline_param.enabled = False
            centerline_param.parameterType = "Optional"
            centerline_param.value = None
            route_id_param.enabled = False
            route_id_param.parameterType = "Optional"
            route_id_param.value = None

        # Copy to AWS logic: disable downstream toggles if not enabled
        enable_copy_to_aws = parameters[10]  # index for Enable Copy to AWS
        enable_deploy_lambda_monitor = parameters[11]
        enable_generate_service = parameters[12]
        if not bool(enable_copy_to_aws.value):
            enable_deploy_lambda_monitor.enabled = False
            enable_deploy_lambda_monitor.value = False
            enable_generate_service.enabled = False
            enable_generate_service.value = False
        else:
            enable_deploy_lambda_monitor.enabled = True
            enable_generate_service.enabled = True

    def updateMessages(self, parameters):
        """
        Validates workflow parameters and sets error or warning messages for user guidance.
        Updates messages for Centerline and Route ID based on enabled toggles and requirements.
        """
        start_step_param = parameters[3]
        oid_in = parameters[4]
        oid_out = parameters[5]
        enable_smooth_gps = parameters[6]
        enable_linear_ref = parameters[7]
        centerline_param = parameters[13]
        route_id_param = parameters[14]

        # Defensive: ensure label-to-name mapping exists even if getParameterInfo not called
        if not hasattr(self, '_step_label_to_name') or not hasattr(self, '_step_name_to_label'):
            self._step_label_to_name = {label: name for _, name, label in self.STEP_FUNCTIONS}
            self._step_name_to_label = {name: label for _, name, label in self.STEP_FUNCTIONS}
        step_label = start_step_param.valueAsText
        step = self._step_label_to_name.get(step_label, "") if step_label and step_label != "--SELECT STEP--" else ""
        linear_ref_enabled = bool(enable_linear_ref.value)
        smooth_gps_enabled = bool(enable_smooth_gps.value)

        # Start step and OID messages
        if step_label == "--SELECT STEP--":
            start_step_param.setErrorMessage("⚠️ Please select a valid Start Step before running.")
        elif step in ("run_mosaic_processor", "create_oid"):
            oid_out.setWarningMessage("⚠️ Existing OID will be overwritten if it exists.")
        else:
            oid_in.setWarningMessage("ℹ️ Existing OID will be used and preserved.")
        if oid_in.enabled and not oid_in.valueAsText:
            oid_in.setErrorMessage("⚠️ Please specify the existing OID dataset.")
        elif oid_out.enabled and not oid_out.valueAsText:
            oid_out.setErrorMessage("⚠️ Please specify the output path for the new OID dataset.")

        # Centerline and Route ID messages
        if linear_ref_enabled:
            if not centerline_param.valueAsText:
                centerline_param.setErrorMessage("⚠️ Centerline is required when Linear Referencing is enabled.")
            if not route_id_param.valueAsText:
                route_id_param.setErrorMessage("⚠️ Route ID is required when Linear Referencing is enabled.")
        elif smooth_gps_enabled:
            centerline_param.clearMessage()
            route_id_param.clearMessage()
        else:
            centerline_param.clearMessage()
            route_id_param.clearMessage()

    def execute(self, parameters: list, messages: Any) -> None:
        """
        Executes the full Mosaic 360 image processing workflow and generates reports.
        Allows dependency injection for testability. Compatible with ArcGIS Pro GUI.
        """
        # Map parameters to dict
        p = self.parameters_to_dict(parameters)
        # Determine which OID param was enabled and populate p["oid_fc"]
        if parameters[5].enabled:
            p["oid_fc"] = parameters[5].valueAsText  # oid_fc_output
        else:
            p["oid_fc"] = parameters[4].valueAsText  # oid_fc_input

        generate_report_flag = str_to_bool(p.get("generate_report", "true"))

        cfg = ConfigManager.from_file(
            path=p.get("config_file"),
            project_base=p["project_folder"],
            messages=messages,
        )
        logger = cfg.get_logger(messages)
        paths = cfg.paths

        logger.custom("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", indent=0, emoji="🚀")
        logger.custom("| --- Starting Mosaic 360 Workflow --- |", indent=0, emoji="🚀")
        logger.custom("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", indent=0, emoji="🚀")
        logger.info(f"Using config: {cfg.source_path}", indent=1)
        logger.info(f"Project root: {cfg.get('__project_root__')}", indent=1)

        if cfg.get("debug_messages", False):
            logger.custom("Debug mode enabled from config", indent=1, emoji="🔍")
       
        cfg.validate()

        # Build steps + order
        step_funcs = self.build_step_funcs_fn(p, cfg)
        step_order = self.get_step_order_fn(step_funcs)

        # Initialize report data
        report_data = self.load_report_json_if_exists_fn(cfg)
        if report_data is None:
            report_data = self.initialize_report_data_fn(p, cfg)
            self.save_report_json_fn(report_data, cfg)
        else:
            logger.custom("Loaded existing report JSON — appending new steps", indent=1, emoji="🔄")

        # Execute steps and capture results
        t_start = self.time_mod.time()

        # Defensive: ensure label-to-name mapping exists even if getParameterInfo not called
        if not hasattr(self, '_step_label_to_name') or not hasattr(self, '_step_name_to_label'):
            self._step_label_to_name = {label: name for _, name, label in self.STEP_FUNCTIONS}
            self._step_name_to_label = {name: label for _, name, label in self.STEP_FUNCTIONS}
        
        # Map label to step name for execution
        step_label = p.get("start_step")
        start_step = self._step_label_to_name.get(step_label, step_order[0]) if step_label and step_label != "--SELECT STEP--" else step_order[0]
        if start_step not in step_order:
            logger.warning(f"Invalid start_step '{start_step}' provided. Falling back to default '{step_order[0]}'.", indent=1)
            start_step = step_order[0]
        start_index = step_order.index(start_step)

        wait_config = cfg.get("orchestrator", {})
        self.run_steps_fn(step_funcs, step_order, start_index, p, report_data, cfg, wait_config=wait_config)

        # After the OID has been created (via run_steps), describe its GDB path
        try:
            report_data.setdefault("paths", {})["oid_gdb"] = self.arcpy_mod.Describe(p["oid_fc"]).path
        except Exception as e:
            logger.warning(f"Could not describe OID FC yet: {e}", indent=1)
            report_data.setdefault("paths", {})["oid_gdb"] = "Unavailable"

        # OID-based metrics
        try:
            raw_metrics = self.collect_oid_metrics_fn(p["oid_fc"])
            summary, reels = self.summarize_oid_metrics_fn(raw_metrics)
            report_data["metrics"].update(summary)
            report_data["reels"] = reels
        except Exception as e:
            logger.warning(f"Could not gather OID stats: {e}", indent=1)

        # Reel folder count
        try:
            reel_folders = [f for f in self.os_mod.listdir(p["input_reels_folder"]) if
                            self.os_mod.path.isdir(self.os_mod.path.join(p["input_reels_folder"], f))]
            report_data["metrics"]["reel_count"] = len(reel_folders)
        except Exception as e:
            report_data["metrics"]["reel_count"] = "—"
            logger.warning(f"Failed to count reel folders: {e}", indent=1)

        # Folder stats for original/enhanced/renamed
        self._compute_and_store_folder_stats(["original", "enhanced", "renamed"], paths, report_data, logger)

        # Elapsed time
        elapsed_total = self.time_mod.time() - t_start
        report_data["metrics"]["elapsed"] = f"{elapsed_total:.1f} sec"
        total_images = report_data["metrics"].get("total_images", 0)
        if total_images:
            report_data["metrics"]["time_per_image"] = f"{elapsed_total / total_images:.2f} sec/image"

        # Determine report output folder
        report_dir = paths.report
        report_data["paths"]["report_dir"] = str(report_dir)

        # Save report data to JSON for future recovery/report generation
        self.save_report_json_fn(report_data, cfg)

        if generate_report_flag:
            try:
                report_data["config"] = cfg
                slug = cfg.get("project.slug", "unknown")
                json_path = self.os_mod.path.join(paths.report, f"report_data_{slug}.json")

                self.generate_report_from_json_fn(
                    json_path=json_path,
                    cfg=cfg
                )
                logger.custom(f"Final report and JSON saved to: {report_dir}", indent=1, emoji="📄")
                # Export log to HTML
                logger.export_html()
            except Exception as e:
                logger.warning(f"Report generation failed: {e}", indent=1)
        else:
            logger.custom("Skipping report generation (disabled by user)", indent=1, emoji="⏭️")
        
        logger.custom("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", indent=0, emoji="🎉")
        logger.custom("| --- Mosaic 360 Workflow Complete --- |", indent=0, emoji="🎉")
        logger.custom("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", indent=0, emoji="🎉")
