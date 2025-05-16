# =============================================================================
# ☁️ AWS S3 Upload Utility (utils/copy_to_aws.py)
# -----------------------------------------------------------------------------
# Purpose:             Uploads JPEG images from a local directory to AWS S3 using TransferManager
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-14
# Last Updated:        2025-05-15
#
# Description:
#   Recursively uploads enhanced or renamed images from a local directory to a configured
#   S3 bucket path using boto3's TransferManager. Supports concurrency scaling, retry logic,
#   resumable logging, ETA progress tracking, and graceful cancellation via file or ArcGIS trigger.
#
# File Location:        /utils/copy_to_aws.py
# Validator:            /utils/validators/copy_to_aws_validator.py
# Called By:            tools/copy_to_aws_tool.py, orchestrator pipeline
# Int. Dependencies:    utils/manager/config_manager, utils/shared/aws_utils
# Ext. Dependencies:    boto3, botocore, threading, multiprocessing, csv, time, os, pathlib, datetime, typing
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/copy_to_aws.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Uses TransferManager with configurable concurrency and retry limits
#   - Respects cancel_copy.txt trigger for manual interruption support
# =============================================================================

__all__ = ["copy_to_aws"]

import os
import time
import csv
import threading
import multiprocessing
from datetime import datetime, timezone
from typing import Optional, Any
from boto3.session import Session
from boto3.s3.transfer import create_transfer_manager, TransferConfig
from botocore.config import Config
from pathlib import Path

from utils.manager.config_manager import ConfigManager
from utils.shared.aws_utils import get_aws_credentials


def collect_upload_tasks(local_dir, include_extensions, bucket_folder):
    """
    Recursively collects all files with given extensions and returns (local_path, s3_key) tuples.
    """
    tasks = []
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if not file_path.suffix.lower() in include_extensions:
            continue
        rel_key = file_path.relative_to(local_dir).as_posix()
        s3_key = f"{bucket_folder}/{rel_key}".lstrip("/")
        tasks.append((str(file_path), s3_key))
    return tasks

def parse_uploaded_keys_from_log(log_file, logger):
    uploaded_keys = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "uploaded":
                        uploaded_keys.add(row.get("s3_key"))
            logger.custom(f"Resuming from previous log, skipping {len(uploaded_keys)} previously uploaded files.", emoji="ℹ🔄", indent=1)
        except Exception as e:
            logger.warning(f"Could not read previous log for resume: {e}", indent=1)
    return uploaded_keys

def calculate_summary(log_rows, total, start_time):
    uploaded = sum(1 for row in log_rows if row[3] == "uploaded")
    skipped = sum(1 for row in log_rows if row[3] == "skipped")
    failed = sum(1 for row in log_rows if row[3] == "error")
    skipped_from_log = sum(1 for row in log_rows if row[3] == "skipped" and row[4] == "from prior log")
    elapsed_time = round(time.time() - start_time, 2)
    total_size_bytes = sum(row[5] for row in log_rows if isinstance(row[5], int))
    avg_time_per_image = round(elapsed_time / total, 3) if total else 0
    avg_speed = round((total_size_bytes / 1024 / 1024) / elapsed_time, 2) if elapsed_time else 0
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "skipped_from_log": skipped_from_log,
        "failed": failed,
        "elapsed_time": elapsed_time,
        "total_size_bytes": total_size_bytes,
        "avg_time_per_image": avg_time_per_image,
        "avg_speed": avg_speed,
        "total_size_mb": total_size_mb
    }

def write_summary_file(summary_file, stats):
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Files", stats['uploaded'] + stats['skipped'] + stats['failed']])
        writer.writerow(["Uploaded", stats['uploaded']])
        writer.writerow(["Skipped", stats['skipped']])
        writer.writerow(["Skipped (from prior log)", stats['skipped_from_log']])
        writer.writerow(["Failed", stats['failed']])
        writer.writerow(["Elapsed Time (sec)", stats['elapsed_time']])
        writer.writerow(["Average Time/Image (sec)", stats['avg_time_per_image']])
        writer.writerow(["Total Size (MB)", stats['total_size_mb']])
        writer.writerow(["Average Speed (MB/sec)", stats['avg_speed']])

def should_cancel(messages, allow_cancel_file_trigger, cancel_txt):
    if hasattr(messages, "isCanceled") and messages.isCanceled():
        return True
    elif allow_cancel_file_trigger and Path(cancel_txt).exists():
        return True
    return False

def copy_to_aws(
        cfg: ConfigManager,
        report_data: Optional[dict] = None,
        local_dir: Optional[str] = None,
        skip_existing: Optional[bool] = None,
        messages: Optional[Any] = None
):
    """
    Uploads files from a local directory to an AWS S3 bucket folder with concurrency, resumable progress, and detailed
    logging.

    Recursively scans the specified local directory, optionally filtering files by extension, and uploads them to the
    given S3 bucket and folder using boto3's TransferManager. Supports skipping files already uploaded (based on a CSV
    log), resuming interrupted uploads, and concurrent transfers with configurable concurrency. Progress and per-file
    status are logged to CSV files, and a summary report is generated upon completion. Uploads can be canceled via
    ArcGIS Pro events or a trigger file. Updates the provided config dictionary with upload status.

    Args:
        cfg:
        report_data:
        local_dir: Path to the local directory containing files to upload.
        skip_existing: If True, skips files already uploaded as recorded in the log.
        messages:
    """
    logger = cfg.get_logger()
    cfg.validate(tool="copy_to_aws")

    logger.custom("Upload process cannot be interrupted once started. Use resume mode to safely restart if "
                "needed.", indent=1, emoji="⚠️")

    # AWS Setup
    logger.info("Verifying AWS credentials...", indent=1)
    access_key, secret_key = get_aws_credentials(cfg)
    bucket = cfg.get("aws.s3_bucket")
    region = cfg.get("aws.region", "us-east-2")
    bucket_folder = cfg.resolve(cfg.get("aws.s3_bucket_folder"))
    batch_size = cfg.get("aws.upload_batch_size", 25)
    max_workers_raw = cfg.get("aws.max_workers", 8)
    retries = cfg.get("aws.retries", 3)
    use_accel = cfg.get("aws.use_acceleration", False)

    include_extensions = [".jpg", ".jpeg"]
    if not local_dir:
        local_dir = Path(cfg.paths.renamed)
    if not skip_existing:
        skip_existing = cfg.get("aws.skip_existing", True)

    allow_cancel_file_trigger = cfg.get("aws.allow_cancel_file_trigger", True)
    cancel_txt = Path(cfg.paths.project_base / "cancel_copy.txt")

    log_file = cfg.paths.get_log_file_path("aws_upload_log", cfg)
    summary_file = cfg.paths.get_log_file_path("aws_upload_summary", cfg)

    logger.info("Starting AWS Upload...", indent=1)

    # Concurrency resolution
    cpu_count = multiprocessing.cpu_count() or 4
    worker_limit = cpu_count * 8  # Safety Limit
    if isinstance(max_workers_raw, int):
        max_concurrency = max_workers_raw
        logger.custom(f"Using Max Workers = {max_concurrency} (from int config)", indent=2, emoji="🧵")
    elif isinstance(max_workers_raw, str) and max_workers_raw.startswith("cpu"):
        try:
            factor = int(max_workers_raw.split("*")[1]) if "*" in max_workers_raw else 1
            max_concurrency = min(cpu_count * factor, worker_limit)
            logger.custom(f"Using Max Workers = {max_concurrency} (cpu_count={cpu_count} × factor={factor})", indent=2, emoji="🧵")
        except Exception as e:
            max_concurrency = cpu_count
            logger.warning(f"Failed to parse max_workers='{max_workers_raw}': {e}. Defaulting to {cpu_count}.", indent=2)
    else:
        max_concurrency = cpu_count
        logger.warning(f"Invalid max_workers value: {max_workers_raw}. Using default {cpu_count}.", indent=2)

    # COLLECT ALL TASKS
    all_tasks = collect_upload_tasks(local_dir, include_extensions, bucket_folder)
    if not all_tasks:
        logger.warning("No matching files found to upload.", indent=2)
        return {}

    # Init AWS session and client
    session = Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    s3_cfg = Config(s3={"use_accelerate_endpoint": True}) if use_accel else None
    s3 = session.client("s3", config=s3_cfg) if s3_cfg else session.client("s3")

    if use_accel:
        logger.custom("Using S3 Transfer Acceleration endpoint", indent=2, emoji="🚀")
    else:
        logger.info("S3 Transfer Acceleration endpoint disabled", indent=2)

    transfer_config = TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=max_concurrency,
        num_download_attempts=retries,
        use_threads=True
    )
    manager = create_transfer_manager(client=s3, config=transfer_config)

    log_rows = []

    # GET UPLOADED TASKS FROM LOG (FOR RESUME)
    uploaded_keys = parse_uploaded_keys_from_log(log_file, logger)

    # CHECK REMAINING TASK ON RESUME
    upload_tasks = []
    for file_path, s3_key in all_tasks:
        if skip_existing and s3_key in uploaded_keys:
            timestamp = datetime.now(timezone.utc).isoformat()
            size = os.path.getsize(file_path)
            log_rows.append((timestamp, file_path, s3_key, "skipped", "from prior log", size, 0.0))
        else:
            upload_tasks.append((file_path, s3_key))

    total = len(upload_tasks)
    if total == 0:
        logger.custom("No files to upload (all skipped or already uploaded).", indent=1, emoji="❗")
        return {
            "uploaded": 0,
            "skipped": 0,
            "skipped_from_log": len(uploaded_keys),
            "failed": 0,
            "cancelled": False,
            "log_file": str(log_file),
            "summary_file": str(summary_file),
            "status": "completed"
        }

    start_time = time.time()
    completed = {"count": 0}
    lock = threading.Lock()
    cancel_event = threading.Event()

    # Write log header
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "local_file", "s3_key", "status", "error", "size_bytes", "duration_sec"])

    def upload_one(f_path, s3_tgt_key):
        """
        Uploads a single file to the specified S3 key and logs the result.

        Attempts to upload the file using the transfer manager, recording the status, error (if any), file size, and
        duration. Appends the upload result to the log and updates the completed count in a thread-safe manner.
        """
        upload_start = None
        try:
            future = manager.upload(f_path, bucket, s3_tgt_key, extra_args={"ContentType": "image/jpeg"})
            upload_start = time.perf_counter()
            future.result()
            upload_end = time.perf_counter()
            status, error = "uploaded", ""
        except Exception as exc:
            upload_end = time.perf_counter()
            if upload_start is None:
                upload_start = upload_end  # duration = 0
            status, error = "error", str(exc)
        duration = round(upload_end - upload_start, 2)
        file_size = os.path.getsize(f_path) if os.path.exists(f_path) else 0
        timestamp_str = datetime.now(timezone.utc).isoformat()

        with lock:
            completed["count"] += 1
            log_rows.append((timestamp_str, f_path, s3_tgt_key, status, error, file_size, duration))
            with open(log_file, "a", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([timestamp_str, f_path, s3_tgt_key, status, error, file_size, duration])

    with cfg.get_progressor(total=total, label="Uploading images to S3") as progressor:
        last = 0

        try:
            for i in range(0, len(upload_tasks), batch_size):
                batch = upload_tasks[i:i + batch_size]
                threads = []

                for local_path, s3_key in batch:
                    if cancel_event.is_set():
                        break
                    t = threading.Thread(target=upload_one, args=(local_path, s3_key))
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                # Check for cancel triggers
                if should_cancel(messages, allow_cancel_file_trigger, cancel_txt):
                    cancel_event.set()
                    logger.custom("Upload canceled by trigger.", emoji="🛑", indent=2)

                if cancel_event.is_set():
                    progressor.update(completed["count"], "🛑 Upload canceled — finishing current batch.")
                    if report_data is not None:
                        upload_status = "canceled" if cancel_event.is_set() else "completed"
                        current_status = report_data.setdefault("upload", {}).get("status")

                        # If previously canceled, and now completed, mark as completed_after_cancel
                        if upload_status == "completed" and current_status == "canceled":
                            upload_status = "completed_after_cancel"

                        report_data["upload"]["status"] = upload_status

                    break

                with lock:
                    current = completed["count"]

                if current > last:
                    elapsed = time.time() - start_time
                    eta = (elapsed / current) * (total - current) if current > 0 else 0
                    eta_min, eta_sec = divmod(int(eta), 60)
                    eta_hr, eta_min = divmod(eta_min, 60)
                    label = f"Uploading {current}/{total} images... ETA: {eta_hr}:{eta_min:02d}:{eta_sec:02d}"
                    progressor.update(current, label)
                    logger.custom(label, indent=3, emoji="☁️")
                    last = current
        finally:
            manager.shutdown()

    # Write Summary File (aws_upload_summary)
    summary_stats = calculate_summary(log_rows, total, start_time)
    write_summary_file(summary_file, summary_stats)

    logger.success(f"Upload complete: {summary_stats['uploaded']} uploaded, {summary_stats['skipped']} skipped, {summary_stats['failed']} failed.", indent=1)
    logger.custom(f"Log written to: {log_file}", indent=1, emoji="📄")
    logger.custom(f"Upload summary written to: {summary_file}", indent=1, emoji="📊")

    # Update report_data if passed
    if report_data is not None:
        report_data.setdefault("upload", {}).update({
            "status": "cancelled" if cancel_event.is_set() else "completed",
            "count": summary_stats['uploaded'],
            "expected_total": total,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "percent_complete": summary_stats['uploaded'] / total * 100 if total else 0
        })

    return {
        "uploaded": summary_stats['uploaded'],
        "skipped": summary_stats['skipped'],
        "skipped_from_log": summary_stats['skipped_from_log'],
        "failed": summary_stats['failed'],
        "cancelled": cancel_event.is_set(),
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "status": "cancelled" if cancel_event.is_set() else "completed"
    }