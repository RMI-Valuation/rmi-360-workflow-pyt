# =============================================================================
# 📡 AWS Lambda Deployment Logic (utils/deploy_lambda_monitor.py)
# -----------------------------------------------------------------------------
# Purpose:             Deploys Lambda functions and schedules upload monitor for 360° image tracking
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Deploys and configures AWS Lambda functions for monitoring upload progress to S3. Sets up a CloudWatch
#   Events schedule to invoke the monitoring Lambda every 5 minutes. Initializes a project-specific progress
#   JSON file and verifies credentials via boto3. Uses keyring or fallback credentials defined in the config.
#
# File Location:        /utils/deploy_lambda_monitor.py
# Validator:            /utils/validators/deploy_lambda_monitor_validator.py
# Called By:            tools/copy_to_aws_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/expression_utils, utils/shared/aws_utils
# Ext. Dependencies:    boto3, json, zipfile, io, contextlib, datetime, pathlib
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md, docs_legacy/tools/copy_to_aws.md, and AWS_SETUP_GUIDE.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Deploys both progress monitor and auto-disable Lambda functions
#   - Configurable via standalone call or ArcGIS Python Toolbox wrapper
# =============================================================================

__all__ = ["deploy_lambda_monitor"]

import json
import zipfile
import io
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from boto3 import Session

from utils.manager.config_manager import ConfigManager
from utils.shared.expression_utils import resolve_expression
from utils.shared.aws_utils import get_aws_credentials


def zip_lambda(source_path: str, arcname: str) -> bytes:
    """
    Packages a Python source file into an in-memory ZIP archive.
    
    Args:
        source_path: Path to the source file to be zipped.
        arcname: Name to assign to the file within the ZIP archive.
    
    Returns:
        The bytes of the ZIP archive containing the source file.
    
    Raises:
        FileNotFoundError: If the specified source file does not exist.
    """
    path = Path(source_path)

    if not path.is_file():
        logger.error(f"Lambda file not found: {path}", error_type=FileNotFoundError, indent=2)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(arcname, path.read_bytes())

    buf.seek(0)
    return buf.read()


def get_final_image_files(final_folder):
    """
    Returns a list of JPEG image files in the given folder (recursively).
    """
    if not final_folder.exists():
        return []
    return [f for f in final_folder.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg"]]

def count_final_images(cfg):
    """
    Counts the number of JPEG images in the final output folder specified by the config.
    
    Returns:
        The number of `.jpg` files found in the resolved final image output folder, or 0 if the folder does not exist.
    """
    logger = cfg.get_logger()
    final_folder = cfg.paths.renamed
    image_files = get_final_image_files(final_folder)
    count = len(image_files)
    if count == 0:
        logger.warning(f"No JPEG images found in {final_folder}", indent=2)
    else:
        logger.custom(f"Found {count} JPEG images in {final_folder}", indent=2, emoji="📸")
    return count


def build_progress_json(cfg: ConfigManager, expected_total, now=None):
    """
    Builds a JSON-compatible dictionary representing the initial upload progress status.
    Optionally accepts a datetime for testability.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    project_info = {
        "number": cfg.get("project.number"),
        "slug": cfg.get("project.slug"),
        "client": cfg.get("project.client"),
        "railroad_name": cfg.get("project.rr_name"),
        "railroad_code": cfg.get("project.rr_mark"),
        "description": cfg.get("project.description")
    }

    camera_info = {
        "make": cfg.get("camera.make"),
        "model": cfg.get("camera.model"),
        "serial_number": cfg.get("camera.sn"),
        "firmware": cfg.get("camera.firmware"),
        "software": cfg.get("camera.software")
    }

    cloud_info = {
        "bucket": cfg.get("aws.s3_bucket"),
        "prefix": resolve_expression(cfg.get("aws.s3_bucket_folder"), cfg),
        "region": cfg.get("aws.region")
    }

    return {
        "project_slug": cfg.get("project.slug"),
        "last_updated": now.isoformat(),
        "start_time": now.isoformat(),
        "end_time": None,
        "upload_status": "In Progress",
        "stalls": 0,
        "count": 0,
        "expected_total": expected_total,
        "percent_complete": 0.0,
        "avg_time_per_image": None,
        "project_info": project_info,
        "camera_info": camera_info,
        "cloud_info": cloud_info
    }


def upload_progress_json(s3_client, progress, bucket, slug, logger):
    """
    Uploads the progress status JSON to an S3 bucket under the 'status/' prefix.
    
    Args:
        s3_client: A boto3 S3 client used to perform the upload.
        progress: The progress status dictionary to upload.
        bucket: The name of the S3 bucket.
        slug: Project slug used to generate the filename.
        logger:
    """
    filename = f"progress_{slug}.json"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=f"status/{filename}",
            Body=json.dumps(progress, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        logger.success(f"Uploaded {filename} to s3://{bucket}/status/", indent=2)
    except Exception as e:
        logger.error(f"Failed to upload progress JSON: {e}", indent=2)


def ensure_lambda_progress_monitor(cfg, lambda_client, role_arn):
    """
    Ensures the 'UploadProgressMonitor' Lambda function exists, deploying it if necessary.
    
    Checks for the existence of the 'UploadProgressMonitor' AWS Lambda function. If it does not exist, packages the
    source code and creates the Lambda function using the provided role ARN. Logs deployment status and raises a
    FileNotFoundError if the source file is missing.
    """
    logger = cfg.get_logger()

    name = "UploadProgressMonitor"
    handler = "lambda_progress_monitor.lambda_handler"

    try:
        lambda_client.get_function(FunctionName=name)
        logger.info(f"Lambda '{name}' already exists.", indent=2)
        return
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.custom("Deploying UploadProgressMonitor Lambda...", indent=2, emoji="🚀")

    try:
        zipped = zip_lambda(cfg.paths.lambda_pm_path, "lambda_progress_monitor.py")
    except FileNotFoundError as e:
        logger.error(f"Lambda zip failed: {e}", indent=3)
        raise
    lambda_client.create_function(
        FunctionName=name,
        Runtime="python3.9",
        Role=role_arn,
        Handler=handler,
        Code={"ZipFile": zipped},
        Timeout=90,
        MemorySize=256,
        Publish=True
    )
    logger.success(f"Lambda '{name}' deployed.", indent=2)


def ensure_lambda_deactivator(cfg, lambda_client, role_arn):
    """
    Ensures the 'DisableUploadMonitorRule' Lambda function exists, deploying it if necessary.
    
    If the Lambda function does not exist, packages its source code and creates the function using the provided role
    ARN. Logs deployment status and raises FileNotFoundError if the source file is missing.
    """
    logger = cfg.get_logger()
    name = "DisableUploadMonitorRule"
    handler = "disable_rule.lambda_handler"

    try:
        lambda_client.get_function(FunctionName=name)
        logger.info(f"Lambda '{name}' already exists.", indent=2)
        return
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.custom("Deploying DisableUploadMonitorRule Lambda...", indent=2, emoji="🚀")

    # 🔍 Resolve Lambda source files relative to config file
    deactivator_path = cfg.paths.lambda_dr_path
    try:
        zipped = zip_lambda(deactivator_path, "disable_rule.py")
    except FileNotFoundError as e:
        logger.error(f"Lambda zip failed: {e}", indent=3)
        raise

    lambda_client.create_function(
        FunctionName=name,
        Runtime="python3.9",
        Role=role_arn,
        Handler=handler,
        Code={"ZipFile": zipped},
        Timeout=30,
        MemorySize=128,
        Publish=True
    )
    logger.success(f"Lambda '{name}' deployed.", indent=2)


def setup_schedule_and_target(cfg, events_client, lambda_client, expected_total):
    """
    Creates or updates a CloudWatch Events rule to trigger the upload progress monitor Lambda.
    
    Configures a scheduled rule to invoke the "UploadProgressMonitor" Lambda function every 5 minutes, passing
    project-specific input parameters. Ensures the Lambda has the necessary permissions to be invoked by the event rule.
    """
    logger = cfg.get_logger()

    rule_name = "UploadProgressScheduleRule"
    target_lambda = "UploadProgressMonitor"

    rule_arn = events_client.put_rule(
        Name=rule_name,
        ScheduleExpression="rate(5 minutes)",
        State="ENABLED"
    )["RuleArn"]

    lambda_arn = lambda_client.get_function(FunctionName=target_lambda)["Configuration"]["FunctionArn"]

    project_config = {
        "project_slug": cfg.get("project.slug"),
        "bucket": cfg.get("aws.s3_bucket"),
        "prefix": f"{cfg.get('project.number')}/",
        "expected_total": expected_total
    }

    events_client.put_targets(
        Rule=rule_name,
        Targets=[{
            "Id": "UploadProgressTarget",
            "Arn": lambda_arn,
            "Input": json.dumps(project_config)
        }]
    )

    with contextlib.suppress(lambda_client.exceptions.ResourceConflictException):
        lambda_client.add_permission(
            FunctionName=target_lambda,
            StatementId="AllowEventsInvokeUploadProgress",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn
        )

    logger.success(f"CloudWatch rule '{rule_name}' updated with target input.", indent=2)


def deploy_lambda_monitor(cfg: ConfigManager):
    """
    Deploys and configures AWS Lambda functions and resources for monitoring image upload progress.

    Resolves configuration, counts final images, builds and uploads initial progress status to S3, ensures required
    Lambda functions are deployed, and sets up a scheduled CloudWatch Events rule to trigger progress monitoring.
    Logs key steps and warnings throughout the process.

    Args:
        cfg:
    """
    logger = cfg.get_logger()
    cfg.validate(tool="deploy_lambda_monitor")

    region = cfg.get("aws.region")
    role_arn = cfg.get("aws.lambda_role_arn")
    slug = cfg.get("project.slug")
    bucket = cfg.get("aws.s3_bucket")

    # Load credentials from keyring or config and verify AWS credentials
    try:
        logger.info("Verifying AWS credentials...", indent=1)
        access_key, secret_key = get_aws_credentials(cfg)
        session = Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        sts = session.client("sts")
        sts.get_caller_identity()
        logger.custom("AWS credentials verified.", emoji="🔑", indent=2)
    except Exception as e:
        logger.error(f"AWS credentials verification failed: {e}", indent=2)
        return
    
    logger.info("Preparing to deploy Lambda monitor...", indent=1)

    expected_total = count_final_images(cfg)

    if expected_total == 0:
        logger.warning("Expected_total is 0 — progress tracking will show 0% until images appear.", indent=2)

    progress = build_progress_json(cfg, expected_total)

    s3 = session.client("s3")
    lambda_client = session.client("lambda")
    events_client = session.client("events")

    upload_progress_json(s3, progress, bucket, slug, logger)
    ensure_lambda_progress_monitor(cfg, lambda_client, role_arn)
    ensure_lambda_deactivator(cfg, lambda_client, role_arn)
    setup_schedule_and_target(cfg, events_client, lambda_client, expected_total)
    logger.success("Deployed AWS Lambda monitor.", indent=1)
