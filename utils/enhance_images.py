# =============================================================================
# 🖼️ Image Enhancement Pipeline (utils/enhance_images.py)
# -----------------------------------------------------------------------------
# Purpose:             Enhances 360° images using white balance, contrast, saturation, and sharpening
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-13
# Last Updated:        2025-05-15
#
# Description:
#   Loads enhancement configuration, checks disk space, and processes all images in an OID
#   feature class using OpenCV-based image enhancement operations. Enhancements include
#   white balance correction, CLAHE contrast, saturation boost, sharpening, and brightness recovery.
#   Supports batch multiprocessing, progress tracking, EXIF metadata preservation, and logging.
#
# File Location:        /utils/enhance_images.py
# Called By:            tools/enhance_images_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    utils/manager/config_manager, utils/shared/check_disk_space
# Ext. Dependencies:    cv2, numpy, arcpy, csv, os, pathlib, subprocess, concurrent.futures
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/enhance_images.md
#   (Ensure these docs are current; update if needed.)
#
# Notes:
#   - Automatically determines parallelism via available CPU cores unless overridden
#   - Recovers EXIF metadata after enhancement using ExifTool
# =============================================================================

import os
import csv
import cv2
import numpy as np
import arcpy
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.manager.config_manager import ConfigManager
from utils.shared.check_disk_space import check_sufficient_disk_space


def compute_image_stats(img):
    """
    Calculates the brightness and contrast of an image.
    
    The image is converted to grayscale before computing the mean (brightness) and standard deviation (contrast) of
    pixel intensities.
    
    Args:
    	img: Input image as a NumPy array in BGR color format.
    
    Returns:
    	A tuple containing the brightness (float) and contrast (float) of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return brightness, contrast


def apply_white_balance(img, method="gray_world"):
    """
    Applies white balance correction to an image using the specified method.
    
    Args:
        img: Input image as a NumPy array in BGR format.
        method: White balance method to use ("gray_world" or "simple").
    
    Returns:
        A tuple containing the white-balanced image, the mean values of the B, G, R channels before correction, and
        the mean values after correction.
    """
    pre_means = tuple(np.mean(img[:, :, c]) for c in range(3))  # B, G, R

    if method == "gray_world":
        avg_b, avg_g, avg_r = pre_means
        avg_gray = (avg_b + avg_g + avg_r) / 3
        eps = 1e-6  # avoid div/0
        img = cv2.merge([
            cv2.addWeighted(img[:, :, 0], avg_gray / max(avg_b, eps), 0, 0, 0),
            cv2.addWeighted(img[:, :, 1], avg_gray / max(avg_g, eps), 0, 0, 0),
            cv2.addWeighted(img[:, :, 2], avg_gray / max(avg_r, eps), 0, 0, 0)
        ])
    elif method == "simple":
        wb = cv2.xphoto.createSimpleWB()
        img = wb.balanceWhite(img)

    post_means = tuple(np.mean(img[:, :, c]) for c in range(3))
    return img, pre_means, post_means


def apply_clahe(img, clip_limit, tile_grid_size):
    """
    Enhances image contrast using CLAHE on the luminance channel.
    
    Converts the input image to LAB color space, applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to
    the L (luminance) channel, and returns the result converted back to BGR color space.
    
    Args:
        img: Input image in BGR format.
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Size of the grid for histogram equalization.
    
    Returns:
        The contrast-enhanced image in BGR format.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_saturation_boost(img, factor):
    """
    Boosts the color saturation of an image by a specified factor.
    
    Converts the image to HSV color space, multiplies the saturation channel by the given factor (clipped to 255), and
    converts the result back to BGR.
    
    Args:
        img: Input image in BGR format.
        factor: Multiplicative factor for the saturation channel.
    
    Returns:
        The image with enhanced color saturation in BGR format.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_sharpening(img, kernel):
    """
    Applies a sharpening filter to an image using a specified convolution kernel.
    
    Args:
        img: Input image in BGR format.
    	kernel: A 2D list or array representing the sharpening kernel to apply.
    
    Returns:
    	The sharpened image as a NumPy array.
    """
    kernel_np = np.array(kernel, dtype=np.float32)
    return cv2.filter2D(img, -1, kernel_np)


def copy_exif_metadata(original: Path, enhanced: Path, exiftool_path: str = "exiftool"):
    """
    Copies all EXIF metadata from the original image to the enhanced image using exiftool.
    
    Args:
        original: Path to the source image with desired EXIF metadata.
        enhanced: Path to the target image to receive the metadata.
        exiftool_path: Path to the exiftool executable (default is "exiftool").
    
    Returns:
        True if metadata was copied successfully, False if the subprocess call failed.
    """
    try:
        # Suppress console window on Windows
        startupinfo = None
        if os.name == "nt":  # Windows only
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(
            [exiftool_path, "-TagsFromFile", str(original), "-overwrite_original", "-all:all", str(enhanced)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        return True
    except subprocess.CalledProcessError:
        return False


def enhance_image(img, cfg: ConfigManager, contrast, logger):
    """
    Enhances an image using configurable white balance, contrast, saturation, and sharpening.
    
    Applies a sequence of image enhancement operations based on the provided configuration, including optional white
    balance correction, CLAHE contrast enhancement, saturation boost, sharpening, and brightness recovery if needed.
    Tracks which methods were applied and collects pre- and post-enhancement brightness and contrast statistics.
    
    Args:
        img: Input image as a NumPy array in BGR format.
        cfg:
        contrast: Initial contrast value of the image, used for adaptive CLAHE.
        logger:
    
    Returns:
        A tuple containing:
            - The enhanced image as a NumPy array.
            - The CLAHE clip limit used (if applied), otherwise None.
            - A dictionary indicating which enhancement methods were applied.
            - A dictionary of pre- and post-enhancement statistics.
    """
    clip_limit_used = None
    methods_applied = {"white_balance": None, "clahe": False, "sharpen": False}

    stats = {
        "pre_rgb_means": None,
        "post_rgb_means": None,
        "brightness_before": np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "contrast_before": np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        "brightness_after": None,
        "contrast_after": None
    }

    if cfg.get("image_enhancement.white_balance.enabled", False):
        method = cfg.get("image_enhancement.white_balance.method", "gray_world")
        img, pre_means, post_means = apply_white_balance(img, method)
        methods_applied["white_balance"] = method
        stats["pre_rgb_means"] = pre_means
        stats["post_rgb_means"] = post_means

    if cfg.get("image_enhancement.clahe.enabled", True):
        grid_size = tuple(cfg.get("image_enhancement.clahe.tile_grid_size", [8, 8]))
        clip_limit = cfg.get("image_enhancement.clahe.clip_limit_low", 2.0)
        if cfg.get("image_enhancement.adaptive", False):
            thresholds = cfg.get("image_enhancement.clahe.contrast_thresholds", [30, 60])
            if contrast < thresholds[0]:
                clip_limit = cfg.get("image_enhancement.clahe.clip_limit_high", 2.5)
        img = apply_clahe(img, clip_limit, grid_size)
        clip_limit_used = clip_limit
        methods_applied["clahe"] = True

    if cfg.get("image_enhancement.saturation_boost.enabled", False):
        factor = cfg.get("image_enhancement.saturation_boost.factor", 1.1)
        img = apply_saturation_boost(img, factor)

    if cfg.get("image_enhancement.sharpen.enabled", True):
        kernel = cfg.get("image_enhancement.sharpen.kernel", [
            [0, -0.5, 0],
            [-0.5, 3.0, -0.5],
            [0, -0.5, 0]
        ])
        img = apply_sharpening(img, kernel)
        methods_applied["sharpen"] = True

    # Post-enhancement stats
    brightness_after = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    contrast_after = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    stats["brightness_after"] = brightness_after
    stats["contrast_after"] = contrast_after

    # Optional brightness recovery
    if cfg.get("image_enhancement.brightness.enabled", False):
        threshold = cfg.get("image_enhancement.brightness.threshold", 110)
        factor = cfg.get("image_enhancement.brightness.factor", 1.15)
        if brightness_after < threshold:
            logger.info(f"🔧 Brightness {brightness_after:.1f} < {threshold}, applying recovery factor {factor}", indent=2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            # Recompute stats after brightening
            brightness_after = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            contrast_after = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            stats["brightness_after"] = brightness_after
            stats["contrast_after"] = contrast_after

    return img, clip_limit_used, methods_applied, stats


def update_oid_image_paths(oid_fc: str, path_map: dict[str, str], logger):
    """
    Updates the "ImagePath" field in an OID feature class to reference enhanced image paths.
    
    Replaces original image paths with corresponding enhanced paths from the provided mapping.
    """
    with arcpy.da.UpdateCursor(oid_fc, ["ImagePath"]) as cursor:
        for row in cursor:
            original = row[0]
            if original in path_map:
                row[0] = path_map[original]
                cursor.updateRow(row)
    logger.custom("OID ImagePath updated to reflect enhanced images.", indent=1, emoji="✅")


def write_log(log_rows, cfg, logger):
    """
    Writes a CSV log file summarizing image enhancement details.
    
    The log includes statistics such as brightness, contrast, applied enhancement methods, and output paths for each
    processed image. Handles permission errors gracefully by logging a warning if the file cannot be written.
    """
    log_path = cfg.paths.get_log_file_path("enhance_log", cfg)
    logger.debug(f"Attempting to write enhance log to: {log_path}", indent=1)
    try:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Filename", "BrightnessBefore", "ContrastBefore", "ClipLimit",
                "WhiteBalance", "CLAHE", "Sharpen",
                "BrightnessAfter", "ContrastAfter",
                "RGBMeansBeforeWB", "RGBMeansAfterWB", "OutputPath"
            ])
            writer.writerows(log_rows)
        logger.info(f"Enhancement log saved to: {log_path}", indent=1)
    except PermissionError as e:
        logger.warning(f"Failed to write enhance log: {e}", indent=1)


def enhance_single_image(original_path: Path, cfg: ConfigManager, logger):
    """
    Enhances a single image file and writes the result to disk, handling output path logic and EXIF metadata copying.

    Reads the image, applies configured enhancement steps, determines the output path based on the specified mode
    (overwrite, suffix, or folder tag replacement), writes the enhanced image, and attempts to copy EXIF metadata from
    the original. Returns enhancement statistics and output paths, or an error message if processing fails.

    Args:
     original_path: Path to the original image file.
     cfg:
     logger:

    Returns:
     A tuple containing:
         - The original image path as a string.
         - The output image path as a string.
         - A list of enhancement statistics for logging.
         - A boolean indicating if EXIF metadata copy failed.
     If an error occurs, returns (None, error_message).
    """
    img = cv2.imread(str(original_path))
    if img is None:
        logger.warning(f"Skipping unreadable image: {original_path}", indent=2)
        return None

    output_mode = cfg.get("image_enhancement.output.mode", "directory")
    suffix = cfg.get("image_enhancement.output.suffix", "_enh")
    original_tag = cfg.get("image_output.folders.original")
    enhanced_tag = cfg.get("image_output.folders.enhanced")


    brightness, contrast = compute_image_stats(img)
    enhanced, clip_limit, methods, stats = enhance_image(img, cfg, contrast, logger)

    if output_mode == "overwrite":
        out_path = original_path
    elif output_mode == "suffix":
        out_path = original_path.with_name(f"{original_path.stem}{suffix}.jpg")
    elif output_mode == "directory":
        path_str = str(original_path)
        if f"/{original_tag}/" in path_str:
            out_path = Path(path_str.replace(f"/{original_tag}/", f"/{enhanced_tag}/", 1))
        elif f"\\{original_tag}\\" in path_str:
            out_path = Path(path_str.replace(f"\\{original_tag}\\", f"\\{enhanced_tag}\\", 1))
        else:
            logger.warning(f"Could not locate '{original_tag}' in image path: {original_path}", indent=2)
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        logger.error(f"Unknown output_mode: '{output_mode}'. Expected one of: overwrite, suffix, directory", indent=2)
        return None

    try:
        if not cv2.imwrite(str(out_path), enhanced):
            logger.error(f"cv2 failed to write image to {out_path}", indent=2)
            return None
        # Copy EXIF metadata from original to enhanced image
        exiftool_path = cfg.paths.exiftool_exe
        copied = copy_exif_metadata(original_path, out_path, exiftool_path)
        if not copied:
            logger.warning(f"Failed to copy EXIF metadata from {original_path.name}", indent=3)
    except Exception as e:
        logger.error(f"Failed to write image to {out_path}: {e}", indent=2)
        return None

    log_row = [
        original_path.name,
        round(stats["brightness_before"], 2),
        round(stats["contrast_before"], 2),
        clip_limit or "",
        methods["white_balance"] or "no",
        "yes" if methods["clahe"] else "no",
        "yes" if methods["sharpen"] else "no",
        round(stats["brightness_after"], 2),
        round(stats["contrast_after"], 2),
        f"{stats['pre_rgb_means']}" if stats["pre_rgb_means"] else "",
        f"{stats['post_rgb_means']}" if stats["post_rgb_means"] else "",
        str(out_path)
    ]
    return (str(original_path), str(out_path), log_row, not copied), None


def process_images_in_parallel(paths, cfg, logger, max_workers, progressor):
    log_rows = []
    path_map = {}
    brightness_deltas = []
    contrast_deltas = []
    failed_exif_copies = []
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [
            executor.submit(enhance_single_image, p, cfg, logger)
            for p in paths
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result, error = future.result()
            if error:
                logger.warning(error)
                continue
            original_path_str, out_path_str, log_row, exif_failed = result
            logger.debug(f"Enhanced image saved: {Path(out_path_str).name}", indent=2)
            path_map[original_path_str] = out_path_str
            log_rows.append(log_row)
            if exif_failed:
                failed_exif_copies.append((Path(original_path_str), Path(out_path_str)))
            b_before = float(log_row[1])
            c_before = float(log_row[2])
            b_after = float(log_row[7])
            c_after = float(log_row[8])
            brightness_deltas.append(b_after - b_before)
            contrast_deltas.append(c_after - c_before)
            progressor.update(idx)
    return path_map, log_rows, brightness_deltas, contrast_deltas, failed_exif_copies


def enhance_images_in_oid(cfg: ConfigManager, oid_fc_path: str):
    """
    Enhances all images referenced in an ArcGIS ObjectID feature class using configurable image processing steps.
    This function loads enhancement configuration, checks disk space, retrieves image paths from the specified feature
    class, and processes each image in parallel. Enhancements may include white balance, contrast adjustment, saturation
    boost, and sharpening. Enhanced images are saved according to the configured output mode, and EXIF metadata is
    copied from originals. The function updates the feature class with new image paths if not overwriting originals,
    writes a CSV log of enhancement details, and logs summary statistics.

    Args:
        cfg:
        oid_fc_path: Path to the ArcGIS ObjectID feature class containing image paths.

    Returns:
        A dictionary mapping original image paths to enhanced image paths.
    """
    logger = cfg.get_logger()
    cfg.validate(tool="enhance_images")

    if not cfg.get("image_enhancement.enabled", False):
        logger.info("Image enhancement is disabled in config. Skipping...", indent=1)
        return {}

    check_sufficient_disk_space(oid_fc_path, cfg)
    output_mode = cfg.get("image_enhancement.output.mode", "directory")

    logger.info(f"Enhancing images in OID feature class: {oid_fc_path}", indent=1)

    with arcpy.da.SearchCursor(oid_fc_path, ["ImagePath"]) as cursor:
        paths = [Path(row[0]) for row in cursor]

    log_rows = []
    path_map = {}
    brightness_deltas = []
    contrast_deltas = []
    failed_exif_copies = []

    max_workers = cfg.get("image_enhancement.max_workers")
    if not max_workers:
        cpu_cores = os.cpu_count() or 8
        max_workers = max(4, int(cpu_cores * 0.75))

    with cfg.get_progressor(total=len(paths), label=f"Enhancing {len(paths)} image(s)") as progressor:
        path_map, log_rows, brightness_deltas, contrast_deltas, failed_exif_copies = process_images_in_parallel(
            paths, cfg, logger, max_workers, progressor
        )

    write_log(log_rows, cfg, logger)

    if output_mode != "overwrite":
        update_oid_image_paths(oid_fc_path, path_map, logger)

    if failed_exif_copies:
        exiftool_path = cfg.paths.exiftool_exe
        retry_successes = 0
        logger.custom(f"Retrying EXIF metadata copy for {len(failed_exif_copies)} image(s)...", indent=2, emoji="🔄")
        for orig, enh in failed_exif_copies:
            if copy_exif_metadata(orig, enh, exiftool_path):
                retry_successes += 1
            else:
                logger.error(f"Final EXIF copy failed: {enh.name}", indent=2)
        logger.custom(f"Retried EXIF copy success count: {retry_successes}/{len(failed_exif_copies)}", indent=2, emoji="✅")

    # Summary
    if brightness_deltas:
        mean_bright_delta = np.mean(brightness_deltas)
        mean_contrast_delta = np.mean(contrast_deltas)
        logger.custom(f"Avg Brightness Δ: {mean_bright_delta:.2f} | Avg Contrast Δ: {mean_contrast_delta:.2f}", indent=1, emoji="📊")

    return path_map
