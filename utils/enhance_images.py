# =============================================================================
# 🖼️ Image Enhancement Pipeline (utils/enhance_images.py)
# -----------------------------------------------------------------------------
# Purpose:             Enhances 360° images using white balance, contrast, saturation, and sharpening
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.0.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-08
#
# Description:
#   Loads enhancement configuration, checks disk space, and processes all images in an OID
#   feature class using OpenCV-based image enhancement operations. Enhancements include
#   white balance correction, CLAHE contrast, saturation boost, sharpening, and brightness recovery.
#   Supports batch multiprocessing, progress tracking, EXIF metadata preservation, and logging.
#
# File Location:        /utils/enhance_images.py
# Called By:            tools/enhance_images_tool.py, tools/process_360_orchestrator.py
# Int. Dependencies:    config_loader, arcpy_utils, check_disk_space, path_utils
# Ext. Dependencies:    cv2, numpy, arcpy, csv, os, pathlib, typing, subprocess, concurrent.futures
#
# Documentation:
#   See: docs/TOOL_GUIDES.md and docs/tools/enhance_images.md
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
import json
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config_loader import resolve_config
from utils.path_utils import get_log_path, get_image_base_path, get_enhancement_profile_path
from utils.arcpy_utils import log_message
from utils.check_disk_space import check_sufficient_disk_space
from utils.openai_utils import ask_chatgpt
from utils.generate_anchor_profile import run_profile_generation
from utils.image_stats import compute_image_stats


def chatgpt_adjustments(pre_stats, post_stats, config=None, messages=None):
    """
    Asks ChatGPT for enhancement parameter suggestions based on pre/post stats.

    Args:
        pre_stats (dict): Dict with brightness, contrast, saturation before enhancement.
        post_stats (dict): Dict with same fields after enhancement.
        config (dict): Full config with OpenAI settings.
        messages (list): Optional logger.

    Returns:
        dict | None: Suggested config with gamma, clahe_clip, saturation_boost
    """
    ai_assist = config.get("image_enhancement", {}).get("ai_assist", False)
    if not ai_assist:
        return None
    prompt = (
        "Here are image stats before and after enhancement:\n"
        f"Before: brightness={pre_stats['brightness']:.2f}, contrast={pre_stats['contrast']:.2f}, saturation={pre_stats['saturation']:.2f}\n"
        f"After: brightness={post_stats['brightness']:.2f}, contrast={post_stats['contrast']:.2f}, saturation={post_stats['saturation']:.2f}\n"
        "Target brightness: ~110, Target contrast: 45-50, Target saturation: ~60.\n"
        "Suggest whether to adjust gamma, CLAHE clip limit, or saturation to improve consistency.\n"
        "Return ONLY a valid JSON object with keys: 'gamma', 'clahe_clip', 'saturation_boost'."
    )
    return ask_chatgpt(prompt, config, messages)


# === Seam-aware Enhancement ===
def wrap_horizontal(img, wrap_size=64):
    left = img[:, -wrap_size:]
    right = img[:, :wrap_size]
    return np.concatenate([left, img, right], axis=1)


def unwrap_horizontal(img, wrap_size=64):
    return img[:, wrap_size:-wrap_size]


def enhance_image_seam_aware(img, enhance_config, contrast, full_config=None, messages=None, wrap_size=64):
    wrapped = wrap_horizontal(img, wrap_size=wrap_size)
    enhanced_wrapped, clip_limit_used, methods_applied, stats = enhance_image(
        wrapped, enhance_config, contrast, full_config=full_config, messages=messages
    )
    enhanced = unwrap_horizontal(enhanced_wrapped, wrap_size=wrap_size)
    return enhanced, clip_limit_used, methods_applied, stats


def load_enhancement_profile(profile_path):
    with open(profile_path, "r") as f:
        return json.load(f)

def interpolate_settings(image_name, profile, fallback):
    keys = sorted(profile.keys())
    if image_name in profile:
        return profile[image_name]
    try:
        index = keys.index(image_name)
    except ValueError:
        return fallback
    before = max(0, index - 1)
    after = min(len(keys) - 1, index + 1)
    k1, k2 = keys[before], keys[after]
    s1, s2 = profile[k1], profile[k2]
    return {k: (s1.get(k, 0) + s2.get(k, 0)) / 2 for k in s1}


# === Main Image Enhancement ===
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


def enhance_image(img, enhance_config, contrast, full_config=None, messages=None):
    """
    Enhances an image using configurable white balance, contrast, saturation, and sharpening.
    
    Applies a sequence of image enhancement operations based on the provided configuration, including optional white
    balance correction, CLAHE contrast enhancement, saturation boost, sharpening, and brightness recovery if needed.
    Tracks which methods were applied and collects pre- and post-enhancement brightness and contrast statistics.
    
    Args:
        img: Input image as a NumPy array in BGR format.
        enhance_config: Dictionary specifying which enhancements to apply and their parameters.
        contrast: Initial contrast value of the image, used for adaptive CLAHE.
        full_config: Optional full configuration dictionary for logging context.
        messages: Optional message handler for logging.
    
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

    if enhance_config.get("apply_white_balance", False):
        method = enhance_config.get("white_balance", {}).get("method", "gray_world")
        img, pre_means, post_means = apply_white_balance(img, method)
        methods_applied["white_balance"] = method
        stats["pre_rgb_means"] = pre_means
        stats["post_rgb_means"] = post_means

    if enhance_config.get("apply_contrast_enhancement", True):
        clahe_cfg = enhance_config.get("clahe", {})
        grid_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        clip_limit = clahe_cfg.get("clip_limit_low", 2.0)
        if enhance_config.get("adaptive", False):
            thresholds = clahe_cfg.get("contrast_thresholds", [30, 60])
            if contrast < thresholds[0]:
                clip_limit = clahe_cfg.get("clip_limit_high", 2.5)
        img = apply_clahe(img, clip_limit, grid_size)
        clip_limit_used = clip_limit
        methods_applied["clahe"] = True

    if enhance_config.get("apply_saturation_boost", False):
        factor = enhance_config.get("saturation_boost", {}).get("factor", 1.1)
        img = apply_saturation_boost(img, factor)

    if enhance_config.get("apply_sharpening", True):
        kernel = enhance_config.get("sharpen", {}).get("kernel", [
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
    if enhance_config.get("brightness_recovery", {}):
        threshold = enhance_config["brightness"].get("threshold", 110)
        factor = enhance_config["brightness"].get("factor", 1.15)
        if brightness_after < threshold:
            log_message(f"🔧 Brightness {brightness_after:.1f} < {threshold}, applying recovery factor {factor}",
                        messages, config=full_config)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            # Recompute stats after brightening
            brightness_after = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            contrast_after = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            stats["brightness_after"] = brightness_after
            stats["contrast_after"] = contrast_after

    return img, clip_limit_used, methods_applied, stats


def update_oid_image_paths(oid_fc: str, path_map: dict[str, str], messages=None):
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
    log_message("✅ OID ImagePath updated to reflect enhanced images.", messages)


def write_log(log_rows, config, messages=None):
    """
    Writes a CSV log file summarizing image enhancement details.
    
    The log includes statistics such as brightness, contrast, applied enhancement methods, and output paths for each
    processed image. Handles permission errors gracefully by logging a warning if the file cannot be written.
    """
    log_path = get_log_path("enhance_log", config)
    log_message(f"[DEBUG] Attempting to write enhance log to: {log_path}", messages, config=config)
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
        log_message(f"Enhancement log saved to: {log_path}", messages, config=config)
    except PermissionError as e:
        log_message(f"❌ Failed to write enhance log: {e}", messages, level="warning", config=config)


def enhance_single_image(original_path: Path, enhance_config, config, output_mode, suffix, original_tag, enhanced_tag,
                         messages, dry_run=False):
    """
    Enhances a single image file and writes the result to disk, handling output path logic and EXIF metadata copying.

    Reads the image, applies configured enhancement steps, determines the output path based on the specified mode
    (overwrite, suffix, or folder tag replacement), writes the enhanced image, and attempts to copy EXIF metadata from
    the original. Returns enhancement statistics and output paths, or an error message if processing fails.

    Args:
     original_path: Path to the original image file.
     enhance_config: Dictionary containing enhancement options (e.g., white balance, CLAHE, sharpening).
     config: Full configuration dictionary including paths and EXIF tool settings.
     output_mode: Determines how the output path is constructed ("overwrite", "suffix", or folder tag replacement).
     suffix: Suffix to append to the filename if output_mode is "suffix".
     original_tag: Folder tag to replace in the path if output_mode is folder tag replacement.
     enhanced_tag: Replacement folder tag for enhanced images.
     messages: Optional logger or list to collect status messages.

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
        return None, f"⚠️ Skipping unreadable image: {original_path}"

    stats = compute_image_stats(img)
    brightness = stats["brightness"]
    contrast = stats["contrast"]
    saturation = stats["saturation"]

    # Load profile and enhance
    profile_path = get_enhancement_profile_path(config)
    if profile_path and Path(profile_path).exists():
        profile = load_enhancement_profile(profile_path)
        image_name = original_path.name
        dynamic_config = interpolate_settings(image_name, profile, enhance_config)
    else:
        dynamic_config = enhance_config

    enhanced, clip_limit, methods, stats = enhance_image_seam_aware(img, dynamic_config, contrast, full_config=config,
                                                                    messages=messages)

    # Track pre-enhancement stats explicitly
    stats["brightness_before"] = brightness
    stats["contrast_before"] = contrast
    stats["saturation_before"] = saturation

    # After enhancement, compute saturation again
    hsv_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    stats["saturation_after"] = float(np.mean(hsv_after[:, :, 1]))

    # Post-analysis and retry if needed
    pre_stats = {
        "brightness": stats["brightness_before"],
        "contrast": stats["contrast_before"],
        "saturation": stats["saturation_before"]
    }
    post_stats = {
        "brightness": stats["brightness_after"],
        "contrast": stats["contrast_after"],
        "saturation": stats["saturation_after"]
    }
    suggestion = chatgpt_adjustments(pre_stats, post_stats, config=config, messages=messages)
    if suggestion:
        dynamic_config["clahe"] = {"clip_limit_low": suggestion.get("clahe_clip", 2.0)}
        dynamic_config["saturation_boost"] = {"factor": suggestion.get("saturation_boost", 1.1)}
        enhanced, clip_limit, methods, stats = enhance_image_seam_aware(img, dynamic_config, contrast,
                                                                        full_config=config, messages=messages)
        hsv_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        stats["saturation_after"] = float(np.mean(hsv_after[:, :, 1]))

    # Output path resolution
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
            return None, f"⚠️ Could not locate '{original_tag}' in image path: {original_path}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        return None, f"❌ Unknown output_mode: '{output_mode}'. Expected one of: overwrite, suffix, directory"

    if not dry_run:
        try:
            if not cv2.imwrite(str(out_path), enhanced):
                return None, f"❌ cv2 failed to write image to {out_path}"
            # ✅ Copy EXIF metadata from original to enhanced image
            exiftool_path = config.get("executables", {}).get("exiftool", {}).get("exe_path", "exiftool")
            copied = copy_exif_metadata(original_path, out_path, exiftool_path)
            if not copied:
                log_message(f"⚠️ Failed to copy EXIF metadata from {original_path.name}", messages, level="warning",
                            config=config)
        except Exception as e:
            return None, f"❌ Failed to write image to {out_path}: {e}"
    else:
        log_message(f"[Dry Run] Would write: {out_path}", messages, config=config)
        copied = True  # assume success for log row

    log_row = [
        original_path.name,
        round(stats["brightness_before"], 2),
        round(stats["contrast_before"], 2),
        round(stats.get("saturation_before", 0.0), 2),
        clip_limit or "",
        methods["white_balance"] or "no",
        "yes" if methods["clahe"] else "no",
        "yes" if methods["sharpen"] else "no",
        round(stats["brightness_after"], 2),
        round(stats["contrast_after"], 2),
        round(stats.get("saturation_after", 0.0), 2),
        f"{stats['pre_rgb_means']}" if stats["pre_rgb_means"] else "",
        f"{stats['post_rgb_means']}" if stats["post_rgb_means"] else "",
        str(out_path)
    ]
    return (str(original_path), str(out_path), log_row, not copied), None


def enhance_images_in_oid(
    oid_fc_path: str,
    config: Optional[dict] = None,
    config_file: Optional[str] = None,
    messages=None
)-> Dict[str, str]:

    """
    Enhances all images referenced in an ArcGIS ObjectID feature class using configurable image processing steps.
    
    This function loads enhancement configuration, checks disk space, retrieves image paths from the specified feature
    class, and processes each image in parallel. Enhancements may include white balance, contrast adjustment, saturation
    boost, and sharpening. Enhanced images are saved according to the configured output mode, and EXIF metadata is
    copied from originals. The function updates the feature class with new image paths if not overwriting originals,
    writes a CSV log of enhancement details, and logs summary statistics.
    
    Args:
        oid_fc_path: Path to the ArcGIS ObjectID feature class containing image paths.
        config: Optional configuration dictionary for enhancement parameters.
        config_file: Optional path to a configuration file.
        messages: Optional message handler for logging and progress reporting.
    
    Returns:
        A dictionary mapping original image paths to enhanced image paths.
    """
    config = resolve_config(
        config=config,
        config_file=config_file,
        oid_fc_path=oid_fc_path,
        messages=messages,
        tool_name="enhance_images"
    )

    if not config.get("image_enhancement", {}).get("enabled", False):
        log_message("Image enhancement is disabled in config. Skipping...", messages, config=config)
        return {}

    enh_config = config.get("image_enhancement", {})
    dry_run = enh_config.get("dry_run", False)
    output_mode = enh_config.get("output", {}).get("mode", {})
    suffix = enh_config.get("output", {}).get("suffix", "_enh")
    folders = config.get("image_output", {}).get("folders", {})
    original_tag = folders.get("original", "original")
    enhanced_tag = folders.get("enhanced", "enhanced")

    check_sufficient_disk_space(oid_fc=oid_fc_path, folder_key=original_tag, config=config, buffer_ratio=1.1,
                                verbose=True, messages=messages)

    with arcpy.da.SearchCursor(oid_fc_path, ["ImagePath"]) as cursor:
        paths = [Path(row[0]) for row in cursor]

    # Auto-generate enhancement profile if needed
    profile_path = get_enhancement_profile_path(config)
    auto_generate = enh_config.get("auto_generate_profile", False)

    sample_image_path = str(paths[0])
    image_dir = get_image_base_path(sample_image_path, original_tag, config, messages)
    if not image_dir:
        log_message("⚠️ Skipping auto-generation due to unresolved image directory.", messages, level="warning",
                    config=config)
        return {}

    if auto_generate and profile_path and not Path(profile_path).exists():
        log_message(f"📁 Enhancement profile not found at {profile_path}. Auto-generating from anchor frames...",
                    messages, config=config)
        run_profile_generation(image_dir=image_dir, config=config, messages=messages)

    if not auto_generate:
        log_message("ℹ️ Auto-generation of enhancement profile is disabled in config.", messages, config=config)

    if Path(profile_path).exists():
        log_message("✅ Enhancement profile found and will be used as-is.", messages, config=config)


    total = len(paths)
    use_progressor = False

    if messages:
        try:
            arcpy.SetProgressor("step", "Enhancing images...", 0, total, 1)
            use_progressor = True
        except (AttributeError, RuntimeError, arcpy.ExecuteError):
            pass

    log_rows = []
    path_map = {}
    brightness_deltas = []
    contrast_deltas = []
    failed_exif_copies = []

    max_workers = enh_config.get("max_workers")
    if not max_workers:
        cpu_cores = os.cpu_count() or 8
        max_workers = max(4, int(cpu_cores * 0.75))

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [
            executor.submit(enhance_single_image, p, enh_config, config, output_mode, suffix, original_tag,
                            enhanced_tag, messages, dry_run=dry_run)
            for p in paths
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result, error = future.result()
            if error:
                log_message(error, messages, level="warning", config=config)
                continue
            original_path_str, out_path_str, log_row, exif_failed = result  # still same if `log_row` just expanded

            log_message(f"Enhanced image saved: {Path(out_path_str).name}", messages, level="debug", config=config)
            path_map[original_path_str] = out_path_str
            log_rows.append(log_row)
            if exif_failed:
                failed_exif_copies.append((Path(original_path_str), Path(out_path_str)))

            # Collect deltas for summary stats
            b_before = float(log_row[1])
            c_before = float(log_row[2])
            b_after = float(log_row[8])
            c_after = float(log_row[9])
            brightness_deltas.append(b_after - b_before)
            contrast_deltas.append(c_after - c_before)

            if use_progressor:
                arcpy.SetProgressorLabel(f"Enhancing {idx}/{total} ({(idx/total)*100:.1f}%)")
                arcpy.SetProgressorPosition(idx)

    if use_progressor:
        arcpy.ResetProgressor()

    write_log(log_rows, config=config, messages=messages)

    if output_mode != "overwrite":
        update_oid_image_paths(oid_fc_path, path_map, messages)

    if failed_exif_copies:
        exiftool_path = config.get("executables", {}).get("exiftool", {}).get("exe_path", "exiftool")
        retry_successes = 0
        log_message(f"🔄 Retrying EXIF metadata copy for {len(failed_exif_copies)} image(s)...", messages, config=config)
        for orig, enh in failed_exif_copies:
            if copy_exif_metadata(orig, enh, exiftool_path):
                retry_successes += 1
            else:
                log_message(f"❌ Final EXIF copy failed: {enh.name}", messages, level="error", config=config)
        log_message(f"✅ Retried EXIF copy success count: {retry_successes}/{len(failed_exif_copies)}", messages,
                    config=config)

    # Summary
    if brightness_deltas:
        mean_bright_delta = np.mean(brightness_deltas)
        mean_contrast_delta = np.mean(contrast_deltas)
        log_message(f"📊 Avg Brightness Δ: {mean_bright_delta:.2f} | Avg Contrast Δ: {mean_contrast_delta:.2f}",
                    messages, config=config)

    if dry_run:
        log_message("✅ Dry run complete. No images or metadata were written.", messages, config=config)

    return path_map


if __name__ == "__main__":
    enhance_images_in_oid("path/to/your/oid_fc")
