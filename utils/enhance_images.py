# =============================================================================
# 🖼️  Image Enhancement Pipeline – Seam-Aware Batch Processor
# -----------------------------------------------------------------------------
# Purpose:     Applies configurable image enhancements to 360° panoramic imagery
#              using seam-aware logic, statistical profiling, and optional AI adjustments.
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-08
#
# Description:
#   This module processes images listed in an ArcGIS ObjectID feature class by:
#     - Loading or generating per-image enhancement profiles
#     - Applying OpenCV-based enhancements (white balance, CLAHE, sharpening, etc.)
#     - Performing seam-aware wrapping/unwrapping to preserve 360° edge continuity
#     - Optionally adjusting parameters using AI (ChatGPT) based on image stats
#     - Writing enhancement logs and updating feature class references
#     - Retrying failed EXIF metadata copies using ExifTool
#
#   Supports batch parallelism, dry-run mode, logging, ArcGIS progressors, and per-image config overrides.
#
# File Location:      /utils/enhance_images.py
# Called By:          tools/enhance_images_tool.py, tools/process_360_orchestrator.py
#
# Key Dependencies:
#   - Internal: config_loader, path_utils, arcpy_utils, check_disk_space, generate_anchor_profiles, image_stats,
#   image_enhancer, exif_utils
#   - External: csv, arcpy, json, typing, concurrent.futures
#
# Related Docs:
#   See: docs/TOOL_GUIDES.md and docs/tools/enhance_images.md
#
# Notes:
#   - Automatically adjusts thread pool size based on available CPU cores (unless overridden)
#   - Seam wrapping adds border context for CLAHE/sharpening in panoramic environments
#   - Uses ExifTool for metadata recovery (path configurable in YAML)
# =============================================================================
import csv
import arcpy
import json
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config_loader import resolve_config
from utils.path_utils import get_log_path, get_image_base_path, get_enhancement_profile_path
from utils.arcpy_utils import log_message
from utils.check_disk_space import check_sufficient_disk_space
from utils.generate_anchor_profile import run_profile_generation
from utils.image_stats import *
from utils.image_enhancer import *
from utils.exif_utils import *
from utils.openai_utils import ask_chatgpt, clean_json_response


# === Seam Wrapping Utilities for Panoramic Enhancements ===
def wrap_horizontal(img: np.ndarray, wrap_size: int = 64) -> np.ndarray:
    """
    Horizontally wraps an image by padding left and right edges with opposite borders.

    This is used to preserve continuity in 360° panoramic enhancements.
    """
    left = img[:, -wrap_size:]
    right = img[:, :wrap_size]
    return np.concatenate([left, img, right], axis=1)


def unwrap_horizontal(img: np.ndarray, wrap_size: int = 64) -> np.ndarray:
    """
    Removes horizontal padding previously added by wrap_horizontal().
    """
    return img[:, wrap_size:-wrap_size]


# === Enhancement Core Functions ===
def apply_seam_aware_enhancements(
    img: np.ndarray,
    enhance_config: Dict,
    contrast: float,
    full_config: Optional[Dict] = None,
    messages: Optional[list] = None,
    wrap_size: int = 64
) -> Tuple[np.ndarray, Optional[float], Dict[str, Optional[bool | str]], Dict[str, float | tuple]]:
    """
    Applies image enhancement operations to a seam-wrapped version of the input image.

    This function performs enhancements (e.g., white balance, CLAHE, sharpening) on a horizontally-wrapped
    version of the input image to preserve edge continuity in 360° panoramic imagery. After enhancement,
    the image is unwrapped to restore its original shape.

    Args:
        img (np.ndarray): Input image in BGR format as a NumPy array.
        enhance_config (dict): Dictionary containing enhancement settings (e.g., white balance method, CLAHE config).
        contrast (float): Pre-computed contrast value used to select CLAHE strength if adaptive mode is enabled.
        full_config (dict, optional): Full resolved config dict used for logging context (default is None).
        messages (list, optional): Logger or list to collect debug/status messages (default is None).
        wrap_size (int): Width in pixels for left/right seam wrapping (default is 64).

    Returns:
        tuple:
            - np.ndarray: The enhanced image after unwrapping.
            - Optional[float]: CLAHE clip limit used, if applied.
            - dict: Dictionary of enhancement methods applied (e.g., {'white_balance': 'gray_world', 'clahe': True}).
            - dict: Dictionary of image statistics (e.g., brightness_before, contrast_after).
    """
    wrapped = wrap_horizontal(img, wrap_size=wrap_size)
    enhanced_wrapped, clip_limit_used, methods_applied, stats = apply_enhancements(
        wrapped, enhance_config, contrast, full_config=full_config, messages=messages
    )
    enhanced = unwrap_horizontal(enhanced_wrapped, wrap_size=wrap_size)
    return enhanced, clip_limit_used, methods_applied, stats


def chatgpt_img_adj(
    pre_stats: Dict[str, float],
    post_stats: Dict[str, float],
    config: Optional[Dict] = None,
    messages: Optional[list] = None
) -> Optional[Dict[str, float]]:
    """
    Uses ChatGPT to suggest enhancement adjustments based on pre- and post-enhancement statistics.

    Compares brightness, contrast, and saturation before and after enhancement and prompts ChatGPT
    to suggest tuning parameters (gamma, CLAHE clip limit, and saturation boost). Expects a JSON-formatted
    response. Returns None if AI assistance is disabled.

    Args:
        pre_stats (dict): Stats before enhancement — keys: brightness, contrast, saturation.
        post_stats (dict): Stats after enhancement — same keys.
        config (dict | None): Full config dictionary including OpenAI settings.
        messages (list | None): Optional logger for capturing output or warnings.

    Returns:
        dict | None: Suggested parameter adjustments, or None if AI is disabled or unavailable.
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
    response_text = ask_chatgpt(prompt, config, messages)
    if not response_text:
        return None

    log_message(f"ChatGPT response: {type(response_text)}", messages, level="debug")
    try:
        return clean_json_response(response_text)
    except Exception as e:
        log_message(f"ChatGPT JSON parse error: {e}", messages, level="warning", config=config)
        return None


def normalize_dynamic_config(config: dict) -> dict:
    """
    Normalizes a flat enhancement config into structured format expected by apply_enhancements().

    Converts keys like 'clahe_clip' and 'saturation_boost' into nested dictionaries matching
    the CLAHE and saturation blocks in config.sample.yaml.
    """
    normalized = config.copy()

    if "clahe_clip" in normalized:
        normalized["clahe"] = {"clip_limit_low": normalized.pop("clahe_clip")}

    if "saturation_boost" in normalized and not isinstance(normalized["saturation_boost"], dict):
        normalized["saturation_boost"] = {"factor": normalized["saturation_boost"]}

    return normalized


def apply_and_adjust_enhancement(
    img: np.ndarray,
    dynamic_config: Dict,
    enhance_config: Dict,
    contrast: float,
    config: Dict,
    messages: Optional[list] = None
) -> Tuple[np.ndarray, Optional[float], Dict[str, Optional[bool | str]], Dict[str, float | tuple]]:
    """
    Enhances an image using seam-aware enhancements and optionally applies adjustments based on AI feedback.

    This function performs a first enhancement pass using the given dynamic config. It then evaluates
    brightness, contrast, and saturation before and after enhancement. If AI-based adjustment is enabled
    in the config and ChatGPT suggests parameter changes, a second enhancement pass is executed using
    the updated configuration.

    Args:
        img (np.ndarray): Input image in BGR format as a NumPy array.
        dynamic_config (dict): Config dict with profile-adjusted enhancement parameters for this image.
        enhance_config (dict): The original (global) enhancement settings from config.image_enhancement.
        contrast (float): Pre-computed contrast value of the input image.
        config (dict): Full configuration dictionary (used for logging and AI toggle).
        messages (list, optional): Logger or list to collect status/debug messages.

    Returns:
        tuple:
            - np.ndarray: Final enhanced image (after 1 or 2 passes).
            - Optional[float]: CLAHE clip limit used (if applied).
            - dict: Dictionary of enhancement methods applied (e.g., white_balance, clahe, sharpen).
            - dict: Dictionary of enhancement stats, including brightness/contrast before and after.
    """
    log_message("🔄 Applying enhancements with the given dynamic config...", messages, config=config)
    # First enhancement pass
    enhanced, clip_limit, methods, stats = apply_seam_aware_enhancements(
        img, dynamic_config, contrast, full_config=config, messages=messages
    )

    # Compute pre-enhancement stats
    pre_stats = compute_image_stats(img)
    stats["brightness_before"] = pre_stats["brightness"]
    stats["contrast_before"] = pre_stats["contrast"]
    stats["saturation_before"] = pre_stats["saturation"]

    # Compute post-enhancement stats
    post_stats = compute_image_stats(enhanced)
    stats["brightness_after"] = post_stats["brightness"]
    stats["contrast_after"] = post_stats["contrast"]
    stats["saturation_after"] = post_stats["saturation"]

    # AI-based adjustment
    suggestion = chatgpt_img_adj(pre_stats, post_stats, config=config, messages=messages)
    log_message(f"ChatGPT response: {suggestion}", messages, level="debug", config=config)
    if suggestion:
        log_message(f"Applying suggested changes from ChatGPT: {suggestion}", messages, config=config)
        dynamic_config["clahe"] = {"clip_limit_low": suggestion.get("clahe_clip", 2.0)}
        dynamic_config["saturation_boost"] = {"factor": suggestion.get("saturation_boost", 1.1)}
        dynamic_config = normalize_dynamic_config(dynamic_config)
        if enhance_config.get("apply_white_balance", False):
            dynamic_config["apply_white_balance"] = True
            dynamic_config["white_balance"] = enhance_config.get("white_balance", {})

        # Second enhancement pass with AI-adjusted config
        enhanced, clip_limit, methods, stats = apply_seam_aware_enhancements(
            img, dynamic_config, contrast, full_config=config, messages=messages
        )

        # Recompute post-enhancement stats after retry
        post_stats = compute_image_stats(enhanced)
        stats["brightness_after"] = post_stats["brightness"]
        stats["contrast_after"] = post_stats["contrast"]
        stats["saturation_after"] = post_stats["saturation"]

    return enhanced, clip_limit, methods, stats


# === Dynamic Profile Configuration ===
def load_enhancement_profile(profile_path: Union[str, Path]) -> dict:
    """Load and parse enhancement profile JSON from disk."""
    with open(profile_path, "r") as f:
        return json.load(f)


def interpolate_profile_settings(
    image_name: str,
    profile: Dict[str, Dict[str, float]],
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Interpolates enhancement settings for a given image from a profile, or falls back to defaults.

    If the image is found in the profile, returns its settings. Otherwise, attempts to interpolate
    enhancement parameters between its nearest neighbors in the profile key list (assumed sorted).
    The result is merged into a copy of the fallback config to preserve top-level flags.

    Args:
        image_name (str): The name of the image to retrieve or interpolate settings for.
        profile (dict): Mapping of image name to parameter dictionaries (e.g., clip_limit, gamma).
        fallback (dict): Default config used for interpolation and to preserve global flags.

    Returns:
        dict: Merged config with interpolated or exact enhancement parameters.
    """
    keys = sorted(profile.keys())
    if image_name in profile:
        interpolated = profile[image_name]
    else:
        try:
            index = keys.index(image_name)
        except ValueError:
            return fallback
        before = max(0, index - 1)
        after = min(len(keys) - 1, index + 1)
        k1, k2 = keys[before], keys[after]
        s1, s2 = profile[k1], profile[k2]
        interpolated = {k: (s1.get(k, 0) + s2.get(k, 0)) / 2 for k in s1}

    # Merge into a copy of the fallback (so we keep top-level flags like apply_white_balance)
    merged = fallback.copy()
    merged.update(interpolated)
    return merged


def load_dynamic_config(
    original_path: Path,
    enhance_config: Dict,
    config: Dict
) -> Dict:
    """
    Loads and resolves the enhancement config for a specific image.

    If an enhancement profile is present and contains the image, interpolated settings are
    loaded and merged with the fallback config. Otherwise, the global enhancement config is returned.

    Args:
        original_path (Path): Full path to the image being processed.
        enhance_config (dict): Global enhancement config used as fallback/default.
        config (dict): Full resolved config used to locate the profile.

    Returns:
        dict: Final per-image enhancement config (interpolated or fallback), normalized to expected structure.
    """
    profile_path = get_enhancement_profile_path(config)
    if profile_path and Path(profile_path).exists():
        profile = load_enhancement_profile(profile_path)
        image_name = original_path.name
        merged = interpolate_profile_settings(image_name, profile, enhance_config)
        return normalize_dynamic_config(merged)
    return enhance_config


# === Image Enhancement ===
def enhance_single_image(
    original_path: Path,
    enhance_config: Dict,
    config: Dict,
    output_mode: str,
    suffix: str,
    original_tag: str,
    enhanced_tag: str,
    messages: Optional[list],
    dry_run: bool = False
) -> Tuple[Optional[Tuple[str, str, list, bool]], Optional[str]]:
    """
    Enhances a single image using seam-aware enhancements and writes the result to disk.

    This function applies white balance, contrast, sharpening, and optional AI-tuned adjustments
    to the image. It computes pre/post stats, resolves the output path based on the selected mode,
    and attempts to copy EXIF metadata from the original to the enhanced output.

    Args:
        original_path (Path): Path to the original image file to enhance.
        enhance_config (dict): Global enhancement configuration.
        config (dict): Full resolved config dictionary.
        output_mode (str): Output strategy: "overwrite", "suffix", or "directory".
        suffix (str): Filename suffix to use if output_mode is "suffix".
        original_tag (str): Folder tag to replace if output_mode is "directory".
        enhanced_tag (str): Replacement tag for enhanced output paths.
        messages (list | None): Logger or message list for status updates.
        dry_run (bool): If True, skips writing the image and EXIF metadata.

    Returns:
        tuple:
            - If successful:
                Tuple[str, str, list, bool]: (original path, output path, log row, exif_copy_failed)
            - If error:
                (None, str): None and an error message string
    """
    img = cv2.imread(str(original_path))
    if img is None:
        return None, f"⚠️ Skipping unreadable image: {original_path}"

    dynamic_config = load_dynamic_config(original_path, enhance_config, config)
    stats = compute_image_stats(img)
    contrast = stats["contrast"]

    enhanced, clip_limit, methods, stats = apply_and_adjust_enhancement(
        img, dynamic_config, enhance_config, contrast, config, messages
    )
    stats["clip_limit"] = clip_limit

    out_path = resolve_output_path(original_path, output_mode, suffix, original_tag, enhanced_tag)
    if not out_path:
        return None, f"⚠️ Could not resolve output path: {original_path}"

    if not dry_run:
        try:
            if not cv2.imwrite(str(out_path), enhanced):
                return None, f"❌ Failed to write image to {out_path}"
            exiftool_path = config.get("executables", {}).get("exiftool", {}).get("exe_path", "exiftool")
            copied = copy_exif_metadata(original_path, out_path, exiftool_path)
            if not copied:
                log_message(f"⚠️ Failed to copy EXIF metadata for {original_path.name}", messages, level="warning", config=config)
        except Exception as e:
            return None, f"❌ Failed to write image: {e}"
    else:
        log_message(f"[Dry Run] Would write: {out_path}", messages, config=config)
        copied = True

    log_row = generate_log_row(stats, methods, original_path, out_path)
    return (str(original_path), str(out_path), log_row, not copied), None


# === Output Utilities ===
def resolve_output_path(
    original_path: Path,
    output_mode: str,
    suffix: str,
    original_tag: str,
    enhanced_tag: str
) -> Optional[Path]:
    """
    Resolves the output path for an enhanced image based on the selected output mode.

    Supports three output modes:
    - "overwrite": Returns the original path.
    - "suffix": Adds a suffix to the filename (before the extension).
    - "directory": Replaces part of the folder path using original/enhanced tag mapping.

    If the "directory" mode fails to find the tag in the path, returns None.

    Args:
        original_path (Path): Full path to the input image.
        output_mode (str): Output strategy — one of "overwrite", "suffix", or "directory".
        suffix (str): Suffix to append if using "suffix" mode.
        original_tag (str): Folder name to look for in the path (e.g., "original").
        enhanced_tag (str): Folder name to substitute into the output path.

    Returns:
        Path or None: Final output path for the enhanced image, or None if resolution fails.
    """
    path_str = str(original_path)
    if output_mode == "overwrite":
        return original_path
    elif output_mode == "suffix":
        return original_path.with_name(f"{original_path.stem}{suffix}.jpg")
    elif output_mode == "directory":
        if f"/{original_tag}/" in path_str:
            out_path = Path(path_str.replace(f"/{original_tag}/", f"/{enhanced_tag}/", 1))
        elif f"\\{original_tag}\\" in path_str:
            out_path = Path(path_str.replace(f"\\{original_tag}\\", f"\\{enhanced_tag}\\", 1))
        else:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    return None


def generate_log_row(
    stats: Dict[str, float | str | tuple],
    methods: Dict[str, str | bool | None],
    original_path: Path,
    out_path: Path
) -> List[str]:
    """
    Builds a CSV-compatible log row summarizing enhancement stats for a single image.

    Includes brightness, contrast, saturation, enhancement methods applied, RGB means (if available),
    and the output file path.

    Args:
        stats (dict): Dictionary of enhancement stats including brightness/contrast before/after.
        methods (dict): Flags indicating which enhancement operations were applied.
        original_path (Path): Path to the original input image.
        out_path (Path): Final path to the enhanced output image.

    Returns:
        list[str]: Ordered list of fields for writing to the enhancement log.
    """
    return [
        original_path.name,
        round(stats["brightness_before"], 2),
        round(stats["contrast_before"], 2),
        round(stats.get("saturation_before", 0.0), 2),
        stats.get("clip_limit", "") or "",
        methods.get("white_balance") or "no",
        "yes" if methods.get("clahe") else "no",
        "yes" if methods.get("sharpen") else "no",
        round(stats["brightness_after"], 2),
        round(stats["contrast_after"], 2),
        round(stats.get("saturation_after", 0.0), 2),
        f"{stats.get('pre_rgb_means', '')}",
        f"{stats.get('post_rgb_means', '')}",
        str(out_path)
    ]


def write_log(
    log_rows: List[List[str]],
    config: dict,
    messages: Optional[list] = None
) -> None:
    """
    Writes a CSV log file summarizing enhancement results for all processed images.

    Each row includes pre- and post-enhancement statistics (brightness, contrast, saturation),
    flags for which methods were applied, and the path of the enhanced output image.

    The file is written to the location resolved by `get_log_path("enhance_log", config)`.

    Args:
        log_rows (List[List[str]]): Rows of enhancement details, typically generated by `generate_log_row()`.
        config (dict): Full configuration dictionary, used to resolve the log path.
        messages (list | None): Optional message collector for status or warnings.

    Returns:
        None
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


def update_oid_image_paths(
    oid_fc: str,
    path_map: Dict[str, str],
    messages: Optional[list] = None
) -> None:
    """
    Updates the 'ImagePath' field in a feature class to reflect enhanced image outputs.

    This function iterates through each row in the provided feature class (OID FC),
    and replaces any matching image paths in the 'ImagePath' field with their corresponding
    enhanced output paths, as provided by `path_map`.

    Args:
        oid_fc (str): Path to the Object ID feature class (typically an ArcGIS layer/table).
        path_map (dict): Dictionary mapping original image paths to enhanced output paths.
        messages (list | None): Optional logger or message collector for progress/debug output.

    Returns:
        None
    """
    with arcpy.da.UpdateCursor(oid_fc, ["ImagePath"]) as cursor:
        for row in cursor:
            original = row[0]
            if original in path_map:
                row[0] = path_map[original]
                cursor.updateRow(row)
    log_message("✅ OID ImagePath updated to reflect enhanced images.", messages)


# === Check Profile ===
def ensure_enhancement_profile_exists(
    paths: List[Path],
    config: Dict,
    messages: List[str]
) -> bool:
    """
    Ensures that an enhancement profile exists for the current batch of images.

    If auto-generation is enabled and the profile file does not exist, this function
    will derive the image base path and trigger generation based on anchor frame statistics.
    If the profile is already present, it is left unchanged.

    Args:
        paths (List[Path]): List of image paths for this batch (used to derive base folder).
        config (dict): Full resolved configuration dictionary.
        messages (list): Message logger for progress or warning output.

    Returns:
        bool: True if a usable profile exists or was successfully generated, False otherwise.
    """
    profile_path = get_enhancement_profile_path(config)
    enh_config = config.get("image_enhancement", {})
    auto_generate = enh_config.get("auto_generate_profile", False)

    sample_image_path = str(paths[0])
    original_tag = config.get("image_output", {}).get("folders", {}).get("original", "original")
    image_dir = get_image_base_path(sample_image_path, original_tag, config, messages)

    if not image_dir:
        log_message("⚠️ Could not resolve image base path. Profile gen skipped.", messages, level="warning", config=config)
        return False

    if auto_generate and not Path(profile_path).exists():
        log_message(f"📁 Generating enhancement profile at: {profile_path}", messages, config=config)
        run_profile_generation(image_dir=image_dir, config=config, messages=messages)

    if Path(profile_path).exists():
        log_message("✅ Enhancement profile is available and will be used.", messages, config=config)
        return True

    return False


# === Batch Processing and Logging ===
def process_image_batch(
    paths: List[Path],
    enh_config: Dict,
    config: Dict,
    output_mode: str,
    suffix: str,
    original_tag: str,
    enhanced_tag: str,
    messages: List[str],
    dry_run: bool,
    use_progressor: bool
) -> Tuple[List[List[str]], Dict[str, str], List[float], List[float], List[Tuple[Path, Path]]]:
    """
    Processes a batch of images using parallel enhancement and logs detailed stats.

    Each image is enhanced using `enhance_single_image()` in parallel threads. This function tracks
    enhancement stats (brightness/contrast deltas), handles EXIF copy failures, and updates the ArcPy
    progressor if enabled.

    Args:
        paths (List[Path]): List of image paths to process.
        enh_config (dict): Image enhancement configuration block.
        config (dict): Full resolved configuration dictionary.
        output_mode (str): Output path mode: "overwrite", "suffix", or "directory".
        suffix (str): Filename suffix for enhanced outputs (if using "suffix" mode).
        original_tag (str): Folder tag in original path (used in directory replacement).
        enhanced_tag (str): Replacement tag for enhanced folder.
        messages (list): Logger or message collector.
        dry_run (bool): If True, skips actual writes and metadata updates.
        use_progressor (bool): Whether to update ArcGIS progress bar.

    Returns:
        tuple:
            - log_rows (List[List[str]]): Enhancement stats for CSV logging.
            - path_map (Dict[str, str]): Original → Enhanced image path mapping.
            - brightness_deltas (List[float]): List of per-image brightness changes.
            - contrast_deltas (List[float]): List of per-image contrast changes.
            - failed_exif_copies (List[Tuple[Path, Path]]): List of images that failed EXIF copy.
    """
    log_message(f"🔄 Processing {len(paths)} images in batch.", messages, config=config)
    log_rows = []
    path_map = {}
    brightness_deltas = []
    contrast_deltas = []
    failed_exif_copies = []

    max_workers = enh_config.get("max_workers") or max(4, int((os.cpu_count() or 8) * 0.75))
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [
            executor.submit(enhance_single_image, p, enh_config, config, output_mode, suffix, original_tag,
                            enhanced_tag, messages, dry_run=dry_run)
            for p in paths
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            log_message(f"🔄 Processing image {idx}/{len(paths)}...", messages, config=config)
            result, error = future.result()
            if error:
                log_message(f"❌ Error in enhancing image {idx}: {error}", messages, config=config)
                continue

            original_path_str, out_path_str, log_row, exif_failed = result
            log_message(f"Enhanced image saved: {Path(out_path_str).name}", messages, level="debug", config=config)

            path_map[original_path_str] = out_path_str
            log_rows.append(log_row)
            if exif_failed:
                failed_exif_copies.append((Path(original_path_str), Path(out_path_str)))

            b_before = float(log_row[1])
            c_before = float(log_row[2])
            b_after = float(log_row[8])
            c_after = float(log_row[9])
            brightness_deltas.append(b_after - b_before)
            contrast_deltas.append(c_after - c_before)

            if use_progressor:
                arcpy.SetProgressorLabel(f"Enhancing {idx}/{len(paths)} ({(idx/len(paths))*100:.1f}%)")
                arcpy.SetProgressorPosition(idx)

    return log_rows, path_map, brightness_deltas, contrast_deltas, failed_exif_copies


def handle_postprocessing(
    log_rows: List[List[str]],
    path_map: Dict[str, str],
    failed_exif_copies: List[Tuple[Path, Path]],
    oid_fc_path: str,
    config: Dict,
    output_mode: str,
    dry_run: bool,
    messages: List[str]
) -> None:
    """
    Finalizes the enhancement batch by writing logs, updating feature class paths, and retrying EXIF copy failures.

    This function:
    - Writes the enhancement CSV log to disk.
    - Updates the "ImagePath" field in the ArcGIS feature class (unless in overwrite or dry-run mode).
    - Retries failed EXIF metadata copy attempts using the configured ExifTool path.

    Args:
        log_rows (List[List[str]]): Per-image enhancement stats ready for CSV logging.
        path_map (Dict[str, str]): Mapping from original paths to enhanced output paths.
        failed_exif_copies (List[Tuple[Path, Path]]): List of image pairs that failed EXIF copy on first attempt.
        oid_fc_path (str): Path to the OID feature class to update.
        config (dict): Full resolved configuration dictionary.
        output_mode (str): Output strategy: "overwrite", "suffix", or "directory".
        dry_run (bool): If True, skips feature class updates and metadata writes.
        messages (list): Logger or message collector for feedback.

    Returns:
        None
    """
    write_log(log_rows, config=config, messages=messages)

    if output_mode != "overwrite":
        if not dry_run:
            update_oid_image_paths(oid_fc_path, path_map, messages)
        else:
            log_message("🛑 Dry run — skipping ImagePath update to enhanced output.", messages)

    if failed_exif_copies:
        exiftool_path = config.get("executables", {}).get("exiftool", {}).get("exe_path", "exiftool")
        retry_successes = 0
        for orig, enh in failed_exif_copies:
            if copy_exif_metadata(orig, enh, exiftool_path):
                retry_successes += 1
            else:
                log_message(f"❌ Final EXIF copy failed: {enh.name}", messages, level="error", config=config)
        log_message(f"✅ Retried EXIF copy success count: {retry_successes}/{len(failed_exif_copies)}", messages, config=config)


# === Entry Point ===
def enhance_images_in_oid(
    oid_fc_path: str,
    config: Optional[dict] = None,
    config_file: Optional[str] = None,
    messages=None
) -> Dict[str, str]:
    """
    Main entry point for enhancing all images referenced in an ArcGIS ObjectID feature class.

    This function:
    - Resolves and validates configuration
    - Ensures disk space availability
    - Auto-generates or loads enhancement profiles (if enabled)
    - Enhances all images using seam-aware enhancements in parallel
    - Writes logs, updates feature class paths (optional), and retries failed EXIF copies

    Args:
        oid_fc_path (str): Path to the ArcGIS ObjectID feature class with an 'ImagePath' field.
        config (dict | None): Optional override configuration dictionary.
        config_file (str | None): Optional path to a YAML config file to resolve.
        messages (list | None): Optional list or logger for progress and status messages.

    Returns:
        Dict[str, str]: Mapping from original image paths to enhanced image paths.
    """
    log_message("🔄 Starting image enhancement process...", messages, config=config)
    config = resolve_config(
        config=config,
        config_file=config_file,
        oid_fc_path=oid_fc_path,
        messages=messages,
        tool_name="enhance_images"
    )
    log_message(f"Configuration loaded. Dry run: {config.get('image_enhancement', {}).get('dry_run', False)}", messages,
                config=config, level="debug")

    if not config.get("image_enhancement", {}).get("enabled", False):
        log_message("Image enhancement is disabled in config. Skipping...", messages, config=config, level="debug")
        return {}

    enh_config = config["image_enhancement"]
    dry_run = enh_config.get("dry_run", False)
    output_mode = enh_config.get("output", {}).get("mode", "directory")
    suffix = enh_config.get("output", {}).get("suffix", "_enh")
    folders = config.get("image_output", {}).get("folders", {})
    original_tag = folders.get("original", "original")
    enhanced_tag = folders.get("enhanced", "enhanced")

    check_sufficient_disk_space(oid_fc=oid_fc_path, folder_key=original_tag, config=config,
                                buffer_ratio=1.1, verbose=True, messages=messages)

    with arcpy.da.SearchCursor(oid_fc_path, ["ImagePath"]) as cursor:
        paths = [Path(row[0]) for row in cursor]

    # 🔄 Prepare or reuse enhancement profile
    if not ensure_enhancement_profile_exists(paths, config, messages):
        log_message("❌ Enhancement profile generation skipped. Returning...", messages, config=config)
        return {}

    log_message(f"✅ Enhancement profile loaded successfully.", messages, config=config)

    # 🧭 Initialize progress bar
    total = len(paths)
    use_progressor = False
    if messages:
        try:
            arcpy.SetProgressor("step", "Enhancing images...", 0, total, 1)
            use_progressor = True
        except (AttributeError, RuntimeError, arcpy.ExecuteError):
            pass

    # 🧵 Parallel enhancement
    log_message("🔄 Processing image batch...", messages, config=config)
    log_rows, path_map, brightness_deltas, contrast_deltas, failed_exif_copies = process_image_batch(
        paths, enh_config, config, output_mode, suffix, original_tag, enhanced_tag, messages, dry_run, use_progressor
    )

    if use_progressor:
        arcpy.ResetProgressor()

    # 📋 Post-run tasks
    log_message("🔄 Post-processing batch...", messages, config=config)
    handle_postprocessing(log_rows, path_map, failed_exif_copies,
                          oid_fc_path, config, output_mode, dry_run, messages)

    # 📊 Summary statistics
    if brightness_deltas:
        mean_bright_delta = np.mean(brightness_deltas)
        mean_contrast_delta = np.mean(contrast_deltas)
        log_message(f"📊 Avg Brightness Δ: {mean_bright_delta:.2f} | Avg Contrast Δ: {mean_contrast_delta:.2f}",
                    messages, config=config)

    if dry_run:
        log_message("✅ Dry run complete. No images or metadata were written.", messages, config=config)

    return path_map
