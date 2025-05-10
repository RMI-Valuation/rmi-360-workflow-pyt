# =============================================================================
# 📊 Anchor Profile Generator (generate_anchor_profile.py)
# -----------------------------------------------------------------------------
# Purpose:     Computes enhancement_profile.json using anchor frame statistics
#              with optional AI segmentation and interpolation of visual zones.
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
#
# Description:
#   - Selects anchor frames from a large image batch based on configurable intervals
#   - Computes brightness, contrast, and saturation stats using OpenCV
#   - Optionally uses ChatGPT to segment anchors into enhancement zones
#   - Interpolates zone configs to produce a full per-image enhancement profile
#   - Writes final profile to enhancement_profile.json
#
# Called By:   enhance_images.py (if auto_generate_profile enabled)
#
# Dependencies:
#   - Internal: arcpy_utils, openai_utils, image_stats, path_utils
#   - External: OpenCV, json, pathlib
#
# Notes:
#   - Uses fallback zones if AI segmentation fails or is disabled
#   - Assumes sorted image names correspond to capture order
# =============================================================================
import json
import cv2
from pathlib import Path
from typing import List, Optional, Dict

from utils.arcpy_utils import log_message
from utils.openai_utils import ask_chatgpt, clean_json_response
from utils.image_stats import compute_image_stats
from utils.path_utils import get_enhancement_profile_path


def select_anchor_frames(
    image_paths: List[Path],
    interval: int = 100,
    messages: Optional[list] = None
) -> List[Path]:
    """
    Selects evenly spaced anchor frames from a sequence of image paths.

    Anchor frames are sampled every `interval` images. If the remaining tail segment
    is at least 50% of the interval and not already included, the last image is added
    as a final anchor. This ensures better coverage across uneven batches.

    Args:
        image_paths (List[Path]): Ordered list of image paths (typically sorted by filename).
        interval (int): Sampling interval in number of images.
        messages (list | None): Optional logger for feedback.

    Returns:
        List[Path]: Subset of input paths selected as anchor frames.
    """
    total = len(image_paths)
    anchors = [image_paths[i] for i in range(0, total, interval)]

    # Handle remainder: if last segment > 50% of interval and not already included, add final anchor
    remainder = total % interval
    if remainder >= interval * 0.5 and image_paths[-1] not in anchors:
        anchors.append(image_paths[-1])

    if remainder and remainder < interval * 0.5:
        log_message("📎 Final images grouped into previous anchor zone.", messages)
    elif remainder:
        log_message("📎 Final anchor frame added for tail segment.", messages)

    return anchors


def generate_anchor_stats(image_paths: List[Path]) -> Dict[str, Dict[str, float | None]]:
    """
    Computes image statistics for a list of anchor image paths.

    For each image, calculates brightness, contrast, and saturation using `compute_image_stats()`.
    If an image cannot be read, fills stats with None values.

    Args:
        image_paths (List[Path]): List of image paths to analyze.

    Returns:
        Dict[str, Dict[str, float | None]]: Mapping from image filename to a dictionary of stats:
            - brightness
            - contrast
            - saturation
    """
    stats = {}
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            stats[p.name] = compute_image_stats(img)
        else:
            stats[p.name] = {"brightness": None, "contrast": None, "saturation": None}
    return stats


def chatgpt_anchor_strategy(
    anchor_stats: Dict[str, Dict[str, float | None]],
    messages: list,
    config: Dict
) -> Optional[Dict[str, Dict[str, object]]]:
    """
    Uses ChatGPT to segment anchor images into visual zones and recommend enhancement parameters.

    Constructs a structured prompt based on anchor frame statistics and sends it to ChatGPT.
    Expects a response that defines 1–3 visual zones, each with a filename range and enhancement values.
    Falls back to None if AI assistance is disabled or if the request fails.

    Args:
        anchor_stats (dict): Mapping of image filename → stats dict (brightness, contrast, saturation).
        messages (list): Message collector for logging warnings or failures.
        config (dict): Full configuration including OpenAI API settings and model options.

    Returns:
        dict | None: Dictionary of enhancement zones (zone_1, zone_2, ...) or None if disabled or failed.
    """
    ai_assist = config.get("image_enhancement", {}).get("ai_assist", False)
    if not ai_assist:
        return None

    prompt = (
        "You are a photo enhancement assistant helping to create consistent visual quality for long sequences of panoramic images.\n"
        "Based on brightness, contrast, and saturation statistics from a list of anchor images, group them into 1 to 3 visual zones.\n\n"

        "Important rules:\n"
        "- This tool will eventually process full-day image sets of 18,000+ images captured under changing daylight (sun position, cloud cover, etc).\n"
        "- Recommend new zones only when there are **sustained and meaningful changes** in lighting/contrast.\n"
        "- Avoid overreacting to short-term fluctuations. Aim for **smooth, gradual transitions** in parameter values between zones.\n"
        "- Use consistent enhancement values **within** each zone.\n"
        "- This current example uses a small sample (e.g., 109 images) for testing — do not overfit to small shifts.\n\n"

        "Return ONLY a valid JSON object — no markdown formatting, no explanation. The format must be:\n\n"
        '{\n'
        '  "zone_1": {\n'
        '    "range": ["image001.jpg", "image150.jpg"],\n'
        '    "gamma": 2.2,\n'
        '    "clahe_clip": 2.0,\n'
        '    "saturation_boost": 1.1\n'
        '  },\n'
        '  "zone_2": { ... }\n'
        '}\n\n'

        "Here are the anchor stats:\n" + json.dumps(anchor_stats, indent=2)
    )
    # Request the ChatGPT response
    response_text = ask_chatgpt(prompt, config, messages)

    # Check response type for debugging
    log_message(f"ChatGPT response (type): {type(response_text)}", messages, level="debug")

    # Ensure the response is returned as a string (not a dict)
    if isinstance(response_text, dict):
        response_text = json.dumps(response_text)

    # Now clean the JSON response
    try:
        cleaned_response = clean_json_response(response_text)
        return cleaned_response
    except ValueError as e:
        log_message(f"Error cleaning ChatGPT response: {e}", messages, level="error", error_type=ValueError,
                    config=config)
        return None


def expand_anchor_config_to_full_profile(
    anchor_config: Dict[str, Dict[str, object]],
    all_images: List[Path],
    messages: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Converts anchor-based enhancement zones into a full per-image enhancement profile.

    For each enhancement zone (defined by a start/end filename range), this function assigns
    enhancement parameters (gamma, clahe_clip, saturation_boost) to all images falling within that range.
    Any images not covered by a zone are assigned default parameters.

    Args:
        anchor_config (dict): Dictionary of enhancement zones. Each key (e.g., "zone_1") maps to a config
                              with a 'range' (start, end) and enhancement values.
        all_images (List[Path]): All image paths in the dataset to be covered by the profile.
        messages (list | None): Optional message collector for fallback warnings.

    Returns:
        Dict[str, Dict[str, float]]: Mapping from image filename → enhancement config dictionary.
    """
    profile = {}
    for zone in anchor_config.values():
        start, end = zone["range"]
        for img in all_images:
            if start <= img.name <= end:
                profile[img.name] = {
                    "gamma": zone.get("gamma", 2.2),
                    "clahe_clip": zone.get("clahe_clip", 2.0),
                    "saturation_boost": zone.get("saturation_boost", 1.1)
                }

    # Fallback for uncovered images
    uncovered = [img.name for img in all_images if img.name not in profile]
    if uncovered:
        for name in uncovered:
            profile[name] = {
                "gamma": 2.2,
                "clahe_clip": 2.0,
                "saturation_boost": 1.1
            }
        log_message(
            f"⚠️ {len(uncovered)} image(s) were not covered by any enhancement zone. Default profile applied.",
            messages, level="warning"
        )

    return profile


def save_enhancement_profile(profile: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Writes the full enhancement profile to disk as formatted JSON."""
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)


def fallback_profile(anchors: List[Path]) -> Dict[str, Dict[str, object]]:
    """Creates a default single-zone enhancement profile covering all anchor frames."""
    return {
        "zone_1": {
            "range": [anchors[0].name, anchors[-1].name],
            "gamma": 2.2,
            "clahe_clip": 2.0,
            "saturation_boost": 1.1
        }
    }


def generate_profile_from_anchors(
        image_paths: List[Path],
        config: Dict,
        messages: Optional[List[str]] = None
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Generates a full per-image enhancement profile using anchor frame statistics and optional AI segmentation.

    This function:
    - Selects anchor frames from the input image list
    - Computes brightness, contrast, and saturation for each anchor
    - Optionally queries ChatGPT to divide anchors into enhancement zones
    - Falls back to a static default zone plan if AI is disabled or fails
    - Expands zone-based config into a full profile covering all images

    Args:
        image_paths (List[Path]): List of .jpg image paths to analyze and profile.
        config (dict): Full resolved config dict with enhancement and AI settings.
        messages (list | None): Optional message log for progress and warnings.

    Returns:
        dict | None: Full enhancement profile mapping image name → parameter dict, or None if setup fails.
    """
    interval = config.get("image_enhancement", {}).get("interval", 100)
    anchors = select_anchor_frames(image_paths, interval=interval, messages=messages)
    log_message(f"📌 Selected {len(anchors)} anchor frames from {len(image_paths)} images", messages)

    if not anchors:
        log_message("❌ No anchor frames selected. Check interval or image availability.", messages, level="warning")
        return None

    anchor_stats = generate_anchor_stats(anchors)
    if not anchor_stats:
        log_message("❌ Failed to compute anchor frame stats.", messages, level="warning")
        return None

    ai_assist = config.get("image_enhancement", {}).get("ai_assist", False)

    if not ai_assist:
        log_message("⚠️ AI assistance disabled — using fallback static profile", messages)
        plan = fallback_profile(anchors)
    else:
        log_message("🤖 Requesting anchor zone strategy from ChatGPT...", messages)
        plan = chatgpt_anchor_strategy(anchor_stats, messages=messages, config=config)

        if not plan:
            log_message("⚠️ ChatGPT failed. Falling back to default enhancement.", messages)
            plan = fallback_profile(anchors)
        else:
            # Validate structure
            valid = True
            for zone, zconf in plan.items():
                log_message(f"Zone '{zone}' config type: {type(zconf)}", messages, level="debug")
                log_message(f"Zone '{zone}' config value: {zconf}", messages, level="debug")

                # Ensure cfg is a dictionary and contains required keys
                if not isinstance(zconf, dict):
                    log_message(f"⚠️ Zone '{zone}' is not a valid config dictionary: {zconf}", messages,
                                level="warning")
                    valid = False
                    continue

                if not all(k in zconf for k in ["range", "gamma", "clahe_clip", "saturation_boost"]):
                    log_message(f"⚠️ Zone '{zone}' missing required keys.", messages, level="warning")
                    valid = False

                # Validate the 'range' key (only if zconf is a dictionary)
                range_value = zconf.get("range")
                if not isinstance(range_value, list) or len(range_value) != 2:
                    log_message(f"⚠️ Invalid range for zone '{zone}': {range_value}", messages, level="warning")
                    valid = False
                else:
                    log_message(f"  - {zone}: {range_value[0]} to {range_value[1]}", messages)

            if not valid:
                log_message("⚠️ ChatGPT plan missing required structure. Falling back to default.", messages,
                            level="warning")
                plan = fallback_profile(anchors)
            else:
                log_message(f"✅ ChatGPT provided {len(plan)} enhancement zones.", messages)

    full_profile = expand_anchor_config_to_full_profile(plan, image_paths)
    return full_profile


def run_profile_generation(
    image_dir: str,
    config: Dict,
    messages: List[str]
) -> None:
    """
    Orchestrates generation of enhancement_profile.json from images in a given directory.

    This function:
    - Collects all .jpg images under the given folder
    - Builds an enhancement profile using anchor frame logic and optional AI assistance
    - Writes the resulting profile to disk in JSON format

    Args:
        image_dir (str): Root directory containing .jpg images.
        config (dict): Full resolved configuration dictionary.
        messages (list): Message collector for logging progress and warnings.

    Returns:
        None
    """
    image_paths = sorted(Path(image_dir).rglob("*.jpg"))
    if not image_paths:
        log_message("❌ No .jpg images found in input folder.", messages)
        return

    profile = generate_profile_from_anchors(image_paths, config, messages)
    if not profile:
        log_message("❌ Enhancement profile generation failed.", messages)
        return

    profile_path = get_enhancement_profile_path(config)
    save_enhancement_profile(profile, profile_path)
    log_message(f"✅ Enhancement profile saved: {profile_path}", messages)
