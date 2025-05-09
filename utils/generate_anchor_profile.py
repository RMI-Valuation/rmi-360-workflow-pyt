# =============================================================================
# 📊 Anchor Profile Generator (generate_anchor_profile.py)
# -----------------------------------------------------------------------------
# Purpose:     Computes enhancement_profile.json using anchor frame statistics
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
# =============================================================================

import json
import cv2
from pathlib import Path

from utils.arcpy_utils import log_message
from utils.openai_utils import ask_chatgpt
from utils.image_stats import compute_image_stats
from utils.path_utils import get_enhancement_profile_path


def select_anchor_frames(image_paths, interval=100):
    return [p for i, p in enumerate(image_paths) if i % interval == 0]


def generate_anchor_stats(image_paths):
    """
    Compute brightness, contrast, and saturation stats for a list of anchor image paths.

    Returns:
        dict[str, dict[str, float]]: A mapping of image filename -> stats dictionary
    """
    stats = {}
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            stats[p.name] = compute_image_stats(img)
        else:
            stats[p.name] = {"brightness": None, "contrast": None, "saturation": None}
    return stats


def chatgpt_anchor_strategy(anchor_stats, messages, config):
    """
    Ask ChatGPT to segment anchor images into zones based on stats and recommend enhancement parameters.

    Args:
        anchor_stats (dict): Dict of image_name -> stats dict (brightness, contrast, saturation)
        messages (list): Message log for warnings/errors.
        config (dict): Full config dict with OpenAI settings.

    Returns:
        dict | None: ChatGPT-parsed enhancement zones or None on error/fallback
    """
    ai_assist = config.get("image_enhancement", {}).get("ai_assist", False)
    if not ai_assist:
        return None
    prompt = (
            "You are a tool enhancement assistant. Based on the following brightness, contrast, and saturation "
            "statistics for a sequence of anchor images, segment them into 1 to 3 visual zones and suggest consistent "
            "enhancement parameters. The format of your reply MUST be only a JSON object with this format:\n\n"
            "{\n  'zone_1': {\n    'range': ['image001.jpg', 'image150.jpg'],\n    'gamma': 2.2,\n    'clahe_clip': 2.0,\n    'saturation_boost': 1.1\n  },\n  'zone_2': { ... }\n}\n\n"
            "Only return the JSON object — no explanation, no markdown formatting."
            " Here are the anchor stats:\n" + json.dumps(anchor_stats, indent=2)
    )
    return ask_chatgpt(prompt, config, messages)


def expand_anchor_config_to_full_profile(anchor_config, all_images):
    profile = {}
    for zone in anchor_config.values():
        start, end = zone["range"]
        apply_to = [img for img in all_images if start <= img.name <= end]
        for img in apply_to:
            profile[img.name] = {
                "gamma": zone.get("gamma", 2.2),
                "clahe_clip": zone.get("clahe_clip", 2.0),
                "saturation_boost": zone.get("saturation_boost", 1.1)
            }
    return profile


def save_enhancement_profile(profile, output_path):
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)


def fallback_profile(anchors):
    """Return a single-zone fallback profile."""
    return {
        "zone_1": {
            "range": [anchors[0].name, anchors[-1].name],
            "gamma": 2.2,
            "clahe_clip": 2.0,
            "saturation_boost": 1.1
        }
    }


def generate_profile_from_anchors(image_paths, config, messages=None):
    """
    Generates a full enhancement profile using anchor frame stats and ChatGPT (optional).

    Args:
        image_paths (list[Path]): List of image files.
        config (dict): Resolved config dict.
        messages (list): Optional message log.

    Returns:
        dict: Full enhancement_profile dict.
    """
    interval = config.get("image_enhancement", {}).get("interval", 100)
    anchors = select_anchor_frames(image_paths, interval=interval)
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
            for zkey, zconf in plan.items():
                if not all(k in zconf for k in ["range", "gamma", "clahe_clip", "saturation_boost"]):
                    log_message(f"⚠️ Zone '{zkey}' missing required keys.", messages, level="warning")
                    valid = False
            if not valid:
                log_message("⚠️ ChatGPT plan missing required structure. Falling back to default.", messages, level="warning")
                plan = fallback_profile(anchors)
            else:
                log_message(f"✅ ChatGPT provided {len(plan)} enhancement zones.", messages)
                for zone, cfg in plan.items():
                    log_message(f"  - {zone}: {cfg['range'][0]} to {cfg['range'][-1]}", messages)

    full_profile = expand_anchor_config_to_full_profile(plan, image_paths)
    return full_profile


def run_profile_generation(
    image_dir: str,
    config,
    messages
):
    """
    Loads images and config, then writes enhancement_profile.json based on anchor analysis.

    Args:
        image_dir (str): Directory containing .jpg images.
        config (dict): Resolved config dict.
        messages (list): Optional message handler.
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
