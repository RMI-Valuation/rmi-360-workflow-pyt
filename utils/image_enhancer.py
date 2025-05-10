# =============================================================================
# 🔧 Image Enhancement Core Functions (image_enhancer.py)
# -----------------------------------------------------------------------------
# Purpose:     Applies individual image enhancement operations for the RMI 360 pipeline,
#              including white balance, CLAHE contrast, saturation boost, sharpening,
#              and brightness recovery.
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
#
# Description:
#   Contains reusable, testable enhancement functions used by the image batch processor.
#   Applies enhancements based on a unified config dictionary.
#
# Exposed Functions:
#   - apply_white_balance
#   - apply_clahe
#   - apply_saturation_boost
#   - apply_sharpening
#   - apply_enhancements (main entry point)
#
# Dependencies:
#   - OpenCV (cv2), NumPy, arcpy_utils
# =============================================================================
import cv2
import numpy as np
from typing import Tuple, Optional

from utils.arcpy_utils import log_message


def apply_white_balance(
    img: np.ndarray,
    method: str = "gray_world"
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Applies white balance correction to an image using the specified method.

    Args:
        img: Input image as a NumPy array in BGR format.
        method: White balance method to use ("gray_world" or "simple").

    Returns:
        A tuple containing the white-balanced image, the mean values of the B, G, R channels before correction, and
        the mean values after correction.
    """
    b_mean, g_mean, r_mean = (float(np.mean(img[:, :, c])) for c in range(3))
    pre_means = (b_mean, g_mean, r_mean)

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

    b_post, g_post, r_post = (float(np.mean(img[:, :, c])) for c in range(3))
    post_means = (b_post, g_post, r_post)

    return img, pre_means, post_means


def apply_clahe(
    img: np.ndarray,
    clip_limit: float,
    tile_grid_size: Tuple[int, int]
) -> np.ndarray:
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


def apply_saturation_boost(
    img: np.ndarray,
    factor: float
) -> np.ndarray:
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


def apply_sharpening(
    img: np.ndarray,
    kernel: list[list[float]]
) -> np.ndarray:
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


def apply_enhancements(
    img: np.ndarray,
    enhance_config: dict,
    contrast: float,
    full_config: Optional[dict] = None,
    messages: Optional[list] = None
) -> Tuple[np.ndarray, Optional[float], dict, dict]:
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
    methods_applied: dict[str, bool | str | None] = {"white_balance": None, "clahe": False, "sharpen": False}

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
        tile_grid = clahe_cfg.get("tile_grid_size", [8, 8])
        grid_size: Tuple[int, int] = (tile_grid[0], tile_grid[1])
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