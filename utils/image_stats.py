# =============================================================================
# 📈 Image Statistics Utilities (image_stats.py)
# -----------------------------------------------------------------------------
# Purpose:     Computes basic image statistics used in enhancement profiling.
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
#
# Description:
#   - Computes brightness (mean gray)
#   - Computes contrast (std dev in grayscale)
#   - Computes saturation (mean HSV S-channel)
#
# Exposed Functions:
#   - compute_image_stats()
#
# Dependencies:
#   - OpenCV (cv2), NumPy
# =============================================================================
from typing import Dict
import cv2
import numpy as np


def compute_image_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Computes basic image statistics used for enhancement decisions.

    - Brightness is the mean of the grayscale image.
    - Contrast is the standard deviation of the grayscale image.
    - Saturation is the mean of the S-channel in HSV color space.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        dict: Dictionary with keys:
            - "brightness" (float)
            - "contrast" (float)
            - "saturation" (float)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    stats = {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "saturation": float(np.mean(hsv[:, :, 1])),
    }
    return stats
