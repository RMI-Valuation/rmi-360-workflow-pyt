import cv2
import numpy as np

def compute_image_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    stats = {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "saturation": float(np.mean(hsv[:, :, 1])),
    }
    return stats