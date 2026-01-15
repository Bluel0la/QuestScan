import numpy as np
import cv2


def compute_quality_score(binary_image: np.ndarray) -> dict:
    """
    Compute OCR preprocessing quality score.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image (H, W), values {0, 255}

    Returns
    -------
    dict
        {
          "score": float (0.0–1.0),
          "status": "pass" | "warn" | "fail",
          "metrics": dict
        }
    """

    if binary_image is None or len(binary_image.shape) != 2:
        raise ValueError("Invalid binary image")

    h, w = binary_image.shape
    total_pixels = h * w

    # Foreground ratio
    foreground_pixels = np.count_nonzero(binary_image)
    foreground_ratio = foreground_pixels / total_pixels

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Exclude background
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_count = len(component_areas)

    avg_component_area = component_areas.mean() if component_count > 0 else 0

    # Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(binary_image, cv2.CV_64F).var()

    # Normalized sub-scores
    fg_score = 1.0 if 0.02 <= foreground_ratio <= 0.35 else 0.0
    cc_score = 1.0 if component_count >= 30 else 0.0
    area_score = 1.0 if avg_component_area >= 15 else 0.0
    blur_score = 1.0 if laplacian_var >= 50 else 0.0

    score = 0.35 * fg_score + 0.25 * cc_score + 0.20 * area_score + 0.20 * blur_score

    if score >= 0.75:
        status = "pass"
    elif score >= 0.45:
        status = "warn"
    else:
        status = "fail"

    return {
        "score": round(score, 2),
        "status": status,
        "metrics": {
            "foreground_ratio": round(foreground_ratio, 3),
            "component_count": component_count,
            "avg_component_area": round(avg_component_area, 1),
            "laplacian_variance": round(laplacian_var, 1),
        },
    }
