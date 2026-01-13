import numpy as np
import cv2


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Deskew a binary image using minimum-area rectangle angle estimation.

    Parameters
    ----------
    image : np.ndarray
        Binary image (H, W), dtype uint8, values {0, 255}.

    Returns
    -------
    np.ndarray
        Deskewed binary image.

    Raises
    ------
    ValueError
        If input image is invalid.
    """

    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array")

    if len(image.shape) != 2:
        raise ValueError(
            f"Deskew expects binary image with shape (H, W), got {image.shape}"
        )

    # Extract foreground pixel coordinates
    coords = np.column_stack(np.where(image > 0))

    # If no foreground detected, return original
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    deskewed = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed