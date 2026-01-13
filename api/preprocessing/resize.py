import numpy as np
import cv2


def resize_to_ocr(image: np.ndarray, target_width: int = 2480) -> np.ndarray:
    """
    Resize image to OCR-friendly resolution (300 DPI equivalent).

    Parameters
    ----------
    image : np.ndarray
        Binary or grayscale image (H, W), dtype uint8.
    target_width : int, optional
        Target width in pixels. Default corresponds to A4 at ~300 DPI.

    Returns
    -------
    np.ndarray
        Resized image preserving aspect ratio.

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
        raise ValueError(f"Resize expects image with shape (H, W), got {image.shape}")

    h, w = image.shape

    if w == target_width:
        return image

    scale = target_width / w
    new_width = target_width
    new_height = int(h * scale)

    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA

    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized
