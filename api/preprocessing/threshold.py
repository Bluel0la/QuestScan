import numpy as np
import cv2


def adaptive_threshold(
    image: np.ndarray, block_size: int = 11, C: int = 2
) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding to binarize handwriting.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (H, W), dtype uint8.
    block_size : int, optional
        Size of pixel neighborhood used to calculate threshold.
        Must be odd and >= 11.
    C : int, optional
        Constant subtracted from the mean.

    Returns
    -------
    np.ndarray
        Binary image (0 or 255).

    Raises
    ------
    ValueError
        If input image or parameters are invalid.
    """

    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array")

    if len(image.shape) != 2:
        raise ValueError(
            f"Adaptive threshold expects grayscale image, got shape {image.shape}"
        )

    if block_size < 11 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 11")

    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C,
    )

    return binary
