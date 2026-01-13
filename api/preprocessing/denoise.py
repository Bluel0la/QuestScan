import numpy as np
import cv2

def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply median blur to remove noise while preserving handwriting strokes.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (H, W), dtype uint8.
    ksize : int, optional
        Kernel size for median blur. Must be odd and >= 3.
        Recommended values: 3 or 5.

    Returns
    -------
    np.ndarray
        Denoised grayscale image.

    Raises
    ------
    ValueError
        If input image is invalid or parameters are unsupported.
    """
    if image is None:
        raise ValueError("Input image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array")
    
    if len(image.shape) != 2:
        raise ValueError(f"Median denoise expects a grayscale image, got shape: {image.shape}")
    
    if not (isinstance(ksize, int) and ksize >= 3 and ksize % 2 == 1):
        raise ValueError("Kernel size must be an odd integer >= 3")
    
    return cv2.medianBlur(image, ksize)