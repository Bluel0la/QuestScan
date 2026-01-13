import numpy as np
import cv2

def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (H, W), dtype uint8.
    clip_limit : float, optional
        Threshold for contrast limiting.
        Recommended range: 1.5–3.0.
    tile_grid_size : tuple, optional
        Size of grid for histogram equalization.

    Returns
    -------
    np.ndarray
        Contrast-enhanced grayscale image.

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
        raise ValueError(f"CLAHE expects a grayscale image, got shape: {image.shape}")
    
    if clip_limit <= 0:
        raise ValueError("clip_limit must be > 0")
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    return clahe.apply(image)