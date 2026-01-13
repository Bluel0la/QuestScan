import numpy as np
import cv2

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image. Expected shape:
        - (H, W, 3) for color images
        - (H, W) for already-grayscale images

    Returns
    -------
    np.ndarray
        Grayscale image with shape (H, W) and dtype uint8.

    Raises
    ------
    ValueError
        If input image is None or has unsupported shape.
    """
    
    if image is None:
        raise ValueError(
            "Input image is None"
        )
        
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Input Image must be a NumPy array"
        )
    
    # If the image is already grayscale
    if len(image.shape) == 2:
        return image
    
    # Color image (BGR expected from OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    raise ValueError(
        f"Unsupported image shape for grayscale conversion: {image.shape}"
    ) 
        
        

