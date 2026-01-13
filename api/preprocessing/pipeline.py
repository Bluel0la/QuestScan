from preprocessing.enhance_contrast import enhance_contrast
from preprocessing.threshold import adaptive_threshold
from preprocessing.denoise import median_denoise
from preprocessing.resize import resize_to_ocr
from preprocessing.grayscale import grayscale
from preprocessing.deskew import deskew
import numpy as np


def preprocess_for_ocr(
    image: np.ndarray, *,
    denoise_ksize: int = 3, clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8), threshold_block_size: int = 11,
    threshold_C: int = 2, target_width: int = 2480 ) -> np.ndarray:
    """
    Full preprocessing pipeline for handwriting OCR.

    Parameters
    ----------
    image : np.ndarray
        Raw input image.
    denoise_ksize : int
        Median blur kernel size (default: 3).
    clahe_clip_limit : float
        CLAHE contrast limit (default: 2.0).
    clahe_tile_grid_size : tuple
        CLAHE tile grid size (default: (8, 8)).
    threshold_block_size : int
        Adaptive threshold block size (default: 11).
    threshold_C : int
        Adaptive threshold constant (default: 2).
    target_width : int
        Target width for OCR normalization (default: 2480).

    Returns
    -------
    np.ndarray
        OCR-ready binary image.
    """

    # 1. Grayscale
    gray = grayscale(image)

    # 2. Noise reduction
    denoised = median_denoise(gray, ksize=denoise_ksize)

    # 3. Contrast enhancement
    contrasted = enhance_contrast(
        denoised, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size
    )

    # 4. Adaptive thresholding
    binary = adaptive_threshold(
        contrasted, block_size=threshold_block_size, C=threshold_C
    )

    # 5. Deskew
    deskewed = deskew(binary)

    # 6. Resize to OCR-friendly resolution
    resized = resize_to_ocr(deskewed, target_width=target_width)

    return resized
