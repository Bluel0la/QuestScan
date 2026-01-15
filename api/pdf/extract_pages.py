from typing import List
import cv2, os, fitz
import numpy as np


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF pages to OpenCV-compatible images using PyMuPDF.

    Parameters
    ----------
    pdf_path : str
        Path to PDF file.
    dpi : int
        Target DPI for rasterization (default: 300).

    Returns
    -------
    list[np.ndarray]
        List of images in BGR format, one per page.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    images: List[np.ndarray] = []

    # PyMuPDF uses 72 DPI as base
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert to NumPy
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        # Convert RGB → BGR for OpenCV
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        images.append(img)

    doc.close()
    return images
