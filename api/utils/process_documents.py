from api.preprocessing.preprocessing_profiles import PREPROCESSING_PROFILES
from api.quality.quality_score import compute_quality_score
from api.utils.pipeline import preprocess_for_ocr
from api.pdf.extract_pages import pdf_to_images
import time, shutil, os, uuid, cv2
from typing import List, Tuple
from api.v1.schemas.base import (
    HandwritingOCRProvider,
    OCRRequest,
    OCRResult,
    OCRStatus,
    OCRRejected,
)
import numpy as np


POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 60  # ~2 minutes


def _load_images(path: str) -> List[np.ndarray]:
    """
    Load document into page-level images.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return pdf_to_images(path)

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unsupported or unreadable file: {path}")

    return [image]


def preprocess_with_retry(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Try multiple preprocessing strategies until quality passes.
    """

    last_quality = None

    for params in PREPROCESSING_PROFILES:
        processed = preprocess_for_ocr(image, **params)
        quality = compute_quality_score(processed)

        if quality["status"] == "pass":
            return processed, quality

        last_quality = quality

    return processed, last_quality


def process_document(path: str, request: OCRRequest) -> OCRResult:
    """
    Main OCR orchestration entry point.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # 1. Load document → pages
    pages = _load_images(path)
    if not pages:
        raise OCRRejected("Document contains no readable pages")

    # 2. Preprocess + quality gate
    processed_pages = []

    for idx, page in enumerate(pages, start=1):
        preprocessed, quality = preprocess_with_retry(page)

        if quality["status"] == "fail":
            raise OCRRejected(
                f"Page {idx} rejected after preprocessing "
                f"(metrics={quality['metrics']})"
            )

        processed_pages.append(preprocessed)

    # 3. Provider selection
    provider = HandwritingOCRProvider()

    if request.action.name == "TABLES" and not provider.capabilities.supports_tables:
        raise RuntimeError("Selected OCR provider does not support table extraction")

    # 4. Persist OCR-ready images (future provider support)
    temp_dir = f"/tmp/ocr_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for i, img in enumerate(processed_pages, start=1):
            cv2.imwrite(os.path.join(temp_dir, f"page_{i}.png"), img)

        # 5. Submit OCR job (original document for now)
        job = provider.submit(path, request)

        # 6. Poll status
        for _ in range(MAX_POLL_ATTEMPTS):
            status = provider.get_status(job)

            if status == OCRStatus.PROCESSED:
                break

            if status == OCRStatus.FAILED:
                raise RuntimeError("OCR job failed during processing")

            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            raise TimeoutError("OCR job timed out")

        # 7. Fetch normalized result
        return provider.fetch_result(job)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
