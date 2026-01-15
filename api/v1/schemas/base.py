from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uuid, requests, os
from enum import Enum
load_dotenv(".env")

HANDWRITING_OCR_API_URL = "https://www.handwritingocr.com/api/v3/documents"


class OCRAction(Enum):
    TRANSCRIBE = "transcribe"
    TABLES = "tables"
    EXTRACT = "extractor"

class OCRCapabilities:
    def __init__(
        self,
        supports_handwriting: bool,
        supports_tables: bool,
        supports_extractors: bool,
        supports_webhooks: bool,
        supports_async: bool,
    ):
        self.supports_handwriting = supports_handwriting
        self.supports_tables = supports_tables
        self.supports_extractors = supports_extractors
        self.supports_webhooks = supports_webhooks
        self.supports_async = supports_async


class OCRRequest:
    def __init__(
        self,
        action: OCRAction,
        #language: Optional[str] = None,
        extractor_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.action = action
        #self.language = language
        self.extractor_id = extractor_id
        self.webhook_url = webhook_url
        self.options = options or {}


class OCRJob:
    def __init__(self, job_id: str, provider: str, provider_job_id: str):
        self.job_id = job_id  # internal UUID
        self.provider = provider  # e.g. "handwritingocr"
        self.provider_job_id = provider_job_id


class OCRStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class OCRPageResult:
    def __init__(
        self,
        page_number: int,
        text: Optional[str] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        fields: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ):
        self.page_number = page_number
        self.text = text
        self.tables = tables or []
        self.fields = fields or {}
        self.confidence = confidence


class OCRResult:
    def __init__(
        self,
        job_id: str,
        pages: List[OCRPageResult],
        raw_provider_response: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.pages = pages
        self.raw_provider_response = raw_provider_response


class OCRProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> OCRCapabilities:
        pass

    @abstractmethod
    def submit(self, document_path: str, request: OCRRequest) -> OCRJob:
        pass

    @abstractmethod
    def get_status(self, job: OCRJob) -> OCRStatus:
        pass

    @abstractmethod
    def fetch_result(self, job: OCRJob) -> OCRResult:
        pass

class OCRRejected(Exception):
    pass


class HandwritingOCRProvider(OCRProvider):
    """
    HandwritingOCR v3 provider implementation.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("HANDWRITING_OCR_API_KEY")
        if not self.api_key:
            raise RuntimeError("HANDWRITING_OCR_API_KEY not configured")

        # Minimal in-memory job tracking (temporary)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Provider metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "handwritingocr"

    @property
    def capabilities(self) -> OCRCapabilities:
        return OCRCapabilities(
            supports_handwriting=True,
            supports_tables=True,
            supports_extractors=True,
            supports_webhooks=True,
            supports_async=True,
        )

    # ------------------------------------------------------------------
    # Core API calls
    # ------------------------------------------------------------------

    def submit(self, document_path: str, request: OCRRequest) -> OCRJob:
        """
        Upload a document and queue it for OCR processing.
        """

        if not os.path.exists(document_path):
            raise FileNotFoundError(document_path)

        # Validate request
        if request.action.value == "extractor" and not request.extractor_id:
            raise ValueError("extractor_id is required when action='extractor'")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        data = {
            "action": request.action.value,
        }

        if request.extractor_id:
            data["extractor_id"] = request.extractor_id

        if request.webhook_url:
            data["webhook_url"] = request.webhook_url

        if request.options:
            # Allow future extension (delete_after, etc.)
            data.update(request.options)

        with open(document_path, "rb") as f:
            files = {"file": f}

            response = requests.post(
                HANDWRITING_OCR_API_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=60,
            )

        if response.status_code != 201:
            raise RuntimeError(
                f"HandwritingOCR submit failed "
                f"(status={response.status_code}): {response.text}"
            )

        payload = response.json()

        provider_job_id = payload.get("id")
        if not provider_job_id:
            raise RuntimeError("Invalid response: missing document id")

        # Track job locally (temporary)
        #self._jobs[provider_job_id] = {
        #    "status": OCRStatus.QUEUED,
        #    "raw": payload,
        #}

        return OCRJob(
            job_id=str(uuid.uuid4()),
            provider=self.name,
            provider_job_id=provider_job_id,
        )

    # ------------------------------------------------------------------
    # Placeholder lifecycle methods (next step)
    # ------------------------------------------------------------------

    def get_status(self, job: OCRJob) -> OCRStatus:
        """
        Retrieve processing status for a document.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        response = requests.get(
            f"{HANDWRITING_OCR_API_URL}/{job.provider_job_id}",
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            payload = response.json()
            status = payload.get("status")
            if status == "processed":
                return OCRStatus.PROCESSED
            return OCRStatus.PROCESSING


        raise RuntimeError(
            f"Unexpected status check response "
            f"(status={response.status_code}): {response.text}"
        )

    def fetch_result(self, job: OCRJob) -> OCRResult:
        """
        Fetch finalized OCR result as normalized OCRResult.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        response = requests.get(
            f"{HANDWRITING_OCR_API_URL}/{job.provider_job_id}",
            headers=headers,
            timeout=60,
        )

        if response.status_code == 202:
            raise RuntimeError("OCR job is still processing")

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch OCR result "
                f"(status={response.status_code}): {response.text}"
            )

        payload = response.json()

        # --- Normalize pages ---
        pages = []

        # Provider may return different shapes depending on action.
        # Prefer 'results' (as in HandwritingOCR examples), then 'pages'.
        pages_payload = payload.get("results") or payload.get("pages") or []

        # If no pages/results array, but the payload includes page-like keys at top-level,
        # treat the entire payload as a single-page response to avoid failing.
        if not pages_payload and any(k in payload for k in ("text", "transcript", "tables", "key_value_pairs", "extractions", "fields")):
            pages_payload = [payload]

        for idx, page in enumerate(pages_payload, start=1):
            # page may already include an explicit page_number
            page_number = page.get("page_number") if isinstance(page, dict) and page.get("page_number") else idx

            # text can be under 'transcript' or 'text'
            text = None
            if isinstance(page, dict):
                text = page.get("transcript") or page.get("text")

            tables = page.get("tables", []) if isinstance(page, dict) else []

            # Normalize fields from several possible places
            fields: Dict[str, Any] = {}
            # direct 'fields' object
            if isinstance(page, dict) and page.get("fields"):
                fields.update(page.get("fields") or {})

            # key_value_pairs -> map to fields
            for kv in (page.get("key_value_pairs") or []) if isinstance(page, dict) else []:
                key = kv.get("key")
                if key:
                    fields[key] = kv.get("value")

            # extractions -> nested arrays of extraction objects
            for extraction_group in (page.get("extractions") or []) if isinstance(page, dict) else []:
                for ext in extraction_group:
                    key = ext.get("key")
                    if key:
                        fields[key] = ext.get("value")

            pages.append(
                OCRPageResult(
                    page_number=page_number,
                    text=text,
                    tables=tables,
                    fields=fields,
                    confidence=(page.get("confidence") if isinstance(page, dict) else None),
                )
            )

        return OCRResult(
            job_id=job.job_id,
            pages=pages,
            raw_provider_response=payload,
        )
