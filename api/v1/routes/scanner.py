from fastapi import APIRouter, File, UploadFile, status, Depends, HTTPException
from api.utils.process_documents import process_document
from api.v1.schemas.base import OCRRequest, OCRAction
from sqlalchemy.orm import Session
import tempfile, shutil, uuid
from pathlib import Path


scan_docs = APIRouter(tags=["scanner"], prefix="/scanner")

# Endpoint to take either scanned images or documents for procesing
@scan_docs.post("/process", status_code=status.HTTP_200_OK)
def scan_document(
    file: UploadFile = File(...)
):
    # Check to maeke sure something is actually uploaded
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file uploaded."
        )

    # Check if the filetype is supported
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix}",
        )

    # Temporarily store the file and work on it
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)/f"{uuid.uuid4()}{suffix}"

        try:
            with tmp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            request = OCRRequest(
                action=OCRAction.TRANSCRIBE
            )

            result = process_document(
                path = str(tmp_path),
                request=request
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {e}"
            )

        return {
            "job_id": result.job_id,
            "pages": [
                {
                    "page": i + 1,
                    "text": text,
                }
                for i, text in enumerate(result.pages)
            ]
        }
