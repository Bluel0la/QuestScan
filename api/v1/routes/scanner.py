from api.utils.process_documents import process_document
from fastapi import APIRouter, File, UploadFile, status
from sqlalchemy import Session, Depends


scan_docs = APIRouter(tags=["scanner"], prefix="scanner")