from api.v1.routes.scanner import scan_docs
from fastapi import APIRouter

api_version_one = APIRouter(prefix="/api/v1")

api_version_one.include_router(scan_docs)
