from api.utils.process_documents import process_document
from api.v1.schemas.base import OCRRequest, OCRAction
import json

request = OCRRequest(action=OCRAction.TRANSCRIBE, options={"delete_after": 600})

result = process_document(
    path="/Users/blue/Code/QuestScan/tests/test_image.png", request=request
)

print("JOB ID:", result.job_id)

for page in result.pages:
    print(f"\n--- Page {page.page_number} ---")
    print(page.text)
