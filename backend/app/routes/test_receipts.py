import json
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.object_detection.detection import run_detection

router = APIRouter(prefix="/api/test-receipts")

TEST_DIR = Path(__file__).parents[2] / "app" / "uploads" / "test"
TMP_DIR = Path(__file__).parents[2] / "app" / "uploads" / "tmp"


@router.get("")
async def list_test_receipts():
    return [p.name for p in sorted(TEST_DIR.iterdir()) if p.is_file() and not p.name.startswith('.')]


@router.get("/{filename}")
async def get_test_receipt(filename: str):
    path = TEST_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Test image not found")
    return FileResponse(str(path))


@router.post("/{filename}/load")
async def load_test_receipt(filename: str):
    src = TEST_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="Test image not found")

    file_id = str(uuid.uuid4())
    dest = TMP_DIR / f"{file_id}{src.suffix}"
    shutil.copy2(src, dest)

    detections = run_detection(str(dest))
    (TMP_DIR / f"{file_id}_detections.json").write_text(json.dumps(detections))

    return {"file_id": file_id}
