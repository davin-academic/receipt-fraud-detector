import json
import uuid
import cv2
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.services.object_detection.detection import run_detection
from app.services.object_detection.detect_regions import draw_detections
from app.services.utils.crop_date_field import crop_date_field
from app.services.manipulation_detection.predict_manipulation import predict
from app.services.utils.overlay_heatmap import overlay_heatmap_on_receipt

router = APIRouter(prefix="/api/receipts")

TMP_DIR = Path(__file__).parents[2] / "app" / "uploads" / "tmp"
SAVED_DIR = Path(__file__).parents[2] / "app" / "uploads" / "saved"


def _get_image(file_id: str) -> Path:
    matches = list(TMP_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return matches[0]


def _load_detections(file_id: str) -> list[dict]:
    path = TMP_DIR / f"{file_id}_detections.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Detections not found")
    return json.loads(path.read_text())


@router.post("")
async def upload_receipt(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    dest = TMP_DIR / f"{file_id}{Path(file.filename).suffix}"
    with dest.open("wb") as f:
        f.write(await file.read())

    detections = run_detection(str(dest))
    (TMP_DIR / f"{file_id}_detections.json").write_text(json.dumps(detections))

    return {"file_id": file_id}


@router.get("/{file_id}/detections")
async def get_detections(file_id: str):
    image_path = _get_image(file_id)
    detections = _load_detections(file_id)
    annotated = draw_detections(str(image_path), detections)
    _, buf = cv2.imencode(".jpg", annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/{file_id}/date-crop")
async def get_date_crop(file_id: str):
    image_path = _get_image(file_id)
    detections = _load_detections(file_id)
    crop = crop_date_field(str(image_path), detections)
    out_path = TMP_DIR / f"{file_id}_date_crop.jpg"
    cv2.imwrite(str(out_path), crop)
    return {"saved_to": str(out_path)}


@router.get("/{file_id}/analyze")
async def analyze_receipt(file_id: str):
    image_path = _get_image(file_id)
    detections = _load_detections(file_id)
    crop = crop_date_field(str(image_path), detections)
    crop_path = TMP_DIR / f"{file_id}_date_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    result, color, alpha_map, zoom = predict(str(crop_path))
    overlaid = overlay_heatmap_on_receipt(str(image_path), color, alpha_map, detections)
    cv2.imwrite(str(TMP_DIR / f"{file_id}_result.jpg"), overlaid)
    if zoom is not None:
        cv2.imwrite(str(TMP_DIR / f"{file_id}_zoom.jpg"), zoom)
    return result


@router.get("/{file_id}/zoom")
async def get_zoom(file_id: str):
    path = TMP_DIR / f"{file_id}_zoom.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No zoom available")
    return FileResponse(str(path))


@router.get("/{file_id}/result")
async def get_result(file_id: str):
    path = TMP_DIR / f"{file_id}_result.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(str(path))


@router.get("")
async def list_receipts():
    return [
        {"file_id": p.stem, "filename": p.name}
        for p in sorted(SAVED_DIR.iterdir())
        if p.is_file()
    ]


@router.get("/{file_id}")
async def get_receipt(file_id: str):
    matches = list(SAVED_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return FileResponse(matches[0])
