from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parents[3] / "ml/models/stage_01/yolo_obb_receipt_and_fields_detector/weights/best.pt"
CONF_THRESHOLD = 0.25

_model = YOLO(MODEL_PATH)


def run_detection(image_path: str) -> list[dict]:
    """Run YOLO once and return structured detections."""
    result = _model.predict(source=image_path, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)[0]

    if result.obb is None or len(result.obb) == 0:
        return []

    classes = result.obb.cls.cpu().numpy().astype(int)
    confs = result.obb.conf.cpu().numpy()
    corners_all = result.obb.xyxyxyxy.cpu().numpy()

    return [
        {
            "class_id": int(cls),
            "class_name": result.names[cls],
            "confidence": float(conf),
            "corners": corners.tolist(),
        }
        for cls, conf, corners in zip(classes, confs, corners_all)
    ]
