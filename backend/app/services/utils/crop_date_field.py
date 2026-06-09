import cv2
import numpy as np

DATE_CLASS_ID = 0
PADDING = 0.10
MIN_OUTPUT_HEIGHT = 64


def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def _expand_quad(corners, padding_ratio):
    return corners + (corners - corners.mean(axis=0)) * padding_ratio


def _warp_crop(img, corners):
    src = _order_corners(corners)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if h < MIN_OUTPUT_HEIGHT:
        w = int(w * MIN_OUTPUT_HEIGHT / h)
        h = MIN_OUTPUT_HEIGHT
    w, h = max(w, 50), max(h, 20)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_CUBIC)


def crop_date_field(image_path: str, detections: list[dict]) -> np.ndarray:
    """Crop the date region using pre-computed detections."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    date_dets = [d for d in detections if d["class_id"] == DATE_CLASS_ID]
    if not date_dets:
        raise RuntimeError("No date field detected.")

    best = max(date_dets, key=lambda d: d["confidence"])
    return _warp_crop(img, _expand_quad(np.array(best["corners"], dtype=np.float32), PADDING))
