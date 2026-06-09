import cv2
import numpy as np

CLASS_COLORS = {
    0: (52, 152, 219),  # date — blue
    1: (46, 204, 113),  # receipt — green
}


def draw_detections(image_path: str, detections: list[dict]) -> np.ndarray:
    """Draw OBB boxes for date and receipt regions on the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    for det in detections:
        cls = det["class_id"]
        if cls not in CLASS_COLORS:
            continue
        color = CLASS_COLORS[cls]
        corners = np.array(det["corners"], dtype=int)
        cv2.polylines(img, [corners.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
        x, y = corners[0]
        cv2.putText(img, f"{det['class_name']} {det['confidence']:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img
