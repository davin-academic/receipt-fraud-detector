import cv2
import numpy as np

from app.services.utils.crop_date_field import DATE_CLASS_ID, PADDING, _order_corners, _expand_quad

HEATMAP_OPACITY = 0.65
_BORDER_FADE = 0.12


def _fade_borders(alpha_map: np.ndarray) -> np.ndarray:
    """Fade alpha to zero at the edges so the warp boundary is invisible."""
    h, w = alpha_map.shape
    fade = np.ones_like(alpha_map)
    bh, bw = int(h * _BORDER_FADE), int(w * _BORDER_FADE)
    if bh > 0:
        ramp = np.linspace(0, 1, bh)
        fade[:bh, :] *= ramp[:, None]
        fade[-bh:, :] *= ramp[::-1, None]
    if bw > 0:
        ramp = np.linspace(0, 1, bw)
        fade[:, :bw] *= ramp[None, :]
        fade[:, -bw:] *= ramp[::-1][None, :]
    return alpha_map * fade


def overlay_heatmap_on_receipt(
    receipt_path: str,
    color: np.ndarray,
    alpha_map: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """Warp the heatmap color and alpha back onto the original receipt and blend softly."""
    receipt = cv2.imread(receipt_path)
    if receipt is None:
        raise ValueError(f"Could not read image: {receipt_path}")

    date_dets = [d for d in detections if d["class_id"] == DATE_CLASS_ID]
    if not date_dets:
        raise RuntimeError("No date field detected.")

    best = max(date_dets, key=lambda d: d["confidence"])
    corners = _order_corners(_expand_quad(np.array(best["corners"], dtype=np.float32), PADDING))

    h_map, w_map = color.shape[:2]
    h_rec, w_rec = receipt.shape[:2]

    src_rect = np.array([[0, 0], [w_map - 1, 0], [w_map - 1, h_map - 1], [0, h_map - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_rect, corners)

    warped_color = cv2.warpPerspective(color, M, (w_rec, h_rec))
    warped_alpha = cv2.warpPerspective(_fade_borders(alpha_map), M, (w_rec, h_rec))

    alpha = (warped_alpha * HEATMAP_OPACITY)[:, :, None]
    return (receipt * (1 - alpha) + warped_color * alpha).astype(np.uint8)
