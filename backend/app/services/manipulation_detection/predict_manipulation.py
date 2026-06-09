import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

MODEL_PATH = Path(__file__).parents[3] / "ml/models/stage_02/unetplusplus_copy_move_manipulation_detector/weights/best.pt"

INPUT_HEIGHT = 64
INPUT_WIDTH = 512

ZOOM_RATIO = 0.4
ZOOM_DISPLAY_HEIGHT = 200

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pixel heatmap threshold
DECISION_THRESHOLD = 0.90

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_laplacian_channel(rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    lap = np.clip(lap, 0, 255).astype(np.float32) / 255.0
    return lap[..., None]


def _to_4ch_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    Input:
        RGB uint8 image, already resized to INPUT_HEIGHT x INPUT_WIDTH

    Output:
        Tensor shape [1, 4, H, W]
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD

    lap = _compute_laplacian_channel(rgb_uint8)

    stacked = np.concatenate([rgb, lap], axis=-1)  # H, W, 4
    stacked = np.ascontiguousarray(stacked.transpose(2, 0, 1))  # 4, H, W

    return torch.from_numpy(stacked).float().unsqueeze(0)


def _load_model():
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=_device)

    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights=None,
        in_channels=4,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
        decoder_interpolation="bilinear",
    ).to(_device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


_model = _load_model()


def _image_verdict(
    prob_map: np.ndarray,
    prob_threshold: float = 0.90,
    min_component_area: int = 80,
    min_component_mean_prob: float = 0.75,
):
    binary = (prob_map > prob_threshold).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    valid_blobs = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        comp_mask = labels == i
        mean_prob = float(prob_map[comp_mask].mean())
        max_prob = float(prob_map[comp_mask].max())

        if mean_prob < min_component_mean_prob:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        valid_blobs.append({
            "area": int(area),
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "bbox": (int(x), int(y), int(bw), int(bh)),
        })

    return len(valid_blobs) > 0, valid_blobs


def _make_zoom_crop(img_bgr: np.ndarray, color: np.ndarray, probs: np.ndarray) -> np.ndarray:
    h, w = probs.shape
    _, peak_x = np.unravel_index(probs.argmax(), probs.shape)

    half = int(w * ZOOM_RATIO / 2)
    x1 = max(0, peak_x - half)
    x2 = min(w, x1 + int(w * ZOOM_RATIO))

    alpha = (probs * 0.65)[:, :, None]
    blended = (img_bgr * (1 - alpha) + color * alpha).astype(np.uint8)
    cropped = blended[:, x1:x2]

    scale = ZOOM_DISPLAY_HEIGHT / h
    return cv2.resize(
        cropped,
        (int(cropped.shape[1] * scale), ZOOM_DISPLAY_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def predict(image_path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run 4-channel manipulation detection on a date crop.

    Returns:
        result_dict, color_map, alpha_map, zoom_crop
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(
        rgb,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    x = _to_4ch_tensor(rgb_resized).to(_device)

    with torch.no_grad():
        probs = torch.sigmoid(_model(x))[0, 0].cpu().numpy()

    probs_resized = cv2.resize(
        probs,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    pixel_mask = probs > DECISION_THRESHOLD
    pct_manipulated = float(pixel_mask.mean() * 100)
    max_prob = float(probs.max())

    is_manipulated, valid_blobs = _image_verdict(
        probs,
        prob_threshold=DECISION_THRESHOLD,
        min_component_area=80,
        min_component_mean_prob=0.75,
    )
    '''
    If you want bounding box in zoomed area
    
    __, valid_blobs = _image_verdict(
        probs_resized,
        prob_threshold=DECISION_THRESHOLD,
        min_component_area=80,
        min_component_mean_prob=0.75,
    )
    '''
    if is_manipulated:
        verdict = "MANIPULATED"
    elif max_prob > 0.85:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    probs_smooth = cv2.GaussianBlur(
        probs_resized,
        (0, 0),
        sigmaX=max(w, h) * 0.04,
    )

    if probs_smooth.max() > 0:
        probs_smooth = np.clip(probs_smooth / probs_smooth.max(), 0, 1)

    color = cv2.applyColorMap(
        (probs_smooth * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )

    # Optional: draw valid blob boxes into color map
    for blob in valid_blobs:
        x1, y1, bw, bh = blob["bbox"]
        cv2.rectangle(
            color,
            (x1, y1),
            (x1 + bw, y1 + bh),
            (0, 255, 255),
            1,
        )

    zoom = _make_zoom_crop(img_bgr, color, probs_smooth) if verdict != "CLEAN" else None

    return {
        "verdict": verdict,
        "max_prob": max_prob,
        "pct_manipulated": pct_manipulated,
        "valid_blobs": valid_blobs,
    }, color, probs_smooth, zoom