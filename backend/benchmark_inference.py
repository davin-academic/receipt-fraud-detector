"""Benchmark inference latency and model size for the receipt-fraud pipeline.

Fills one half of this table per run (the half matching the machine it runs on):

    Environment   Component    Input               Avg. time   Model size
    Desktop GPU   YOLO-OBB     Full receipt image  X ms        X MB
    Desktop GPU   U-Net++      Cropped date field  X ms        X MB
    Desktop GPU   Full pipeline One receipt image  X ms        X MB
    MacBook CPU   YOLO-OBB     ...

Usage:
    python -m app.benchmark_inference                # auto-detect device
    python -m app.benchmark_inference --cpu          # force CPU
    python -m app.benchmark_inference --runs 5       # repeats per image
    python -m app.benchmark_inference --images DIR   # custom image folder
"""

import argparse
import os
import statistics
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent  # .../backend/app
DEFAULT_IMAGES_DIR = APP_DIR.parent / "stage_01_inference" / "stage_01_images" / "test"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
parser.add_argument("--runs", type=int, default=3, help="Timed repeats per image (default: 3)")
parser.add_argument("--warmup", type=int, default=3, help="Warm-up iterations, not timed (default: 3)")
parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR, help="Folder of receipt images")
args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import torch

from app.services.object_detection.detection import run_detection, MODEL_PATH as YOLO_PATH
from app.services.utils.crop_date_field import crop_date_field
from app.services.manipulation_detection.predict_4channel_manipulation import (
    predict,
    MODEL_PATH as UNET_PATH,
)

USING_GPU = torch.cuda.is_available()
DEVICE_LABEL = "Desktop GPU" if USING_GPU else "MacBook CPU"


def _sync():
    """Block until queued GPU work finishes so timings are accurate."""
    if USING_GPU:
        torch.cuda.synchronize()


def _collect_images(images_dir: Path) -> list[Path]:
    paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in images_dir.glob(ext)
    )
    if not paths:
        raise RuntimeError(f"No images found in {images_dir}")
    return paths


def _model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _time_calls(fn, items, runs: int, warmup: int) -> list[float]:
    """Return per-call latencies in ms. fn(item) is the timed unit of work."""
    # Warm-up (caches kernels, allocates buffers)
    for _ in range(warmup):
        for item in items:
            fn(item)
    _sync()

    timings = []
    for _ in range(runs):
        for item in items:
            start = time.perf_counter()
            fn(item)
            _sync()
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _summary(timings: list[float]) -> str:
    return (
        f"mean {statistics.mean(timings):7.2f} ms  |  "
        f"median {statistics.median(timings):7.2f} ms  |  "
        f"min {min(timings):7.2f}  max {max(timings):7.2f}  "
        f"(n={len(timings)})"
    )


def main():
    image_paths = _collect_images(args.images)
    print(f"Environment : {DEVICE_LABEL}")
    print(f"Device      : {'cuda:' + torch.cuda.get_device_name(0) if USING_GPU else 'cpu'}")
    print(f"Images      : {len(image_paths)} from {args.images}")
    print(f"Runs/image  : {args.runs} (after {args.warmup} warm-up)")
    print("=" * 78)

    # --- Stage 1: YOLO-OBB on the full receipt image -----------------------
    yolo_timings = _time_calls(
        lambda p: run_detection(str(p)),
        image_paths,
        args.runs,
        args.warmup,
    )

    # --- Prepare date crops for the U-Net++ benchmark ----------------------
    # Detect once per image, crop the date field, write crops to a temp dir
    # predict() follows the exact production path (it reads from disk).
    crop_dir = APP_DIR / "uploads" / "tmp" / "_bench_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    skipped = 0
    for p in image_paths:
        try:
            detections = run_detection(str(p))
            crop = crop_date_field(str(p), detections)
        except RuntimeError:
            skipped += 1  # no date field detected on this receipt
            continue
        out = crop_dir / f"{p.stem}_datecrop.jpg"
        cv2.imwrite(str(out), crop)
        crop_paths.append(out)

    if not crop_paths:
        raise RuntimeError("No date crops could be produced — cannot benchmark U-Net++.")
    if skipped:
        print(f"(note: {skipped} image(s) had no detectable date field, excluded from U-Net++ row)")

    # --- Stage 2: U-Net++ on the cropped date field ------------------------
    unet_timings = _time_calls(
        lambda p: predict(str(p)),
        crop_paths,
        args.runs,
        args.warmup,
    )

    # --- Full pipeline: detect -> crop -> manipulation, per receipt --------
    def _full_pipeline(p: Path):
        detections = run_detection(str(p))
        crop = crop_date_field(str(p), detections)
        tmp = crop_dir / f"{p.stem}_pipe.jpg"
        cv2.imwrite(str(tmp), crop)
        return predict(str(tmp))

    pipeline_inputs = [p for p in image_paths if (crop_dir / f"{p.stem}_datecrop.jpg") in crop_paths]
    pipeline_timings = _time_calls(_full_pipeline, pipeline_inputs, args.runs, args.warmup)

    # --- Report ------------------------------------------------------------
    yolo_mb = _model_size_mb(YOLO_PATH)
    unet_mb = _model_size_mb(UNET_PATH)

    print("=" * 78)
    print(f"{'Component':<14}{'Input':<22}{'Avg time':<14}{'Model size'}")
    print("-" * 78)
    print(f"{'YOLO-OBB':<14}{'Full receipt image':<22}{statistics.mean(yolo_timings):>8.2f} ms   {yolo_mb:>7.2f} MB")
    print(f"{'U-Net++':<14}{'Cropped date field':<22}{statistics.mean(unet_timings):>8.2f} ms   {unet_mb:>7.2f} MB")
    print(f"{'Full pipeline':<14}{'One receipt image':<22}{statistics.mean(pipeline_timings):>8.2f} ms   {yolo_mb + unet_mb:>7.2f} MB")
    print("=" * 78)
    print("Detail:")
    print(f"  YOLO-OBB      : {_summary(yolo_timings)}")
    print(f"  U-Net++       : {_summary(unet_timings)}")
    print(f"  Full pipeline : {_summary(pipeline_timings)}")
    print()
    print("Markdown row for table:")
    print(f"| {DEVICE_LABEL} | YOLO-OBB | Full receipt image | {statistics.mean(yolo_timings):.1f} ms | {yolo_mb:.1f} MB |")
    print(f"| {DEVICE_LABEL} | U-Net++ | Cropped date field | {statistics.mean(unet_timings):.1f} ms | {unet_mb:.1f} MB |")
    print(f"| {DEVICE_LABEL} | Full pipeline | One receipt image | {statistics.mean(pipeline_timings):.1f} ms | {yolo_mb + unet_mb:.1f} MB |")


if __name__ == "__main__":
    main()
