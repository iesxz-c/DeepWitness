"""Merged dual-model weapon/knife detection pipeline.

Two YOLOv8 models are used instead of one because no single training run
produced acceptable recall across all weapon classes:

  weapon_detect_v1_best.pt  — trained on gun + heavy-weapon + knife data.
  Classes kept: "gun", "heavy-weapon".
  The knife class in this model has ~0 recall (only 13 training instances),
  so its knife predictions are discarded entirely.

  knife_detect_v1_best.pt   — trained on a knife-focused dataset.
  Class kept: "knife" only.
  The remaining classes (Handgun, Slap, Violence, Person, etc.) each had
  fewer than 10 training instances — statistically meaningless — so all
  non-knife predictions from this model are discarded.

The pipeline takes frames (optionally from a video via ingest.py conventions),
runs both models on every Nth frame (configurable skip), filters each model's
raw output to its "good" classes, and merges both into one detection list per
frame with a source_model tag indicating which weights file produced each
detection.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEAPON_WEIGHTS = WEIGHTS_DIR / "weapon_detect_v1_best.pt"
KNIFE_WEIGHTS = WEIGHTS_DIR / "knife_detect_v1_best.pt"

WEAPON_KEEP = {"gun", "heavy-weapon"}
KNIFE_KEEP = {"knife"}

WEAPON_MODEL_TAG = "weapon_v1"
KNIFE_MODEL_TAG = "knife_v1"

DEFAULT_SKIP = 5
DEFAULT_CONFIDENCE = 0.25


def _load_models():
    weapon = YOLO(str(WEAPON_WEIGHTS))
    knife = YOLO(str(KNIFE_WEIGHTS))
    return weapon, knife


def _filter(results, allowed_classes: set[str]) -> list[dict]:
    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, f"cls_{cls_id}")
            if cls_name not in allowed_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "class_name": cls_name,
                "confidence": round(float(box.conf[0]), 3),
            })
    return detections


def detect_frame(frame, weapon_model, knife_model,
                 confidence: float = DEFAULT_CONFIDENCE) -> list[dict]:
    """Run both models on a single BGR frame and return merged detections."""
    weapon_results = weapon_model(frame, conf=confidence, verbose=False)
    knife_results = knife_model(frame, conf=confidence, verbose=False)

    weapon_dets = _filter(weapon_results, WEAPON_KEEP)
    for d in weapon_dets:
        d["source_model"] = WEAPON_MODEL_TAG

    knife_dets = _filter(knife_results, KNIFE_KEEP)
    for d in knife_dets:
        d["source_model"] = KNIFE_MODEL_TAG

    return weapon_dets + knife_dets


def detect_video(video_path: str | Path, skip: int = DEFAULT_SKIP,
                 confidence: float = DEFAULT_CONFIDENCE):
    """Yield (frame_index, detections) for every processed frame."""
    weapon_model, knife_model = _load_models()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                dets = detect_frame(frame, weapon_model, knife_model, confidence)
                yield frame_idx, dets
            frame_idx += 1
    finally:
        cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run merged detection on a video clip.")
    parser.add_argument("video", nargs="?",
                        default=str(Path(__file__).resolve().parent / "test_clips" / "sample.mp4"),
                        help="Path to input video (default: cv_pipeline/test_clips/sample.mp4)")
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP,
                        help=f"Process every Nth frame (default: {DEFAULT_SKIP})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Minimum confidence threshold (default: {DEFAULT_CONFIDENCE})")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    print(f"Video:     {video_path}")
    print(f"Skip:      every {args.skip} frames")
    print(f"Confidence: {args.conf}")
    print(f"Weights:   {WEAPON_MODEL_TAG}={WEAPON_WEIGHTS.name}, "
          f"{KNIFE_MODEL_TAG}={KNIFE_WEIGHTS.name}")
    print("=" * 70)

    total_frames = 0
    processed_frames = 0
    total_detections = 0

    cap = cv2.VideoCapture(str(video_path))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    for frame_idx, dets in detect_video(video_path, skip=args.skip, confidence=args.conf):
        processed_frames += 1
        total_frames = max(total_frames, frame_idx + 1)
        total_detections += len(dets)

        if dets:
            print(f"\nFrame {frame_idx} ({len(dets)} detection{'s' if len(dets) != 1 else ''}):")
            for d in dets:
                print(f"  [{d['source_model']:>10}] {d['class_name']:<15} "
                      f"conf={d['confidence']:.3f}  bbox={d['bbox']}")
        else:
            print(f"\nFrame {frame_idx}: no detections")

    print("\n" + "=" * 70)
    print(f"Total frames in video: {video_len}")
    print(f"Frames processed:      {processed_frames} (every {args.skip}th frame)")
    print(f"Total detections:      {total_detections}")
    print("=" * 70)
