"""Merged triple-model weapon/knife/person detection pipeline.

Three YOLOv8 models are used:

  weapon_detect_v1_best.pt  — trained on gun + heavy-weapon + knife data.
  Classes kept: "gun", "heavy-weapon".
  The knife class in this model has ~0 recall (only 13 training instances),
  so its knife predictions are discarded entirely.

  knife_detect_v2_best.pt   — 35-epoch retrain on a more diverse knife dataset.
  Better generalization than v1 (recall 0.866 vs v1's narrower focus).
  Class kept: "Knife" only.
  The remaining classes each had fewer than 10 training instances —
  statistically meaningless — so all non-knife predictions are discarded.

  yolov8n.pt               — stock COCO-pretrained YOLOv8n (no fine-tuning).
  Class kept: "person" only (class 0). All other 79 COCO classes discarded.
  Used to detect people in frame for situational awareness. May not fire on
  partial-body / hand-only framing (e.g. close-up gun clips).

The pipeline runs all three models on every Nth frame (configurable skip),
filters each model's raw output to its allowed classes, and merges into one
detection list per frame with a source_model tag.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEAPON_WEIGHTS = WEIGHTS_DIR / "weapon_detect_v1_best.pt"
KNIFE_WEIGHTS = WEIGHTS_DIR / "knife_detect_v2_best.pt"

WEAPON_KEEP = {"gun", "heavy-weapon"}
KNIFE_KEEP = {"Knife"}
PERSON_KEEP = {"person"}

WEAPON_MODEL_TAG = "weapon_v1"
KNIFE_MODEL_TAG = "knife_v2"
PERSON_MODEL_TAG = "person_coco"

DEFAULT_SKIP = 5
DEFAULT_CONF_WEAPON = 0.5
DEFAULT_CONF_KNIFE = 0.65
DEFAULT_CONF_PERSON = 0.5


def _load_models():
    weapon = YOLO(str(WEAPON_WEIGHTS))
    knife = YOLO(str(KNIFE_WEIGHTS))
    person = YOLO("yolov8n.pt")
    return weapon, knife, person


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


def detect_frame(frame, weapon_model, knife_model, person_model,
                 conf_weapon: float = DEFAULT_CONF_WEAPON,
                 conf_knife: float = DEFAULT_CONF_KNIFE,
                 conf_person: float = DEFAULT_CONF_PERSON) -> list[dict]:
    """Run all three models on a single BGR frame and return merged detections."""
    weapon_results = weapon_model(frame, conf=conf_weapon, verbose=False)
    knife_results = knife_model(frame, conf=conf_knife, verbose=False)
    person_results = person_model(frame, conf=conf_person, verbose=False)

    weapon_dets = _filter(weapon_results, WEAPON_KEEP)
    for d in weapon_dets:
        d["source_model"] = WEAPON_MODEL_TAG

    knife_dets = _filter(knife_results, KNIFE_KEEP)
    for d in knife_dets:
        d["source_model"] = KNIFE_MODEL_TAG

    person_dets = _filter(person_results, PERSON_KEEP)
    for d in person_dets:
        d["source_model"] = PERSON_MODEL_TAG

    return weapon_dets + knife_dets + person_dets


def detect_video(video_path: str | Path, skip: int = DEFAULT_SKIP,
                 conf_weapon: float = DEFAULT_CONF_WEAPON,
                 conf_knife: float = DEFAULT_CONF_KNIFE,
                 conf_person: float = DEFAULT_CONF_PERSON):
    """Yield (frame_index, detections) for every processed frame."""
    weapon_model, knife_model, person_model = _load_models()

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
                dets = detect_frame(frame, weapon_model, knife_model, person_model,
                                    conf_weapon, conf_knife, conf_person)
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
    parser.add_argument("--conf-weapon", type=float, default=DEFAULT_CONF_WEAPON,
                        help=f"Weapon model confidence threshold (default: {DEFAULT_CONF_WEAPON})")
    parser.add_argument("--conf-knife", type=float, default=DEFAULT_CONF_KNIFE,
                        help=f"Knife model confidence threshold (default: {DEFAULT_CONF_KNIFE})")
    parser.add_argument("--conf-person", type=float, default=DEFAULT_CONF_PERSON,
                        help=f"Person (COCO) model confidence threshold (default: {DEFAULT_CONF_PERSON})")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    print(f"Video:     {video_path}")
    print(f"Skip:      every {args.skip} frames")
    print(f"Confidence: weapon={args.conf_weapon}, knife={args.conf_knife}, "
          f"person={args.conf_person}")
    print(f"Weights:   {WEAPON_MODEL_TAG}={WEAPON_WEIGHTS.name}, "
          f"{KNIFE_MODEL_TAG}={KNIFE_WEIGHTS.name}, "
          f"{PERSON_MODEL_TAG}=yolov8n.pt")
    print("=" * 70)

    total_frames = 0
    processed_frames = 0
    total_detections = 0

    cap = cv2.VideoCapture(str(video_path))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    for frame_idx, dets in detect_video(video_path, skip=args.skip,
                                        conf_weapon=args.conf_weapon,
                                        conf_knife=args.conf_knife,
                                        conf_person=args.conf_person):
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
