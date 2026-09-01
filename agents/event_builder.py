"""Convert CV detection output into validated Event objects.

Bridges cv_pipeline/detect.py (raw per-frame detections) with
schemas/event.py (structured Event model) and agents/store.py
(EventStore + VectorIndex for persistence and semantic search).

Every model in the perception stack is fused to build events:
  * YOLO weapon model (gun / heavy-weapon)            -> weapon_v1
  * YOLO knife model                                  -> knife_v2
  * YOLO COCO general model (person/car/bag/tv/...)   -> coco_general
  * VideoMAE UCF-Crime action model (Burglary, ...)   -> videomae-ucf-crime
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.event import Event
from cv_pipeline.detect import detect_video

CLASS_TO_EVENT_TYPE = {
    "gun": "weapon_detected",
    "heavy-weapon": "weapon_detected",
    "knife": "weapon_detected",
    "Knife": "weapon_detected",
    "person": "person_detected",
    "backpack": "object_detected",
    "handbag": "object_detected",
    "suitcase": "object_detected",
    "car": "vehicle_detected",
    "truck": "vehicle_detected",
    "cell phone": "object_detected",
    "tv": "object_detected",
    "laptop": "object_detected",
}

# Action-model classes that map to a real incident type; anything else is
# kept under its own lowercase label as the event_type.
ACTION_LABELS = {
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Road Accidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
}


def _frame_to_time(frame_idx: int, fps: float) -> str:
    total_seconds = frame_idx / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_action_event(
    camera_name: str,
    action_result: dict,
    time: str = "00:00:00",
    case_id: str | None = None,
) -> Event | None:
    """Build a single high-level event from the VideoMAE action verdict."""
    if not action_result:
        return None
    label = action_result.get("label", "")
    if label == "Normal Videos":
        return None  # nothing suspicious to log
    event_type = label.lower().replace(" ", "_")
    description = (
        f"{label} action detected (confidence {action_result['confidence']:.3f}, "
        f"model: videomae-large-finetuned-UCF-Crime)"
    )
    return Event(
        time=time,
        camera=camera_name,
        event_type=event_type,
        description=description,
        confidence=round(action_result["confidence"], 3),
        case_id=case_id,
    )


def build_events_from_detections(
    camera_name: str,
    detections_by_frame: list[tuple[int, list[dict]]],
    fps: float,
    action_result: dict | None = None,
    case_id: str | None = None,
) -> list[Event]:
    """Convert detection pipeline output into validated Event objects.

    Fuses the VideoMAE action verdict (if provided) plus every object-level
    detection into a single Event list.
    """
    events: list[Event] = []

    if action_result:
        action_ev = build_action_event(camera_name, action_result, case_id=case_id)
        if action_ev:
            events.append(action_ev)

    for frame_idx, dets in detections_by_frame:
        for det in dets:
            time_str = _frame_to_time(frame_idx, fps)
            class_name = det["class_name"]
            event_type = CLASS_TO_EVENT_TYPE.get(class_name, "unknown")
            source = det.get("source_model", "unknown")
            description = (
                f"{class_name} detected "
                f"(confidence {det['confidence']:.2f}, model: {source})"
            )
            events.append(Event(
                time=time_str,
                camera=camera_name,
                event_type=event_type,
                description=description,
                confidence=det["confidence"],
                case_id=case_id,
            ))
    return events


if __name__ == "__main__":
    import argparse
    import cv2

    from cv_pipeline.action import classify_clip
    from agents.store import EventStore, VectorIndex

    parser = argparse.ArgumentParser(
        description="Run ALL perception models and store fused events."
    )
    parser.add_argument("video", nargs="?",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "cv_pipeline" / "test_clips" / "sample.mp4"),
                        help="Path to input video (default: cv_pipeline/test_clips/sample.mp4)")
    parser.add_argument("--camera", default="cam-test",
                        help="Camera name to assign to events (default: cam-test)")
    parser.add_argument("--skip", type=int, default=5,
                        help="Process every Nth frame (default: 5)")
    parser.add_argument("--conf-weapon", type=float, default=0.5,
                        help="Weapon model confidence threshold (default: 0.5)")
    parser.add_argument("--conf-knife", type=float, default=0.65,
                        help="Knife model confidence threshold (default: 0.65)")
    parser.add_argument("--conf-coco", type=float, default=0.5,
                        help="COCO general model confidence threshold (default: 0.5)")
    parser.add_argument("--case-id", default=None,
                        help="Case ID to tag all events with")
    parser.add_argument("--persist", action="store_true",
                        help="Persist fused events into backend/events.db")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video:   {video_path}")
    print(f"FPS:     {fps}")
    print(f"Frames:  {total_frames}")
    print(f"Camera:  {args.camera}")
    print(f"Skip:    every {args.skip} frames")
    print("=" * 60)

    # 1) Object-level detections (weapon + knife + COCO models)
    print("Running YOLO detection pipeline (weapon + knife + coco)...")
    detections_by_frame = list(detect_video(video_path, skip=args.skip,
                                            conf_weapon=args.conf_weapon,
                                            conf_knife=args.conf_knife,
                                            conf_coco=args.conf_coco))

    # 2) Action-level classification (VideoMAE UCF-Crime)
    print("Running VideoMAE action model...")
    action_result = classify_clip(video_path)
    print(f"  Action verdict: {action_result['label']} @ "
          f"{action_result['confidence']:.4f}")

    # 3) Fuse everything
    events = build_events_from_detections(
        args.camera, detections_by_frame, fps,
        action_result=action_result,
        case_id=args.case_id,
    )
    print(f"\nFused events: {len(events)}")

    # Break down by event_type
    from collections import Counter
    breakdown = Counter(e.event_type for e in events)
    print("Breakdown:", dict(breakdown))

    if args.persist:
        store = EventStore(str(Path(__file__).resolve().parent.parent / "backend" / "events.db"))
        vindex = VectorIndex()
        for existing in store.get_events():
            vindex.add_event(existing)
        for ev in events:
            store.add_event(ev)
            vindex.add_event(ev)
        print(f"Persisted {len(events)} events. DB total: {len(store.get_events())}")
        store.close()
    else:
        print("\nEvents built (dry run; use --persist to save to events.db):")
        for ev in events[:20]:
            print(f"  [{ev.time}] {ev.event_type}: {ev.description}")
