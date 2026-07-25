"""Convert CV detection output into validated Event objects.

Bridges cv_pipeline/detect.py (raw per-frame detections) with
schemas/event.py (structured Event model) and agents/store.py
(EventStore + VectorIndex for persistence and semantic search).
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
}


def _frame_to_time(frame_idx: int, fps: float) -> str:
    total_seconds = frame_idx / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_events_from_detections(
    camera_name: str,
    detections_by_frame: list[tuple[int, list[dict]]],
    fps: float,
) -> list[Event]:
    """Convert detection pipeline output into validated Event objects.

    Args:
        camera_name: Camera identifier (e.g. "cam-entrance").
        detections_by_frame: Output of detect_video() — list of
            (frame_idx, detections) where each detection is
            {bbox, class_name, confidence, source_model}.
        fps: Frames per second of the source video.

    Returns:
        List of Event objects, one per detection. Frames with zero
        detections are skipped.
    """
    events = []
    for frame_idx, dets in detections_by_frame:
        for det in dets:
            time_str = _frame_to_time(frame_idx, fps)
            class_name = det["class_name"]
            event_type = CLASS_TO_EVENT_TYPE.get(class_name, "unknown")
            description = (
                f"{class_name} detected "
                f"(confidence {det['confidence']:.2f}, model: {det['source_model']})"
            )
            events.append(Event(
                time=time_str,
                camera=camera_name,
                event_type=event_type,
                description=description,
                confidence=det["confidence"],
            ))
    return events


if __name__ == "__main__":
    import argparse
    import cv2

    from agents.store import EventStore, VectorIndex

    parser = argparse.ArgumentParser(
        description="Run detection pipeline and store results as Events."
    )
    parser.add_argument("video", nargs="?",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "cv_pipeline" / "test_clips" / "sample.mp4"),
                        help="Path to input video (default: cv_pipeline/test_clips/sample.mp4)")
    parser.add_argument("--camera", default="cam-test",
                        help="Camera name to assign to events (default: cam-test)")
    parser.add_argument("--skip", type=int, default=5,
                        help="Process every Nth frame (default: 5)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
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
    print(f"Conf:    {args.conf}")
    print("=" * 60)

    detections_by_frame = list(detect_video(video_path, skip=args.skip, confidence=args.conf))
    print(f"Frames with detections: "
          f"{sum(1 for _, d in detections_by_frame if d)} / {len(detections_by_frame)} processed")

    events = build_events_from_detections(args.camera, detections_by_frame, fps)
    print(f"Events created: {len(events)}")

    if not events:
        print("\nNo events to store (clean video — no weapons detected).")
        print("Pipeline ran end-to-end successfully.")
    else:
        store = EventStore(":memory:")
        vindex = VectorIndex()
        for ev in events:
            store.add_event(ev)
            vindex.add_event(ev)

        print(f"\nStored {len(events)} events in EventStore + VectorIndex:")
        for ev in events:
            print(f"  [{ev.time}] {ev.event_type}: {ev.description}")

        store.close()
