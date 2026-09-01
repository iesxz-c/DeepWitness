"""Build real burglary-clip events and persist them into backend/events.db.

Runs the actual YOLO weapon/person/object detection pipeline on a UCF-Crime
Burglary clip, converts detections to Events, and appends them to the same
persistent SQLite DB that the FastAPI backend loads on startup.

Usage:
    python scripts/build_burglary_events.py <clip.mp4> [--camera cam-store-1]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents.event_builder import build_events_from_detections  # noqa: E402
from cv_pipeline.detect import detect_video  # noqa: E402
from agents.store import EventStore, VectorIndex  # noqa: E402

EVENTS_DB = str(REPO / "backend" / "events.db")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build events from a burglary clip and persist to events.db")
    parser.add_argument("clip", help="Path to the burglary clip (.mp4)")
    parser.add_argument("--camera", default="cam-store-1", help="Camera name to assign")
    parser.add_argument("--skip", type=int, default=5, help="Process every Nth frame (default 5)")
    parser.add_argument("--conf-weapon", type=float, default=0.5)
    parser.add_argument("--conf-knife", type=float, default=0.5)
    parser.add_argument("--conf-coco", type=float, default=0.5)
    args = parser.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"ERROR: clip not found: {clip}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print("ERROR: cannot open clip")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Clip:    {clip}")
    print(f"FPS:     {fps:.2f}")
    print(f"Frames:  {frames}  (~{frames/max(fps,1)/60:.1f} min)")
    print(f"Camera:  {args.camera}")

    print("\nRunning detection pipeline (this may take a while on CPU)...")
    print("=" * 60)
    detections = list(detect_video(str(clip), skip=args.skip,
                                   conf_weapon=args.conf_weapon,
                                   conf_knife=args.conf_knife,
                                   conf_coco=args.conf_coco))
    frames_with = sum(1 for _, d in detections if d)
    print(f"Frames with detections: {frames_with} / {len(detections)} processed")

    new_events = build_events_from_detections(args.camera, detections, fps)
    print(f"Events created: {len(new_events)}")
    if not new_events:
        print("No weapon/person/object detections above threshold -> no events built.")
        print("Re-run with lower --conf-* thresholds.")
        return

    # Persist into the SAME DB the backend uses, then rebuild vector index.
    store = EventStore(EVENTS_DB)
    vindex = VectorIndex()
    for existing in store.get_events():
        vindex.add_event(existing)

    print(f"\nAppending {len(new_events)} new events to {EVENTS_DB} ...")
    for ev in new_events:
        store.add_event(ev)
        vindex.add_event(ev)

    total = len(store.get_events())
    print(f"DB now holds {total} events total.")

    listed = {}
    for ev in new_events:
        listed.setdefault(ev.event_type, []).append(ev.confidence)
    print("\nNew events by type + count (mean confidence):")
    for et, confs in sorted(listed.items()):
        print(f"  {et:<20} n={len(confs):<4} mean_conf={sum(confs)/len(confs):.3f}")

    print("\nSampling up to 12 new events:")
    for ev in new_events[:12]:
        print(f"  [{ev.time}] {ev.camera} {ev.event_type}: {ev.description}")
    store.close()


if __name__ == "__main__":
    main()
