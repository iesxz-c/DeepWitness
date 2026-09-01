"""Attach the VideoMAE UCF-Crime action verdict to the burglary incident in the KB.

The action model (OPear/videomae-large-finetuned-UCF-Crime) classifies the clip
as an ACTION (e.g. 'Burglary' @ 0.950), which the YOLO event pipeline cannot
produce. This script adds a high-level action-recognition Event to the same
persistent events.db, so the Query/Report agents can answer
"what kind of incident was this?" with the real action-model verdict.

Usage:
    python scripts/add_action_event.py <clip.mp4> [--camera cam-burglary-demo]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from cv_pipeline.action import classify_clip, MODEL_NAME  # noqa: E402
from agents.store import EventStore  # noqa: E402
from schemas.event import Event  # noqa: E402

EVENTS_DB = str(REPO / "backend" / "events.db")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add action-model verdict event to events.db")
    parser.add_argument("clip", help="Path to the clip to classify (e.g. Burglary018_x264.mp4)")
    parser.add_argument("--camera", default="cam-burglary-demo")
    parser.add_argument("--event-type", default="burglary", help="Event type to assign to the action event")
    args = parser.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"ERROR: clip not found: {clip}")
        sys.exit(1)

    print(f"Classifying action of: {clip.name}  (model: {MODEL_NAME})")
    result = classify_clip(clip)
    label = result["label"]
    conf = result["confidence"]
    print(f"Action verdict: {label} @ {conf:.4f}")

    # Sampled event times: we attach the action verdict at the incident's start
    # and end (00:00:13 was the heaviest detection window).
    action_events = [
        Event(
            time="00:00:00",
            camera=args.camera,
            event_type=args.event_type,
            description=(
                f"{label} action detected (confidence {conf:.3f}, "
                f"model: videomae-large-finetuned-UCF-Crime)"
            ),
            confidence=conf,
            case_id="case-burglary-demo",
        ),
    ]

    store = EventStore(EVENTS_DB)
    for ev in action_events:
        store.add_event(ev)
        print(f"Stored: [{ev.time}] {ev.camera} {ev.event_type}: {ev.description}")

    total = len(store.get_events())
    print(f"DB now holds {total} events total.")
    store.close()


if __name__ == "__main__":
    main()
