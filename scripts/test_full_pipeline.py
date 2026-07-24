"""End-to-end pipeline test on real video.

Runs the full spec pipeline:
  CCTV Video -> Detection -> Events -> Knowledge Base -> Timeline -> Evidence -> Report

Uses cv_pipeline/test_clips/positive_sample.mp4 (handgun footage) to prove
the system works on genuine video input, not fake/hardcoded data.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from cv_pipeline.detect import detect_video
from agents.event_builder import build_events_from_detections
from agents.store import EventStore, VectorIndex
from agents.timeline_agent import build_timeline
from agents.evidence_agent import get_evidence
from agents.report_agent import generate_report

VIDEO = Path(__file__).resolve().parent.parent / "cv_pipeline" / "test_clips" / "positive_sample.mp4"
CAMERA = "cam-demo"
SKIP = 5
CONF = 0.5
EVIDENCE_SAMPLE_SIZE = 5


def main():
    if not VIDEO.exists():
        print(f"ERROR: video not found: {VIDEO}")
        sys.exit(1)

    # --- Step 0: video metadata ---
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 70)
    print("FULL PIPELINE TEST: CCTV Video -> Detection -> Events -> Report")
    print("=" * 70)
    print(f"Video:   {VIDEO.name}")
    print(f"FPS:     {fps}")
    print(f"Frames:  {total_frames}")
    print(f"Camera:  {CAMERA}")
    print()

    # --- Step 1: Detection ---
    print("-" * 70)
    print("STEP 1: detect_video() — running YOLOv8 on every 5th frame")
    print("-" * 70)
    detections_by_frame = list(detect_video(VIDEO, skip=SKIP, confidence=CONF))
    frames_with_dets = sum(1 for _, d in detections_by_frame if d)
    total_dets = sum(len(d) for _, d in detections_by_frame)
    print(f"  Frames processed:   {len(detections_by_frame)}")
    print(f"  Frames w/ detections: {frames_with_dets}")
    print(f"  Total detections:   {total_dets}")

    # --- Step 2: Build Events ---
    print()
    print("-" * 70)
    print("STEP 2: build_events_from_detections() — converting to Event objects")
    print("-" * 70)
    events = build_events_from_detections(CAMERA, detections_by_frame, fps)
    print(f"  Events created: {len(events)}")
    if events:
        print(f"  First: [{events[0].time}] {events[0].event_type} conf={events[0].confidence}")
        print(f"  Last:  [{events[-1].time}] {events[-1].event_type} conf={events[-1].confidence}")
        class_counts = {}
        for e in events:
            class_counts[e.description.split("(")[0].strip()] = class_counts.get(
                e.description.split("(")[0].strip(), 0) + 1
        for desc, count in class_counts.items():
            print(f"    {desc}: {count}")

    # --- Step 3: Store in EventStore + VectorIndex ---
    print()
    print("-" * 70)
    print("STEP 3: EventStore + VectorIndex — persisting events")
    print("-" * 70)
    store = EventStore(":memory:")
    vindex = VectorIndex()
    for ev in events:
        store.add_event(ev)
        vindex.add_event(ev)
    stored = store.get_events()
    print(f"  Stored: {len(stored)} events in SQLite")
    print(f"  Indexed: {len(vindex.events)} events in FAISS vector index")

    # --- Step 4: Build Timeline ---
    print()
    print("-" * 70)
    print("STEP 4: build_timeline() — merging into timeline entries")
    print("-" * 70)
    timeline = build_timeline(stored)
    print(f"  Timeline entries: {len(timeline)} (from {len(events)} events)")
    print(f"  Note: single-camera footage, so no cross-camera merging expected")
    for i, entry in enumerate(timeline[:5]):
        print(f"  [{entry.time}] {entry.event_type} conf={entry.confidence:.2f}"
              f" sources={entry.sources}")
    if len(timeline) > 5:
        print(f"  ... and {len(timeline) - 5} more entries")

    # --- Step 5: Get Evidence ---
    print()
    print("-" * 70)
    print("STEP 5: get_evidence() — retrieving evidence bundles")
    print("-" * 70)
    print("  NOTE: Using simulated frame cache (no real thumbnails extracted yet).")
    print("  In production, frame extraction from video would populate this cache.")

    with tempfile.TemporaryDirectory() as tmpdir:
        for entry in timeline:
            for cam in entry.sources:
                ts = entry.time.replace(":", "")
                (Path(tmpdir) / f"{cam}_{ts}.jpg").write_bytes(b"\xff")

        sample = timeline[:EVIDENCE_SAMPLE_SIZE]
        evidence = [get_evidence(entry, tmpdir) for entry in sample]
        print(f"  Evidence bundles: {len(evidence)} (sampled first {EVIDENCE_SAMPLE_SIZE})")
        for b in evidence:
            thumbs = len(b.thumbnail_paths)
            print(f"    [{b.event.time}] conf={b.confidence:.2f} thumbnails={thumbs}")

    # --- Step 6: Generate Report ---
    print()
    print("-" * 70)
    print("STEP 6: generate_report() — LLM generates investigation report")
    print("-" * 70)
    markdown, structured, provider = generate_report(timeline, evidence, verbose=True)
    print(f"  Provider used: {provider}")
    print(f"  Markdown length: {len(markdown)} chars")
    print(f"  Structured keys: {list(structured.keys()) if isinstance(structured, dict) else 'parse failed'}")

    # --- Step 7: Full Report ---
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(markdown)

    # --- Summary ---
    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Source video:          {VIDEO.name} ({total_frames} frames, {fps} fps)")
    print(f"  Detections:            {total_dets} from {frames_with_dets} frames")
    print(f"  Events created:        {len(events)}")
    print(f"  Timeline entries:      {len(timeline)}")
    print(f"  Evidence bundles:      {len(evidence)} (sampled)")
    print(f"  Report generated by:   {provider}")
    print(f"  Report length:         {len(markdown)} chars")
    print("=" * 70)

    store.close()


if __name__ == "__main__":
    main()
