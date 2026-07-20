from datetime import datetime
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event

from pydantic import BaseModel, Field


TIME_FORMAT = "%H:%M:%S"
TIME_DELTA_THRESHOLD = 120  # seconds
SIMILARITY_THRESHOLD = 0.5


class TimelineEntry(BaseModel):
    time: str
    event_type: str
    description: str
    confidence: float
    sources: list[str]


_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _time_diff_seconds(t1: str, t2: str) -> int:
    d1 = datetime.strptime(t1, TIME_FORMAT)
    d2 = datetime.strptime(t2, TIME_FORMAT)
    return abs(int((d2 - d1).total_seconds()))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_timeline(events: list[Event]) -> list[TimelineEntry]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.time)

    model = _get_model()
    descriptions = [e.description for e in sorted_events]
    embeddings = model.encode(descriptions)

    timeline: list[TimelineEntry] = []
    used = set()

    for i, ev in enumerate(sorted_events):
        if i in used:
            continue

        entry = TimelineEntry(
            time=ev.time,
            event_type=ev.event_type,
            description=ev.description,
            confidence=ev.confidence,
            sources=[ev.camera],
        )

        for j in range(i + 1, len(sorted_events)):
            if j in used:
                continue
            other = sorted_events[j]

            if other.camera in entry.sources:
                continue

            if _time_diff_seconds(entry.time, other.time) > TIME_DELTA_THRESHOLD:
                continue

            if _cosine_sim(embeddings[i], embeddings[j]) < SIMILARITY_THRESHOLD:
                continue

            entry.sources.append(other.camera)
            entry.confidence = max(entry.confidence, other.confidence)
            if other.time > entry.time:
                entry.time = other.time
            used.add(j)

        used.add(i)
        timeline.append(entry)

    return timeline


if __name__ == "__main__":
    test_events = [
        # Same incident, two cameras — should merge
        Event(time="10:15:00", camera="cam-lobby", event_type="theft",
              description="Person grabbed a backpack from the lobby sofa and walked out", confidence=0.91),
        Event(time="10:16:30", camera="cam-entrance", event_type="theft",
              description="Individual exiting building carrying a backpack that was not theirs", confidence=0.85),

        # Unrelated event — should NOT merge
        Event(time="10:17:00", camera="cam-parking", event_type="vandalism",
              description="Unknown person keying the paint on a silver sedan", confidence=0.73),

        # Second same-incident pair, different pair — should merge
        Event(time="14:40:00", camera="cam-north", event_type="intrusion",
              description="Male subject climbed over the north perimeter fence", confidence=0.88),
        Event(time="14:41:15", camera="cam-yard", event_type="intrusion",
              description="Person seen jumping the fence into the north yard area", confidence=0.82),

        # Lone event — no merge candidate
        Event(time="18:30:00", camera="cam-lobby", event_type="loitering",
              description="Individual sitting in lobby for over 30 minutes without interacting", confidence=0.60),
    ]

    print("=== Input Events ===")
    for ev in test_events:
        print(f"  [{ev.time}] {ev.camera} | {ev.event_type}: {ev.description}")

    timeline = build_timeline(test_events)

    print(f"\n=== Timeline ({len(timeline)} entries from {len(test_events)} events) ===")
    for entry in timeline:
        sources_str = ", ".join(entry.sources)
        merged = " [MERGED]" if len(entry.sources) > 1 else ""
        print(f"  [{entry.time}] {entry.event_type}{merged}")
        print(f"    cameras: {sources_str}")
        print(f"    confidence: {entry.confidence:.2f}")
        print(f"    {entry.description}")
