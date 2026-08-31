"""Cross-case memory for investigations.

Two capabilities:

1. assign_case_id() — groups an event into a case based on camera identity and
   time proximity (same camera within CASE_WINDOW_SECONDS = 1 hour reuses that
   case), unless the event already carries an explicit case_id.

2. cross_case_search() — queries the existing FAISS VectorIndex for past
   events similar to a new event, EXCLUDING everything already in the new
   event's case, returning the top-k matches with their case_id and a cosine
   similarity score. FAISS provides candidate ranking; scores are recomputed
   as true cosine similarity over normalized embeddings so they are directly
   interpretable (1.0 = identical).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event
from agents.store import VectorIndex


CASE_WINDOW_SECONDS = 3600


class CrossCaseMatch(BaseModel):
    event: Event
    similarity: float


def _to_seconds(time_str: str) -> int:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _time_distance(a: str, b: str) -> int:
    raw = abs(_to_seconds(a) - _to_seconds(b))
    return min(raw, 86400 - raw)


def _new_case_id(event: Event) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in event.camera).strip("-").lower()
    return f"case-{slug}-{event.time.replace(':', '')}"


def assign_case_id(
    event: Event,
    history: list[Event],
    window_seconds: float = CASE_WINDOW_SECONDS,
) -> str:
    if event.case_id:
        return event.case_id
    for past in history:
        if past.case_id and past.camera == event.camera \
                and _time_distance(event.time, past.time) <= window_seconds:
            event.case_id = past.case_id
            return event.case_id
    event.case_id = _new_case_id(event)
    return event.case_id


def cross_case_search(
    query_event: Event,
    index: VectorIndex,
    k: int = 3,
) -> list[CrossCaseMatch]:
    n = index.index.ntotal
    if n == 0 or not index.events:
        return []

    q_vec = np.array(index.model.encode([query_event.description]), dtype=np.float32)
    distances, indices = index.search(q_vec, int(n))

    candidates: list[Event] = []
    for idx in indices[0]:
        if 0 <= idx < len(index.events):
            past = index.events[idx]
            same_case = (
                past.case_id is not None
                and query_event.case_id is not None
                and past.case_id == query_event.case_id
            )
            if not same_case:
                candidates.append(past)

    q_norm, *_ = index.model.encode([query_event.description], normalize_embeddings=True)
    scored: list[CrossCaseMatch] = []
    for past in candidates:
        v_norm, *_ = index.model.encode([past.description], normalize_embeddings=True)
        similarity = float(np.dot(q_norm, v_norm))
        scored.append(CrossCaseMatch(event=past, similarity=round(similarity, 4)))

    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:k]


if __name__ == "__main__":
    case_a = "case-cam-entrance-080200"
    case_b = "case-cam-parking-140500"

    history = [
        Event(time="08:02:00", camera="cam-entrance", event_type="intrusion",
              description="Person jumped over the perimeter fence near the east gate",
              confidence=0.92, case_id=case_a),
        Event(time="08:11:00", camera="cam-entrance", event_type="intrusion",
              description="Individual climbing over the security fence beside the gate",
              confidence=0.84, case_id=case_a),
        Event(time="14:05:00", camera="cam-parking", event_type="vandalism",
              description="Suspect smashed the window of a parked sedan in the lot",
              confidence=0.88, case_id=case_b),
        Event(time="14:22:00", camera="cam-parking", event_type="theft",
              description="Unknown person prying open a car door with a metal bar",
              confidence=0.79, case_id=case_b),
    ]

    print("Building vector index from", len(history), "past events ...")
    vindex = VectorIndex()
    for e in history:
        vindex.add_event(e)

    print("\n--- New event E1: no explicit case_id (auto-grouping expected) ---")
    e1 = Event(time="14:40:00", camera="cam-parking", event_type="intrusion",
               description="Intruder hopped the perimeter fence next to the parking lot "
                           "and tried several car door handles",
               confidence=0.90)
    assigned = assign_case_id(e1, history)
    print(f"assigned case_id : {assigned}  (expected {case_b})")

    print("\n--- Cross-case retrieval for E1 ---")
    print(f"(excluding everything in '{e1.case_id}')")
    for i, match in enumerate(cross_case_search(e1, vindex, k=3), 1):
        print(f"  {i}. [{match.similarity:.4f}] case={match.event.case_id} "
              f"[{match.event.time} {match.event.camera}] "
              f"{match.event.event_type}: {match.event.description}")

    print("\n--- New event E2: explicit case_id is respected ---")
    e2 = Event(time="08:07:00", camera="cam-entrance", event_type="intrusion",
               description="Someone forcing the side door of the building at night",
               confidence=0.87, case_id="case-manual-999")
    kept = assign_case_id(e2, history)
    print(f"case_id after grouping: {kept}  (explicit value preserved)")
