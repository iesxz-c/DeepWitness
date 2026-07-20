from pathlib import Path
from typing import Union

from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event
from agents.timeline_agent import TimelineEntry


class EvidenceBundle(BaseModel):
    event: Union[Event, TimelineEntry]
    thumbnail_paths: list[str]
    confidence: float


def _time_to_filename(time_str: str) -> str:
    return time_str.replace(":", "")


def get_evidence(
    event_or_entry: Union[Event, TimelineEntry],
    frame_cache_dir: str | Path,
) -> EvidenceBundle:
    cache = Path(frame_cache_dir)
    if not cache.is_dir():
        return EvidenceBundle(
            event=event_or_entry,
            thumbnail_paths=[],
            confidence=event_or_entry.confidence,
        )

    if isinstance(event_or_entry, Event):
        cameras = [event_or_entry.camera]
        time_str = event_or_entry.time
    else:
        cameras = event_or_entry.sources
        time_str = event_or_entry.time

    ts = _time_to_filename(time_str)
    found = []
    for cam in cameras:
        frame = cache / f"{cam}_{ts}.jpg"
        if frame.is_file():
            found.append(str(frame))

    return EvidenceBundle(
        event=event_or_entry,
        thumbnail_paths=found,
        confidence=event_or_entry.confidence,
    )


if __name__ == "__main__":
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_files = [
            "cam-lobby_101630.jpg",
            "cam-entrance_101630.jpg",
            "cam-north_144000.jpg",
        ]
        for name in dummy_files:
            (Path(tmpdir) / name).write_bytes(b"")

        print("=== Dummy frame cache ===")
        for f in sorted(os.listdir(tmpdir)):
            print(f"  {f}")

        match_entry = TimelineEntry(
            time="10:16:30",
            event_type="theft",
            description="Person grabbed a backpack and walked out",
            confidence=0.91,
            sources=["cam-lobby", "cam-entrance"],
        )
        bundle_match = get_evidence(match_entry, tmpdir)
        print(f"\n=== Match (theft @ 10:16:30) ===")
        print(f"  confidence: {bundle_match.confidence}")
        print(f"  thumbnails: {bundle_match.thumbnail_paths}")

        no_match_entry = TimelineEntry(
            time="18:30:00",
            event_type="loitering",
            description="Individual sitting in lobby too long",
            confidence=0.60,
            sources=["cam-lobby"],
        )
        bundle_none = get_evidence(no_match_entry, tmpdir)
        print(f"\n=== No match (loitering @ 18:30:00) ===")
        print(f"  confidence: {bundle_none.confidence}")
        print(f"  thumbnails: {bundle_none.thumbnail_paths}")

        single_event = Event(
            time="14:40:00",
            camera="cam-north",
            event_type="intrusion",
            description="Person climbed over fence",
            confidence=0.88,
        )
        bundle_single = get_evidence(single_event, tmpdir)
        print(f"\n=== Single Event (intrusion @ 14:40:00) ===")
        print(f"  confidence: {bundle_single.confidence}")
        print(f"  thumbnails: {bundle_single.thumbnail_paths}")
