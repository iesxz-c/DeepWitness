import sqlite3
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event


class EventStore:
    def __init__(self, db_path: str = "events.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_event(self, event: Event) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (data) VALUES (?)",
            (event.model_dump_json(),),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_events(
        self,
        camera: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[Event]:
        rows = self.conn.execute("SELECT data FROM events").fetchall()
        results = []
        for row in rows:
            e = Event.model_validate_json(row["data"])
            if camera and e.camera != camera:
                continue
            if start_time and e.time < start_time:
                continue
            if end_time and e.time > end_time:
                continue
            if event_type and e.event_type != event_type:
                continue
            results.append(e)
        return results

    def close(self):
        self.conn.close()


class VectorIndex:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.events: list[Event] = []

    def add_event(self, event: Event):
        embedding = self.model.encode([event.description])
        self.index.add(np.array(embedding, dtype=np.float32))
        self.events.append(event)

    def semantic_search(self, query: str, k: int = 5) -> list[Event]:
        q_vec = self.model.encode([query])
        k = min(k, self.index.ntotal)
        distances, indices = self.search(np.array(q_vec, dtype=np.float32), k)
        return [self.events[i] for i in indices[0] if i < len(self.events)]

    def search(self, query_vec: np.ndarray, k: int):
        return self.index.search(query_vec, k)


if __name__ == "__main__":
    fake_events = [
        Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
              description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
        Event(time="10:30:00", camera="cam-parking", event_type="loitering",
              description="Individual standing near parked cars for an extended period", confidence=0.78),
        Event(time="12:45:00", camera="cam-lobby", event_type="theft",
              description="Unattended bag picked up and carried out by unknown person", confidence=0.95),
        Event(time="15:20:00", camera="cam-lobby", event_type="loitering",
              description="Unknown individual loitering near reception desk", confidence=0.65),
        Event(time="18:00:00", camera="cam-loading", event_type="intrusion",
              description="Person forced open the loading dock side door", confidence=0.88),
        Event(time="22:10:00", camera="cam-parking", event_type="vandalism",
              description="Individual scratching paint on multiple vehicles in lot", confidence=0.71),
    ]

    print("=== EventStore ===")
    store = EventStore(":memory:")
    for ev in fake_events:
        store.add_event(ev)

    lobby_events = store.get_events(camera="cam-lobby")
    print(f"Events at cam-lobby ({len(lobby_events)}):")
    for e in lobby_events:
        print(f"  [{e.time}] {e.event_type}: {e.description}")

    loitering = store.get_events(event_type="loitering")
    print(f"\nLoitering events ({len(loitering)}):")
    for e in loitering:
        print(f"  [{e.camera}] {e.description}")

    store.close()

    print("\n=== VectorIndex ===")
    vindex = VectorIndex()
    for ev in fake_events:
        vindex.add_event(ev)

    results = vindex.semantic_search("someone stealing a bag", k=3)
    print(f"Top-3 for 'someone stealing a bag':")
    for e in results:
        print(f"  [{e.camera}] {e.event_type}: {e.description}")

    results2 = vindex.semantic_search("breaking into the building", k=3)
    print(f"\nTop-3 for 'breaking into the building':")
    for e in results2:
        print(f"  [{e.camera}] {e.event_type}: {e.description}")
