import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.event import Event
from agents.store import EventStore, VectorIndex
from agents.timeline_agent import TimelineEntry, build_timeline
from agents.evidence_agent import EvidenceBundle, get_evidence
from agents.query_agent import InvestigatorAgent
from agents.report_agent import generate_report

FAKE_EVENTS = [
    Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
          description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
    Event(time="10:30:00", camera="cam-parking", event_type="loitering",
          description="Individual standing near parked cars for an extended period", confidence=0.78),
    # Mergeable pair: same theft, two cameras, ~90s apart
    Event(time="12:45:00", camera="cam-lobby", event_type="theft",
          description="Person grabbed a backpack from the lobby sofa and walked out", confidence=0.95),
    Event(time="12:46:30", camera="cam-entrance", event_type="theft",
          description="Individual exiting building carrying a backpack that was not theirs", confidence=0.88),
    Event(time="15:20:00", camera="cam-lobby", event_type="loitering",
          description="Unknown individual loitering near reception desk", confidence=0.65),
    Event(time="18:00:00", camera="cam-loading", event_type="intrusion",
          description="Person forced open the loading dock side door", confidence=0.88),
    Event(time="22:10:00", camera="cam-parking", event_type="vandalism",
          description="Individual scratching paint on multiple vehicles in lot", confidence=0.71),
]


class State:
    store: EventStore
    vindex: VectorIndex
    agent: InvestigatorAgent


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = EventStore(":memory:")
    state.vindex = VectorIndex()
    for ev in FAKE_EVENTS:
        state.store.add_event(ev)
        state.vindex.add_event(ev)
    state.agent = InvestigatorAgent(state.vindex, state.store, Path("."))
    yield
    state.store.close()


app = FastAPI(title="CCTV Investigation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def list_events(
    camera: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
):
    events = state.store.get_events(
        camera=camera,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
    )
    return [e.model_dump() for e in events]


@app.get("/timeline")
def timeline():
    all_events = state.store.get_events()
    entries = build_timeline(all_events)
    return [t.model_dump() for t in entries]


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest):
    state.agent.tool_log.clear()
    answer = state.agent.ask(req.question)
    return {
        "answer": answer,
        "tools_called": [t["tool"] for t in state.agent.tool_log],
    }


@app.get("/report")
def report():
    all_events = state.store.get_events()
    timeline_entries = build_timeline(all_events)
    evidence_bundles = [get_evidence(ev, Path(".")) for ev in all_events]
    markdown, structured = generate_report(timeline_entries, evidence_bundles)
    return {"markdown": markdown, "structured": structured}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
