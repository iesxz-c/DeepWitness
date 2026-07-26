import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import cv2
from fastapi import FastAPI, File, Form, Query, UploadFile
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
from cv_pipeline.detect import detect_video
from agents.event_builder import build_events_from_detections

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

CLASS_TO_EVENT_TYPE = {
    "gun": "weapon_detected",
    "heavy-weapon": "weapon_detected",
    "knife": "weapon_detected",
    "Knife": "weapon_detected",
}

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


class UploadedVideo:
    def __init__(self, video_id: str, camera: str, path: Path, events_created: int):
        self.video_id = video_id
        self.camera = camera
        self.path = path
        self.events_created = events_created


class State:
    store: EventStore
    vindex: VectorIndex
    agent: InvestigatorAgent
    videos: list[UploadedVideo]


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = EventStore(":memory:")
    state.vindex = VectorIndex()
    state.videos = []
    for ev in FAKE_EVENTS:
        state.store.add_event(ev)
        state.vindex.add_event(ev)
    state.agent = InvestigatorAgent(state.vindex, state.store, Path("."))
    yield
    state.store.close()


app = FastAPI(title="CCTV Investigation API", lifespan=lifespan)

# CORS: local dev only — lock allow_origins to specific origins before deployment.
# allow_credentials must be False when using "*" per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    result = state.agent.ask(req.question)
    return {
        "answer": result["answer"],
        "tools_called": result["tools_called"],
        "provider_used": result["provider_used"],
    }


@app.get("/report")
def report():
    all_events = state.store.get_events()
    timeline_entries = build_timeline(all_events)
    evidence_bundles = [get_evidence(ev, Path(".")) for ev in all_events]
    markdown, structured, provider = generate_report(timeline_entries, evidence_bundles)

    # Strip accidental code fences the LLM sometimes wraps output in
    stripped = markdown.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    markdown = stripped

    return {
        "markdown": markdown,
        "structured": structured,
        "provider_used": provider,
    }


# ---------------------------------------------------------------------------
# Video upload endpoints
# ---------------------------------------------------------------------------
# NOTE: POST /videos runs full YOLO inference synchronously on the request
# thread. This is fine for a demo/defense but would need an async job queue
# (Celery, BackgroundTasks, etc.) in production.
#
# Test with curl:
#   curl -X POST http://localhost:8000/videos \
#     -F "file=@cv_pipeline/test_clips/sample.mp4" \
#     -F "camera=cam-upload"
#
#   curl http://localhost:8000/videos
# ---------------------------------------------------------------------------

@app.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    camera: str = Form(default=None),
):
    if not camera or not camera.strip():
        camera = f"cam-upload-{datetime.now():%Y%m%d-%H%M%S}"
    video_id = uuid.uuid4().hex[:12]
    dest = UPLOADS_DIR / f"{video_id}_{file.filename}"
    with open(dest, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        os.remove(dest)
        return {"error": f"Cannot open video: {file.filename}"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detections_by_frame = list(detect_video(dest, skip=5, conf_weapon=0.5, conf_knife=0.65, conf_coco=0.5))
    events = build_events_from_detections(camera, detections_by_frame, fps)

    for ev in events:
        state.store.add_event(ev)
        state.vindex.add_event(ev)

    state.videos.append(UploadedVideo(video_id, camera, dest, len(events)))

    return {
        "video_id": video_id,
        "camera": camera,
        "events_created": len(events),
        "status": "processed",
    }


@app.get("/videos")
def list_videos():
    return [
        {"video_id": v.video_id, "camera": v.camera, "events_created": v.events_created}
        for v in state.videos
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
