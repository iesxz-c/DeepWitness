import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event
from agents.timeline_agent import TimelineEntry
from agents.evidence_agent import EvidenceBundle
from agents.query_agent import RATE_LIMIT_DELAY, MAX_RETRIES, INITIAL_BACKOFF

REPORT_PROMPT = """\
You are an AI investigator generating a CCTV surveillance report.
Based on the following timeline entries and evidence bundles, produce a structured
investigation report in markdown with these exact sections:

## Summary
A 2-3 sentence executive summary of the incident(s).

## Timeline of Events
A table with columns: Time | Event Type | Cameras | Confidence | Description.
Merge rows where the same incident was captured by multiple cameras.

## Evidence
List each piece of evidence with: Event description, camera(s), timestamp, thumbnail path(s), confidence.

## Confidence Notes
Assess the overall reliability of the evidence. Note any gaps (e.g. cameras with no footage,
low-confidence detections). Flag the highest-priority events for human review.

Keep the language professional and concise. This report may be reviewed by law enforcement.
"""

REPORT_FORMAT_INSTRUCTIONS = """\

After the markdown report, append a JSON object (fenced with ```json) with this structure:
{
  "summary": "...",
  "timeline": [{"time": "...", "event_type": "...", "cameras": ["..."], "confidence": 0.0, "description": "..."}],
  "evidence": [{"description": "...", "cameras": ["..."], "time": "...", "thumbnails": ["..."], "confidence": 0.0}],
  "confidence_notes": "..."
}
This JSON is for programmatic consumption — do not omit it."""


def _wait(verbose: bool = False):
    if verbose:
        print(f"  waiting {RATE_LIMIT_DELAY}s before API call...")
    time.sleep(RATE_LIMIT_DELAY)


def _api_call_gen(model, prompt: str, verbose: bool = False) -> str:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        _wait(verbose)
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                print(f"  [retry {attempt}/{MAX_RETRIES}] 429, backing off {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                raise


def _parse_structured(raw: str) -> dict[str, Any]:
    match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {
        "summary": "",
        "timeline": [],
        "evidence": [],
        "confidence_notes": "",
    }


def generate_report(
    timeline: list[TimelineEntry],
    evidence: list[EvidenceBundle],
    verbose: bool = False,
) -> tuple[str, dict[str, Any]]:
    timeline_data = [t.model_dump() for t in timeline]
    evidence_data = []
    for b in evidence:
        entry = {
            "description": b.event.description if hasattr(b.event, "description") else str(b.event),
            "cameras": b.event.sources if hasattr(b.event, "sources") else [b.event.camera],
            "time": b.event.time,
            "thumbnails": b.thumbnail_paths,
            "confidence": b.confidence,
        }
        evidence_data.append(entry)

    context = (
        "## Timeline Entries\n"
        f"```json\n{json.dumps(timeline_data, indent=2)}\n```\n\n"
        "## Evidence Bundles\n"
        f"```json\n{json.dumps(evidence_data, indent=2)}\n```"
    )

    prompt = REPORT_PROMPT + context + REPORT_FORMAT_INSTRUCTIONS

    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    raw_md = _api_call_gen(model, prompt, verbose=verbose)

    structured = _parse_structured(raw_md)
    return raw_md, structured


if __name__ == "__main__":
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    fake_events = [
        Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
              description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
        Event(time="10:30:00", camera="cam-parking", event_type="loitering",
              description="Individual standing near parked cars for an extended period", confidence=0.78),
        Event(time="12:45:00", camera="cam-lobby", event_type="theft",
              description="Unattended bag picked up and carried out by unknown person", confidence=0.95),
        Event(time="18:00:00", camera="cam-loading", event_type="intrusion",
              description="Person forced open the loading dock side door", confidence=0.88),
    ]

    from agents.timeline_agent import build_timeline
    from agents.evidence_agent import get_evidence

    timeline = build_timeline(fake_events)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()
        for name in ["cam-entrance_081500.jpg", "cam-lobby_124500.jpg", "cam-loading_180000.jpg"]:
            (frame_dir / name).write_bytes(b"")

        evidence = [get_evidence(ev, frame_dir) for ev in fake_events]

    print("=== Generating Report ===")
    markdown, structured = generate_report(timeline, evidence, verbose=True)

    print("\n" + markdown)

    print("\n=== Structured Output ===")
    print(json.dumps(structured, indent=2))
