"""Test report_agent fallback: invalidate OpenRouter (new tier 1), confirm
Groq (tier 2) generates a valid report with all 4 sections and a structured dict."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schemas.event import Event
from agents.timeline_agent import build_timeline
from agents.evidence_agent import get_evidence
from agents.report_agent import generate_report

REAL_OPENROUTER = os.environ.get("OPENROUTER_API_KEY", "")
EXPECTED_PROVIDER = "groq"
REQUIRED_SECTIONS = ["## Summary", "## Timeline of Events", "## Evidence", "## Confidence Notes"]
REQUIRED_STRUCT_KEYS = ["summary", "timeline", "evidence", "confidence_notes"]

events = [
    Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
          description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
    Event(time="10:30:00", camera="cam-parking", event_type="loitering",
          description="Individual standing near parked cars for an extended period", confidence=0.78),
    Event(time="12:45:00", camera="cam-lobby", event_type="theft",
          description="Unattended bag picked up and carried out by unknown person", confidence=0.95),
    Event(time="18:00:00", camera="cam-loading", event_type="intrusion",
          description="Person forced open the loading dock side door", confidence=0.88),
]


def _restore():
    if REAL_OPENROUTER:
        os.environ["OPENROUTER_API_KEY"] = REAL_OPENROUTER
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)


try:
    os.environ["OPENROUTER_API_KEY"] = "bad-key-for-testing-0000000000"

    timeline = build_timeline(events)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()
        for name in ["cam-entrance_081500.jpg", "cam-lobby_124500.jpg", "cam-loading_180000.jpg"]:
            (frame_dir / name).write_bytes(b"\xff")
        evidence = [get_evidence(ev, frame_dir) for ev in events]

    print(f"OpenRouter key: INVALIDATED (forcing fallback to Groq)")
    print(f"Timeline entries: {len(timeline)}")
    print(f"Evidence bundles: {len(evidence)}")
    print("-" * 60)

    markdown, structured, provider = generate_report(timeline, evidence, verbose=True)

    print(f"\n{'='*60}")
    print(f"provider_used:  {provider}")
    print(f"markdown len:   {len(markdown)} chars")
    print(f"struct keys:    {list(structured.keys())}")
    print(f"{'='*60}")
    print("\n--- Markdown preview (first 500 chars) ---")
    print(markdown[:500])
    print(f"\n--- Structured output ---")
    print(json.dumps(structured, indent=2)[:500])

    # --- checks ---
    errors = []

    if provider != EXPECTED_PROVIDER:
        errors.append(f"provider_used: expected '{EXPECTED_PROVIDER}', got '{provider}'")

    for section in REQUIRED_SECTIONS:
        if section.lower() not in markdown.lower():
            errors.append(f"missing markdown section: '{section}'")

    if not isinstance(structured, dict):
        errors.append(f"structured output is not a dict: {type(structured)}")
    else:
        for key in REQUIRED_STRUCT_KEYS:
            if key not in structured:
                errors.append(f"missing structured key: '{key}'")
            elif not structured[key]:
                errors.append(f"structured key '{key}' is empty")

    if isinstance(structured.get("timeline"), list) and len(structured["timeline"]) == 0:
        errors.append("structured.timeline is empty")

    if isinstance(structured.get("evidence"), list) and len(structured["evidence"]) == 0:
        errors.append("structured.evidence is empty")

    print(f"\n--- Verification ---")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED")
        sys.exit(0)

finally:
    _restore()
