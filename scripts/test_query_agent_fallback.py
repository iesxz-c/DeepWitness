"""Test query_agent fallback: invalidate OpenRouter (new tier 1), confirm
Groq (tier 2) picks up and runs search_events -> get_evidence tool chain."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.store import EventStore, VectorIndex
from agents.query_agent import InvestigatorAgent
from schemas.event import Event

REAL_OPENROUTER = os.environ.get("OPENROUTER_API_KEY", "")
QUESTION = "Find the intrusion event and retrieve the video frame evidence (thumbnails) for it."
EXPECTED_PROVIDER = "groq"
REQUIRED_TOOLS = ["search_events", "get_evidence"]

events = [
    Event(
        time="08:15:00", camera="cam-entrance", event_type="intrusion",
        description="Person jumped over the perimeter fence near the east gate",
        confidence=0.92,
    ),
    Event(
        time="12:45:00", camera="cam-lobby", event_type="theft",
        description="Unattended bag picked up and carried out by unknown person",
        confidence=0.95,
    ),
    Event(
        time="18:00:00", camera="cam-loading", event_type="intrusion",
        description="Person forced open the loading dock side door",
        confidence=0.88,
    ),
]


def _restore():
    if REAL_OPENROUTER:
        os.environ["OPENROUTER_API_KEY"] = REAL_OPENROUTER
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)


try:
    os.environ["OPENROUTER_API_KEY"] = "bad-key-for-testing-0000000000"

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()
        for name in [
            "cam-entrance_081500.jpg",
            "cam-lobby_124500.jpg",
            "cam-loading_180000.jpg",
        ]:
            (frame_dir / name).write_bytes(b"\xff")

        store = EventStore(":memory:")
        vindex = VectorIndex()
        for ev in events:
            store.add_event(ev)
            vindex.add_event(ev)

        agent = InvestigatorAgent(vindex, store, frame_dir, verbose=True)

        print(f"\nQ: {QUESTION}")
        print(f"OpenRouter key: INVALIDATED (forcing fallback to Groq)")
        print("-" * 60)

        result = agent.ask(QUESTION)

        provider = result["provider_used"]
        tools = result["tools_called"]
        answer = result["answer"]

        print(f"\n{'='*60}")
        print(f"provider_used:  {provider}")
        print(f"tools_called:   {tools}")
        print(f"answer:         {answer[:300]}")
        print(f"{'='*60}")

        # --- checks ---
        errors = []

        if provider != EXPECTED_PROVIDER:
            errors.append(
                f"provider_used: expected '{EXPECTED_PROVIDER}', got '{provider}'"
            )

        for tool in REQUIRED_TOOLS:
            if tool not in tools:
                errors.append(
                    f"missing tool '{tool}' in tools_called {tools}"
                )

        if not answer or len(answer.strip()) < 10:
            errors.append(f"answer too short or empty: {answer!r}")

        answer_lower = answer.lower()
        if "intrusion" not in answer_lower and "fence" not in answer_lower and "evidence" not in answer_lower:
            errors.append(
                "answer does not reference the intrusion event or evidence"
            )

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
    store.close()
