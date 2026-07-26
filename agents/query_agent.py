import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event
from agents.store import EventStore, VectorIndex
from agents.timeline_agent import TimelineEntry, build_timeline
from agents.evidence_agent import EvidenceBundle, get_evidence
from agents.llm_client import call_llm_with_tools

SYSTEM_PROMPT = """\
You are an AI investigator assistant for a CCTV surveillance system.
You have access to tools for searching events, building timelines, and retrieving evidence.
Use the tools to answer the user's question. You may call multiple tools in sequence.
Always provide a clear, concise final answer based on the tool results."""

TOOL_DEFS = [
    {
        "name": "search_events",
        "description": "Semantic search across all events by natural language query. Returns the most relevant events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "k": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_timeline",
        "description": "Build a chronological timeline of events within a time range, merging duplicates across cameras.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "Start time in HH:MM:SS format"},
                "end_time": {"type": "string", "description": "End time in HH:MM:SS format"},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "get_evidence",
        "description": "Look up video frame evidence for a specific camera and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "description": "Camera name"},
                "time": {"type": "string", "description": "Time in HH:MM:SS format"},
            },
            "required": ["camera", "time"],
        },
    },
]


class InvestigatorAgent:
    def __init__(
        self,
        vector_index: VectorIndex,
        event_store: EventStore,
        frame_cache_dir: str | Path,
        verbose: bool = False,
    ):
        self.vector_index = vector_index
        self.event_store = event_store
        self.frame_cache_dir = Path(frame_cache_dir)
        self.verbose = verbose
        self.tool_log: list[dict[str, Any]] = []
        self.provider_used: str | None = None

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        self.tool_log.append({"tool": name, "args": args})

        if name == "search_events":
            results = self.vector_index.semantic_search(
                query=args["query"],
                k=int(args.get("k", 5)),
            )
            result_json = json.dumps([e.model_dump() for e in results], indent=2)
            approx_tokens = len(result_json) // 4
            print(f"  [TOKENS] search_events result: ~{approx_tokens} tokens "
                  f"({len(results)} events returned)")
            return result_json

        elif name == "get_timeline":
            all_events = self.event_store.get_events(
                start_time=args["start_time"],
                end_time=args["end_time"],
            )
            timeline = build_timeline(all_events)
            MAX_TIMELINE_ENTRIES = 20
            truncated = False
            if len(timeline) > MAX_TIMELINE_ENTRIES:
                timeline = timeline[:MAX_TIMELINE_ENTRIES]
                truncated = True
            result_json = json.dumps([t.model_dump() for t in timeline], indent=2)
            if truncated:
                result_json = json.dumps({
                    "timeline": json.loads(result_json),
                    "note": f"Showing first {MAX_TIMELINE_ENTRIES} of {len(all_events)} total events. "
                            f"Use search_events for targeted queries.",
                })
            approx_tokens = len(result_json) // 4
            print(f"  [TOKENS] get_timeline result: ~{approx_tokens} tokens "
                  f"({len(all_events)} raw events -> {len(timeline)} timeline entries"
                  f"{', TRUNCATED' if truncated else ''})")
            return result_json

        elif name == "get_evidence":
            events = self.event_store.get_events(
                camera=args["camera"],
                start_time=args["time"],
                end_time=args["time"],
            )
            if not events:
                return json.dumps({"error": "no event found for that camera and time"})

            bundle = get_evidence(events[0], self.frame_cache_dir)
            result_json = json.dumps({
                "confidence": bundle.confidence,
                "thumbnail_paths": bundle.thumbnail_paths,
                "event": bundle.event.model_dump(),
            }, indent=2)
            approx_tokens = len(result_json) // 4
            print(f"  [TOKENS] get_evidence result: ~{approx_tokens} tokens")
            return result_json

        return json.dumps({"error": f"unknown tool: {name}"})

    def ask(self, question: str) -> dict[str, Any]:
        self.tool_log = []
        approx_tokens = len(question) // 4
        print(f"  [TOKENS] query_agent question: ~{approx_tokens} tokens")
        t0 = time.time()
        result = call_llm_with_tools(
            question=question,
            system_prompt=SYSTEM_PROMPT,
            tool_defs=TOOL_DEFS,
            tool_executor=self._execute_tool,
            verbose=self.verbose,
        )
        elapsed = time.time() - t0
        self.provider_used = result["provider_used"]
        print(f"  [TIMING] query_agent total: {elapsed:.1f}s (provider: {result['provider_used']})")
        return result


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

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()
        for name in [
            "cam-lobby_124500.jpg", "cam-entrance_081500.jpg",
            "cam-loading_180000.jpg",
        ]:
            (frame_dir / name).write_bytes(b"")

        store = EventStore(":memory:")
        vindex = VectorIndex()
        for ev in fake_events:
            store.add_event(ev)
            vindex.add_event(ev)

        agent = InvestigatorAgent(vindex, store, frame_dir, verbose=True)

        questions = [
            "Who was near the lobby around 10:15?",
            "What happened between 2pm and 3pm?",
            "Is there any evidence for the intrusion event?",
        ]

        for q in questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print(f"{'='*60}")
            agent.tool_log.clear()
            result = agent.ask(q)
            print(f"\nProvider: {result['provider_used']}")
            print(f"Tools called: {result['tools_called']}")
            print(f"Answer: {result['answer']}")

        store.close()
