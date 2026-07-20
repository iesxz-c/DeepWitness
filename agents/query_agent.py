import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event
from agents.store import EventStore, VectorIndex
from agents.timeline_agent import TimelineEntry, build_timeline
from agents.evidence_agent import EvidenceBundle, get_evidence

SYSTEM_PROMPT = """\
You are an AI investigator assistant for a CCTV surveillance system.
You have access to tools for searching events, building timelines, and retrieving evidence.
Use the tools to answer the user's question. You may call multiple tools in sequence.
Always provide a clear, concise final answer based on the tool results."""

TOOL_DECLARATIONS = [
    genai.protos.FunctionDeclaration(
        name="search_events",
        description="Semantic search across all events by natural language query. Returns the most relevant events.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Natural language search query"),
                "k": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of results to return"),
            },
            required=["query"],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name="get_timeline",
        description="Build a chronological timeline of events within a time range, merging duplicates across cameras.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "start_time": genai.protos.Schema(type=genai.protos.Type.STRING, description="Start time in HH:MM:SS format"),
                "end_time": genai.protos.Schema(type=genai.protos.Type.STRING, description="End time in HH:MM:SS format"),
            },
            required=["start_time", "end_time"],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name="get_evidence",
        description="Look up video frame evidence for a specific camera and time.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "camera": genai.protos.Schema(type=genai.protos.Type.STRING, description="Camera name"),
                "time": genai.protos.Schema(type=genai.protos.Type.STRING, description="Time in HH:MM:SS format"),
            },
            required=["camera", "time"],
        ),
    ),
]

GEMINI_TOOLS = [genai.protos.Tool(function_declarations=TOOL_DECLARATIONS)]


RATE_LIMIT_DELAY = 13  # seconds between API calls (free tier = 5 RPM)
MAX_RETRIES = 5
INITIAL_BACKOFF = 15  # seconds


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
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS,
        )
        self.tool_log: list[dict[str, Any]] = []

    def _wait(self, label: str = "API call"):
        if self.verbose:
            print(f"  waiting {RATE_LIMIT_DELAY}s before next {label}...")
        time.sleep(RATE_LIMIT_DELAY)

    def _api_call(self, chat, content):
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait("API call")
            try:
                return chat.send_message(content)
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    print(f"  [retry {attempt}/{MAX_RETRIES}] 429 rate limit, backing off {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                else:
                    raise

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        self.tool_log.append({"tool": name, "args": args})

        if name == "search_events":
            results = self.vector_index.semantic_search(
                query=args["query"],
                k=int(args.get("k", 5)),
            )
            return json.dumps([e.model_dump() for e in results], indent=2)

        elif name == "get_timeline":
            all_events = self.event_store.get_events(
                start_time=args["start_time"],
                end_time=args["end_time"],
            )
            timeline = build_timeline(all_events)
            return json.dumps([t.model_dump() for t in timeline], indent=2)

        elif name == "get_evidence":
            events = self.event_store.get_events(
                camera=args["camera"],
                start_time=args["time"],
                end_time=args["time"],
            )
            if not events:
                return json.dumps({"error": "no event found for that camera and time"})

            bundle = get_evidence(events[0], self.frame_cache_dir)
            return json.dumps({
                "confidence": bundle.confidence,
                "thumbnail_paths": bundle.thumbnail_paths,
                "event": bundle.event.model_dump(),
            }, indent=2)

        return json.dumps({"error": f"unknown tool: {name}"})

    def ask(self, question: str) -> str:
        self.tool_log = []
        chat = self.model.start_chat(history=[])
        response = self._api_call(chat, question)

        while True:
            function_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            if not function_calls:
                text_parts = [
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text") and part.text
                ]
                return "\n".join(text_parts)

            function_responses = []
            for fc in function_calls:
                name = fc.name
                args = dict(fc.args)
                print(f"  [tool] {name}({json.dumps(args)})")
                result = self._execute_tool(name, args)
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=name,
                            response={"result": result},
                        )
                    )
                )

            response = self._api_call(chat, function_responses)


if __name__ == "__main__":
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

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
            answer = agent.ask(q)
            print(f"\nTools called: {[t['tool'] for t in agent.tool_log]}")
            print(f"Answer: {answer}")

        store.close()
