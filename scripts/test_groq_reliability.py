"""Stress-test openai/gpt-oss-20b on Groq for multi-round tool-calling reliability.

Forces OpenRouter key invalid so Groq becomes tier 1 via call_llm_with_tools.
Tests 1-call, 2-call, 3+-call, repeated 3+-call, and hallucination-resistance scenarios.
"""

import sys
import os
import json
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from schemas.event import Event
from agents.store import EventStore, VectorIndex
from agents.llm_client import call_llm_with_tools

# ---------------------------------------------------------------------------
# Setup: seed events + tool executor
# ---------------------------------------------------------------------------

FAKE_EVENTS = [
    Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
          description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
    Event(time="10:30:00", camera="cam-parking", event_type="loitering",
          description="Individual standing near parked cars for an extended period", confidence=0.78),
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

store = EventStore(":memory:")
vindex = VectorIndex()
for ev in FAKE_EVENTS:
    store.add_event(ev)
    vindex.add_event(ev)


def tool_executor(name, args):
    if name == "search_events":
        results = vindex.semantic_search(query=args["query"], k=int(args.get("k", 5)))
        return json.dumps([e.model_dump() for e in results], indent=2)
    elif name == "get_timeline":
        events = store.get_events(start_time=args["start_time"], end_time=args["end_time"])
        return json.dumps([e.model_dump() for e in events], indent=2)
    elif name == "get_evidence":
        events = store.get_events(camera=args["camera"], start_time=args["time"], end_time=args["time"])
        if not events:
            return json.dumps({"error": "no event found"})
        return json.dumps({"event": events[0].model_dump(), "confidence": events[0].confidence})
    return json.dumps({"error": f"unknown tool: {name}"})


TOOLS = [
    {
        "name": "search_events",
        "description": "Semantic search across all events by natural language query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "k": {"type": "integer", "description": "Number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_timeline",
        "description": "Build a chronological timeline within a time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "HH:MM:SS"},
                "end_time": {"type": "string", "description": "HH:MM:SS"},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "get_evidence",
        "description": "Look up video frame evidence for a camera and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "description": "Camera name"},
                "time": {"type": "string", "description": "HH:MM:SS"},
            },
            "required": ["camera", "time"],
        },
    },
]

SYSTEM_PROMPT = "You are a CCTV investigator. Answer concisely using the tools available."

# ---------------------------------------------------------------------------
# Force OpenRouter invalid so Groq is tier 1
# ---------------------------------------------------------------------------
saved_or = os.environ.pop("OPENROUTER_API_KEY", None)
os.environ["OPENROUTER_API_KEY"] = "bad-key-for-testing"


def run_test(test_name, question, expected_min_rounds=1, expected_max_rounds=10):
    """Run a single test case and return result dict."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = call_llm_with_tools(
            question=question,
            system_prompt=SYSTEM_PROMPT,
            tool_defs=TOOLS,
            tool_executor=tool_executor,
            verbose=True,
        )
        elapsed = time.time() - t0
        answer = result["answer"]
        tools_called = result["tools_called"]
        provider = result["provider_used"]
        rounds = len(tools_called)

        coherent = len(answer.strip()) > 10
        in_range = expected_min_rounds <= rounds <= expected_max_rounds

        print(f"  provider: {provider}")
        print(f"  tool_rounds: {rounds}")
        print(f"  tools_called: {tools_called}")
        print(f"  answer_preview: {answer[:200]}")
        print(f"  elapsed: {elapsed:.1f}s")

        return {
            "name": test_name,
            "provider": provider,
            "rounds": rounds,
            "tools_called": tools_called,
            "coherent": coherent,
            "in_range": in_range,
            "elapsed": elapsed,
            "error": None,
            "answer_preview": answer[:100],
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {type(e).__name__}: {e}")
        return {
            "name": test_name,
            "provider": "groq",
            "rounds": 0,
            "tools_called": [],
            "coherent": False,
            "in_range": False,
            "elapsed": elapsed,
            "error": f"{type(e).__name__}: {e}",
            "answer_preview": "",
        }


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
results = []

# Test 1: 1 tool call (baseline)
results.append(run_test(
    "1-call baseline",
    "How many events are in the system?",
    expected_min_rounds=1, expected_max_rounds=2,
))

# Test 2: 2 sequential tool calls
results.append(run_test(
    "2-call sequential",
    "Find intrusion events, then get the timeline from 08:00:00 to 09:00:00",
    expected_min_rounds=2, expected_max_rounds=3,
))

# Test 3: 3+ sequential tool calls (the scenario that broke llama-3.3-70b)
results.append(run_test(
    "3-call chained",
    "Search for all weapon events, get the timeline from 08:00:00 to 23:00:00, "
    "then get evidence for cam-entrance at 08:15:00",
    expected_min_rounds=3, expected_max_rounds=5,
))

# Test 4: Repeat test 3 five times (flakiness check)
print(f"\n{'#'*60}")
print("# REPEAT TEST 4: 3-call chained x5 (flakiness check)")
print(f"{'#'*60}")

for i in range(1, 6):
    results.append(run_test(
        f"3-call chained (run {i}/5)",
        "Search for all weapon events, get the timeline from 08:00:00 to 23:00:00, "
        "then get evidence for cam-entrance at 08:15:00",
        expected_min_rounds=3, expected_max_rounds=5,
    ))

# Test 5: Hallucination resistance (no weather tool exists)
results.append(run_test(
    "hallucination resistance",
    "What's the weather like near the incidents?",
    expected_min_rounds=0, expected_max_rounds=3,
))

# ---------------------------------------------------------------------------
# Restore key
# ---------------------------------------------------------------------------
if saved_or:
    os.environ["OPENROUTER_API_KEY"] = saved_or

store.close()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Test':<30} {'Rounds':>7} {'Result':>8} {'Time':>6}  Notes")
print(f"{'-'*30} {'-'*7} {'-'*8} {'-'*6}  {'-'*30}")

total_pass = 0
total_fail = 0

for r in results:
    if r["error"]:
        result_str = "FAIL"
        notes = r["error"][:50]
        total_fail += 1
    elif r["coherent"] and r["in_range"]:
        result_str = "PASS"
        notes = r["answer_preview"][:50]
        total_pass += 1
    else:
        result_str = "WARN"
        notes = f"coherent={r['coherent']} in_range={r['in_range']}"
        total_fail += 1

    print(f"{r['name']:<30} {r['rounds']:>5}   {result_str:>8} {r['elapsed']:>5.1f}s  {notes}")

print(f"{'-'*30} {'-'*7} {'-'*8} {'-'*6}")
print(f"{'TOTAL':<30} {'':>7} {total_pass:>3}Pass {total_fail:>3}Fail")
print(f"{'='*80}")
