"""
End-to-end Query Agent + Report Generation Agent test against the REAL
persistent knowledge base (backend/events.db).

Prints everything raw and uncut:
  1. The exact query text sent in
  2. The raw evidence events retrieved from the KB (used as context)
  3. The full Query Agent natural-language answer
  4. The full Report Agent generated markdown (with evidence/JSON)

Run on Colab in /content/DeepWitness (after git clone):
  python scripts/e2e_agents_test.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from schemas.event import Event  # noqa: E402
from agents.store import EventStore, VectorIndex  # noqa: E402
from agents.query_agent import InvestigatorAgent  # noqa: E402
from agents.report_agent import generate_report  # noqa: E402
from agents.timeline_agent import build_timeline  # noqa: E402
from agents.evidence_agent import get_evidence  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # loads GEMINI_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY from .env

EVENTS_DB = str(REPO / "backend" / "events.db")

# ---------------------------------------------------------------------------
# 1. Load the REAL persistent event store + rebuild the vector index
# ---------------------------------------------------------------------------
store = EventStore(EVENTS_DB)
all_events = store.get_events()
print("=" * 80)
print(f"KNOWLEDGE BASE LOADED: {len(all_events)} events from {EVENTS_DB}")
cameras = Counter(e.camera for e in all_events)
types = Counter(e.event_type for e in all_events)
print("Cameras:", dict(cameras))
print("Event types:", dict(types))
print("=" * 80)

vindex = VectorIndex()
for ev in all_events:
    vindex.add_event(ev)

# ---------------------------------------------------------------------------
# 2. The exact investigator query (matches the actual data in the KB)
# ---------------------------------------------------------------------------
QUERY = (
    "Was any weapon or suspicious activity detected at the cam-persist-test "
    "camera? Summarize what was found and when."
)

print("\n" + "=" * 80)
print("STEP 1 - EXACT QUERY SENT IN")
print("=" * 80)
print(QUERY)
print("=" * 80)

# ---------------------------------------------------------------------------
# 3. Raw evidence the Query Agent's tool actually retrieves (context)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2 - RAW EVIDENCE RETRIEVED FROM KB (context the agent used)")
print("=" * 80)

retrieved = vindex.semantic_search(QUERY, k=5)
print("\n--- semantic_search top-5 events (raw Event records) ---")
for e in retrieved:
    print(json.dumps(e.model_dump(), indent=2))

all_tl = build_timeline(all_events)
print("\n--- timeline (raw TimelineEntry records) ---")
print(json.dumps([t.model_dump() for t in all_tl], indent=2))

knife_evs = [e for e in all_events if "knife" in e.description.lower() or "weapon" in e.description.lower()]
print("\n--- weapon/knife-related raw events ---")
for e in knife_evs:
    print(json.dumps(e.model_dump(), indent=2))
print("=" * 80)

# ---------------------------------------------------------------------------
# 4. Run the Query Agent end-to-end (LLM with tools) - raw answer
# ---------------------------------------------------------------------------
agent = InvestigatorAgent(vindex, store, Path("."), verbose=True)
print("\n" + "=" * 80)
print("STEP 3 - QUERY AGENT OUTPUT (LLM, tool calls + raw answer)")
print("=" * 80)
result = agent.ask(QUERY)
print("\n[TOOLS CALLED BY AGENT]")
for log in agent.tool_log:
    print(json.dumps(log, indent=2))
print(f"\n[PROVIDER USED]: {result['provider_used']}")
print("\n[FULL RAW ANSWER]")
print(result["answer"])
print("=" * 80)

# ---------------------------------------------------------------------------
# 5. Run the Report Generation Agent on the real events - raw markdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4 - REPORT GENERATION AGENT OUTPUT (raw markdown)")
print("=" * 80)
timeline_entries = build_timeline(all_events)
evidence_bundles = [get_evidence(ev, Path(".")) for ev in all_events]
raw_md, structured, provider = generate_report(timeline_entries, evidence_bundles, verbose=True)
print(f"\n[PROVIDER USED]: {provider}")
print("\n[FULL RAW REPORT MARKDOWN]")
print(raw_md)
print("\n[STRUCTURED JSON EXTRACTED]")
print(json.dumps(structured, indent=2))
print("=" * 80)

store.close()
