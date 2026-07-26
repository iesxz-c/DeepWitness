"""Test whether report/query generation time scales with event count.

Generates datasets of increasing size (5, 20, 50, 85 events) and measures:
- Report generation time + approximate input tokens
- Query tool-call time for a timeline query + approximate tool output tokens
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.event import Event
from agents.store import EventStore, VectorIndex
from agents.timeline_agent import build_timeline
from agents.evidence_agent import get_evidence
from agents.report_agent import generate_report
from agents.query_agent import InvestigatorAgent


def generate_fake_events(n: int) -> list[Event]:
    """Generate n fake weapon-detection events mimicking real CCTV output."""
    import random
    random.seed(42)

    event_types = ["weapon_detected"]
    cameras = [f"cam-{i}" for i in range(1, 6)]
    classes = ["gun", "knife", "heavy-weapon"]

    events = []
    for i in range(n):
        # Spread events across a 10-minute window (00:00:00 to 00:10:00)
        total_seconds = int((i / max(n - 1, 1)) * 600)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        time_str = f"{h:02d}:{m:02d}:{s:02d}"

        cls = random.choice(classes)
        cam = random.choice(cameras)
        conf = round(random.uniform(0.50, 0.95), 2)

        events.append(Event(
            time=time_str,
            camera=cam,
            event_type="weapon_detected",
            description=f"{cls} detected (confidence {conf:.2f}, model: weapon_v1)",
            confidence=conf,
        ))
    return events


def test_report_scaling():
    """Test report generation time at different event counts."""
    sizes = [5, 20, 50, 85]
    results = []

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"REPORT TEST: {n} events")
        print(f"{'='*60}")

        events = generate_fake_events(n)

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            for ev in events[:3]:
                ts = ev.time.replace(":", "")
                (frame_dir / f"{ev.camera}_{ts}.jpg").write_bytes(b"")

            timeline = build_timeline(events)
            evidence = [get_evidence(ev, frame_dir) for ev in events[:3]]

            t0 = time.time()
            markdown, structured, provider = generate_report(timeline, evidence, verbose=False)
            elapsed = time.time() - t0

            # Count timeline/evidence JSON size
            timeline_json = json.dumps([t.model_dump() for t in timeline], indent=2)
            evidence_json = json.dumps([{
                "description": b.event.description if hasattr(b.event, "description") else str(b.event),
                "cameras": b.event.sources if hasattr(b.event, "sources") else [b.event.camera],
                "time": b.event.time,
                "thumbnails": b.thumbnail_paths,
                "confidence": b.confidence,
            } for b in evidence], indent=2)
            context = f"{timeline_json}\n{evidence_json}"
            approx_tokens = len(context) // 4

            results.append({
                "events": n,
                "timeline_entries": len(timeline),
                "approx_input_tokens": approx_tokens,
                "time_s": round(elapsed, 1),
                "provider": provider,
            })
            print(f"  DONE: {elapsed:.1f}s, ~{approx_tokens} tokens, provider={provider}")

    return results


def test_query_scaling():
    """Test query tool-call time at different event counts."""
    sizes = [5, 20, 50, 85]
    results = []

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"QUERY TEST: {n} events")
        print(f"{'='*60}")

        events = generate_fake_events(n)

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()

            store = EventStore(":memory:")
            vindex = VectorIndex()
            for ev in events:
                store.add_event(ev)
                vindex.add_event(ev)

            agent = InvestigatorAgent(vindex, store, frame_dir, verbose=False)

            t0 = time.time()
            result = agent.ask("What weapon events occurred?")
            elapsed = time.time() - t0

            results.append({
                "events": n,
                "time_s": round(elapsed, 1),
                "provider": result["provider_used"],
                "tools_called": result["tools_called"],
            })
            print(f"  DONE: {elapsed:.1f}s, provider={result['provider_used']}")

            store.close()

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SCALING TEST — report_agent and query_agent")
    print("=" * 60)

    print("\n--- REPORT GENERATION SCALING ---")
    report_results = test_report_scaling()

    print("\n\n--- QUERY GENERATION SCALING ---")
    query_results = test_query_scaling()

    print("\n\n" + "=" * 60)
    print("RESULTS TABLE — Report Generation")
    print("=" * 60)
    print(f"{'Events':>8} | {'Timeline':>10} | {'~Tokens':>10} | {'Time (s)':>10} | {'Provider':>20}")
    print("-" * 60)
    for r in report_results:
        print(f"{r['events']:>8} | {r['timeline_entries']:>10} | {r['approx_input_tokens']:>10} | {r['time_s']:>10} | {r['provider']:>20}")

    print("\n" + "=" * 60)
    print("RESULTS TABLE — Query Generation")
    print("=" * 60)
    print(f"{'Events':>8} | {'Time (s)':>10} | {'Provider':>20} | {'Tools':>10}")
    print("-" * 60)
    for r in query_results:
        print(f"{r['events']:>8} | {r['time_s']:>10} | {r['provider']:>20} | {str(r['tools_called']):>10}")

    # Scaling analysis
    if len(report_results) >= 2:
        print("\n\n--- SCALING ANALYSIS ---")
        base = report_results[0]
        for r in report_results[1:]:
            event_ratio = r["events"] / base["events"]
            time_ratio = r["time_s"] / max(base["time_s"], 0.1)
            token_ratio = r["approx_input_tokens"] / max(base["approx_input_tokens"], 1)
            print(f"  {base['events']}->{r['events']} events: "
                  f"events x{event_ratio:.1f}, "
                  f"tokens x{token_ratio:.1f}, "
                  f"time x{time_ratio:.1f}")
