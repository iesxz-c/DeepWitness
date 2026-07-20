import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"


def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def test_health():
    r = get("/health")
    assert r == {"status": "ok"}, f"health failed: {r}"
    return "PASS", ""


def test_events():
    r = get("/events")
    assert isinstance(r, list) and len(r) == 7, f"expected 7 events, got {len(r)}"
    assert r[0]["camera"] == "cam-entrance"
    return "PASS", ""


def test_events_filtered():
    r = get("/events?camera=cam-lobby")
    assert len(r) == 2, f"expected 2 lobby events, got {len(r)}"
    return "PASS", ""


def test_timeline():
    r = get("/timeline")
    assert len(r) == 6, f"expected 6 timeline entries (merge), got {len(r)}"
    theft = [e for e in r if e["event_type"] == "theft"]
    assert len(theft) == 1, f"expected 1 merged theft, got {len(theft)}"
    assert "cam-lobby" in theft[0]["sources"] and "cam-entrance" in theft[0]["sources"]
    return "PASS", ""


def test_query():
    r = post_json("/query", {"question": "What happened between 2pm and 3pm?"})
    assert "answer" in r, f"missing answer: {r}"
    assert "tools_called" in r, f"missing tools_called: {r}"
    assert "provider_used" in r, f"missing provider_used: {r}"
    return "PASS", r["provider_used"]


def test_report():
    r = get("/report")
    assert "markdown" in r, f"missing markdown: {r}"
    assert "structured" in r, f"missing structured: {r}"
    assert "provider_used" in r, f"missing provider_used: {r}"
    assert len(r["markdown"]) > 100, f"markdown too short: {len(r['markdown'])}"
    return "PASS", r["provider_used"]


if __name__ == "__main__":
    tests = [
        ("GET /health", test_health),
        ("GET /events", test_events),
        ("GET /events?camera=cam-lobby", test_events_filtered),
        ("GET /timeline", test_timeline),
        ("POST /query", test_query),
        ("GET /report", test_report),
    ]

    results = []
    for name, fn in tests:
        try:
            status, provider = fn()
            prov_str = f" (provider={provider})" if provider else ""
            results.append((name, status, prov_str))
            print(f"  {status} {name}{prov_str}")
        except Exception as e:
            results.append((name, f"FAIL: {e}", ""))
            print(f"  FAIL {name}: {e}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{passed}/{len(results)} passed")
