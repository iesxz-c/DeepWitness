"""Test backend fallback through HTTP endpoints.

Launches a local FastAPI server with an INVALID OPENROUTER_API_KEY
(new tier 1), then tests POST /query and GET /report against the
running server.  The server is killed when done (or on error).
"""

import json
import os
import subprocess
import sys
import time

import requests

BASE = "http://localhost:8000"
TIMEOUT = 120
CWD = os.path.join(os.path.dirname(__file__), "..")


def main():
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "sk-invalid-test-key"
    # keep GEMINI and GROQ from the real environment

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        cwd=CWD,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server()
        results = []
        results.append(_test_query())
        results.append(_test_report())
        _print_summary(results)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_server():
    print("Waiting for server to start...")
    for i in range(30):
        time.sleep(1)
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s\n")
                return
        except Exception:
            pass
    print("FATAL: server did not start in 30s")
    sys.exit(1)


def _test_query():
    print("=" * 60)
    print("Scenario 1: POST /query  (OpenRouter invalid -> Groq)")
    print("=" * 60)
    try:
        r = requests.post(
            f"{BASE}/query",
            json={"question": "Is there any evidence for the intrusion event?"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        body = r.json()

        provider = body.get("provider_used", "")
        tools = body.get("tools_called", [])
        answer = body.get("answer", "")

        print(f"  provider_used: {provider}")
        print(f"  tools_called:  {tools}")
        print(f"  answer:        {answer[:200]}...")

        errs = []
        if provider not in ("groq",):
            errs.append(f"provider_used: expected 'groq', got '{provider}'")
        if not answer or len(answer.strip()) < 10:
            errs.append("answer is empty or too short")
        if not isinstance(tools, list) or len(tools) == 0:
            errs.append("tools_called is empty")

        if errs:
            for e in errs:
                print(f"  FAIL: {e}")
            return ("1. POST /query", False)
        print("  PASS")
        return ("1. POST /query", True)

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return ("1. POST /query", False)


def _test_report():
    print(f"\n{'=' * 60}")
    print("Scenario 2: GET /report  (OpenRouter invalid -> Groq)")
    print("=" * 60)
    try:
        r = requests.get(f"{BASE}/report", timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()

        provider = body.get("provider_used", "")
        markdown = body.get("markdown", "")
        structured = body.get("structured", {})

        print(f"  provider_used:   {provider}")
        print(f"  markdown len:    {len(markdown)} chars")
        skeys = list(structured.keys()) if isinstance(structured, dict) else type(structured)
        print(f"  structured keys: {skeys}")

        errs = []
        if provider not in ("groq",):
            errs.append(f"provider_used: expected 'groq', got '{provider}'")
        if not markdown or len(markdown) < 100:
            errs.append(f"markdown too short ({len(markdown)} chars)")
        if not isinstance(structured, dict):
            errs.append(f"structured is not a dict: {type(structured)}")
        else:
            for key in ["summary", "timeline", "evidence", "confidence_notes"]:
                if key not in structured:
                    errs.append(f"missing structured key: '{key}'")
                elif not structured[key]:
                    errs.append(f"structured key '{key}' is empty")
        for section in [
            "## Summary", "## Timeline of Events",
            "## Evidence", "## Confidence Notes",
        ]:
            if section.lower() not in markdown.lower():
                errs.append(f"missing markdown section: '{section}'")

        if errs:
            for e in errs:
                print(f"  FAIL: {e}")
            return ("2. GET /report", False)
        print("  PASS")
        return ("2. GET /report", True)

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return ("2. GET /report", False)


def _print_summary(results):
    print(f"\n{'=' * 60}")
    print(f"{'Scenario':<25} {'Result'}")
    print("-" * 60)
    for name, passed in results:
        print(f"{name:<25} {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    total = sum(1 for _, p in results if p)
    print(f"\n{total}/{len(results)} passed")


if __name__ == "__main__":
    main()
