"""Test the 4-tier LLM fallback chain via call_llm_with_tools().

Chain: gemini -> openrouter/free -> groq -> openrouter/paid

Scenarios invalidate real API keys with os.environ overrides and
restore them in try/finally blocks after each scenario.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.llm_client import call_llm_with_tools

REAL_KEYS = {
    k: os.environ.get(k, "")
    for k in ["GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"]
}

BAD_KEY = "bad-key-for-testing-0000000000"
QUESTION = "Say exactly: test ok"
SYSTEM_PROMPT = "Reply with exactly the words: test ok"
TOOLS = []


def dummy_executor(name, args):
    return json.dumps({"result": "dummy"})


def _set(key, value):
    os.environ[key] = value


def _restore():
    for k, v in REAL_KEYS.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _invalidate(*keys):
    for k in keys:
        _set(k, BAD_KEY)


def _call():
    return call_llm_with_tools(
        question=QUESTION,
        system_prompt=SYSTEM_PROMPT,
        tool_defs=TOOLS,
        tool_executor=dummy_executor,
        verbose=True,
    )


def run_scenario(name, invalid_keys, expected_provider, warn=None):
    """Run one scenario. Returns (actual_provider_or_None, passed)."""
    _restore()
    _invalidate(*invalid_keys)

    if warn:
        print(f"\n  *** WARNING: {warn}")

    try:
        result = _call()
        actual = result["provider_used"]
        passed = actual == expected_provider
        return actual, passed
    except RuntimeError:
        return None, expected_provider is None
    except Exception:
        return None, False
    finally:
        _restore()


results = []

# Scenario 1: Gemini key invalid -> openrouter/free
actual, passed = run_scenario(
    name="Scenario 1: Gemini key invalid",
    invalid_keys=["GEMINI_API_KEY"],
    expected_provider="openrouter/free",
)
results.append(("1. Gemini invalid -> openrouter/free", "openrouter/free", actual, passed))

# Scenario 2: Gemini + OpenRouter both invalid -> groq
# NOTE: openrouter/free and openrouter/paid share OPENROUTER_API_KEY.
# Invalidating it kills tier 2 (openrouter/free) AND tier 4 (openrouter/paid),
# so only gemini and groq survive. Gemini is bad -> groq wins.
actual, passed = run_scenario(
    name="Scenario 2: Gemini + OpenRouter invalid",
    invalid_keys=["GEMINI_API_KEY", "OPENROUTER_API_KEY"],
    expected_provider="groq",
)
results.append(("2. Gemini+OpenRouter invalid -> groq", "groq", actual, passed))

# Scenario 3: Gemini + OpenRouter + Groq all invalid -> openrouter/paid
# UNREACHABLE: openrouter/free and openrouter/paid share a single key.
# Scenario 2 already invalidated that key, killing both tier 2 and tier 4.
# With groq also dead, all 4 tiers are non-functional -> RuntimeError.
actual, passed = run_scenario(
    name="Scenario 3: Gemini + OpenRouter + Groq invalid",
    invalid_keys=["GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"],
    expected_provider=None,
    warn=(
        "openrouter/free and openrouter/paid share OPENROUTER_API_KEY.\n"
        "  Invalidating that key already killed both tier 2 and tier 4.\n"
        "  This scenario (isolate tier 4 alone) is structurally impossible.\n"
        "  Expected outcome: RuntimeError (all tiers dead)."
    ),
)
results.append(("3. Gemini+OR+Groq invalid -> RuntimeError", "RuntimeError", "RuntimeError" if actual is None else actual, passed))

# Scenario 4: ALL keys invalid -> clean RuntimeError, no raw traceback
actual, passed = run_scenario(
    name="Scenario 4: ALL keys invalid",
    invalid_keys=["GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"],
    expected_provider=None,
)
results.append(("4. ALL invalid -> RuntimeError", "RuntimeError", "RuntimeError" if actual is None else actual, passed))

# Summary table
print("\n" + "=" * 80)
print(f"{'Scenario':<45} {'Expected':<15} {'Actual':<15} {'Result'}")
print("-" * 80)
for scenario, expected, actual, passed in results:
    status = "PASS" if passed else "FAIL"
    print(f"{scenario:<45} {expected:<15} {str(actual):<15} {status}")
print("=" * 80)
total = sum(1 for *_, p in results if p)
print(f"\n{total}/{len(results)} passed")
