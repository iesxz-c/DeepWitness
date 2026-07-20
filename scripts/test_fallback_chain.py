import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.llm_client import call_llm_with_tools

REAL_KEYS = {}
for k in ["GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "AINATIVE_API_KEY"]:
    REAL_KEYS[k] = os.environ.get(k, "")

BAD_KEY = "bad-key-for-testing-0000000000"

QUESTION = "Say exactly: test ok"
SYSTEM_PROMPT = "Reply with exactly the words: test ok"
TOOLS = []


def dummy_executor(name, args):
    return json.dumps({"result": "dummy"})


def _set(key, value):
    if value:
        os.environ[key] = value
    else:
        os.environ.pop(key, None)


def _restore():
    for k, v in REAL_KEYS.items():
        _set(k, v)


def run_scenario(name, invalid_keys, expected_provider):
    _restore()
    for key_name in invalid_keys:
        _set(key_name, BAD_KEY)

    print(f"\n--- {name} ---")
    print(f"  Invalidated: {invalid_keys}")
    try:
        result = call_llm_with_tools(
            question=QUESTION,
            system_prompt=SYSTEM_PROMPT,
            tool_defs=TOOLS,
            tool_executor=dummy_executor,
            verbose=True,
        )
        actual = result["provider_used"]
        print(f"  >> Result: {actual}")
        _restore()
        return actual
    except RuntimeError as e:
        print(f"  >> All providers failed: {e}")
        _restore()
        return None


results = []

# Scenario 1: Gemini invalid -> openrouter/free
actual = run_scenario(
    "Scenario 1: Gemini invalid",
    invalid_keys=["GEMINI_API_KEY"],
    expected_provider="openrouter/free",
)
results.append(("1. Gemini invalid", "openrouter/free", actual, actual == "openrouter/free"))

# Scenario 2: Gemini + OpenRouter invalid -> groq
# NOTE: OpenRouter free/paid share OPENROUTER_API_KEY, so invalidating it
# kills both tier 2 (openrouter/free) AND tier 5 (openrouter/paid).
actual = run_scenario(
    "Scenario 2: Gemini + OpenRouter invalid",
    invalid_keys=["GEMINI_API_KEY", "OPENROUTER_API_KEY"],
    expected_provider="groq",
)
results.append(("2. Gemini+OpenRouter invalid", "groq", actual, actual == "groq"))

# Scenario 3: Gemini + OpenRouter + Groq invalid -> ainative
actual = run_scenario(
    "Scenario 3: Gemini+OpenRouter+Groq invalid",
    invalid_keys=["GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"],
    expected_provider="ainative",
)
# ainative key may or may not be valid
ainative_ok = actual == "ainative"
results.append(("3. Gemini+OpenRouter+Groq invalid", "ainative", actual, ainative_ok))

# Scenario 4: ainative invalid -> openrouter/paid
# WARNING: openrouter/free and openrouter/paid share OPENROUTER_API_KEY.
# With a valid OpenRouter key, tier 2 (openrouter/free) always succeeds
# before tier 5 (openrouter/paid) is reached.
# Reaching tier 5 independently is structurally impossible with a shared key.
print("\n--- Scenario 4: ainative invalid (OpenRouter valid) ---")
print("  WARNING: openrouter/free and openrouter/paid share OPENROUTER_API_KEY.")
print("  Tier 2 always succeeds before tier 5 is reached.")
print("  This scenario CANNOT independently reach openrouter/paid.")
_restore()
_set("AINATIVE_API_KEY", BAD_KEY)
try:
    result = call_llm_with_tools(
        question=QUESTION,
        system_prompt=SYSTEM_PROMPT,
        tool_defs=TOOLS,
        tool_executor=dummy_executor,
        verbose=False,
    )
    actual = result["provider_used"]
    print(f"  >> Actual: {actual} (tier 2 absorbed before tier 5)")
    _restore()
    results.append(("4. ainative invalid", "openrouter/paid (impossible)", actual, actual == "openrouter/free"))
except Exception as e:
    print(f"  >> Unexpected error: {e}")
    _restore()
    results.append(("4. ainative invalid", "openrouter/paid (impossible)", None, False))

# Scenario 5: ALL keys invalid -> clean error
print("\n--- Scenario 5: ALL keys invalid ---")
_restore()
for k in REAL_KEYS:
    _set(k, BAD_KEY)
try:
    call_llm_with_tools(
        question=QUESTION,
        system_prompt=SYSTEM_PROMPT,
        tool_defs=TOOLS,
        tool_executor=dummy_executor,
        verbose=False,
    )
    print("  >> FAIL: should have raised RuntimeError")
    results.append(("5. ALL invalid", "RuntimeError", None, False))
except RuntimeError as e:
    print(f"  >> Clean RuntimeError: {e}")
    results.append(("5. ALL invalid", "RuntimeError", "RuntimeError", True))
except Exception as e:
    print(f"  >> FAIL: raw {type(e).__name__}: {e}")
    results.append(("5. ALL invalid", "RuntimeError", type(e).__name__, False))
finally:
    _restore()

# Summary table
print("\n" + "=" * 80)
print(f"{'Scenario':<40} {'Expected':<20} {'Actual':<15} {'Result'}")
print("-" * 80)
for scenario, expected, actual, passed in results:
    status = "PASS" if passed else "FAIL"
    print(f"{scenario:<40} {expected:<20} {str(actual):<15} {status}")
print("=" * 80)
total = sum(1 for *_, p in results if p)
print(f"\n{total}/{len(results)} passed")
