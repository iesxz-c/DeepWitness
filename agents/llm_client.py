import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import google.generativeai as genai
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE_LIMIT_DELAY = 13
MAX_RETRIES = 2
INITIAL_BACKOFF = 10

GEMINI_MAX_RETRIES = 2
GEMINI_INITIAL_BACKOFF = 10

GEMINI_MODEL = "gemini-2.5-flash"
OPENROUTER_FREE_MODEL = "openrouter/free"
OPENROUTER_PAID_MODEL = "openai/gpt-4o-mini"
GROQ_MODEL = "llama-3.3-70b-versatile"
AINATIVE_MODEL = "gpt-4o-mini"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/iesxz-c/fnl",
    "X-Title": "CCTV Investigation System",
}


def _get_gemini_tools(tool_defs: list[dict]) -> list:
    TYPE_MAP = {
        "string": genai.protos.Type.STRING,
        "integer": genai.protos.Type.INTEGER,
        "number": genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
        "object": genai.protos.Type.OBJECT,
    }
    declarations = []
    for t in tool_defs:
        props = {}
        for pname, pdef in t["parameters"].get("properties", {}).items():
            props[pname] = genai.protos.Schema(
                type=TYPE_MAP.get(pdef.get("type", "string"), genai.protos.Type.STRING),
                description=pdef.get("description", ""),
            )
        required = t["parameters"].get("required", [])
        declarations.append(genai.protos.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties=props,
                required=required,
            ),
        ))
    return [genai.protos.Tool(function_declarations=declarations)]


def _gemini_tool_loop(
    system_prompt: str,
    question: str,
    tool_defs: list[dict],
    tool_executor: Callable[[str, dict[str, Any]], str],
    verbose: bool = False,
) -> tuple[str, list[str]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    gemini_tools = _get_gemini_tools(tool_defs)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        tools=gemini_tools,
    )
    chat = model.start_chat(history=[])
    tools_called = []

    def _wait():
        if verbose:
            print(f"  [gemini] waiting {RATE_LIMIT_DELAY}s...")
        time.sleep(RATE_LIMIT_DELAY)

    def _call(content):
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            if attempt > 1:
                if verbose:
                    print(f"  [gemini] waiting {RATE_LIMIT_DELAY}s before retry...")
                time.sleep(RATE_LIMIT_DELAY)
            try:
                return chat.send_message(content)
            except Exception as e:
                err_str = str(e)
                if "400" in err_str or "API_KEY_INVALID" in err_str:
                    raise
                if "429" in err_str and attempt < GEMINI_MAX_RETRIES:
                    if verbose:
                        print(f"  [gemini] retry {attempt}/{GEMINI_MAX_RETRIES}, backoff {GEMINI_INITIAL_BACKOFF}s")
                    time.sleep(GEMINI_INITIAL_BACKOFF)
                else:
                    raise

    response = _call(question)

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
            return "\n".join(text_parts), tools_called

        function_responses = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args)
            tools_called.append(name)
            if verbose:
                print(f"  [gemini] [tool] {name}({json.dumps(args)})")
            result = tool_executor(name, args)
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": result},
                    )
                )
            )

        response = _call(function_responses)


def _openai_compatible_loop(
    provider_tag: str,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    system_prompt: str,
    tool_defs: list[dict],
    tool_executor: Callable[[str, dict[str, Any]], str],
    extra_headers: dict[str, str] | None = None,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    client = OpenAI(base_url=base_url, api_key=api_key, default_headers=extra_headers or {})

    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tool_defs
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    tools_called = []

    while True:
        if verbose:
            print(f"  [{provider_tag}] calling {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=oai_tools if oai_tools else None,
        )

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return msg.content or "", tools_called

        messages.append(msg.model_dump())

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_called.append(name)
            if verbose:
                print(f"  [{provider_tag}] [tool] {name}({json.dumps(args)})")
            result = tool_executor(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


def _try_gemini(question, system_prompt, tool_defs, tool_executor, verbose):
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set")
    return _gemini_tool_loop(system_prompt, question, tool_defs, tool_executor, verbose)


def _try_openrouter_free(question, system_prompt, tool_defs, tool_executor, verbose):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return _openai_compatible_loop(
        provider_tag="openrouter/free",
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        model=OPENROUTER_FREE_MODEL,
        question=question,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
        tool_executor=tool_executor,
        extra_headers=OPENROUTER_HEADERS,
        verbose=verbose,
    )


def _try_groq(question, system_prompt, tool_defs, tool_executor, verbose):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return _openai_compatible_loop(
        provider_tag="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=key,
        model=GROQ_MODEL,
        question=question,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
        tool_executor=tool_executor,
        verbose=verbose,
    )


def _try_ainative(question, system_prompt, tool_defs, tool_executor, verbose):
    key = os.environ.get("AINATIVE_API_KEY")
    if not key:
        raise RuntimeError("AINATIVE_API_KEY not set")
    return _openai_compatible_loop(
        provider_tag="ainative",
        base_url="https://api.ainative.studio/v1",
        api_key=key,
        model=AINATIVE_MODEL,
        question=question,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
        tool_executor=tool_executor,
        verbose=verbose,
    )


def _try_openrouter_paid(question, system_prompt, tool_defs, tool_executor, verbose):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return _openai_compatible_loop(
        provider_tag="openrouter/paid",
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        model=OPENROUTER_PAID_MODEL,
        question=question,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
        tool_executor=tool_executor,
        extra_headers=OPENROUTER_HEADERS,
        verbose=verbose,
    )


FALLBACK_CHAIN = [
    ("gemini", _try_gemini),
    ("openrouter/free", _try_openrouter_free),
    ("groq", _try_groq),
    ("ainative", _try_ainative),
    ("openrouter/paid", _try_openrouter_paid),
]


def call_llm_with_tools(
    question: str,
    system_prompt: str,
    tool_defs: list[dict],
    tool_executor: Callable[[str, dict[str, Any]], str],
    verbose: bool = False,
) -> dict[str, Any]:
    last_error = None
    for name, fn in FALLBACK_CHAIN:
        try:
            answer, tools_called = fn(
                question=question,
                system_prompt=system_prompt,
                tool_defs=tool_defs,
                tool_executor=tool_executor,
                verbose=verbose,
            )
            return {
                "answer": answer,
                "tools_called": tools_called,
                "provider_used": name,
            }
        except Exception as e:
            last_error = e
            if verbose:
                print(f"  [{name}] FAILED: {e}")
            print(f"  falling back to next provider...")

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


if __name__ == "__main__":
    from schemas.event import Event
    from agents.store import EventStore, VectorIndex

    fake_events = [
        Event(time="08:15:00", camera="cam-entrance", event_type="intrusion",
              description="Person jumped over the perimeter fence near the east gate", confidence=0.92),
        Event(time="12:45:00", camera="cam-lobby", event_type="theft",
              description="Unattended bag picked up and carried out by unknown person", confidence=0.95),
    ]

    store = EventStore(":memory:")
    vindex = VectorIndex()
    for ev in fake_events:
        store.add_event(ev)
        vindex.add_event(ev)

    def test_executor(name, args):
        if name == "search_events":
            results = vindex.semantic_search(query=args["query"], k=int(args.get("k", 3)))
            return json.dumps([e.model_dump() for e in results])
        return json.dumps({"error": "unknown tool"})

    tools = [
        {
            "name": "search_events",
            "description": "Search events by natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of results"},
                },
                "required": ["query"],
            },
        },
    ]

    prompt = "You are a CCTV investigator. Answer concisely."

    print("=== Test 1: Gemini (should work) ===")
    result = call_llm_with_tools(
        "What events are in the system?", prompt, tools, test_executor, verbose=True
    )
    print(f"  provider: {result['provider_used']}")
    print(f"  tools_called: {result['tools_called']}")
    print(f"  answer: {result['answer'][:200]}")

    print("\n=== Test 2: Force Gemini failure -> full fallback chain ===")
    saved_gemini = os.environ.pop("GEMINI_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = "bad-key-for-testing"
    try:
        result2 = call_llm_with_tools(
            "What events are in the system?", prompt, tools, test_executor, verbose=True
        )
        print(f"  provider: {result2['provider_used']}")
        print(f"  tools_called: {result2['tools_called']}")
        print(f"  answer: {result2['answer'][:200]}")
    except RuntimeError as e:
        print(f"  All 5 tiers failed: {e}")
    finally:
        if saved_gemini:
            os.environ["GEMINI_API_KEY"] = saved_gemini

    store.close()
