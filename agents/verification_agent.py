"""Independent dual-provider verification for LLM-generated claims.

Flow:
  1. Primary provider (normal fallback chain) generates structured claims
     from an evidence bundle.
  2. A SECOND provider — picked from FALLBACK_CHAIN, guaranteed to be a
     different provider FAMILY than the primary (e.g. openrouter/* never
     verifies openrouter/*) — independently generates claims from the SAME
     evidence, seeing neither the primary output nor its identity.
  3. Claims are compared field-by-field (structured check, not text diff):
       location          -> set of camera ids extracted from both values
       confidence_level  -> bucketed to high/medium/low before comparing
       everything else   -> normalized fuzzy string match >= FUZZY_THRESHOLD
  4. Result carries "verification_status": "agreed" | "disputed".
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm_client import call_llm_with_tools, FALLBACK_CHAIN


CLAIM_FIELDS = ["threat_or_weapon", "location", "time_window", "confidence_level"]
FUZZY_THRESHOLD = 0.75
CONFIDENCE_KEYWORDS = [
    ("low", ("low", "weak", "poor", "uncertain")),
    ("medium", ("medium", "moderate", "mid", "fair")),
    ("high", ("high", "strong", "certain", "very")),
]

CLAIM_SYSTEM_PROMPT = "You are an independent fact extractor for CCTV investigations."

CLAIM_EXTRACTION_PROMPT = """\
Based ONLY on the evidence below, extract factual claims as a JSON object \
with exactly these four fields:

{"threat_or_weapon": "<specific weapon or threat described, e.g. 'handgun', 'knife', 'none'>",
 "location": "<comma-separated camera id(s) the evidence refers to>",
 "time_window": "<timestamp(s) referenced, HH:MM:SS format>",
 "confidence_level": "<high, medium, or low>"}

Rules:
- Use ONLY facts present in the evidence; do not invent details.
- Values must be short literals, not sentences.
- Respond with a single ```json fenced code block and nothing else.

"""


class VerificationResult(BaseModel):
    verification_status: str  # "agreed" | "disputed"
    primary_provider: str
    verifier_provider: str
    agreements: dict[str, list[str]]
    disputes: dict[str, list[str]]


def _noop_executor(name: str, args: dict) -> str:
    return json.dumps({"error": "verification does not use tools"})


def _parse_json_block(raw: str) -> dict[str, str]:
    match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    candidates = [match.group(1)] if match else []
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            continue
    return {}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:\.\-\s]", " ", value.lower())).strip()


def _fuzzy_equal(a: str, b: str) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= FUZZY_THRESHOLD


def _bucket_confidence(value: str) -> str:
    v = value.lower()
    for bucket, keywords in CONFIDENCE_KEYWORDS:
        if any(k in v for k in keywords):
            return bucket
    return "unknown"


def _cmp_location(a: str, b: str) -> bool:
    cams = lambda s: set(re.findall(r"cam[-\s]?[a-z0-9\-]+", s.lower()))
    return cams(a) == cams(b) if (cams(a) or cams(b)) else _fuzzy_equal(a, b)


def _cmp_confidence(a: str, b: str) -> bool:
    return _bucket_confidence(a) == _bucket_confidence(b)


FIELD_COMPARATORS: dict[str, Callable[[str, str], bool]] = {
    "location": _cmp_location,
    "confidence_level": _cmp_confidence,
}


def compare_claims(
    primary_claims: dict[str, str],
    verifier_claims: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    agreements: dict[str, list[str]] = {}
    disputes: dict[str, list[str]] = {}
    for field in CLAIM_FIELDS:
        a = primary_claims.get(field, "<missing>")
        b = verifier_claims.get(field, "<missing>")
        comparator = FIELD_COMPARATORS.get(field, _fuzzy_equal)
        if comparator(a, b):
            agreements[field] = [a, b]
        else:
            disputes[field] = [a, b]
    return agreements, disputes


def _provider_family(name: str) -> str:
    return name.split("/")[0]


def pick_verifier(primary: str) -> tuple[str, Callable]:
    """First chain entry from a DIFFERENT provider family than the primary,
    starting the search just after the primary's position (wrapping)."""
    names = [n for n, _ in FALLBACK_CHAIN]
    start = names.index(primary) + 1 if primary in names else 0
    for offset in range(len(names)):
        name, fn = FALLBACK_CHAIN[(start + offset) % len(FALLBACK_CHAIN)]
        if _provider_family(name) != _provider_family(primary):
            return name, fn
    raise RuntimeError(f"No verifier provider available outside family of '{primary}'")


def generate_verified_claims(
    evidence_context: str,
    verbose: bool = False,
) -> tuple[VerificationResult, str, str]:
    prompt = CLAIM_EXTRACTION_PROMPT + evidence_context

    primary = call_llm_with_tools(
        question=prompt,
        system_prompt=CLAIM_SYSTEM_PROMPT,
        tool_defs=[],
        tool_executor=_noop_executor,
        verbose=verbose,
    )
    primary_name = primary["provider_used"]

    verifier_name, verifier_fn = pick_verifier(primary_name)
    if verbose:
        print(f"  [verifier] primary={primary_name} -> verifier={verifier_name}")

    verifier_answer, _ = verifier_fn(
        question=prompt,
        system_prompt=CLAIM_SYSTEM_PROMPT + " Answer independently.",
        tool_defs=[],
        tool_executor=_noop_executor,
        verbose=verbose,
    )

    agreements, disputes = compare_claims(
        _parse_json_block(primary["answer"]),
        _parse_json_block(verifier_answer),
    )
    status = "agreed" if not disputes else "disputed"

    result = VerificationResult(
        verification_status=status,
        primary_provider=primary_name,
        verifier_provider=verifier_name,
        agreements=agreements,
        disputes=disputes,
    )
    return result, primary["answer"], verifier_answer


if __name__ == "__main__":
    import tempfile

    from schemas.event import Event
    from agents.evidence_agent import get_evidence

    def _print_comparison(label: str, result: VerificationResult,
                          raw_primary: str, raw_verifier: str,
                          evidence_description: str):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"Evidence: {evidence_description}")
        print(f"\n--- PRIMARY OUTPUT ({result.primary_provider}) ---")
        print(raw_primary.strip())
        print(f"\n--- VERIFIER OUTPUT ({result.verifier_provider}) ---")
        print(raw_verifier.strip())
        print("\n--- FIELD-BY-FIELD COMPARISON ---")
        for field, values in result.agreements.items():
            print(f"  OK   {field:<22} {values[0]!r} == {values[1]!r}")
        for field, values in result.disputes.items():
            print(f"  DIFF {field:<22} primary={values[0]!r} vs verifier={values[1]!r}")
        print(f"\nverification_status: {result.verification_status}")

    # --- Scenario 1: genuine agreement ---
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()
        weapon_event = Event(
            time="21:47:12",
            camera="cam-parking",
            event_type="gun",
            description="Individual carrying a handgun near the blue sedan, "
                        "walking quickly toward the lot exit",
            confidence=0.81,
        )
        ts = weapon_event.time.replace(":", "")
        (frame_dir / f"{weapon_event.camera}_{ts}.jpg").write_bytes(b"")
        bundle = get_evidence(weapon_event, frame_dir)

    evidence_context_1 = (
        "## Evidence Bundle\n"
        f"```json\n{json.dumps(bundle.model_dump(), indent=2, default=str)}\n```"
    )

    result_1, raw_p1, raw_v1 = generate_verified_claims(
        evidence_context_1, verbose=True
    )
    _print_comparison(
        "Scenario 1 — clear handgun (high confidence)",
        result_1, raw_p1, raw_v1,
        f"[{weapon_event.time} {weapon_event.camera}] "
        f"{weapon_event.event_type} conf={weapon_event.confidence}",
    )

    # --- Scenario 2: forced mismatch via --force-mismatch ---
    if "--force-mismatch" in sys.argv:
        ambiguous_event = Event(
            time="15:03:44",
            camera="cam-loading",
            event_type="weapon",
            description="Subject holding an indistinct elongated object while "
                        "approaching the cargo bay, partially occluded by shadow",
            confidence=0.52,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            ts = ambiguous_event.time.replace(":", "")
            (frame_dir / f"{ambiguous_event.camera}_{ts}.jpg").write_bytes(b"")
            bundle2 = get_evidence(ambiguous_event, frame_dir)

        evidence_context_2 = (
            "## Evidence Bundle\n"
            f"```json\n{json.dumps(bundle2.model_dump(), indent=2, default=str)}\n```"
        )

        result_2, raw_p2, raw_v2 = generate_verified_claims(
            evidence_context_2, verbose=False
        )

        # Inject artificial disagreement for a deterministic "disputed" test
        raw_v2_forced = re.sub(
            r'"threat_or_weapon"\s*:\s*"[^"]*"',
            '"threat_or_weapon": "knife"',
            raw_v2,
            count=1,
        )
        injected_claims = _parse_json_block(raw_v2_forced)
        agreements, disputes = compare_claims(
            _parse_json_block(raw_p2), injected_claims,
        )
        result_2 = VerificationResult(
            verification_status="agreed" if not disputes else "disputed",
            primary_provider=result_2.primary_provider,
            verifier_provider=f"{result_2.verifier_provider} [injected]",
            agreements=agreements,
            disputes=disputes,
        )

        _print_comparison(
            "Scenario 2 — ambiguous weapon (--force-mismatch, injected 'knife')",
            result_2, raw_p2, raw_v2_forced,
            f"[{ambiguous_event.time} {ambiguous_event.camera}] "
            f"{ambiguous_event.event_type} conf={ambiguous_event.confidence}",
        )

    else:
        print("\n\n--- Scenario 2 skipped (pass --force-mismatch to run forced-disagreement test) ---")
