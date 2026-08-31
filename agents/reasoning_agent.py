"""Rule-based threat composition layer.

Composes a risk_level (low/medium/high) and human-readable reason for a
target event by correlating it with nearby events (same camera, within a
configurable time window).

All rules live in the RULES list as explicit (condition, risk_level,
reason_template) tuples evaluated first-match-wins, so any panel can show
exactly which rule fired and why ("rule_id" is returned with every result).
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.event import Event


DEFAULT_WINDOW_SECONDS = 30
WEAPON_HIGH_CONFIDENCE = 0.75

WEAPON_EVENT_TYPES = {
    "weapon", "gun", "heavy-weapon", "firearm", "knife",
}
ACTION_EVENT_TYPES = {
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "road accidents", "robbery", "shooting", "shoplifting",
    "stealing", "vandalism", "action",
}


class ThreatAssessment(BaseModel):
    risk_level: str
    rule_id: str
    reason: str


def _to_seconds(time_str: str) -> int:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _time_distance(a: str, b: str) -> int:
    da, db = _to_seconds(a), _to_seconds(b)
    raw = abs(da - db)
    return min(raw, 86400 - raw)


def find_nearby(
    target: Event,
    candidates: list[Event],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> list[Event]:
    """Same-camera events within the time window, excluding the target itself."""
    return [
        e for e in candidates
        if e.camera == target.camera
        and e.model_dump(exclude={"description"}) != target.model_dump(exclude={"description"})
        and _time_distance(target.time, e.time) <= window_seconds
    ]


@dataclass
class ThreatContext:
    """Precomputed facts handed to every rule's condition function."""
    event: Event
    nearby: list[Event] = field(default_factory=list)
    window_seconds: float = DEFAULT_WINDOW_SECONDS

    def __post_init__(self):
        pool = [self.event, *self.nearby]
        weapons = [e for e in pool if self._kind(e) == "weapon"]
        actions = [e for e in pool if self._kind(e) == "action"]
        self.has_weapon = bool(weapons)
        self.has_action = bool(actions)
        self.weapon_conf = max((e.confidence for e in weapons), default=0.0)
        self.weapon_label = max(weapons, key=lambda e: e.confidence).event_type if weapons else ""
        self.action_conf = max((e.confidence for e in actions), default=0.0)
        self.action_label = max(actions, key=lambda e: e.confidence).event_type if actions else ""

    @staticmethod
    def _kind(e: Event) -> str:
        et = e.event_type.strip().lower()
        if et in WEAPON_EVENT_TYPES:
            return "weapon"
        if et in ACTION_EVENT_TYPES:
            return "action"
        return "other"


def _has_weapon_with_action(ctx: ThreatContext) -> bool:
    return ctx.has_weapon and ctx.has_action


def _has_high_conf_weapon_alone(ctx: ThreatContext) -> bool:
    return ctx.has_weapon and not ctx.has_action and ctx.weapon_conf >= WEAPON_HIGH_CONFIDENCE


def _has_low_conf_weapon_alone(ctx: ThreatContext) -> bool:
    return ctx.has_weapon and not ctx.has_action


def _has_action_only(ctx: ThreatContext) -> bool:
    return ctx.has_action and not ctx.has_weapon


def _fallback(ctx: ThreatContext) -> bool:
    return True


@dataclass(frozen=True)
class Rule:
    id: str
    condition: Callable[[ThreatContext], bool]
    risk_level: str
    reason_template: str
    summary: str


RULES: list[Rule] = [
    Rule(
        id="R1-weapon-plus-action",
        condition=_has_weapon_with_action,
        risk_level="high",
        reason_template=(
            "Weapon '{weapon_label}' (conf {weapon_conf:.2f}) corroborated by "
            "action '{action_label}' (conf {action_conf:.2f}) on {camera} at "
            "{time}, within a {window_seconds:.0f}s window"
        ),
        summary="weapon AND action event present within window",
    ),
    Rule(
        id="R2-weapon-alone-high-confidence",
        condition=_has_high_conf_weapon_alone,
        risk_level="high",
        reason_template=(
            "High-confidence weapon '{weapon_label}' (conf {weapon_conf:.2f}) "
            "on {camera} at {time} with no action corroboration within "
            "{window_seconds:.0f}s"
        ),
        summary="weapon alone AND confidence >= weapon_high_confidence",
    ),
    Rule(
        id="R3-weapon-alone-low-confidence",
        condition=_has_low_conf_weapon_alone,
        risk_level="medium",
        reason_template=(
            "Low-confidence weapon '{weapon_label}' (conf {weapon_conf:.2f} < "
            "{weapon_high_confidence}) on {camera} at {time} with no "
            "action support within {window_seconds:.0f}s"
        ),
        summary="weapon alone AND confidence < weapon_high_confidence",
    ),
    Rule(
        id="R4-action-only",
        condition=_has_action_only,
        risk_level="low",
        reason_template=(
            "Action '{action_label}' (conf {action_conf:.2f}) on {camera} at "
            "{time} with no weapon detected within {window_seconds:.0f}s; "
            "standalone action recognition treated as advisory"
        ),
        summary="action event present AND no weapon",
    ),
    Rule(
        id="R5-no-corroboration",
        condition=_fallback,
        risk_level="low",
        reason_template=(
            "Isolated '{event_type}' (conf {confidence:.2f}) on {camera} at "
            "{time}; no weapon or action events within {window_seconds:.0f}s"
        ),
        summary="fallback: no weapon, no action (always matches)",
    ),
]


def assess_threat(
    event: Event,
    other_events: list[Event],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> ThreatAssessment:
    nearby = find_nearby(event, other_events, window_seconds)
    ctx = ThreatContext(event=event, nearby=nearby, window_seconds=window_seconds)

    template_vars = {
        **event.model_dump(),
        "window_seconds": window_seconds,
        "nearby_count": len(nearby),
        "weapon_label": ctx.weapon_label,
        "weapon_conf": ctx.weapon_conf,
        "weapon_high_confidence": WEAPON_HIGH_CONFIDENCE,
        "action_label": ctx.action_label,
        "action_conf": ctx.action_conf,
    }

    for rule in RULES:
        if rule.condition(ctx):
            return ThreatAssessment(
                risk_level=rule.risk_level,
                rule_id=rule.id,
                reason=rule.reason_template.format(**template_vars),
            )

    raise RuntimeError("RULES must end with an unconditional fallback rule")


if __name__ == "__main__":
    scenarios = {
        "1. Weapon + action support": (
            Event(time="14:32:10", camera="cam-entrance", event_type="gun",
                  description="Handgun visible in suspect's waistband",
                  confidence=0.82),
            [
                Event(time="14:32:25", camera="cam-entrance", event_type="Fighting",
                      description="Two individuals grappling on the floor",
                      confidence=0.71),
                Event(time="14:31:50", camera="cam-entrance", event_type="loitering",
                      description="Suspect pacing near entrance",
                      confidence=0.64),
            ],
        ),
        "2. Weapon only (low confidence)": (
            Event(time="09:15:40", camera="cam-parking", event_type="knife",
                  description="Possible bladed object in hand",
                  confidence=0.58),
            [
                Event(time="09:16:02", camera="cam-parking", event_type="person",
                      description="Single individual walking away",
                      confidence=0.91),
            ],
        ),
        "3. Action only (no weapon)": (
            Event(time="22:04:12", camera="cam-loading", event_type="Vandalism",
                  description="Individual striking glass pane repeatedly",
                  confidence=0.77),
            [
                Event(time="22:04:30", camera="cam-loading", event_type="intrusion",
                      description="Forced entry through side door",
                      confidence=0.69),
            ],
        ),
    }

    print("RULES (evaluation order, first match wins):")
    for i, rule in enumerate(RULES, 1):
        print(f"  {i}. {rule.id:<32} [{rule.risk_level:<6}] {rule.summary}")

    for name, (target, others) in scenarios.items():
        assessment = assess_threat(target, others)
        print(f"\n{name}")
        print(f"  risk_level : {assessment.risk_level}")
        print(f"  rule_id    : {assessment.rule_id}")
        print(f"  reason     : {assessment.reason}")
