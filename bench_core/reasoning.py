"""Translate an abstract reasoning *mode* into the concrete directive a host
expects — the one place the audit's six divergent Gemini-floor encodings (plus
the OpenAI/Fireworks/Anthropic/OpenRouter variants) now live.

A benchmark asks for a mode ("off" / "low" / "medium" / "high"); the model's
declared `ReasoningCap` says how that host expresses it and what its floor is.
`build_reasoning` returns a `ReasoningInjection` describing where the directive
goes; each provider adapter reads only the field for its mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench_core.capabilities import ReasoningCap, ReasoningStyle

_HIGH_MODES = {"high", "on", "max"}
_OFF_MODES = {"off", "none", "disabled"}


@dataclass
class ReasoningInjection:
    """How to attach reasoning to a request. A given cap fills exactly one of
    the mechanism fields; the rest stay empty/None so an adapter that reads the
    wrong field simply sees "nothing to do"."""

    kwargs: dict = field(
        default_factory=dict
    )  # top-level create() kwargs (reasoning_effort)
    extra_body: dict = field(
        default_factory=dict
    )  # OpenRouter reasoning + provider pin
    thinking_level: str | None = None  # Gemini 3.x
    thinking_budget: int | None = None  # Gemini 2.5
    anthropic_thinking: dict | None = (
        None  # {"type":"disabled"} | {"type":"enabled",...}
    )
    allow_temperature: bool = True


def _resolve_value(mode: str, cap: ReasoningCap) -> Any:
    """Concrete value for a mode: `off`/`on` come from the cap; an intermediate
    mode passes through literally (caller owns vocabulary validity)."""
    if mode in _OFF_MODES:
        return cap.off_value
    if mode in _HIGH_MODES:
        return cap.on_value
    return mode


def _allow_temperature(mode: str, cap: ReasoningCap) -> bool:
    """Temperature is allowed when thinking is not engaged, or when the host
    tolerates temperature alongside active thinking. An off run that truly
    disables thinking counts as "not engaged"."""
    if mode in _OFF_MODES and cap.true_off:
        return True
    return cap.temperature_when_on


def build_reasoning(mode: str, cap: ReasoningCap) -> ReasoningInjection:
    style = cap.style
    allow_temp = _allow_temperature(mode, cap)

    if style is ReasoningStyle.NONE:
        return ReasoningInjection(allow_temperature=allow_temp)

    if style is ReasoningStyle.EFFORT:
        return ReasoningInjection(
            kwargs={"reasoning_effort": _resolve_value(mode, cap)},
            allow_temperature=allow_temp,
        )

    if style is ReasoningStyle.THINKING_LEVEL:
        return ReasoningInjection(
            thinking_level=_resolve_value(mode, cap),
            allow_temperature=allow_temp,
        )

    if style is ReasoningStyle.THINKING_BUDGET:
        return ReasoningInjection(
            thinking_budget=_resolve_value(mode, cap),
            allow_temperature=allow_temp,
        )

    if style is ReasoningStyle.DISABLED_BLOCK:
        if mode in _OFF_MODES:
            thinking = {"type": "disabled"}
        else:
            thinking = {"type": "enabled", "budget_tokens": cap.on_value}
        return ReasoningInjection(
            anthropic_thinking=thinking,
            allow_temperature=allow_temp,
        )

    if style is ReasoningStyle.REASONING_BODY:
        toggle = cap.extra.get("toggle", "enabled")
        value = _resolve_value(mode, cap)
        if toggle == "effort":
            reasoning = {"effort": value}
        else:  # "enabled" boolean toggle
            reasoning = {"enabled": bool(value)}
        extra_body: dict = {"reasoning": reasoning}
        pin = cap.extra.get("provider_only")
        if pin:
            extra_body["provider"] = {"only": list(pin)}
        return ReasoningInjection(extra_body=extra_body, allow_temperature=allow_temp)

    raise ValueError(f"unhandled reasoning style {style!r}")
