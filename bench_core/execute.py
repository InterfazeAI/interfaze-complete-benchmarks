"""The fallback ladder: one request through an adapter, self-healing declared-cap gaps."""

from __future__ import annotations

import dataclasses

from bench_core.capabilities import Capabilities, ReasoningStyle
from bench_core.errors import Classification, ErrorKind
from bench_core.request import Request
from bench_core.response import Response

_THINKING_FLOORS = ["minimal", "low", "medium", "high"]


class BenchError(Exception):
    def __init__(self, classification: Classification):
        super().__init__(classification.message or classification.kind.value)
        self.classification = classification


def _raise_thinking_floor(caps: Capabilities) -> tuple[Capabilities, str]:
    cur = caps.reasoning.off_value
    try:
        nxt = _THINKING_FLOORS[_THINKING_FLOORS.index(cur) + 1]
    except (ValueError, IndexError):
        nxt = "low"
    reasoning = dataclasses.replace(caps.reasoning, off_value=nxt)
    return dataclasses.replace(
        caps, reasoning=reasoning
    ), f"thinking floor raised to {nxt}"


def _force_reasoning_floor(caps: Capabilities) -> tuple[Capabilities, str]:
    r = caps.reasoning
    if r.style is ReasoningStyle.REASONING_BODY:
        extra = dict(r.extra)
        extra["toggle"] = "effort"
        reasoning = dataclasses.replace(
            r, extra=extra, off_value="minimal", true_off=False
        )
    elif r.style is ReasoningStyle.EFFORT:
        reasoning = dataclasses.replace(r, off_value="low", true_off=False)
    else:
        reasoning = dataclasses.replace(r, true_off=False)
    return dataclasses.replace(
        caps, reasoning=reasoning
    ), "reasoning cannot be disabled; using floor"


def _apply_action(
    c: Classification, req: Request, caps: Capabilities
) -> tuple[Request, Capabilities, str] | None:
    """Return a mutated (req, caps, hint), or None if the action can't be
    applied here (escalates to fatal)."""
    if c.action == "raise_thinking_floor":
        caps, hint = _raise_thinking_floor(caps)
        return req, caps, hint
    if c.action == "force_reasoning_floor":
        caps, hint = _force_reasoning_floor(caps)
        return req, caps, hint
    if c.action == "drop_temperature":
        req = dataclasses.replace(req, temperature=None)
        return req, caps, "temperature dropped (rejected with reasoning)"
    return None


def execute(
    adapter,
    client,
    req: Request,
    caps: Capabilities,
    model_id: str,
    *,
    max_param_retries: int = 4,
) -> tuple[Response, list[str]]:
    hints: list[str] = []
    work_req, work_caps = req, caps

    for _ in range(max_param_retries + 1):
        encoded = adapter.encode(work_req, work_caps, model_id)
        try:
            raw = adapter.call(client, encoded)
        except Exception as exc:
            c = adapter.classify_error(exc)
            if c.kind is ErrorKind.PARAM_REJECTED:
                applied = _apply_action(c, work_req, work_caps)
                if applied is not None:
                    work_req, work_caps, hint = applied
                    hints.append(hint)
                    continue
            raise BenchError(c) from exc

        resp = adapter.decode(raw, work_caps)
        if not resp.text.strip():
            raise BenchError(
                Classification(ErrorKind.EMPTY_CONTENT, message="empty content")
            )
        return resp, hints

    raise BenchError(
        Classification(ErrorKind.FATAL, message="parameter adjustments exhausted")
    )
