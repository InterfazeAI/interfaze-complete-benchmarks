from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorKind(str, Enum):
    PARAM_REJECTED = "param_rejected"
    RATE_LIMITED = "rate_limited"
    TRANSIENT = "transient"  # timeout / 5xx / connection -> backoff + retry
    EMPTY_CONTENT = "empty_content"
    FATAL = "fatal"  # auth / missing model / unrecognized 4xx -> fail the sample


@dataclass
class Classification:
    kind: ErrorKind
    action: str | None = (
        None  # raise_thinking_floor | force_reasoning_floor | drop_temperature | drop_param
    )
    param: str | None = None
    message: str = ""


# 400s that name a specific, adjustable parameter. Ordered — first match wins.
_PARAM_PATTERNS: list[tuple[re.Pattern, str, str | None]] = [
    (
        re.compile(r"thinking level \w+ is not supported", re.IGNORECASE),
        "raise_thinking_floor",
        "thinking_level",
    ),
    (
        re.compile(
            r"reasoning is mandatory|reasoning.*cannot be disabled|cannot disable reasoning",
            re.IGNORECASE,
        ),
        "force_reasoning_floor",
        "reasoning",
    ),
    (re.compile(r"temperature", re.IGNORECASE), "drop_temperature", "temperature"),
    (
        re.compile(
            r"(?:unknown|unsupported|unexpected|invalid) (?:parameter|argument|keyword)[:\s]+['\"]?([\w.]+)",
            re.IGNORECASE,
        ),
        "drop_param",
        None,
    ),
]

_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_MSG = re.compile(
    r"timed out|timeout|connection|temporarily unavailable|overloaded", re.IGNORECASE
)


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    m = re.match(r"\s*(\d{3})\b", str(exc))
    return int(m.group(1)) if m else None


def classify(exc: Exception) -> Classification:
    msg = str(exc)
    status = _status_of(exc)

    if isinstance(exc, TimeoutError) or _TRANSIENT_MSG.search(msg):
        return Classification(ErrorKind.TRANSIENT, message=msg)
    if status == 429:
        return Classification(ErrorKind.RATE_LIMITED, message=msg)
    if status in _TRANSIENT_STATUS:
        return Classification(ErrorKind.TRANSIENT, message=msg)
    if status in (401, 403, 404):
        return Classification(ErrorKind.FATAL, message=msg)

    if status == 400 or status is None:
        for pattern, action, param in _PARAM_PATTERNS:
            m = pattern.search(msg)
            if m:
                captured = param or (m.group(1) if m.groups() else None)
                return Classification(
                    ErrorKind.PARAM_REJECTED, action=action, param=captured, message=msg
                )
        if status == 400:
            return Classification(ErrorKind.FATAL, message=msg)

    return Classification(ErrorKind.FATAL, message=msg)
