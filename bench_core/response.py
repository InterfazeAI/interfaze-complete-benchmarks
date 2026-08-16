"""Normalized response — one shape regardless of host."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Response:
    text: str
    reasoning_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    raw_id: str | None = None
