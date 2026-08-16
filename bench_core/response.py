"""Normalized response — one shape regardless of host, so benchmark `parse`
never touches choices[0] / content-blocks / .text directly.
"""

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
