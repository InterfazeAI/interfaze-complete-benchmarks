"""Normalize a host response into `Response` (unifies the three shapes)."""

from __future__ import annotations

from typing import Any

from bench_core.capabilities import ResponseShape
from bench_core.response import Response


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _decode_openai(raw: Any) -> Response:
    choice = raw.choices[0]
    usage = _get(raw, "usage")
    reasoning = None
    if usage is not None:
        details = _get(usage, "completion_tokens_details")
        if details is not None:
            reasoning = _get(details, "reasoning_tokens")
        if reasoning is None:  # Fireworks/Inkling report it at the top level
            reasoning = _get(usage, "reasoning_tokens")
    return Response(
        text=_get(choice.message, "content") or "",
        reasoning_tokens=reasoning,
        input_tokens=_get(usage, "prompt_tokens") if usage else None,
        output_tokens=_get(usage, "completion_tokens") if usage else None,
        finish_reason=_get(choice, "finish_reason"),
        raw_id=_get(raw, "id"),
    )


def _decode_anthropic(raw: Any) -> Response:
    text = "".join(b.text for b in (raw.content or []) if _get(b, "type") == "text")
    usage = _get(raw, "usage")
    return Response(
        text=text,
        reasoning_tokens=None,  # Anthropic does not report thinking tokens separately
        input_tokens=_get(usage, "input_tokens") if usage else None,
        output_tokens=_get(usage, "output_tokens") if usage else None,
        raw_id=_get(raw, "id"),
    )


def _decode_gemini(raw: Any) -> Response:
    meta = _get(raw, "usage_metadata")
    return Response(
        text=_get(raw, "text") or "",
        reasoning_tokens=_get(meta, "thoughts_token_count") if meta else None,
        input_tokens=_get(meta, "prompt_token_count") if meta else None,
        output_tokens=_get(meta, "candidates_token_count") if meta else None,
        raw_id=_get(raw, "response_id"),
    )


_DECODERS = {
    ResponseShape.OPENAI_CHAT: _decode_openai,
    ResponseShape.ANTHROPIC_BLOCKS: _decode_anthropic,
    ResponseShape.GEMINI_TEXT: _decode_gemini,
}


def decode_response(raw: Any, shape: ResponseShape) -> Response:
    return _DECODERS[shape](raw)
