"""Anthropic adapter. System messages become the top-level `system` param;
`max_tokens` is mandatory and must exceed `budget_tokens` when thinking is
enabled — both handled declaratively from the cap.
"""

from __future__ import annotations

from bench_core.capabilities import Capabilities
from bench_core.media import image_block
from bench_core.providers.base import ProviderAdapter
from bench_core.reasoning import build_reasoning
from bench_core.request import ImagePart, Message, Request, TextPart

_DEFAULT_OFF_MAX = 1024
_DEFAULT_ON_MAX = 16000
_BUDGET_MARGIN = 4096


class AnthropicAdapter(ProviderAdapter):
    def __init__(self, name, base_url, key_spec, capability_defaults):
        self.name = name
        self.base_url = base_url
        self.key_spec = key_spec
        self.capability_defaults = capability_defaults

    def build_client(self, api_key: str | None = None):
        from anthropic import Anthropic

        return Anthropic(
            api_key=api_key or self.resolve_key(), timeout=180.0, max_retries=2
        )

    def _content(self, msg: Message, caps: Capabilities):
        if all(isinstance(p, TextPart) for p in msg.parts):
            return "".join(p.text for p in msg.parts)
        blocks = []
        for p in msg.parts:
            if isinstance(p, TextPart):
                blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                blocks.append(image_block(p, caps.media.image))
        return blocks

    def encode(self, req: Request, caps: Capabilities, model_id: str) -> dict:
        system = "".join(
            p.text
            for m in req.messages
            if m.role == "system"
            for p in m.parts
            if isinstance(p, TextPart)
        )
        messages = [
            {"role": m.role, "content": self._content(m, caps)}
            for m in req.messages
            if m.role != "system"
        ]
        kwargs: dict = {"model": model_id, "messages": messages}
        if system:
            kwargs["system"] = system

        inj = build_reasoning(req.reasoning.mode, caps.reasoning)
        extra = caps.reasoning.extra
        if inj.anthropic_thinking is not None:
            kwargs["thinking"] = inj.anthropic_thinking

        thinking_on = (inj.anthropic_thinking or {}).get("type") == "enabled"
        if thinking_on:
            budget = inj.anthropic_thinking["budget_tokens"]
            max_tokens = req.max_tokens or extra.get("on_max_tokens", _DEFAULT_ON_MAX)
            if max_tokens <= budget:
                max_tokens = budget + _BUDGET_MARGIN
        else:
            max_tokens = req.max_tokens or extra.get("off_max_tokens", _DEFAULT_OFF_MAX)
        kwargs["max_tokens"] = max_tokens

        if req.temperature is not None and inj.allow_temperature:
            kwargs["temperature"] = req.temperature

        return kwargs

    def call(self, client, encoded: dict):
        return client.messages.create(**encoded)
