# OpenAI Chat Completions: openai, fireworks, openrouter, interfaze.

from __future__ import annotations

from typing import Any

from src.capabilities import Capabilities
from src.media import audio_block, image_block
from src.providers.base import ProviderAdapter
from src.reasoning import build_reasoning
from src.request import AudioPart, ImagePart, Message, Request, TextPart


class OpenAICompatAdapter(ProviderAdapter):
    def __init__(self, name, base_url, key_spec, capability_defaults):
        self.name = name
        self.base_url = base_url
        self.key_spec = key_spec
        self.capability_defaults = capability_defaults

    def build_client(self, api_key: str | None = None):
        from openai import OpenAI

        return OpenAI(
            api_key=api_key or self.resolve_key(),
            base_url=self.base_url,
            timeout=180.0,
            max_retries=2,
        )

    def _content(self, msg: Message, caps: Capabilities) -> Any:
        if all(isinstance(p, TextPart) for p in msg.parts):
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            return "".join(p.text for p in text_parts)
        blocks = []
        for p in msg.parts:
            if isinstance(p, TextPart):
                blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                blocks.append(image_block(p, caps.media.image))
            elif isinstance(p, AudioPart):
                blocks.append(audio_block(p, caps.media.audio))
        return blocks

    def encode(self, req: Request, caps: Capabilities, model_id: str) -> dict:
        kwargs: dict = {
            "model": model_id,
            "messages": [
                {"role": m.role, "content": self._content(m, caps)}
                for m in req.messages
            ],
        }

        inj = build_reasoning(req.reasoning.mode, caps.reasoning)
        kwargs.update(inj.kwargs)
        if inj.extra_body:
            kwargs["extra_body"] = inj.extra_body

        if req.temperature is not None and inj.allow_temperature:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs[caps.max_tokens_param] = req.max_tokens

        return kwargs

    def call(self, client, encoded: dict):
        return client.chat.completions.create(**encoded)
