"""Gemini (native google-genai) adapter. Reasoning becomes a ThinkingConfig
(level for 3.x, budget for 2.5); media parts become genai Parts. The client is
cached at the factory level to avoid the "client has been closed" GC bug.
"""

from __future__ import annotations

from bench_core.capabilities import Capabilities
from bench_core.media import GeminiPart, audio_block, image_block
from bench_core.providers.base import ProviderAdapter
from bench_core.reasoning import build_reasoning
from bench_core.request import AudioPart, ImagePart, Request, TextPart


class GeminiAdapter(ProviderAdapter):
    def __init__(self, name, base_url, key_spec, capability_defaults):
        self.name = name
        self.base_url = base_url
        self.key_spec = key_spec
        self.capability_defaults = capability_defaults

    def build_client(self, api_key: str | None = None):
        from google import genai

        return genai.Client(api_key=api_key or self.resolve_key())

    def _contents(self, req: Request, caps: Capabilities) -> list:
        from google.genai import types

        contents: list = []
        for m in req.messages:
            for p in m.parts:
                if isinstance(p, TextPart):
                    contents.append(p.text)
                elif isinstance(p, ImagePart):
                    gp = image_block(p, caps.media.image)
                    contents.append(self._to_part(gp, types))
                elif isinstance(p, AudioPart):
                    gp = audio_block(p, caps.media.audio)
                    contents.append(self._to_part(gp, types))
        return contents

    @staticmethod
    def _to_part(gp: GeminiPart, types):
        return types.Part.from_bytes(data=gp.data, mime_type=gp.mime)

    def encode(self, req: Request, caps: Capabilities, model_id: str) -> dict:
        from google.genai import types

        inj = build_reasoning(req.reasoning.mode, caps.reasoning)
        thinking = None
        if inj.thinking_level is not None:
            thinking = types.ThinkingConfig(thinking_level=inj.thinking_level)
        elif inj.thinking_budget is not None:
            thinking = types.ThinkingConfig(thinking_budget=inj.thinking_budget)

        config_kwargs: dict = {}
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking
        if req.temperature is not None and inj.allow_temperature:
            config_kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            config_kwargs["max_output_tokens"] = req.max_tokens

        return {
            "model": model_id,
            "contents": self._contents(req, caps),
            "config": types.GenerateContentConfig(**config_kwargs),
        }

    def call(self, client, encoded: dict):
        return client.models.generate_content(**encoded)
