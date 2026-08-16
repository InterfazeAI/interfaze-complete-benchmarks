"""Declared model capabilities + the merge that resolves them.

Every axis the audit found varying across providers/models is a field here, so
that a new model is described by *data* (a target YAML file) rather than a new
code path. Provider adapters ship a default capability dict; the target file
overrides fields; the CLI can override again. `deep_merge` collapses the chain;
`Capabilities.from_dict` gives adapters a typed, validated view.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningStyle(str, Enum):
    """How a host expresses "think this much".

    The vocabulary is genuinely different per host — this enum names the
    mechanism so the adapter can translate an abstract mode into the concrete
    param without a benchmark-level `if provider == ...`.
    """

    EFFORT = (
        "effort"  # OpenAI/Fireworks/Interfaze: reasoning_effort="none|low|high|..."
    )
    THINKING_LEVEL = "thinking_level"  # Gemini 3.x: thinking_config.thinking_level enum
    THINKING_BUDGET = (
        "thinking_budget"  # Gemini 2.5: thinking_config.thinking_budget int
    )
    DISABLED_BLOCK = (
        "disabled_block"  # Anthropic: thinking={"type":"disabled"|"enabled"}
    )
    REASONING_BODY = (
        "reasoning_body"  # OpenRouter: extra_body.reasoning{enabled|effort}
    )
    NONE = "none"  # non-reasoning model: send nothing


class AudioShape(str, Enum):
    FILE_BLOCK = (
        "file_block"  # Interfaze: {"type":"file","file":{"file_data": data-uri}}
    )
    AUDIO_URL = (
        "audio_url"  # Fireworks: {"type":"audio_url","audio_url":{"url": data-uri}}
    )
    INPUT_AUDIO = (
        "input_audio"  # OpenRouter: {"type":"input_audio","input_audio":{data,format}}
    )
    GEMINI_PART = "gemini_part"  # Gemini: types.Part.from_bytes(mime_type="audio/...")
    NONE = "none"


class ImageShape(str, Enum):
    IMAGE_URL = (
        "image_url"  # OpenAI-family: {"type":"image_url","image_url":{"url": data-uri}}
    )
    ANTHROPIC_SOURCE = "anthropic_source"  # {"type":"image","source":{base64}}
    GEMINI_PART = "gemini_part"  # types.Part.from_bytes(mime_type="image/...")
    NONE = "none"


class ResponseShape(str, Enum):
    OPENAI_CHAT = "openai_chat"  # choices[0].message.content
    ANTHROPIC_BLOCKS = "anthropic_blocks"  # content[] blocks, join type=="text"
    GEMINI_TEXT = "gemini_text"  # response.text


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge `override` onto `base`, returning a new dict.

    Nested dicts merge key-by-key; every other value (scalar, list) is replaced
    wholesale by `override`. Neither input is mutated — the resolution chain
    (adapter defaults ← target ← CLI) reuses the same base dict repeatedly.
    """
    result = copy.deepcopy(dict(base))
    for key, val in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(val, Mapping):
            result[key] = deep_merge(existing, val)
        else:
            result[key] = copy.deepcopy(val)
    return result


@dataclass(frozen=True)
class ReasoningCap:
    style: ReasoningStyle
    # NB: keys are `off_value`/`on_value`, NOT `off`/`on` — YAML 1.1 parses bare
    # `off:`/`on:` as booleans, so `off: low` would become `{False: "low"}`.
    off_value: Any = None  # concrete value sent for the "off" mode (may be a FLOOR)
    on_value: Any = None  # concrete value sent for the "high"/"on" mode
    true_off: bool = True  # is `off` a real disable, or does it still think?
    temperature_when_on: bool = True  # may temperature accompany active thinking?
    extra: dict = field(
        default_factory=dict
    )  # provider pins, budget/max coupling, etc.


@dataclass(frozen=True)
class MediaCap:
    audio: AudioShape = AudioShape.NONE
    image: ImageShape = ImageShape.NONE


@dataclass(frozen=True)
class Capabilities:
    reasoning: ReasoningCap
    media: MediaCap
    response: ResponseShape
    max_tokens_param: str = "max_tokens"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Capabilities:
        r = dict(d.get("reasoning") or {})
        try:
            style = ReasoningStyle(r.get("style", "none"))
        except ValueError as exc:
            raise ValueError(
                f"unknown reasoning style {r.get('style')!r}; "
                f"expected one of {[s.value for s in ReasoningStyle]}"
            ) from exc
        reasoning = ReasoningCap(
            style=style,
            off_value=r.get("off_value"),
            on_value=r.get("on_value"),
            true_off=bool(r.get("true_off", True)),
            temperature_when_on=bool(r.get("temperature_when_on", True)),
            extra=dict(r.get("extra") or {}),
        )

        m = dict(d.get("media") or {})
        media = MediaCap(
            audio=AudioShape(m.get("audio", "none")),
            image=ImageShape(m.get("image", "none")),
        )

        return cls(
            reasoning=reasoning,
            media=media,
            response=ResponseShape(d.get("response", "openai_chat")),
            max_tokens_param=d.get("max_tokens_param", "max_tokens"),
        )
