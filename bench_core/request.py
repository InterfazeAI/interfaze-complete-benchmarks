"""Provider-agnostic request the benchmark layer builds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/jpeg"


@dataclass
class AudioPart:
    data: bytes
    mime: str = "audio/wav"


Part = TextPart | ImagePart | AudioPart


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    parts: list[Part] = field(default_factory=list)


@dataclass
class ReasoningSpec:
    mode: str = "off"  # off | low | medium | high (host floor applied by the cap)


@dataclass
class Request:
    messages: list[Message]
    reasoning: ReasoningSpec = field(default_factory=ReasoningSpec)
    temperature: float | None = 0.0
    max_tokens: int | None = None
    schema: dict | None = None  # optional structured-output JSON schema
