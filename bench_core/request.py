"""Provider-agnostic request the benchmark layer builds. Benchmarks construct
these and never name a provider; the adapter's `encode` turns them into the
host-native payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    """Already-encoded image bytes + its mime (e.g. "image/jpeg" for OCRBench,
    "image/png" for olmOCR — a *benchmark* choice; the wire *shape* is the
    adapter's job). Build via `media.encode_image` so RGB-convert/resize happen
    once, centrally, instead of in 1-of-9 variants."""

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
