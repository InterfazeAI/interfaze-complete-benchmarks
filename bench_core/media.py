"""Media block builders — one per host shape (image + audio)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

from bench_core.capabilities import AudioShape, ImageShape
from bench_core.request import AudioPart, ImagePart


@dataclass(frozen=True)
class GeminiPart:
    """Marker for a native `google.genai` part. Kept SDK-free here so media
    logic is testable without the SDK; the Gemini adapter maps it to
    `types.Part.from_bytes(data=..., mime_type=...)`."""

    data: bytes
    mime: str


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_b64(data)}"


def _subtype(mime: str) -> str:
    """ "audio/wav" -> "wav"."""
    return mime.split("/", 1)[-1]


def audio_block(part: AudioPart, shape: AudioShape) -> Any:
    if shape is AudioShape.FILE_BLOCK:
        return {
            "type": "file",
            "file": {
                "filename": f"audio.{_subtype(part.mime)}",
                "file_data": _data_uri(part.data, part.mime),
            },
        }
    if shape is AudioShape.AUDIO_URL:
        return {
            "type": "audio_url",
            "audio_url": {"url": _data_uri(part.data, part.mime)},
        }
    if shape is AudioShape.INPUT_AUDIO:
        # Raw base64 (NOT a data-URI) + explicit format. Sending the audio_url
        # data-URI here is accepted but silently drops the audio.
        return {
            "type": "input_audio",
            "input_audio": {"data": _b64(part.data), "format": _subtype(part.mime)},
        }
    if shape is AudioShape.GEMINI_PART:
        return GeminiPart(data=part.data, mime=part.mime)
    raise ValueError(f"cannot send audio to a model with audio shape {shape!r}")


def image_block(part: ImagePart, shape: ImageShape) -> Any:
    if shape is ImageShape.IMAGE_URL:
        return {
            "type": "image_url",
            "image_url": {"url": _data_uri(part.data, part.mime)},
        }
    if shape is ImageShape.ANTHROPIC_SOURCE:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime,
                "data": _b64(part.data),
            },
        }
    if shape is ImageShape.GEMINI_PART:
        return GeminiPart(data=part.data, mime=part.mime)
    raise ValueError(f"cannot send image to a model with image shape {shape!r}")


def encode_image(
    pil_image, mime: str = "image/jpeg", max_side: int | None = None
) -> ImagePart:
    """PIL image -> ImagePart, doing the conversions every consumer needs once:
    RGB-convert for JPEG (JPEG can't hold alpha; the omission crashed 8/9 OCR
    variants on RGBA/P images), optional longest-side downscale, then encode."""
    from PIL import Image

    img = pil_image
    if max_side is not None:
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / longest
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    fmt = _subtype(mime).upper()
    if fmt in ("JPG", "JPEG"):
        fmt = "JPEG"
        if img.mode != "RGB":
            img = img.convert("RGB")

    buf = io.BytesIO()
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    img.save(buf, format=fmt, **save_kwargs)
    return ImagePart(data=buf.getvalue(), mime=mime)
