"""Media block shapes — the audit's most dangerous fragmentation. Four audio
shapes and several image shapes look almost interchangeable but are host-
specific and fail *silently* when wrong (an audio_url data-URI sent to
OpenRouter is accepted, the audio dropped, and the model hallucinates a
transcript at WER~1.0). These tests pin each shape exactly.
"""

import base64
import io

from bench_core.capabilities import AudioShape, ImageShape
from bench_core.media import GeminiPart, audio_block, image_block
from bench_core.request import AudioPart, ImagePart

WAV = b"RIFF....WAVEfake-audio-bytes"
JPEG = b"\xff\xd8\xff\xe0jpeg-bytes"
PNG = b"\x89PNG\r\n\x1a\npng-bytes"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --- audio ---------------------------------------------------------------


def test_audio_file_block_is_interfaze_data_uri():
    block = audio_block(AudioPart(WAV, "audio/wav"), AudioShape.FILE_BLOCK)
    assert block["type"] == "file"
    assert block["file"]["file_data"] == f"data:audio/wav;base64,{_b64(WAV)}"


def test_audio_url_block_is_fireworks_data_uri():
    block = audio_block(AudioPart(WAV, "audio/wav"), AudioShape.AUDIO_URL)
    assert block == {
        "type": "audio_url",
        "audio_url": {"url": f"data:audio/wav;base64,{_b64(WAV)}"},
    }


def test_input_audio_block_is_openrouter_raw_b64_with_format():
    # This is the anti-silent-drop shape: raw base64 (NOT a data-URI) + format.
    block = audio_block(AudioPart(WAV, "audio/wav"), AudioShape.INPUT_AUDIO)
    assert block == {
        "type": "input_audio",
        "input_audio": {"data": _b64(WAV), "format": "wav"},
    }
    assert "data:" not in block["input_audio"]["data"]


def test_audio_gemini_part_carries_bytes_and_mime():
    block = audio_block(AudioPart(WAV, "audio/wav"), AudioShape.GEMINI_PART)
    assert block == GeminiPart(data=WAV, mime="audio/wav")


# --- image ---------------------------------------------------------------


def test_image_url_block_jpeg_data_uri():
    block = image_block(ImagePart(JPEG, "image/jpeg"), ImageShape.IMAGE_URL)
    assert block == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_b64(JPEG)}"},
    }


def test_image_url_block_preserves_png_mime_for_olmocr():
    block = image_block(ImagePart(PNG, "image/png"), ImageShape.IMAGE_URL)
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_anthropic_source_block():
    block = image_block(ImagePart(JPEG, "image/jpeg"), ImageShape.ANTHROPIC_SOURCE)
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(JPEG)},
    }


def test_image_gemini_part_marker():
    block = image_block(ImagePart(JPEG, "image/jpeg"), ImageShape.GEMINI_PART)
    assert block == GeminiPart(data=JPEG, mime="image/jpeg")


# --- encode_image (central RGB-convert + resize) -------------------------


def test_encode_image_converts_rgba_to_jpeg_without_crashing():
    # PIL raises OSError saving RGBA as JPEG; the convert must happen here so a
    # transparent PNG sample doesn't crash the run (it did in 8/9 OCR variants).
    from PIL import Image

    from bench_core.media import encode_image

    rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    part = encode_image(rgba, mime="image/jpeg")
    assert part.mime == "image/jpeg"
    assert Image.open(io.BytesIO(part.data)).mode == "RGB"


def test_encode_image_downscales_to_max_side():
    from PIL import Image

    from bench_core.media import encode_image

    big = Image.new("RGB", (3000, 1500))
    part = encode_image(big, mime="image/jpeg", max_side=1536)
    assert max(Image.open(io.BytesIO(part.data)).size) == 1536


def test_encode_image_png_keeps_alpha():
    from PIL import Image

    from bench_core.media import encode_image

    rgba = Image.new("RGBA", (10, 10), (0, 255, 0, 64))
    part = encode_image(rgba, mime="image/png")
    assert Image.open(io.BytesIO(part.data)).mode == "RGBA"
