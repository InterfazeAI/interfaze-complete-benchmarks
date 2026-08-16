import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# The project .env may spell this key lowercase (`fireworks_api_key`);
# load_dotenv() copies the name verbatim, so accept either casing.
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("fireworks_api_key")

if FIREWORKS_API_KEY is None:
    raise ValueError(
        "FIREWORKS_API_KEY is not set in environment variables get it from https://app.fireworks.ai/settings/users/api-keys"
    )

FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)

# Inkling (Thinking Machines) — text+image+audio in, text out, 1M context.
# inkling-small is NOT on Fireworks serverless; it needs an on-demand
# dedicated deployment, after which its model id is the same string.
INKLING = "accounts/fireworks/models/inkling"
INKLING_SMALL = "accounts/fireworks/models/inkling-small"

# Inkling always thinks. reasoning_effort accepts none|low|medium|high|xhigh|max
# — "none" is the floor (still emits reasoning tokens), there is no true off.
REASONING_FLOOR = "none"
REASONING_MAX = "high"

fireworks_client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)


def invoke_fireworks(
    messages: list[dict],
    model: str = INKLING,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
):
    """Invoke a Fireworks-hosted model over the OpenAI-compatible endpoint.

    Image input uses the standard `image_url` block with a data URL. Audio
    input uses an `audio_url` block with a data URL (Fireworks rejects
    OpenAI's `input_audio` shape for this model):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
            ],
        }]
    """
    kwargs: dict = {"model": model, "messages": messages, "stream": stream}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    try:
        return fireworks_client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error invoking Fireworks API: {e}") from e
