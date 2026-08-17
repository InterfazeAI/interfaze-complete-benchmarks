"""Gemini encode: native genai config object. Asserts on the config's thinking
field (level vs budget) and that media parts become genai Parts. Uses the real
google-genai types (offline — no network)."""

from src.capabilities import Capabilities
from src.providers.gemini import GeminiAdapter
from src.request import ImagePart, Message, ReasoningSpec, Request, TextPart

LEVEL_CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "thinking_level",
            "off_value": "low",
            "on_value": "high",
            "true_off": False,
        },
        "media": {"image": "gemini_part"},
        "response": "gemini_text",
    }
)
BUDGET_CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "thinking_budget",
            "off_value": 0,
            "on_value": -1,
            "true_off": True,
        },
        "media": {"image": "gemini_part"},
        "response": "gemini_text",
    }
)
ADAPTER = GeminiAdapter(
    name="gemini", base_url=None, key_spec=["GEMINI_API_KEY"], capability_defaults={}
)


def test_thinking_level_floor_goes_into_config():
    req = Request(
        [Message("user", [TextPart("hi")])], ReasoningSpec("off"), temperature=0.0
    )
    enc = ADAPTER.encode(req, LEVEL_CAPS, "gemini-3.7-flash")
    assert enc["model"] == "gemini-3.7-flash"
    # the SDK coerces "low" -> ThinkingLevel.LOW enum; compare on value, case-insensitive
    assert str(enc["config"].thinking_config.thinking_level.value).lower() == "low"
    assert enc["config"].temperature == 0.0


def test_thinking_budget_zero_goes_into_config():
    req = Request([Message("user", [TextPart("hi")])], ReasoningSpec("off"))
    enc = ADAPTER.encode(req, BUDGET_CAPS, "gemini-2.5-flash")
    assert enc["config"].thinking_config.thinking_budget == 0


def test_image_part_becomes_genai_part():
    req = Request(
        [Message("user", [TextPart("read"), ImagePart(b"\x89PNGx", "image/png")])],
        ReasoningSpec("off"),
    )
    contents = ADAPTER.encode(req, LEVEL_CAPS, "gemini-3.7-flash")["contents"]
    # text passes through as a string; the image becomes a genai Part with inline bytes
    assert "read" in contents
    part = contents[-1]
    assert getattr(part, "inline_data", None) is not None
