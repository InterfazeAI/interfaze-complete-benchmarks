"""Anthropic encode: system-message separation, the disabled/enabled thinking
block, temperature omission under thinking, and the max_tokens > budget_tokens
coupling (all declarative via the cap).
"""

from bench_core.capabilities import Capabilities
from bench_core.providers.anthropic import AnthropicAdapter
from bench_core.request import ImagePart, Message, ReasoningSpec, Request, TextPart

CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "disabled_block",
            "on_value": 10000,
            "true_off": True,
            "temperature_when_on": False,
            "extra": {"off_max_tokens": 1024, "on_max_tokens": 16000},
        },
        "media": {"image": "anthropic_source"},
        "response": "anthropic_blocks",
    }
)
ADAPTER = AnthropicAdapter(
    name="anthropic",
    base_url=None,
    key_spec=["ANTHROPIC_API_KEY"],
    capability_defaults={},
)


def test_off_disables_thinking_and_keeps_temperature():
    req = Request(
        [Message("user", [TextPart("hi")])], ReasoningSpec("off"), temperature=0.0
    )
    kw = ADAPTER.encode(req, CAPS, "claude-sonnet-4-6")
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["max_tokens"] == 1024
    assert kw["temperature"] == 0.0


def test_high_enables_budget_bumps_max_tokens_and_omits_temperature():
    req = Request(
        [Message("user", [TextPart("hi")])], ReasoningSpec("high"), temperature=0.0
    )
    kw = ADAPTER.encode(req, CAPS, "claude-sonnet-4-6")
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert kw["max_tokens"] > 10000  # must exceed budget
    assert "temperature" not in kw


def test_system_message_is_separated():
    req = Request(
        [
            Message("system", [TextPart("You are terse.")]),
            Message("user", [TextPart("hi")]),
        ],
        ReasoningSpec("off"),
    )
    kw = ADAPTER.encode(req, CAPS, "claude-sonnet-4-6")
    assert kw["system"] == "You are terse."
    assert [m["role"] for m in kw["messages"]] == ["user"]


def test_image_uses_anthropic_source_block():
    req = Request(
        [Message("user", [TextPart("read"), ImagePart(b"\xff\xd8j", "image/jpeg")])],
        ReasoningSpec("off"),
    )
    content = ADAPTER.encode(req, CAPS, "claude-sonnet-4-6")["messages"][0]["content"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
