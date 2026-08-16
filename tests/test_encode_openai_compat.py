"""OpenAI-compatible encode — the assembly point where reasoning placement,
temperature gating, media blocks and the max-tokens param name all come
together for openai / fireworks / openrouter / interfaze.
"""

from bench_core.capabilities import Capabilities
from bench_core.providers.openai_compat import OpenAICompatAdapter
from bench_core.request import ImagePart, Message, ReasoningSpec, Request, TextPart

OPENAI_CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "effort",
            "off_value": "none",
            "on_value": "high",
            "true_off": True,
            "temperature_when_on": False,
        },
        "media": {"image": "image_url"},
        "response": "openai_chat",
    }
)
FIREWORKS_CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "effort",
            "off_value": "none",
            "on_value": "high",
            "true_off": False,
            "temperature_when_on": True,
        },
        "media": {"image": "image_url"},
        "response": "openai_chat",
        "max_tokens_param": "max_completion_tokens",
    }
)
OPENROUTER_CAPS = Capabilities.from_dict(
    {
        "reasoning": {
            "style": "reasoning_body",
            "off_value": False,
            "on_value": True,
            "true_off": True,
            "extra": {"toggle": "enabled"},
        },
        "media": {"image": "image_url"},
        "response": "openai_chat",
    }
)

ADAPTER = OpenAICompatAdapter(
    name="test", base_url="http://x", key_spec=["K"], capability_defaults={}
)


def _text_req(mode, temperature=0.0):
    return Request(
        messages=[Message("user", [TextPart("What is 2+2?")])],
        reasoning=ReasoningSpec(mode=mode),
        temperature=temperature,
    )


def test_text_only_content_is_a_plain_string():
    kw = ADAPTER.encode(_text_req("off"), OPENAI_CAPS, "gpt-5.5")
    assert kw["model"] == "gpt-5.5"
    assert kw["messages"] == [{"role": "user", "content": "What is 2+2?"}]


def test_openai_off_sends_effort_and_temperature():
    kw = ADAPTER.encode(_text_req("off"), OPENAI_CAPS, "gpt-5.5")
    assert kw["reasoning_effort"] == "none"
    assert kw["temperature"] == 0.0


def test_openai_high_omits_temperature():
    kw = ADAPTER.encode(_text_req("high"), OPENAI_CAPS, "gpt-5.5")
    assert kw["reasoning_effort"] == "high"
    assert "temperature" not in kw


def test_fireworks_high_keeps_temperature():
    kw = ADAPTER.encode(_text_req("high"), FIREWORKS_CAPS, "inkling")
    assert kw["reasoning_effort"] == "high"
    assert kw["temperature"] == 0.0


def test_max_tokens_uses_capability_param_name():
    req = _text_req("off")
    req.max_tokens = 4096
    assert (
        ADAPTER.encode(req, FIREWORKS_CAPS, "inkling")["max_completion_tokens"] == 4096
    )
    assert "max_tokens" not in ADAPTER.encode(req, FIREWORKS_CAPS, "inkling")


def test_openrouter_reasoning_goes_to_extra_body():
    kw = ADAPTER.encode(_text_req("off"), OPENROUTER_CAPS, "x-ai/grok-4.3")
    assert "reasoning_effort" not in kw
    assert kw["extra_body"] == {"reasoning": {"enabled": False}}


def test_image_message_becomes_block_list():
    req = Request(
        messages=[
            Message(
                "user",
                [TextPart("Read this."), ImagePart(b"\xff\xd8jpg", "image/jpeg")],
            )
        ],
        reasoning=ReasoningSpec(mode="off"),
    )
    content = ADAPTER.encode(req, OPENAI_CAPS, "gpt-5.5")["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Read this."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_none_temperature_is_omitted():
    kw = ADAPTER.encode(_text_req("off", temperature=None), OPENAI_CAPS, "gpt-5.5")
    assert "temperature" not in kw
