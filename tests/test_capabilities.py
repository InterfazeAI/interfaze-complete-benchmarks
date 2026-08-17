"""Capability merge + typed parsing.

The capability layer is the heart of the redesign: provider adapters ship
default capability dicts, target YAML files override fields, and the CLI can
override again. All three collapse through `deep_merge`, then `Capabilities`
gives the adapters a typed, validated view.
"""

from src.capabilities import (
    AudioShape,
    Capabilities,
    ImageShape,
    ReasoningStyle,
    ResponseShape,
    deep_merge,
)


def test_deep_merge_overrides_nested_scalar_keeping_siblings():
    base = {"reasoning": {"style": "effort", "off_value": "none", "true_off": True}}
    override = {"reasoning": {"off_value": "low"}}
    assert deep_merge(base, override) == {
        "reasoning": {"style": "effort", "off_value": "low", "true_off": True}
    }


def test_deep_merge_does_not_mutate_inputs():
    base = {"reasoning": {"off_value": "none"}}
    override = {"reasoning": {"off_value": "low"}}
    deep_merge(base, override)
    assert base == {"reasoning": {"off_value": "none"}}  # base untouched


def test_deep_merge_adds_new_nested_key():
    base = {"media": {"image": "image_url"}}
    override = {"media": {"audio": "input_audio"}}
    assert deep_merge(base, override) == {
        "media": {"image": "image_url", "audio": "input_audio"}
    }


def test_capabilities_from_dict_coerces_enums():
    caps = Capabilities.from_dict(
        {
            "reasoning": {
                "style": "thinking_level",
                "off_value": "low",
                "on_value": "high",
                "true_off": False,
                "temperature_when_on": True,
            },
            "media": {"audio": "gemini_part", "image": "gemini_part"},
            "response": "gemini_text",
            "max_tokens_param": "max_tokens",
        }
    )
    assert caps.reasoning.style is ReasoningStyle.THINKING_LEVEL
    assert caps.reasoning.true_off is False
    assert caps.media.audio is AudioShape.GEMINI_PART
    assert caps.media.image is ImageShape.GEMINI_PART
    assert caps.response is ResponseShape.GEMINI_TEXT


def test_capabilities_from_dict_rejects_unknown_reasoning_style():
    import pytest

    with pytest.raises(ValueError, match="reasoning style"):
        Capabilities.from_dict(
            {
                "reasoning": {"style": "telepathy"},
                "media": {"audio": "gemini_part", "image": "gemini_part"},
                "response": "gemini_text",
            }
        )
