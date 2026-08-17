"""The reasoning translation — the single place the audit's "6 different ways"
of encoding a thinking floor collapse into. Every case here is a real quirk
that currently lives as a scattered special-case:

- OpenAI: reasoning_effort="none", and temperature only when reasoning is off
- Fireworks: reasoning_effort="none" is a FLOOR (still thinks), temp always sent
- Gemini 3.7-flash: thinking_level floor "low" (rejects "minimal")
- Gemini 3.x-flash <3.7: thinking_level floor "minimal"
- Gemini 2.5-flash: thinking_budget=0 (true off); 2.5-pro: budget=128 (floor)
- Anthropic: thinking={"type":"disabled"} off / enabled+budget on, temp omitted on
- OpenRouter default: reasoning.enabled bool; grok/google: reasoning.effort (can't disable)
- Moonshot: reasoning.enabled=false + provider pin
"""

from src.capabilities import ReasoningCap, ReasoningStyle
from src.reasoning import build_reasoning


def test_none_style_injects_nothing():
    cap = ReasoningCap(style=ReasoningStyle.NONE)
    inj = build_reasoning("high", cap)
    assert inj.kwargs == {}
    assert inj.extra_body == {}
    assert inj.thinking_level is None
    assert inj.thinking_budget is None
    assert inj.anthropic_thinking is None
    assert inj.allow_temperature is True


def test_openai_effort_off_sends_none_and_allows_temperature():
    # OpenAI "none" is a true off, so temperature is allowed on the off run.
    cap = ReasoningCap(
        style=ReasoningStyle.EFFORT,
        off_value="none",
        on_value="high",
        true_off=True,
        temperature_when_on=False,
    )
    inj = build_reasoning("off", cap)
    assert inj.kwargs == {"reasoning_effort": "none"}
    assert inj.allow_temperature is True


def test_openai_effort_high_omits_temperature():
    # GPT-5.x rejects temperature!=default once reasoning engages.
    cap = ReasoningCap(
        style=ReasoningStyle.EFFORT,
        off_value="none",
        on_value="high",
        true_off=True,
        temperature_when_on=False,
    )
    inj = build_reasoning("high", cap)
    assert inj.kwargs == {"reasoning_effort": "high"}
    assert inj.allow_temperature is False


def test_fireworks_effort_floor_still_allows_temperature_on_high():
    # Fireworks/Inkling sends temperature regardless of reasoning.
    cap = ReasoningCap(
        style=ReasoningStyle.EFFORT,
        off_value="none",
        on_value="high",
        true_off=False,
        temperature_when_on=True,
    )
    inj = build_reasoning("high", cap)
    assert inj.kwargs == {"reasoning_effort": "high"}
    assert inj.allow_temperature is True


def test_effort_intermediate_mode_passes_through():
    cap = ReasoningCap(style=ReasoningStyle.EFFORT, off_value="none", on_value="high")
    assert build_reasoning("low", cap).kwargs == {"reasoning_effort": "low"}


def test_gemini_37_flash_thinking_level_floor_is_low():
    cap = ReasoningCap(
        style=ReasoningStyle.THINKING_LEVEL,
        off_value="low",
        on_value="high",
        true_off=False,
    )
    inj = build_reasoning("off", cap)
    assert inj.thinking_level == "low"
    assert inj.thinking_budget is None


def test_gemini_old_flash_thinking_level_floor_is_minimal():
    cap = ReasoningCap(
        style=ReasoningStyle.THINKING_LEVEL,
        off_value="minimal",
        on_value="high",
        true_off=False,
    )
    assert build_reasoning("off", cap).thinking_level == "minimal"


def test_gemini_25_flash_budget_zero_is_true_off():
    cap = ReasoningCap(
        style=ReasoningStyle.THINKING_BUDGET, off_value=0, on_value=-1, true_off=True
    )
    inj = build_reasoning("off", cap)
    assert inj.thinking_budget == 0
    assert inj.thinking_level is None


def test_gemini_25_pro_budget_floor_is_128():
    cap = ReasoningCap(
        style=ReasoningStyle.THINKING_BUDGET, off_value=128, on_value=-1, true_off=False
    )
    assert build_reasoning("off", cap).thinking_budget == 128
    assert build_reasoning("high", cap).thinking_budget == -1


def test_anthropic_disabled_block_off_allows_temperature():
    cap = ReasoningCap(
        style=ReasoningStyle.DISABLED_BLOCK,
        on_value=10000,
        true_off=True,
        temperature_when_on=False,
    )
    inj = build_reasoning("off", cap)
    assert inj.anthropic_thinking == {"type": "disabled"}
    assert inj.allow_temperature is True


def test_anthropic_disabled_block_on_enables_budget_and_omits_temperature():
    cap = ReasoningCap(
        style=ReasoningStyle.DISABLED_BLOCK,
        on_value=10000,
        true_off=True,
        temperature_when_on=False,
    )
    inj = build_reasoning("high", cap)
    assert inj.anthropic_thinking == {"type": "enabled", "budget_tokens": 10000}
    assert inj.allow_temperature is False


def test_openrouter_enabled_toggle_off_disables():
    cap = ReasoningCap(
        style=ReasoningStyle.REASONING_BODY,
        off_value=False,
        on_value=True,
        true_off=True,
        extra={"toggle": "enabled"},
    )
    inj = build_reasoning("off", cap)
    assert inj.extra_body == {"reasoning": {"enabled": False}}


def test_openrouter_effort_toggle_cannot_disable_uses_minimal_floor():
    # grok/google via OpenRouter reject enabled=false; floor is effort=minimal.
    cap = ReasoningCap(
        style=ReasoningStyle.REASONING_BODY,
        off_value="minimal",
        on_value="high",
        true_off=False,
        extra={"toggle": "effort"},
    )
    inj = build_reasoning("off", cap)
    assert inj.extra_body == {"reasoning": {"effort": "minimal"}}


def test_openrouter_provider_pin_is_emitted():
    cap = ReasoningCap(
        style=ReasoningStyle.REASONING_BODY,
        off_value=False,
        on_value=True,
        extra={"toggle": "enabled", "provider_only": ["moonshotai"]},
    )
    inj = build_reasoning("off", cap)
    assert inj.extra_body["reasoning"] == {"enabled": False}
    assert inj.extra_body["provider"] == {"only": ["moonshotai"]}
