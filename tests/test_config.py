"""Target loading + the capability merge chain. A target YAML declares only what
differs from its provider's defaults; per-benchmark overrides and CLI overrides
layer on top. build_adapter maps the provider to the right adapter class.
"""

import textwrap

from bench_core.capabilities import AudioShape, ReasoningStyle
from bench_core.config import build_adapter, load_target, resolve_capabilities
from bench_core.providers.gemini import GeminiAdapter
from bench_core.providers.openai_compat import OpenAICompatAdapter


def _write(tmp_path, text):
    p = tmp_path / "target.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_load_target_parses_core_fields(tmp_path):
    t = load_target(
        _write(
            tmp_path,
            """
        name: inkling
        provider: fireworks
        model_id: accounts/fireworks/models/inkling
    """,
        )
    )
    assert t.name == "inkling"
    assert t.provider == "fireworks"
    assert t.model_id == "accounts/fireworks/models/inkling"


def test_target_inherits_provider_defaults(tmp_path):
    # A fireworks target declares no media; it should inherit audio_url + the
    # "none is a floor" reasoning semantics from the provider defaults.
    t = load_target(
        _write(
            tmp_path,
            """
        name: inkling
        provider: fireworks
        model_id: x
    """,
        )
    )
    caps = resolve_capabilities(t)
    assert caps.media.audio is AudioShape.AUDIO_URL
    assert caps.reasoning.style is ReasoningStyle.EFFORT
    assert caps.reasoning.true_off is False  # Fireworks "none" still thinks


def test_target_overrides_win_over_defaults(tmp_path):
    # gemini-2.5-flash uses the budget knob, overriding the thinking_level default.
    t = load_target(
        _write(
            tmp_path,
            """
        name: gemini-2.5-flash
        provider: gemini
        model_id: gemini-2.5-flash
        capabilities:
          reasoning:
            style: thinking_budget
            off_value: 0
            on_value: -1
            true_off: true
    """,
        )
    )
    caps = resolve_capabilities(t)
    assert caps.reasoning.style is ReasoningStyle.THINKING_BUDGET
    assert caps.reasoning.off_value == 0


def test_per_benchmark_override_applies(tmp_path):
    t = load_target(
        _write(
            tmp_path,
            """
        name: gemini-3.7-flash
        provider: gemini
        model_id: gemini-3.7-flash
        capabilities:
          reasoning: { style: thinking_level, off_value: low, on_value: high, true_off: false }
        overrides:
          asr:
            reasoning: { off_value: high }
    """,
        )
    )
    assert resolve_capabilities(t, benchmark="gpqa").reasoning.off_value == "low"
    assert resolve_capabilities(t, benchmark="asr").reasoning.off_value == "high"


def test_cli_override_wins_over_everything(tmp_path):
    t = load_target(
        _write(
            tmp_path,
            """
        name: inkling
        provider: fireworks
        model_id: x
    """,
        )
    )
    caps = resolve_capabilities(t, cli_overrides={"reasoning": {"on_value": "max"}})
    assert caps.reasoning.on_value == "max"


def test_build_adapter_maps_provider_to_class(tmp_path):
    fw = load_target(_write(tmp_path, "name: a\nprovider: fireworks\nmodel_id: x\n"))
    a = build_adapter(fw)
    assert isinstance(a, OpenAICompatAdapter)
    assert "fireworks_api_key" in [k.lower() for k in a.key_spec]
    assert a.base_url and "fireworks" in a.base_url

    gtarget = load_target(_write(tmp_path, "name: b\nprovider: gemini\nmodel_id: y\n"))
    assert isinstance(build_adapter(gtarget), GeminiAdapter)
