"""Target loading + the capability merge chain. Targets live in one file under a
`targets:` map (key = name); a target declares only what differs from its
provider's defaults, and per-benchmark / CLI overrides layer on top.
build_adapter maps the provider to the right adapter class.
"""

import textwrap

import pytest

from bench_core.capabilities import AudioShape, ReasoningStyle
from bench_core.config import (
    build_adapter,
    build_routes,
    load_all_targets,
    load_target,
    resolve_capabilities,
)
from bench_core.providers.gemini import GeminiAdapter
from bench_core.providers.openai_compat import OpenAICompatAdapter


def _targets(tmp_path, body):
    p = tmp_path / "targets.yaml"
    p.write_text("targets:\n" + textwrap.indent(textwrap.dedent(body), "  "))
    return p


def test_load_target_parses_core_fields(tmp_path):
    path = _targets(
        tmp_path,
        """
        inkling:
          provider: fireworks
          model_id: accounts/fireworks/models/inkling
    """,
    )
    t = load_target("inkling", path)
    assert t.name == "inkling"
    assert t.provider == "fireworks"
    assert t.model_id == "accounts/fireworks/models/inkling"


def test_load_all_targets_returns_every_entry(tmp_path):
    path = _targets(
        tmp_path,
        """
        a:
          provider: fireworks
          model_id: x
        b:
          provider: gemini
          model_id: y
    """,
    )
    targets = load_all_targets(path)
    assert set(targets) == {"a", "b"}


def test_unknown_target_name_raises(tmp_path):
    import pytest

    path = _targets(
        tmp_path,
        """
        a:
          provider: fireworks
          model_id: x
    """,
    )
    with pytest.raises(ValueError, match="unknown target"):
        load_target("nope", path)


def test_target_inherits_provider_defaults(tmp_path):
    # A fireworks target declares no media; it should inherit audio_url + the
    # "none is a floor" reasoning semantics from the provider defaults.
    path = _targets(
        tmp_path,
        """
        inkling:
          provider: fireworks
          model_id: x
    """,
    )
    caps = resolve_capabilities(load_target("inkling", path))
    assert caps.media.audio is AudioShape.AUDIO_URL
    assert caps.reasoning.style is ReasoningStyle.EFFORT
    assert caps.reasoning.true_off is False  # Fireworks "none" still thinks


def test_target_overrides_win_over_defaults(tmp_path):
    # gemini-2.5-flash uses the budget knob, overriding the thinking_level default.
    path = _targets(
        tmp_path,
        """
        gemini-2.5-flash:
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
    caps = resolve_capabilities(load_target("gemini-2.5-flash", path))
    assert caps.reasoning.style is ReasoningStyle.THINKING_BUDGET
    assert caps.reasoning.off_value == 0


def test_per_benchmark_override_applies(tmp_path):
    path = _targets(
        tmp_path,
        """
        gemini-3.7-flash:
          provider: gemini
          model_id: gemini-3.7-flash
          capabilities:
            reasoning: { style: thinking_level, off_value: low, on_value: high, true_off: false }
          overrides:
            asr:
              reasoning: { off_value: high }
    """,
    )
    t = load_target("gemini-3.7-flash", path)
    assert resolve_capabilities(t, benchmark="gpqa").reasoning.off_value == "low"
    assert resolve_capabilities(t, benchmark="asr").reasoning.off_value == "high"


def test_cli_override_wins_over_everything(tmp_path):
    path = _targets(
        tmp_path,
        """
        inkling:
          provider: fireworks
          model_id: x
    """,
    )
    caps = resolve_capabilities(
        load_target("inkling", path), cli_overrides={"reasoning": {"on_value": "max"}}
    )
    assert caps.reasoning.on_value == "max"


def test_build_adapter_maps_provider_to_class(tmp_path):
    path = _targets(
        tmp_path,
        """
        a:
          provider: fireworks
          model_id: x
        b:
          provider: gemini
          model_id: y
    """,
    )
    a = build_adapter(load_target("a", path))
    assert isinstance(a, OpenAICompatAdapter)
    assert "fireworks_api_key" in [k.lower() for k in a.key_spec]
    assert a.base_url and "fireworks" in a.base_url
    assert isinstance(build_adapter(load_target("b", path)), GeminiAdapter)


def test_fallbacks_build_ordered_routes_with_own_caps(tmp_path):
    path = _targets(
        tmp_path,
        """
        inkling-small:
          provider: openrouter
          model_id: thinkingmachines/inkling-small
          fallbacks:
            - provider: fireworks
              model_id: accounts/fireworks/models/inkling-small
        """,
    )
    routes = build_routes(load_target("inkling-small", path))
    assert [(r.provider, r.model_id) for r in routes] == [
        ("openrouter", "thinkingmachines/inkling-small"),
        ("fireworks", "accounts/fireworks/models/inkling-small"),
    ]
    # each route resolves its OWN provider's caps, not the primary's
    assert routes[0].caps.reasoning.style is ReasoningStyle.REASONING_BODY
    assert routes[1].caps.reasoning.style is ReasoningStyle.EFFORT


def test_no_fallbacks_is_single_route(tmp_path):
    path = _targets(
        tmp_path,
        """
        solo:
          provider: interfaze
          model_id: interfaze-x
        """,
    )
    assert len(build_routes(load_target("solo", path))) == 1


def test_fallback_is_validated(tmp_path):
    path = _targets(
        tmp_path,
        """
        bad:
          provider: openrouter
          model_id: or/x
          fallbacks:
            - provider: nope
              model_id: y
        """,
    )
    with pytest.raises(ValueError):
        load_all_targets(path)
