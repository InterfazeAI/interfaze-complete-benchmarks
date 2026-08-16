"""Targets + provider registry + the capability merge chain.

A provider ships defaults (base_url, key precedence, and the host's baseline
capabilities); a target YAML declares only what differs; per-benchmark and CLI
overrides layer on top. `resolve_capabilities` collapses the chain into a typed
`Capabilities`; `build_adapter` maps the provider name to its adapter class.

Adding a MODEL is a YAML file. Adding a PROVIDER is one registry entry here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bench_core.capabilities import Capabilities, deep_merge
from bench_core.providers.anthropic import AnthropicAdapter
from bench_core.providers.gemini import GeminiAdapter
from bench_core.providers.openai_compat import OpenAICompatAdapter

# --- provider registry ----------------------------------------------------


@dataclass
class ProviderSpec:
    adapter_cls: type
    base_url: str | None
    key_spec: list[str]
    capability_defaults: dict


_OPENAI_MEDIA = {"image": "image_url"}

PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        OpenAICompatAdapter,
        "https://api.openai.com/v1",
        ["OPENAI_API_KEY"],
        {
            "reasoning": {
                "style": "effort",
                "off_value": "none",
                "on_value": "high",
                "true_off": True,
                "temperature_when_on": False,
            },
            "media": _OPENAI_MEDIA,
            "response": "openai_chat",
        },
    ),
    "fireworks": ProviderSpec(
        OpenAICompatAdapter,
        "https://api.fireworks.ai/inference/v1",
        ["FIREWORKS_API_KEY", "fireworks_api_key"],
        {
            # Fireworks "none" is a FLOOR (still thinks); temperature is sent regardless.
            "reasoning": {
                "style": "effort",
                "off_value": "none",
                "on_value": "high",
                "true_off": False,
                "temperature_when_on": True,
            },
            "media": {"image": "image_url", "audio": "audio_url"},
            "response": "openai_chat",
        },
    ),
    "openrouter": ProviderSpec(
        OpenAICompatAdapter,
        "https://openrouter.ai/api/v1",
        ["OPENROUTER_API_KEY", "openrouter_api_key"],
        {
            "reasoning": {
                "style": "reasoning_body",
                "off_value": False,
                "on_value": True,
                "true_off": True,
                "extra": {"toggle": "enabled"},
            },
            "media": {"image": "image_url", "audio": "input_audio"},
            "response": "openai_chat",
        },
    ),
    "interfaze": ProviderSpec(
        OpenAICompatAdapter,
        None,
        ["INTERFAZE_API_KEY"],
        {
            # Interfaze reasoning vocab is off/minimal/low/medium/high/on/auto (NOT "none").
            "reasoning": {
                "style": "effort",
                "off_value": "off",
                "on_value": "high",
                "true_off": True,
                "temperature_when_on": True,
            },
            "media": {"image": "image_url", "audio": "file_block"},
            "response": "openai_chat",
        },
    ),
    "anthropic": ProviderSpec(
        AnthropicAdapter,
        None,
        ["ANTHROPIC_API_KEY"],
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
        },
    ),
    "gemini": ProviderSpec(
        GeminiAdapter,
        None,
        ["GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"],
        {
            # 3.x default; 2.5 targets override to thinking_budget. "low" floor is
            # safe for 3.7+ flash and Pro (both reject "minimal").
            "reasoning": {
                "style": "thinking_level",
                "off_value": "low",
                "on_value": "high",
                "true_off": False,
            },
            "media": {"image": "gemini_part", "audio": "gemini_part"},
            "response": "gemini_text",
        },
    ),
}


# --- targets ---------------------------------------------------------------


@dataclass
class Target:
    name: str
    provider: str
    model_id: str
    capabilities: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    harness_specific: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def load_target(path: str | Path) -> Target:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if data.get("provider") not in PROVIDERS:
        raise ValueError(
            f"unknown provider {data.get('provider')!r}; known: {sorted(PROVIDERS)}"
        )
    return Target(
        name=data["name"],
        provider=data["provider"],
        model_id=data["model_id"],
        capabilities=data.get("capabilities") or {},
        overrides=data.get("overrides") or {},
        harness_specific=data.get("harness_specific") or {},
        raw=data,
    )


def resolve_capabilities(
    target: Target,
    benchmark: str | None = None,
    cli_overrides: dict | None = None,
) -> Capabilities:
    spec = PROVIDERS[target.provider]
    merged: dict[str, Any] = deep_merge(spec.capability_defaults, target.capabilities)
    if benchmark and benchmark in target.overrides:
        merged = deep_merge(merged, target.overrides[benchmark])
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
    return Capabilities.from_dict(merged)


def build_adapter(target: Target):
    spec = PROVIDERS[target.provider]
    base_url = spec.base_url
    if target.provider == "interfaze":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.interfaze.ai/v1")
    return spec.adapter_cls(
        name=target.provider,
        base_url=base_url,
        key_spec=list(spec.key_spec),
        capability_defaults=spec.capability_defaults,
    )
