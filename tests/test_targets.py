"""Every shipped target must load, resolve typed capabilities and build its adapter
offline, so a bad targets.yaml entry fails CI rather than mid-run.
"""

import pytest

from src.capabilities import Capabilities
from src.config import build_adapter, load_all_targets, resolve_capabilities

TARGETS = load_all_targets()
TARGET_NAMES = sorted(TARGETS)


def test_targets_exist():
    assert TARGET_NAMES, "no targets defined in src/targets.yaml"


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_target_resolves_and_builds_adapter(name):
    target = TARGETS[name]
    caps = resolve_capabilities(target)
    assert isinstance(caps, Capabilities)
    resolve_capabilities(target, benchmark="asr")  # per-benchmark path must resolve too
    adapter = build_adapter(target)
    assert adapter.name == target.provider
    assert adapter.key_spec


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_reasoning_keys_are_not_yaml_booleans_by_accident(name):
    # bare off:/on: keys parse as YAML booleans and silently drop the reasoning values
    reasoning = TARGETS[name].capabilities.get("reasoning", {})
    assert False not in reasoning and True not in reasoning, (
        f"target {name!r} used bare off:/on: keys (parsed as booleans) — "
        "use off_value:/on_value:"
    )
