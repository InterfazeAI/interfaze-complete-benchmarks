"""Every shipped target YAML must load, resolve to typed capabilities, and build
its adapter — with no network. This is the guard behind the "add a YAML, no
code" promise: a new target file is validated here (and in CI on push) rather
than failing mid-run.
"""

from pathlib import Path

import pytest

from bench_core.capabilities import Capabilities
from bench_core.config import build_adapter, load_target, resolve_capabilities

TARGET_DIR = Path(__file__).resolve().parent.parent / "bench_core" / "targets"
TARGET_FILES = sorted(TARGET_DIR.glob("*.yaml"))


def test_targets_exist():
    assert TARGET_FILES, "no target YAML files found"


@pytest.mark.parametrize("path", TARGET_FILES, ids=lambda p: p.stem)
def test_target_loads_resolves_and_builds_adapter(path):
    target = load_target(path)
    # the filename should match the declared name (keeps the dispatch dropdown honest)
    assert target.name == path.stem
    caps = resolve_capabilities(target)
    assert isinstance(caps, Capabilities)
    # per-benchmark override paths must also resolve
    resolve_capabilities(target, benchmark="asr")
    adapter = build_adapter(target)
    assert adapter.name == target.provider
    assert adapter.key_spec  # some env var to read a key from


@pytest.mark.parametrize("path", TARGET_FILES, ids=lambda p: p.stem)
def test_reasoning_off_and_on_values_are_not_yaml_booleans_by_accident(path):
    # Guards the off:/on: -> bool trap: if a future edit uses bare off:/on: keys,
    # the reasoning dict loses its values and this catches it.
    target = load_target(path)
    reasoning = target.capabilities.get("reasoning", {})
    assert False not in reasoning and True not in reasoning, (
        f"{path.name} used bare off:/on: keys (parsed as booleans) — "
        "use off_value:/on_value:"
    )
