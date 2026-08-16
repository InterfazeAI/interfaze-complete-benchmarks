"""Interactive add-target: append a validated entry to targets.yaml, and the
merge-diff helper that decides which targets a push should benchmark.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from add_target import add_target
from ci_matrix import changed_targets

from bench_core.capabilities import ReasoningStyle
from bench_core.config import load_all_targets, resolve_capabilities


def _base(tmp_path):
    p = tmp_path / "targets.yaml"
    p.write_text(
        "targets:\n  x:\n    provider: fireworks\n    model_id: accounts/fireworks/models/x\n"
    )
    return p


def test_append_and_validate(tmp_path):
    p = _base(tmp_path)
    add_target("newmodel", "gemini", "gemini-3.7-flash", path=p)
    targets = load_all_targets(p)
    assert "x" in targets and "newmodel" in targets  # existing preserved
    assert targets["newmodel"].provider == "gemini"
    resolve_capabilities(targets["newmodel"])  # resolves without error


def test_append_with_capabilities_json(tmp_path):
    p = _base(tmp_path)
    add_target(
        "g25",
        "gemini",
        "gemini-2.5-flash",
        capabilities_json='{"reasoning":{"style":"thinking_budget","off_value":0,"on_value":-1,"true_off":true}}',
        ci_regression=True,
        path=p,
    )
    t = load_all_targets(p)["g25"]
    assert t.raw.get("ci_regression") is True
    caps = resolve_capabilities(t)
    assert caps.reasoning.style is ReasoningStyle.THINKING_BUDGET
    assert caps.reasoning.off_value == 0


def test_rejects_unknown_provider(tmp_path):
    with pytest.raises(SystemExit, match="unknown provider"):
        add_target("z", "telepathy", "m", path=_base(tmp_path))


def test_rejects_duplicate(tmp_path):
    with pytest.raises(SystemExit, match="already exists"):
        add_target("x", "fireworks", "y", path=_base(tmp_path))


def test_changed_targets_detects_added_and_modified(tmp_path):
    old = tmp_path / "old.yaml"
    new = tmp_path / "new.yaml"
    old.write_text("targets:\n  a:\n    provider: fireworks\n    model_id: x\n")
    new.write_text(
        "targets:\n  a:\n    provider: fireworks\n    model_id: x\n"
        "  b:\n    provider: gemini\n    model_id: y\n"
    )
    assert changed_targets(str(old), str(new)) == ["b"]  # only the added one

    new.write_text("targets:\n  a:\n    provider: openai\n    model_id: x\n")
    assert changed_targets(str(old), str(new)) == ["a"]  # modified entry


def test_changed_targets_empty_baseline_runs_nothing(tmp_path):
    # first push / force-push / no old file -> don't benchmark the whole registry
    new = tmp_path / "new.yaml"
    new.write_text("targets:\n  a:\n    provider: fireworks\n    model_id: x\n")
    assert changed_targets(None, str(new)) == []
    assert changed_targets(str(tmp_path / "missing.yaml"), str(new)) == []


def test_changed_targets_ignores_ci_regression_flip(tmp_path):
    # toggling ci_regression must NOT trigger a (paid) benchmark — model unchanged
    old = tmp_path / "old.yaml"
    new = tmp_path / "new.yaml"
    old.write_text("targets:\n  a:\n    provider: fireworks\n    model_id: x\n")
    new.write_text(
        "targets:\n  a:\n    provider: fireworks\n    model_id: x\n    ci_regression: true\n"
    )
    assert changed_targets(str(old), str(new)) == []
