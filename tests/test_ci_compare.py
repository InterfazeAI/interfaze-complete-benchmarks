"""CI regression check: fresh results/ vs the tracked baselines/ tree."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from ci_compare import compare_target


def _write(root, bench, target, metrics):
    path = root / bench / target / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"benchmark": bench, **metrics}))


def test_no_baseline_reports_new(tmp_path):
    _write(tmp_path / "results", "gpqa", "m", {"accuracy": 0.5, "n": 198})
    rows, warnings = compare_target("m", 0.02, tmp_path / "results", tmp_path / "base")
    assert rows == [["gpqa", "accuracy", "0.5000", "new", "—"]]
    assert warnings == []


def test_drop_beyond_tolerance_warns(tmp_path):
    _write(tmp_path / "results", "gpqa", "m", {"accuracy": 0.40, "n": 198})
    _write(tmp_path / "base", "gpqa", "m", {"accuracy": 0.50, "n": 198})
    rows, warnings = compare_target("m", 0.02, tmp_path / "results", tmp_path / "base")
    assert rows[0][3] == "0.5000" and "REGRESSION" in rows[0][4]
    assert warnings == ["gpqa/m: accuracy 0.5000 -> 0.4000 (Δ-0.1000)"]


def test_lower_wer_is_not_a_regression(tmp_path):
    _write(tmp_path / "results", "voxpopuli_aa", "m", {"corpus_wer": 0.05, "n": 100})
    _write(tmp_path / "base", "voxpopuli_aa", "m", {"corpus_wer": 0.10, "n": 100})
    _, warnings = compare_target("m", 0.02, tmp_path / "results", tmp_path / "base")
    assert warnings == []


def test_smoke_run_is_found_under_smoke_root_and_not_judged(tmp_path):
    _write(tmp_path / "results" / "_smoke", "gpqa", "m", {"accuracy": 0.0, "n": 5})
    _write(tmp_path / "base", "gpqa", "m", {"accuracy": 0.50, "n": 198})
    rows, warnings = compare_target("m", 0.02, tmp_path / "results", tmp_path / "base")
    assert "smoke (n=5 vs 198)" in rows[0][4]
    assert warnings == []

