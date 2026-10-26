"""CI matrix builder: dispatch (target x chosen benchmarks) and schedule (every
ci_regression target x the CI benchmark set)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from ci_matrix import CI_BENCHMARKS, build_matrix


def test_dispatch_specific_benchmarks():
    m = build_matrix("workflow_dispatch", "inkling", "gpqa,asr")
    assert m["include"] == [
        {"target": "inkling", "benchmark": "gpqa"},
        {"target": "inkling", "benchmark": "asr"},
    ]


def test_dispatch_all_expands_to_ci_set():
    m = build_matrix("workflow_dispatch", "gpt-5.5", "all")
    assert [i["benchmark"] for i in m["include"]] == CI_BENCHMARKS
    assert {i["target"] for i in m["include"]} == {"gpt-5.5"}


def test_dispatch_no_target_runs_nothing():
    # no model selected -> empty matrix -> zero jobs (no default model)
    assert build_matrix("workflow_dispatch", None, None)["include"] == []


def test_schedule_uses_ci_regression_targets():
    m = build_matrix("schedule")
    targets = {i["target"] for i in m["include"]}
    # inkling + gemini-3.7-flash are flagged ci_regression in targets.yaml
    assert "inkling" in targets and "gemini-3.7-flash" in targets
    assert all(i["benchmark"] in CI_BENCHMARKS for i in m["include"])
