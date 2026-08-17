"""The single result contract: one slug, one layout
(results/<benchmark>/<target>/{responses.jsonl,metrics.json,run.json}), and ONE
content-based discovery so reporting and re-scoring can't disagree (the audit
found report_scores dropping 5 of 7 RefCOCO runners by prefix mismatch).
"""

import json

from bench_core.results import RunStore, discover, model_slug


def test_sampled_run_is_isolated_from_full_run(tmp_path):
    """A --sample smoke (written under an isolated _smoke root) must neither
    clobber a full run's metrics on disk nor appear in the leaderboard."""
    full = RunStore("gpqa", "inkling", root=tmp_path)
    full.write_metrics({"n": 198})
    smoke = RunStore("gpqa", "inkling", root=tmp_path / "_smoke")
    smoke.write_metrics({"n": 3})
    assert json.loads(full.metrics_path.read_text())["n"] == 198
    assert [d["n"] for d in discover(tmp_path)] == [198]


def test_model_slug_is_canonical():
    assert model_slug("accounts/fireworks/models/inkling-small") == "inkling-small"
    assert model_slug("google/gemini-3.7-flash") == "gemini-3.7-flash"
    assert model_slug("gpt-5.5") == "gpt-5.5"
    assert model_slug("x-ai/grok-4.3") == "grok-4.3"


def test_append_and_load_responses_round_trip(tmp_path):
    store = RunStore("gpqa", "inkling", root=tmp_path)
    store.append_response({"id": "a", "response": "A"})
    store.append_response({"id": "b", "response": "B"})
    assert store.load_responses() == [
        {"id": "a", "response": "A"},
        {"id": "b", "response": "B"},
    ]


def test_completed_ids_uses_done_predicate(tmp_path):
    store = RunStore("gpqa", "inkling", root=tmp_path)
    store.append_response({"id": "a", "response": "A"})
    store.append_response({"id": "b", "response": None})  # not done
    done = store.completed_ids("id", lambda r: r.get("response") is not None)
    assert done == {"a"}


def test_write_metrics_and_run(tmp_path):
    store = RunStore("gpqa", "inkling", root=tmp_path)
    store.write_metrics({"benchmark": "gpqa", "score": 0.87})
    store.write_run({"provider": "fireworks", "model_id": "x"})
    assert store.metrics_path.exists()
    assert store.run_path.exists()


def test_discover_finds_all_metrics_content_based(tmp_path):
    RunStore("gpqa", "inkling", root=tmp_path).write_metrics(
        {"benchmark": "gpqa", "target": "inkling", "score": 0.87}
    )
    RunStore("refcoco", "gemini-3.7-flash", root=tmp_path).write_metrics(
        {"benchmark": "refcoco", "target": "gemini-3.7-flash", "score": 0.3}
    )
    found = discover(root=tmp_path)
    keys = {(m["benchmark"], m["target"]) for m in found}
    assert keys == {("gpqa", "inkling"), ("refcoco", "gemini-3.7-flash")}


def test_paths_live_under_benchmark_and_slugged_target(tmp_path):
    store = RunStore("mmmu_pro", "accounts/fireworks/models/inkling", root=tmp_path)
    assert store.dir == tmp_path / "mmmu_pro" / "inkling"
