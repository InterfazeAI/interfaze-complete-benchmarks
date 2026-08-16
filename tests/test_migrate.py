"""Backfill: legacy flat results -> new contract, with (benchmark, target)
dedup (full run beats a smoke) and refcoco oracle merged as a submetric.
"""

import json

from bench_core.migrate import migrate_results


def _write(root, name, obj):
    (root / name).write_text(json.dumps(obj))


def test_dedup_full_run_beats_smoke(tmp_path):
    _write(
        tmp_path,
        "fireworks_inkling_thinkingoff_gpqa_diamond_metrics.json",
        {"model": "accounts/fireworks/models/inkling", "accuracy": 0.87, "total": 198},
    )
    (
        tmp_path / "fireworks_inkling_thinkingoff_gpqa_diamond_responses.jsonl"
    ).write_text(json.dumps({"id": "a", "response": "A", "prediction": "A"}) + "\n")
    # a 2-sample smoke of the SAME model -> same (gpqa, inkling) slot
    _write(
        tmp_path,
        "smoke_inkling_gpqa_diamond_metrics.json",
        {"model": "accounts/fireworks/models/inkling", "accuracy": 0.5, "total": 2},
    )

    log = migrate_results(tmp_path)
    metrics = json.loads((tmp_path / "gpqa" / "inkling" / "metrics.json").read_text())
    assert metrics["benchmark"] == "gpqa"
    assert metrics["target"] == "inkling"
    assert metrics["accuracy"] == 0.87  # full run won, not the smoke
    assert metrics["n"] == 198
    assert any("shadowed" in line and "n=2" in line for line in log)


def test_refcoco_oracle_merged_as_submetric(tmp_path):
    _write(
        tmp_path,
        "refcoco_val_fireworks_x_metrics.json",
        {
            "model": "accounts/fireworks/models/inkling",
            "accuracy": 0.32,
            "mean_iou": 0.3,
            "total": 10,
        },
    )
    _write(
        tmp_path,
        "refcoco_val_fireworks_x_oracle_metrics.json",
        {"accuracy": 0.81, "mean_iou": 0.7, "total": 10},
    )
    (tmp_path / "refcoco_val_fireworks_x_responses.jsonl").write_text(
        json.dumps({"id": "1", "response": "[1,2,3,4]", "pred_bbox_xyxy": [1, 2, 3, 4]})
        + "\n"
    )

    migrate_results(tmp_path)
    m = json.loads((tmp_path / "refcoco_val" / "inkling" / "metrics.json").read_text())
    assert m["accuracy"] == 0.32
    assert m["oracle"]["accuracy"] == 0.81
    # response migrated with the new prediction field for resume/scoring
    rows = [
        json.loads(x)
        for x in (tmp_path / "refcoco_val" / "inkling" / "responses.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["prediction"] == [1, 2, 3, 4]


def test_empty_responses_not_migrated(tmp_path):
    _write(
        tmp_path,
        "voxpopuli_aa_x_metrics.json",
        {
            "model": "accounts/fireworks/models/inkling",
            "corpus_wer": 0.1,
            "num_samples": 2,
        },
    )
    (tmp_path / "voxpopuli_aa_x_responses.jsonl").write_text(
        json.dumps({"id": "1", "response": "hello", "prediction": "hello"})
        + "\n"
        + json.dumps({"id": "2", "response": "", "prediction": ""})
        + "\n"
    )
    migrate_results(tmp_path)
    rows = (
        (tmp_path / "voxpopuli_aa" / "inkling" / "responses.jsonl")
        .read_text()
        .splitlines()
    )
    assert len(rows) == 1  # the empty-response row is left to resume
