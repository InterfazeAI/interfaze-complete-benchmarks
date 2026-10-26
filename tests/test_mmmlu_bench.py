"""MMMLU: parse units + offline scoring-parity replay of the archived runs."""

import json
import pathlib

import pytest

from benchmarks.mmmlu import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHIVED_TAGS = [
    "mmmlulite_fireworks_accounts-fireworks-models-inkling_reasoningoff",
    "mmmlulite_fireworks_accounts-fireworks-models-deepseek-v4-flash_reasoningoff",
    "mmmlulite_gemini_gemini-3-7-flash_reasoninghigh",
    "mmmlulite_openrouter_thinkingmachines-inkling-small_reasoninghigh",
]


def test_parse_answer():
    assert bench.parse_answer("B") == "B"
    assert bench.parse_answer("The answer is (C).") == "C"
    assert bench.parse_answer("") is None


def test_build_sample_lite_and_full_variants():
    # lite (opencompass) schema
    lite = bench._build_sample(
        {
            "subject": "math",
            "input": "2+2?",
            "A": "3",
            "B": "4",
            "C": "5",
            "D": "6",
            "target": "b",
        },
        "EN",
        3,
        "lite",
    )
    assert lite["id"] == "EN:3" and lite["question"] == "2+2?" and lite["answer"] == "B"
    # full (openai/MMMLU) schema — id from Unnamed: 0, Question/Subject/Answer columns
    full = bench._build_sample(
        {
            "Unnamed: 0": 7,
            "Subject": "math",
            "Question": "2+2?",
            "A": "3",
            "B": "4",
            "C": "5",
            "D": "6",
            "Answer": "b",
        },
        "EN",
        0,
        "full",
    )
    assert full["id"] == "EN:7" and full["subject"] == "math" and full["answer"] == "B"


def test_score_macro_averages_languages():
    samples = [
        {"id": "EN:0", "language": "EN", "subject": "math", "answer": "A"},
        {"id": "FR:0", "language": "FR", "subject": "math", "answer": "B"},
    ]
    records = [{"id": "EN:0", "prediction": "A"}, {"id": "FR:0", "prediction": "C"}]
    m = bench.score(records, samples)
    assert m["per_language"]["EN"]["accuracy"] == 1.0
    assert m["per_language"]["FR"]["accuracy"] == 0.0
    assert m["macro_accuracy"] == 0.5


@pytest.mark.parametrize("tag", ARCHIVED_TAGS)
def test_scoring_parity_with_archived_run(tag):
    resp_path = RESULTS / f"{tag}_responses.jsonl"
    metrics_path = RESULTS / f"{tag}_metrics.json"
    if not (resp_path.exists() and metrics_path.exists()):
        pytest.skip(f"archived run {tag} not present")

    archived = json.loads(metrics_path.read_text())
    samples, records = [], []
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("response") is None:
            continue
        samples.append(
            {
                "id": rec["id"],
                "language": rec["language"],
                "subject": rec["subject"],
                "answer": rec["answer"],
            }
        )
        records.append(
            {"id": rec["id"], "prediction": bench.parse_answer(rec["response"])}
        )

    m = bench.score(records, samples)
    assert m["num_samples"] == archived["num_samples"]
    assert m["macro_accuracy"] == pytest.approx(archived["macro_accuracy"])
    assert m["micro_accuracy"] == pytest.approx(archived["micro_accuracy"])
