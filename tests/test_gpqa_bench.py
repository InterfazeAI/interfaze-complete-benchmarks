"""GPQA benchmark: parse unit tests + offline scoring-parity replay.

The parity test is the real migration gate: replay each archived responses.jsonl
through the NEW parse/score and assert the score is bit-identical to the archived
metrics.json — proving the port preserves scores with zero API spend.
"""

import json
import pathlib
from types import SimpleNamespace as NS

import pytest

from benchmarks.gpqa import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHIVED_TAGS = [
    "fireworks_inkling_thinkingoff_gpqa_diamond",
    "fireworks_deepseekv4flash_thinkingoff_gpqa_diamond",
    "gemini37flash_thinkingdefault_gpqa_diamond",
    "thinkingmachinesinklingsmall_thinkingon_gpqa_diamond",
]


def _resp(text):
    return NS(text=text)


def test_parse_single_letter():
    assert bench.parse(_resp("A"), None) == "A"
    assert bench.parse(_resp("d"), None) == "D"


def test_parse_letter_in_sentence():
    assert bench.parse(_resp("The answer is B."), None) == "B"
    assert bench.parse(_resp("Answer: C"), None) == "C"


def test_parse_empty_is_none():
    assert bench.parse(_resp(""), None) is None
    assert bench.parse(_resp("   "), None) is None


def test_score_joins_records_to_samples():
    samples = [
        {"id": "1", "correct_letter": "A", "domain": "Physics"},
        {"id": "2", "correct_letter": "B", "domain": "Chemistry"},
    ]
    records = [{"id": "1", "prediction": "A"}, {"id": "2", "prediction": "C"}]
    m = bench.score(records, samples)
    assert m["correct"] == 1
    assert m["total"] == 2
    assert m["accuracy"] == 0.5
    assert m["per_domain"]["Physics"]["accuracy"] == 1.0


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
        samples.append(
            {
                "id": rec["id"],
                "correct_letter": rec["correct_letter"],
                "domain": rec.get("domain"),
            }
        )
        records.append(
            {"id": rec["id"], "prediction": bench.parse(_resp(rec["response"]), None)}
        )

    m = bench.score(records, samples)
    assert m["total"] == archived["total"]
    assert m["correct"] == archived["correct"]
    assert m["accuracy"] == archived["accuracy"]
    assert m["unparseable"] == archived["unparseable"]
