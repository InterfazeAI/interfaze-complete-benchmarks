"""Spider2-Lite: offline unit tests for the pure logic (SQL extraction, result
comparison). Full scoring re-executes SQL against the ~4GB gitignored data/, so a
score-replay is gated behind data/ presence + BENCH_SPIDER2_PARITY=1.
"""

import json
import os
import pathlib

import pytest

from benchmarks.spider2_lite import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def test_extract_sql_fenced():
    assert bench.extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert bench.extract_sql("prose\n```\nSELECT 2\n```\nmore") == "SELECT 2"


def test_extract_sql_unterminated_fence():
    # a model that opens ```sql, emits SQL, never closes the block
    assert bench.extract_sql("Here you go:\n```sql\nSELECT 3;") == "SELECT 3;"


def test_extract_sql_no_fence_returns_whole():
    assert bench.extract_sql("SELECT 4") == "SELECT 4"


def test_vectors_match_float_tolerance_and_order():
    assert bench._vectors_match([1.0, 2.0], [1.005, 2.0], ignore_order=False)
    assert not bench._vectors_match([1.0], [1.5], ignore_order=False)


def test_compare_table_column_match_unordered():
    import pandas as pd

    gold = pd.DataFrame({"a": [1, 2, 3]})
    pred = pd.DataFrame({"x": [3, 2, 1]})
    assert bench.compare_table(pred, gold, None, ignore_order=True) == 1
    assert bench.compare_table(pred, gold, None, ignore_order=False) == 0


_PARITY_ON = bench._ALL_EXAMPLES.exists() and os.getenv("BENCH_SPIDER2_PARITY")


@pytest.mark.skipif(
    not _PARITY_ON, reason="needs data/ + BENCH_SPIDER2_PARITY=1 (re-executes SQL)"
)
def test_full_scoring_parity_when_data_present():
    tag = "spider2_lite_local_fireworks_inkling"
    resp_path = RESULTS / f"{tag}_responses.jsonl"
    metrics_path = RESULTS / f"{tag}_metrics.json"
    archived = json.loads(metrics_path.read_text())
    samples, records = [], []
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        samples.append({"instance_id": rec["instance_id"], "db": rec["db"]})
        records.append(
            {
                "instance_id": rec["instance_id"],
                "prediction": bench.extract_sql(rec["response"]),
            }
        )
    m = bench.score(records, samples)
    # NOT exact: Spider2 re-executes SQL with a 120s timeout cap, so a heavy query
    # near the boundary can flip between runs (warm cache vs cold). A faithful port
    # reproduces the archived score within a small execution-noise tolerance.
    assert abs(m["correct"] - archived["correct"]) <= 2
