"""OCRBench v2: offline parity of the category aggregation.

The per-sample scorer is reused verbatim (not re-tested here). This replays each
archived scored.json — which already carries per-sample `score` + `type` — through
the ported `aggregate` and matches en_overall/cn_overall + per-category means.
"""

import json
import pathlib

import pytest

from benchmarks.ocrbench_v2 import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHIVED = [
    "ocrbench_v2_fireworks_inkling",
    "ocrbench_v2_fireworks_inkling-small",
    "ocrbench_v2_gemini",
]


def test_aggregate_macro_of_means():
    scored = [
        {"type": "text recognition en", "score": 1.0},
        {"type": "text recognition en", "score": 0.0},  # text_recognition avg = 0.5
        {"type": "math QA en", "score": 1.0},  # mathematical_calculation avg = 1.0
        {"type": "text spotting en", "score": 0.0, "ignore": "True"},  # skipped
    ]
    agg = bench.aggregate(scored)
    assert agg["en_scores"]["text_recognition"] == {"avg": 0.5, "count": 2}
    assert agg["en_scores"]["mathematical_calculation"]["avg"] == 1.0
    assert agg["en_scores"]["text_spotting"]["count"] == 0  # ignored one didn't count
    # overall = mean of the two non-empty category means
    assert agg["en_overall"] == pytest.approx((0.5 + 1.0) / 2)


@pytest.mark.parametrize("tag", ARCHIVED)
def test_aggregation_parity_with_archived_run(tag):
    scored_path = RESULTS / f"{tag}_scored.json"
    metrics_path = RESULTS / f"{tag}_metrics.json"
    if not (scored_path.exists() and metrics_path.exists()):
        pytest.skip(f"archived run {tag} not present")

    scored = json.loads(scored_path.read_text())
    archived = json.loads(metrics_path.read_text())
    agg = bench.aggregate(scored)

    assert agg["en_overall"] == pytest.approx(archived["en_overall"])
    assert agg["cn_overall"] == pytest.approx(archived["cn_overall"])
    for cat, v in archived["en_scores"].items():
        assert agg["en_scores"][cat]["count"] == v["count"]
        assert agg["en_scores"][cat]["avg"] == pytest.approx(v["avg"])
