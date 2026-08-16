"""Emit the GitHub Actions job matrix ({"include": [{target, benchmark}, ...]}).

- schedule  -> every target flagged `ci_regression: true` x the CI benchmark set
- dispatch  -> the chosen target x the chosen benchmarks (or the CI set for "all")

The CI set is the API-only, --sample-friendly benchmarks. olmocr (needs poppler +
playwright + the full dataset to score) and spider2 (needs the ~4GB data/) are
left to local/self-hosted runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CI_BENCHMARKS = ["gpqa", "mmmlu", "mmmu_pro", "asr", "ocrbench_v2", "refcoco"]


def build_matrix(
    event: str, target: str | None = None, benchmarks: str | None = None
) -> dict:
    if event == "schedule":
        from bench_core.config import load_all_targets

        targets = [
            n for n, t in load_all_targets().items() if t.raw.get("ci_regression")
        ]
        targets = targets or ["inkling"]
        benches = CI_BENCHMARKS
    else:  # workflow_dispatch (or manual)
        targets = [target or "inkling"]
        b = (benchmarks or "").strip()
        benches = (
            CI_BENCHMARKS
            if (not b or b == "all")
            else [x.strip() for x in b.split(",") if x.strip()]
        )
    return {
        "include": [{"target": t, "benchmark": b} for t in targets for b in benches]
    }


if __name__ == "__main__":
    matrix = build_matrix(
        os.getenv("EVENT_NAME", "workflow_dispatch"),
        os.getenv("INPUT_TARGET") or None,
        os.getenv("INPUT_BENCHMARKS") or None,
    )
    print(json.dumps(matrix))
