"""Emit the GitHub Actions job matrix ({"include": [{target, benchmark}, ...]}).

- schedule -> every target flagged `ci_regression: true` x the CI benchmark set
- dispatch -> the chosen target x the chosen benchmarks (or the CI set for "all")
- push     -> the target(s) whose entry changed in this merge x the CI set

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


def _matrix(targets, benches):
    return {
        "include": [{"target": t, "benchmark": b} for t in targets for b in benches]
    }


def changed_targets(old_path: str | None, new_path: str) -> list[str]:
    """Target names whose *run-affecting* spec was added or modified between two
    targets.yaml files (the merge diff)."""
    import yaml

    def load(p):
        if not p or not Path(p).exists():
            return {}
        return (yaml.safe_load(Path(p).read_text()) or {}).get("targets") or {}

    old, new = load(old_path), load(new_path)
    if not old:
        # No usable baseline (first push / force-push / unavailable). Don't
        # assume everything changed and benchmark the whole registry — run
        # nothing; the author can dispatch explicitly.
        return []

    def runspec(spec):
        # ci_regression only controls nightly membership, not the run itself,
        # so flipping it must NOT trigger a (full, paid) benchmark.
        return {k: v for k, v in (spec or {}).items() if k != "ci_regression"}

    return [
        n for n, spec in new.items() if n not in old or runspec(old[n]) != runspec(spec)
    ]


def build_matrix(
    event: str,
    target: str | None = None,
    benchmarks: str | None = None,
    targets: list[str] | None = None,
) -> dict:
    if event == "schedule":
        from bench_core.config import load_all_targets

        flagged = [
            n for n, t in load_all_targets().items() if t.raw.get("ci_regression")
        ]
        return _matrix(
            flagged, CI_BENCHMARKS
        )  # explicit opt-in only; nothing flagged -> nothing runs
    if event == "push":
        return _matrix(targets or [], CI_BENCHMARKS)
    # workflow_dispatch (or manual): no target -> nothing runs (no default model)
    b = (benchmarks or "").strip()
    benches = (
        CI_BENCHMARKS
        if (not b or b == "all")
        else [x.strip() for x in b.split(",") if x.strip()]
    )
    return _matrix([target] if target else [], benches)


if __name__ == "__main__":
    event = os.getenv("EVENT_NAME", "workflow_dispatch")
    if event == "push":
        tl = changed_targets(
            os.getenv("OLD_TARGETS_FILE"),
            os.getenv("NEW_TARGETS_FILE", "bench_core/targets.yaml"),
        )
        print(json.dumps(build_matrix("push", targets=tl)))
    else:
        print(
            json.dumps(
                build_matrix(
                    event,
                    os.getenv("INPUT_TARGET") or None,
                    os.getenv("INPUT_BENCHMARKS") or None,
                )
            )
        )
