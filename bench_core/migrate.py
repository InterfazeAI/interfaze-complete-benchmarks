"""One-time backfill: fold the legacy flat results/*_metrics.json (+ their
responses/predictions siblings) into the new results/<benchmark>/<target>/
contract, so the report reads one layout and a partial run resumes rather than
re-bill.

The new contract keys by (benchmark, target) only — it does not encode host or
reasoning — so when several legacy runs map to the same slot (a 2-sample smoke
vs a full run, or reasoning off vs high) the one with the most *evaluated*
samples wins and the rest are logged as shadowed. Idempotent; no silent drops.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bench_core.results import DEFAULT_ROOT, RunStore, model_slug

# count of samples actually evaluated (NOT a fixed denominator like Spider2's
# total_local_subset, which is 135 even for a 1-sample smoke)
_N_KEYS = ("num_samples", "total", "total_evaluated", "n")


def _classify(name: str) -> str | None:
    if name.startswith("ocrbench_v2_"):
        return "ocrbench_v2"
    if name.startswith("mmmupro_standard_"):
        return "mmmu_pro_standard"
    if name.startswith("mmmupro_vision_"):
        return "mmmu_pro_vision"
    if name.startswith("mmmlulite_"):
        return "mmmlu_lite"
    if name.startswith("mmmlufull_"):
        return "mmmlu_full"
    if name.startswith("refcoco_"):
        return f"refcoco_{name.split('_')[1]}"
    if name.startswith("voxpopuli_aa_"):
        return "voxpopuli_aa"
    if name.startswith("spider2_lite_local_"):
        return "spider2_lite"
    if "gpqa_diamond" in name:
        return "gpqa"
    return None


def _n(d: dict) -> int:
    for k in _N_KEYS:
        v = d.get(k)
        if isinstance(v, int):
            return v
    return 0


def _resp_shape(benchmark: str):
    if benchmark == "spider2_lite":
        return "instance_id", "pred_sql"
    if benchmark.startswith("refcoco"):
        return "id", "pred_bbox_xyxy"
    return "id", "prediction"


def _migrate_responses(
    metrics_path: Path, benchmark: str, store: RunStore
) -> str | None:
    if benchmark == "ocrbench_v2":
        src = metrics_path.with_name(
            metrics_path.name.replace("_metrics.json", "_predictions.json")
        )
        if not src.exists():
            return None
        records = [
            {
                "id": r["id"],
                "response": r.get("predict", ""),
                "prediction": r.get("predict", ""),
            }
            for r in json.loads(src.read_text())
        ]
    else:
        src = metrics_path.with_name(
            metrics_path.name.replace("_metrics.json", "_responses.jsonl")
        )
        if not src.exists():
            return None
        id_key, pred_field = _resp_shape(benchmark)
        records = []
        for line in src.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records.append(
                    {
                        id_key: r[id_key],
                        "response": r.get("response"),
                        "prediction": r.get(pred_field),
                    }
                )
    # Only genuine answers migrate; empty/None responses are left to resume (a
    # blank prediction must re-run, not count as done).
    records = [r for r in records if (r.get("response") or "") != ""]
    store._ensure_dir()
    store.responses_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    return f"  responses: {src.name} ({len(records)} rows)"


def migrate_results(
    results_dir: Path | str = DEFAULT_ROOT, dry_run: bool = False
) -> list[str]:
    results_dir = Path(results_dir)
    log: list[str] = []

    groups: dict[tuple, list] = defaultdict(list)
    for p in sorted(results_dir.glob("*_metrics.json")):
        name = p.name
        if "_oracle_" in name:  # folded into the strict refcoco metrics
            continue
        benchmark = _classify(name)
        if benchmark is None:
            log.append(f"SKIP (unrecognized): {name}")
            continue
        d = json.loads(p.read_text())
        target = model_slug(d.get("model") or name)
        groups[(benchmark, target)].append((p, d, _n(d)))

    for (benchmark, target), items in sorted(groups.items()):
        winner = items[0]
        for c in items[1:]:
            if c[2] >= winner[2]:  # tie -> later (sorted) wins, matching the old report
                winner = c
        p, d, n = winner

        newm = dict(d)
        newm.update(
            {
                "benchmark": benchmark,
                "target": target,
                "provider": d.get("provider"),
                "n": n,
                "reasoning": {"mode": "high" if "high" in p.name else "off"},
            }
        )
        if benchmark.startswith("refcoco"):
            oracle_p = p.with_name(
                p.name.replace("_metrics.json", "_oracle_metrics.json")
            )
            if oracle_p.exists():
                od = json.loads(oracle_p.read_text())
                newm["oracle"] = {
                    "accuracy": od.get("accuracy"),
                    "mean_iou": od.get("mean_iou"),
                    "total": od.get("total"),
                    "interpretation_counts": od.get("interpretation_counts"),
                }

        store = RunStore(benchmark, target, root=results_dir)
        log.append(
            f"metrics: {p.name} (n={n}) -> {benchmark}/{model_slug(target)}/metrics.json"
        )
        if not dry_run:
            store.write_metrics(newm)
            r = _migrate_responses(p, benchmark, store)
            if r:
                log.append(r)

        for c in items:
            if c is not winner:
                log.append(
                    f"  shadowed: {c[0].name} (n={c[2]}) — slot taken by the n={n} run"
                )

    return log
