"""Compare a fresh run's metrics against the committed baseline and emit a
markdown summary (for $GITHUB_STEP_SUMMARY). Regressions beyond a tolerance are
flagged as ::warning:: — non-blocking, since benchmark scores aren't
deterministic (reasoning models, SQL-timeout boundaries).

    uv run python scripts/ci_compare.py --target inkling [--tolerance 0.02]

Baseline = the metrics.json at git HEAD for the same path; "fresh" = the working
tree after the run. Smoke runs (n far below baseline) are reported but not judged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# primary metric per benchmark: (key, higher_is_better)
_PRIMARY = [
    ("corpus_wer", False),
    ("accuracy_of_local_135", True),
    ("macro_accuracy", True),
    ("en_overall", True),
    ("accuracy", True),
]


def primary(metrics: dict):
    for key, higher in _PRIMARY:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, metrics[key], higher
    return None, None, True


def _baseline(path: Path) -> dict | None:
    rel = path.relative_to(ROOT)
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{rel.as_posix()}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def compare_target(target: str, tolerance: float) -> tuple[list[list], list[str]]:
    rows, warnings = [], []
    for mpath in sorted(RESULTS.glob(f"*/{target}/metrics.json")):
        fresh = json.loads(mpath.read_text())
        base = _baseline(mpath)
        key, new, higher = primary(fresh)
        bench = fresh.get("benchmark", mpath.parent.parent.name)
        n = fresh.get("n")
        if base is None:
            rows.append([bench, key or "-", _fmt(new), "new", "—"])
            continue
        _, old, _ = primary(base)
        delta = (new - old) if (new is not None and old is not None) else None
        base_n = base.get("n")
        note = ""
        if base_n and n and n < base_n * 0.5:
            note = f"smoke (n={n} vs {base_n})"
        elif delta is not None:
            regressed = (delta < -tolerance) if higher else (delta > tolerance)
            if regressed:
                note = "REGRESSION"
                warnings.append(
                    f"{bench}/{target}: {key} {old:.4f} -> {new:.4f} (Δ{delta:+.4f})"
                )
        rows.append(
            [
                bench,
                key or "-",
                _fmt(new),
                _fmt(old),
                _fmt(delta, sign=True) + (f"  {note}" if note else ""),
            ]
        )
    return rows, warnings


def _fmt(v, sign=False):
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    rows, warnings = compare_target(args.target, args.tolerance)
    print(f"### Benchmark results — `{args.target}` vs baseline\n")
    if not rows:
        print("_no metrics found for this target_")
        return
    print("| benchmark | metric | new | baseline | Δ |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    for w in warnings:
        print(f"\n::warning::regression — {w}")


if __name__ == "__main__":
    main()
