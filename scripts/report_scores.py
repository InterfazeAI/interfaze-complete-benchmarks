"""
Detailed benchmark scores for every model with results on disk.

    uv run python scripts/report_scores.py                # interactive picker
    uv run python scripts/report_scores.py --all           # every model
    uv run python scripts/report_scores.py --model inkling
    uv run python scripts/report_scores.py --list          # just the model names
    uv run python scripts/report_scores.py --model inkling --tsv   # paste into Sheets

Reads the results/<benchmark>/<target>/metrics.json contract; olmOCR scores come only
from the run logs harness.py tees (its metrics.json isn't read and lacks the ± CI).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` is importable when run as a script
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# Column orders, exactly as the sheet expects them
OCR_COLS = [
    ("Avg Real", "en_overall"),
    ("Recog", "text_recognition"),
    ("Refer", "text_detection"),
    ("Spot", "text_spotting"),
    ("Extract", "relationship_extraction"),
    ("Parse", "element_parsing"),
    ("Calc", "mathematical_calculation"),
    ("Understand", "visual_text_understanding"),
    ("Reason", "knowledge_reasoning"),
]

OLMOCR_COLS = [
    ("Overall", None),
    ("ArXiv", "arxiv_math"),
    ("OldScansMath", "old_scans_math"),
    ("Tables", "table_tests"),
    ("OldScans", "old_scans"),
    ("Headers", "headers_footers"),
    ("MultiCol", "multi_column"),
    ("LongTinyText", "long_tiny_text"),
    ("Base", "baseline"),
    ("Real Overall", None),
]

GPQA_COLS = [
    ("Overall", None),
    ("Physics (n=86)", "Physics"),
    ("Chemistry (n=93)", "Chemistry"),
    ("Biology (n=19)", "Biology"),
]

MMMLU_LANGS = [
    "FR_FR",
    "PT_BR",
    "BN_BD",
    "JA_JP",
    "DE_DE",
    "YO_NG",
    "ES_LA",
    "ID_ID",
    "ZH_CN",
    "SW_KE",
    "IT_IT",
    "AR_XY",
    "KO_KR",
    "HI_IN",
]

MMMU_STD_SUBJECTS = [
    "overall",
    "Art",
    "Electronics",
    "Economics",
    "Marketing",
    "Finance",
    "Art_Theory",
    "Public_Health",
    "Basic_Medical_Science",
    "Sociology",
    "Literature",
    "Physics",
    "Energy_and_Power",
    "History",
    "Design",
    "Materials",
    "Biology",
    "Psychology",
    "Geography",
    "Manage",
    "Pharmacy",
    "Agriculture",
    "Clinical_Medicine",
    "Accounting",
    "Computer_Science",
    "Architecture_and_Engineering",
    "Chemistry",
    "Math",
    "Diagnostics_and_Laboratory_Medicine",
    "Mechanical_Engineering",
    "Music",
]

MMMU_VIS_SUBJECTS = [
    "overall",
    "Economics",
    "Art_Theory",
    "Basic_Medical_Science",
    "Art",
    "Literature",
    "Pharmacy",
    "Clinical_Medicine",
    "Sociology",
    "Public_Health",
    "Design",
    "Physics",
    "History",
    "Chemistry",
    "Electronics",
    "Marketing",
    "Geography",
    "Math",
    "Biology",
    "Computer_Science",
    "Manage",
    "Finance",
    "Agriculture",
    "Accounting",
    "Psychology",
    "Mechanical_Engineering",
    "Diagnostics_and_Laboratory_Medicine",
    "Energy_and_Power",
    "Materials",
    "Architecture_and_Engineering",
    "Music",
]


def canon(model: str) -> str:
    """Bare model name, provider prefixes and slug mangling removed."""
    m = model.strip().rsplit("/", 1)[-1].lower()
    m = re.sub(r"^accounts-fireworks-models-", "", m)
    # filename slugs turn dots into dashes: gemini-3-7-flash -> gemini-3.7-flash
    m = re.sub(r"^gemini-(\d)-(\d)-", r"gemini-\1.\2-", m)
    m = re.sub(r"^inklingsmall$", "inkling-small", m)
    m = re.sub(r"^thinkingmachinesinklingsmall$", "inkling-small", m)
    m = re.sub(r"^gemini37flash$", "gemini-3.7-flash", m)
    return m


def _pct(x):
    return None if x is None else round(float(x) * 100, 2)


def load_metrics() -> dict:
    """{model: {benchmark_key: payload}} from the results/ metrics.json contract."""
    from src.results import discover

    out: dict[str, dict] = {}
    for d in discover(RESULTS):
        benchmark, model = d.get("benchmark"), d.get("target")
        if not benchmark or not model:
            continue
        b = out.setdefault(model, {})
        reasoning = (d.get("reasoning") or {}).get("mode", "off")

        if benchmark == "gpqa":
            b["gpqa"] = {
                "overall": _pct(d.get("accuracy")),
                "n": d.get("total") or d.get("n"),
                "domains": {
                    k: _pct(v["accuracy"])
                    for k, v in (d.get("per_domain") or {}).items()
                },
            }
        elif benchmark == "voxpopuli_aa":
            wer = d.get("corpus_wer")
            b["asr"] = {
                "wer": _pct(wer),
                "inv": _pct(1 - wer) if wer is not None else None,
                "cer": _pct(d.get("corpus_cer")),
                "n": d.get("num_samples") or d.get("n"),
            }
        elif benchmark.startswith("mmmlu"):  # mmmlu_lite / mmmlu_full
            variant = benchmark[len("mmmlu_") :] if "_" in benchmark else "lite"
            b[f"mmmlu_{variant}"] = {
                "macro": _pct(d.get("macro_accuracy")),
                "n": d.get("num_samples") or d.get("n"),
                "langs": {
                    k: _pct(v["accuracy"])
                    for k, v in (d.get("per_language") or {}).items()
                },
                "reasoning": reasoning,
            }
        elif benchmark.startswith("mmmu_pro_"):
            setting = benchmark[len("mmmu_pro_") :]
            b[f"mmmupro_{setting}_{reasoning}"] = {
                "overall": _pct(d.get("accuracy")),
                "n": d.get("num_samples") or d.get("n"),
                "subjects": {
                    k: _pct(v["accuracy"])
                    for k, v in (d.get("per_subject") or {}).items()
                },
                "reasoning": reasoning,
            }
        elif benchmark.startswith("refcoco_"):
            split = benchmark[len("refcoco_") :]
            b[f"refcoco_{split}"] = {
                "acc": _pct(d.get("accuracy")),
                "mean_iou": round(d.get("mean_iou", 0), 4),
                "n": d.get("total") or d.get("n"),
                "split": split,
            }
            orc = d.get("oracle")
            if orc:
                b[f"refcoco_{split}_oracle"] = {
                    "acc": _pct(orc.get("accuracy")),
                    "mean_iou": round(orc.get("mean_iou", 0), 4),
                    "n": orc.get("total") or d.get("total"),
                    "split": split,
                }
        elif benchmark == "ocrbench_v2":
            en = d.get("en_scores", {})
            covered = sum(1 for v in en.values() if v.get("count"))
            b["ocrbench"] = {
                "en_overall": _pct(d.get("en_overall")),
                "cn_overall": _pct(d.get("cn_overall")),
                "cats": {
                    k: _pct(v["avg"]) if v.get("count") else None for k, v in en.items()
                },
                "counts": {k: v.get("count", 0) for k, v in en.items()},
                "partial": covered < 8,
                "covered": covered,
            }
        elif benchmark == "spider2_lite":
            b["spider2"] = {
                "acc": _pct(d.get("accuracy_of_local_135")),
                "correct": d.get("correct"),
                "n": d.get("total_local_subset"),
            }
    return out


def load_olmocr() -> dict:
    """olmOCR scores by model: captured logs/pasted summaries, overridden by run metrics.json."""
    from benchmarks.olmocr.harness import parse_scores
    from src.results import discover

    out: dict[str, dict] = {}
    candidates = []
    for pat in ("**/olmocr*.log", "**/*olmocr*.txt", "**/suite*.log"):
        candidates += sorted(LOGS.glob(pat))
    for log in dict.fromkeys(candidates):
        scores = parse_scores(log.read_text(errors="ignore"))
        if scores["overall"] is None:
            continue
        out[canon(scores.pop("candidate"))] = {
            **scores,
            "source": str(log.relative_to(ROOT)),
        }
    for d in discover(RESULTS):
        if d.get("benchmark") != "olmocr" or d.get("overall") is None:
            continue
        if d.get("ci") is None and d["target"] in out:
            continue  # pre-CI metrics.json; its run's log has the same table plus the CI
        out[d["target"]] = {
            "overall": d["overall"],
            "ci": d.get("ci"),
            "splits": d.get("splits") or {},
            "source": str(Path(d["_path"]).relative_to(ROOT)),
        }
    return out


def _cell(value, missing: str) -> str:
    if value is None:
        return missing
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _aligned(cells, widths) -> str:
    """Label column left-aligned, value columns right-aligned."""
    return "  ".join(
        c.ljust(w) if i == 0 else c.rjust(w)
        for i, (c, w) in enumerate(zip(cells, widths))
    )


def table(headers, rows, tsv=False):
    if tsv:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(_cell(c, "") for c in r))
        return
    text_rows = [[_cell(c, "-") for c in r] for r in rows]
    if text_rows:
        widths = [
            max(len(h), *(len(c) for c in col))
            for h, col in zip(headers, zip(*text_rows))
        ]
    else:
        widths = [len(h) for h in headers]
    print(_aligned(headers, widths))
    print("  ".join("-" * w for w in widths))
    for r in text_rows:
        print(_aligned(r, widths))


def report(models: list[str], data: dict, olm: dict, tsv=False):
    def get(model, key):
        return data.get(model, {}).get(key)

    # ---- OCRBench V2
    rows, notes = [], []
    for m in models:
        d = get(m, "ocrbench")
        if not d:
            continue
        row = [m, d["en_overall"]] + [d["cats"].get(k) for _, k in OCR_COLS[1:]]
        rows.append(row)
        if d["partial"]:
            missing = [k for k, c in d["counts"].items() if not c]
            notes.append(
                f"  ! {m}: PARTIAL — only {d['covered']}/8 EN categories have samples "
                f"(missing: {', '.join(missing)}). 'Avg Real' averages the covered ones, "
                f"so it is NOT comparable to a full run."
            )
    if rows:
        print("\n=== OCRBench V2 (EN) ===")
        table([c for c, _ in [("model", None)] + OCR_COLS], rows, tsv)
        for n in notes:
            print(n)

    # ---- olmOCR
    rows = []
    for m in models:
        d = olm.get(m)
        if not d:
            continue
        row = [m, d["overall"]] + [d["splits"].get(k) for _, k in OLMOCR_COLS[1:-1]]
        ci = f"±{d['ci']:.1f}" if d["ci"] is not None else ""
        row.append(f"{d['overall']:.1f}{ci}")
        rows.append(row)
    if rows:
        print("\n=== olmOCR-bench ===")
        table([c for c, _ in [("model", None)] + OLMOCR_COLS], rows, tsv)

    # ---- RefCOCO (every split found, so a val number is never mistaken for TestA)
    rows = []
    for m in models:
        for key, d in sorted((data.get(m) or {}).items()):
            if not key.startswith("refcoco"):
                continue
            label = (
                f"{d['split']}{' (oracle)' if key.endswith('_oracle') else ' (strict)'}"
            )
            rows.append([m, label, d["acc"], d["mean_iou"], d["n"]])
    if rows:
        print("\n=== RefCOCO ===")
        table(["Model", "split/scoring", "Acc@0.5", "mean IoU", "n"], rows, tsv)

    # ---- ASR
    rows = [
        [m, d["inv"], d["wer"], d["cer"], d["n"]]
        for m in models
        if (d := get(m, "asr"))
    ]
    if rows:
        print("\n=== ASR (VoxPopuliCleaned-AA) ===")
        table(["Model", "1-WER", "WER", "CER", "n"], rows, tsv)
        print(
            "  (sheet convention is 1-WER; WER shown too since lower-is-better there)"
        )

    # ---- GPQA
    rows = []
    for m in models:
        d = get(m, "gpqa")
        if d:
            rows.append(
                [m, d["overall"]] + [d["domains"].get(k) for _, k in GPQA_COLS[1:]]
            )
    if rows:
        print("\n=== GPQA Diamond ===")
        table([c for c, _ in [("Model", None)] + GPQA_COLS], rows, tsv)

    # ---- MMMLU (lite and/or full)
    for variant in ("lite", "full"):
        rows = []
        for m in models:
            d = get(m, f"mmmlu_{variant}")
            if d:
                rows.append(
                    [m, d["macro"]] + [d["langs"].get(lg) for lg in MMMLU_LANGS]
                )
        if rows:
            print(f"\n=== MMMLU ({variant}) ===")
            table(["Model", "macro"] + MMMLU_LANGS, rows, tsv)

    # ---- MMMU-Pro, subjects as rows in the sheet's order
    for setting, subjects in (
        ("standard", MMMU_STD_SUBJECTS),
        ("vision", MMMU_VIS_SUBJECTS),
    ):
        cols, found = [], []
        for m in models:
            for reasoning in ("off", "high"):
                d = get(m, f"mmmupro_{setting}_{reasoning}")
                if d:
                    cols.append((f"{m} ({reasoning})", d))
                    found.append(m)
        if not cols:
            continue
        print(
            f"\n=== MMMU-Pro-{'Standard Split' if setting == 'standard' else 'Vision'} ==="
        )
        rows = []
        for s in subjects:
            row = [s]
            for _, d in cols:
                row.append(d["overall"] if s == "overall" else d["subjects"].get(s))
            rows.append(row)
        table(["Subject"] + [c for c, _ in cols], rows, tsv)

    # ---- Spider2
    rows = [
        [m, d["acc"], f"{d['correct']}/{d['n']}"]
        for m in models
        if (d := get(m, "spider2"))
    ]
    if rows:
        print("\n=== Spider-2.0-lite (SQLite subset) ===")
        table(["Model", "Spider-2.0-lite", "correct"], rows, tsv)
        for m in models:
            d = get(m, "spider2")
            if d and d["n"] and d["correct"] is not None and d["n"] < 135:
                print(f"  ! {m}: only {d['n']} examples scored — not a full run")


def main():
    ap = argparse.ArgumentParser(description="Detailed benchmark scores from results/")
    ap.add_argument("--model", action="append", help="model name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every model found")
    ap.add_argument("--list", action="store_true", help="list model names and exit")
    ap.add_argument(
        "--tsv", action="store_true", help="tab-separated, for pasting into Sheets"
    )
    args = ap.parse_args()

    data, olm = load_metrics(), load_olmocr()
    known = sorted(set(data) | set(olm))
    if not known:
        sys.exit(f"No results found in {RESULTS}")

    if args.list:
        for m in known:
            n = len(data.get(m, {})) + (1 if m in olm else 0)
            print(f"{m:24} {n} benchmark result(s)")
        return

    if args.all:
        chosen = known
    elif args.model:
        chosen = []
        for want in args.model:
            hits = [m for m in known if want.lower() in m]
            if not hits:
                sys.exit(f"No model matching {want!r}. Known: {', '.join(known)}")
            chosen += hits
    else:
        print("Models with results:\n")
        for i, m in enumerate(known, 1):
            n = len(data.get(m, {})) + (1 if m in olm else 0)
            print(f"  {i:2}. {m:24} ({n} result(s))")
        print(f"  {len(known) + 1:2}. ALL")
        try:
            raw = input("\nPick a number (or name, blank = ALL): ").strip()
        except EOFError:
            raw = ""
        if not raw or raw == str(len(known) + 1):
            chosen = known
        elif raw.isdigit() and 1 <= int(raw) <= len(known):
            chosen = [known[int(raw) - 1]]
        else:
            chosen = [m for m in known if raw.lower() in m] or sys.exit(
                f"No match for {raw!r}"
            )

    print(f"\nModels: {', '.join(chosen)}")
    report(chosen, data, olm, tsv=args.tsv)


if __name__ == "__main__":
    main()
