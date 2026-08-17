"""
Emit the Inkling / Inkling-Small columns for the "running" tab of the
Interfaze Benchmarks sheet, read straight from results/*_metrics.json.

    uv run python scripts/sheet_columns.py

Writes results/sheet_inkling_columns.tsv and prints the same thing, in the
sheet's row order, so the two columns can be pasted in as a block.

A cell is left BLANK where our run is not comparable to that row's stated
configuration — a blank is recoverable, a mismatched number silently corrupts
the leaderboard. Every blank is explained in the NOTES column.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "sheet_inkling_columns.tsv"


def j(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ocr = j("ocrbench_v2_fireworks_inkling_metrics.json")
    mmmu_std = j(
        "mmmupro_standard_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    mmmu_vis = j(
        "mmmupro_vision_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    mmmlu = j(
        "mmmlulite_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    gpqa = j("fireworks_inkling_thinkingoff_gpqa_diamond_metrics.json")
    spider = j("spider2_lite_local_fireworks_inkling_metrics.json")
    asr = j("voxpopuli_aa_fireworks_accounts_fireworks_models_inkling_metrics.json")

    mmmu_composite = (mmmu_std["accuracy"] + mmmu_vis["accuracy"]) / 2 * 100

    # (sheet Benchmark label, Inkling cell, Inkling Small cell, note)
    rows = [
        (
            "OCRBench V2",
            f"{ocr['en_overall'] * 100:.2f}",
            "",
            (
                "Inkling = EN overall, n=10000. Sheet row says Thinking OFF; Inkling "
                "cannot disable thinking, this is reasoning_effort=none (its floor). "
                "Small: not measured."
            ),
        ),
        (
            "olmOCR",
            "74.90",
            "",
            "1403/1403 pages, 8413 tests, +/-1.1. Small: not measured.",
        ),
        (
            "RefCoco",
            "",
            "",
            (
                "BLANK ON PURPOSE: this row is TestA; our run is val "
                f"(strict {32.37:.2f}, format-tolerant oracle {81.19:.2f}). Not the "
                "same split — say the word and I'll run TestA to fill it properly."
            ),
        ),
        (
            "VoxPoppuliCleaned-AA",
            f"{(1 - asr['corpus_wer']) * 100:.2f}",
            "",
            (
                "Reported as 1-WER to match this row's scale (corpus WER 4.61%, "
                "n=628). Small: not measured."
            ),
        ),
        (
            "TheSOB",
            "",
            "",
            "BLANK: TheSOB is not implemented in this repo, so neither model was run.",
        ),
        (
            "Spider-2.0-lite",
            f"{spider['accuracy_of_local_135'] * 100:.2f}",
            "",
            (
                "64/135 SQLite subset. 5 of the 71 misses hit a 120s per-query cap "
                "added to the scorer. Small: not measured."
            ),
        ),
        (
            "GPQA-Diamond",
            f"{gpqa['accuracy'] * 100:.2f}",
            "88.30",
            (
                "CONFIG MISMATCH: this row is Thinking On; our Inkling run is the "
                "floor (mean 6537 reasoning tokens/sample even so). Small 88.30 is "
                "Thinking Machines' PUBLISHED figure at effort=0.99, NOT our measurement."
            ),
        ),
        (
            "MMMLU",
            f"{mmmlu['macro_accuracy'] * 100:.2f}",
            "",
            (
                "CONFIG MISMATCH: row is Thinking On; ours is the floor. Macro over "
                "14 langs, n=19950. Small: not measured."
            ),
        ),
        (
            "MMMU-Pro",
            f"{mmmu_composite:.2f}",
            "73.10",
            (
                f"Inkling = standard+vision composite ({mmmu_std['accuracy'] * 100:.2f}"
                f" / {mmmu_vis['accuracy'] * 100:.2f}), the published convention. Small "
                "73.10 is the PUBLISHED standard-setting figure, not measured, and not "
                "a composite."
            ),
        ),
    ]

    lines = ["Benchmark\tInkling\tInkling Small\tNOTES"]
    lines += ["\t".join(r) for r in rows]
    OUT.write_text("\n".join(lines) + "\n")

    w = max(len(r[0]) for r in rows)
    print(f"{'Benchmark'.ljust(w)}  {'Inkling':>8}  {'Inkling Small':>13}")
    print("-" * (w + 27))
    for label, ink, small, _ in rows:
        print(f"{label.ljust(w)}  {ink or '(blank)':>8}  {small or '(blank)':>13}")
    print(f"\nwrote {OUT}")


# ---------------------------------------------------------------------------
# Extra tabs. The "running" tab above is one column per model; these tabs are
# one ROW per model, with sub-scores as columns. Emitted as separate TSV blocks
# so each can be pasted as a row into its own tab.
# ---------------------------------------------------------------------------

OUT_TABS = RESULTS / "sheet_inkling_tabs.tsv"


def emit_tabs():
    ocr = j("ocrbench_v2_fireworks_inkling_metrics.json")
    gpqa = j("fireworks_inkling_thinkingoff_gpqa_diamond_metrics.json")
    mmmlu = j(
        "mmmlulite_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    spider = j("spider2_lite_local_fireworks_inkling_metrics.json")
    en = ocr["en_scores"]

    def p(x):
        return f"{x * 100:.2f}"

    blocks = []

    # --- OCRBench V2 tab: model | Avg Real | Recog | Refer | Spot | Extract
    #     | Parse | Calc | Understand | Reason   (EN scores; matches the tab's
    #     Interfaze row, whose Avg Real equals the running-tab OCRBench value)
    blocks.append(
        (
            "OCRBench V2 tab",
            [
                "model",
                "Avg Real",
                "Recog",
                "Refer",
                "Spot",
                "Extract",
                "Parse",
                "Calc",
                "Understand",
                "Reason",
            ],
            [
                "Inkling",
                p(ocr["en_overall"]),
                p(en["text_recognition"]["avg"]),
                p(en["text_detection"]["avg"]),
                p(en["text_spotting"]["avg"]),
                p(en["relationship_extraction"]["avg"]),
                p(en["element_parsing"]["avg"]),
                p(en["mathematical_calculation"]["avg"]),
                p(en["visual_text_understanding"]["avg"]),
                p(en["knowledge_reasoning"]["avg"]),
            ],
        )
    )

    # --- olmOCR tab. olmOCR-bench prints its per-split table to stdout instead
    #     of persisting a metrics file, so these come from the run log
    #     (logs/fireworks_inkling/olmocr.log, 1403/1403 pages, 8413 tests).
    blocks.append(
        (
            "olmOCR tab",
            [
                "model",
                "Overall",
                "ArXiv",
                "OldScansMath",
                "Tables",
                "OldScans",
                "Headers",
                "MultiCol",
                "LongTinyText",
                "Base",
                "Real Overall",
            ],
            [
                "Inkling",
                "74.90",
                "71.20",
                "78.80",
                "82.60",
                "41.60",
                "91.60",
                "72.40",
                "61.80",
                "99.10",
                "74.9±1.1",
            ],
        )
    )

    # --- GPQA tab: model | Overall | Physics (n=86) | Chemistry (n=93) | Biology (n=19)
    d = gpqa["per_domain"]
    blocks.append(
        (
            "GPQA tab",
            [
                "model",
                "Overall",
                "Physics (n=86)",
                "Chemistry (n=93)",
                "Biology (n=19)",
            ],
            [
                "Inkling",
                p(gpqa["accuracy"]),
                p(d["Physics"]["accuracy"]),
                p(d["Chemistry"]["accuracy"]),
                p(d["Biology"]["accuracy"]),
            ],
        )
    )

    # --- MMMLU tab: languages are ROWS and models are COLUMNS, so this is a
    #     column (macro first, then the 14 languages in the sheet's order).
    order = [
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
    pl = mmmlu["per_language"]
    blocks.append(("MMMLU tab (column: macro then langs)", ["lang", "Inkling"], None))
    mmmlu_rows = [["macro", p(mmmlu["macro_accuracy"])]]
    mmmlu_rows += [[lg, p(pl[lg]["accuracy"])] for lg in order if lg in pl]

    # --- Spider-2.0 tab: Model | Spider-2.0-lite
    blocks.append(
        (
            "Spider-2.0 tab",
            ["Model", "Spider-2.0-lite"],
            ["Inkling", p(spider["accuracy_of_local_135"])],
        )
    )

    lines = []
    for title, header, row in blocks:
        lines.append(f"# {title}")
        lines.append("\t".join(header))
        if row:
            lines.append("\t".join(row))
        else:
            lines += ["\t".join(r) for r in mmmlu_rows]
        lines.append("")
    OUT_TABS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT_TABS}")


if __name__ == "__main__":
    main()
    print()
    emit_tabs()
