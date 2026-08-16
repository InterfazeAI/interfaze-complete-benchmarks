"""
Gemini-3.5-flash vs DeepSeek-V4-Flash across the benchmark suite.

Bars, not a radar: DeepSeek V4 Flash is text-only, so 6 of 9 benchmarks are not
runnable for it at all. On a radar those axes would collapse to the centre and
read as "scored badly"; as bars, an absent bar can be labelled for what it is
("no image input") and never be mistaken for a zero.

Gemini figures come from the CSVs in the repo root (the team's own runs).
DeepSeek figures are measured by this repo.

    uv run python scripts/plot_gemini_vs_deepseek.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CSV_MASTER = ROOT / "Interfaze Benchmarks - running.csv"
OUT = RESULTS / "gemini35flash_vs_deepseekv4flash.png"

# Validated palette (dataviz references/palette.md), light mode.
SURFACE = "#fcfcfb"
GEMINI = "#2a78d6"  # slot 1
DEEPSEEK = "#eb6834"  # slot 2
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
MONO = ["DejaVu Sans Mono", "Menlo", "monospace"]

GEMINI_COL = "Gemini-3.5-flash"

# Why DeepSeek has no number, verified by direct probe against the Fireworks
# endpoint rather than assumed from docs:
#   image -> HTTP 400 "This model does not support image inputs"
#   audio -> HTTP 400, content must be a plain string
UNSUPPORTED = {
    "OCRBench V2": "no image input",
    "olmOCR": "no image input",
    "RefCoco": "no image input",
    "MMMU-Pro": "no image input",
    "VoxPoppuliCleaned-AA": "no audio input",
    "TheSOB": "not in this repo",
}

# Benchmark label -> our measured metrics file + extractor.
# NB: gpqa_fireworks' model_slug strips hyphens (deepseekv4flash) while the
# other runners keep them (deepseek-v4-flash) — inherited from gpqa_openrouter.
MEASURED = {
    "Spider-2.0-lite": (
        "spider2_lite_local_fireworks_deepseek-v4-flash_metrics.json",
        lambda m: m["accuracy_of_local_135"] * 100,
    ),
    "MMMLU": (
        "mmmlulite_fireworks_accounts-fireworks-models-deepseek-v4-flash_reasoningoff_metrics.json",
        lambda m: m["macro_accuracy"] * 100,
    ),
    "GPQA-Diamond": (
        "fireworks_deepseekv4flash_thinkingoff_gpqa_diamond_metrics.json",
        lambda m: m["accuracy"] * 100,
    ),
}

DISPLAY = {
    "OCRBench V2": "OCRBench V2",
    "olmOCR": "olmOCR",
    "RefCoco": "RefCOCO",
    "VoxPoppuliCleaned-AA": "VoxPopuli-AA",
    "TheSOB": "TheSOB",
    "Spider-2.0-lite": "Spider-2.0-Lite",
    "GPQA-Diamond": "GPQA Diamond",
    "MMMLU": "MMMLU",
    "MMMU-Pro": "MMMU-Pro",
}


def load_gemini() -> dict:
    """Pull the Gemini-3.5-flash column out of the team's master CSV."""
    out = {}
    with open(CSV_MASTER, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bench = (row.get("Benchmark") or "").strip()
            raw = (row.get(GEMINI_COL) or "").strip()
            if not bench or not raw:
                continue
            try:
                out[bench] = float(raw)
            except ValueError:
                continue  # price rows, "Does not support ..." cells
    return out


# A run in flight is its own state. Its partial accuracy is NOT usable as an
# estimate: mmmlu_multi walks languages in order, so a half-finished run covers
# ~7 of 14 languages and its macro average is biased by construction.
IN_FLIGHT = {
    "MMMLU": (
        "mmmlulite_fireworks_accounts-fireworks-models-deepseek-v4-flash_reasoningoff_responses.jsonl",
        19950,
    ),
    "GPQA-Diamond": (
        "fireworks_deepseekv4flash_thinkingoff_gpqa_diamond_responses.jsonl",
        198,
    ),
}


def load_deepseek() -> dict:
    out = {}
    for bench, (fname, pick) in MEASURED.items():
        p = RESULTS / fname
        if p.exists():
            out[bench] = round(pick(json.loads(p.read_text())), 2)
    return out


def pending_note(bench: str) -> str:
    """'running (n/total)' for an unfinished run, else a plain not-run."""
    spec = IN_FLIGHT.get(bench)
    if not spec:
        return UNSUPPORTED.get(bench, "not run")
    fname, total = spec
    p = RESULTS / fname
    done = sum(1 for _ in p.open()) if p.exists() else 0
    return f"running… {done:,}/{total:,}"


def main():
    gem = load_gemini()
    ds = load_deepseek()

    # Comparable rows first — that is the actual comparison, so it reads first.
    comparable = [b for b in DISPLAY if b in ds and b in gem]
    others = [b for b in DISPLAY if b not in ds and b in gem]
    order = comparable + others

    fig, ax = plt.subplots(figsize=(12.5, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    h = 0.34
    gap = 0.04  # surface gap between the paired bars, not a border
    ys = list(range(len(order)))[::-1]

    for y, bench in zip(ys, order):
        g = gem[bench]
        ax.barh(y + (h + gap) / 2, g, height=h, color=GEMINI, zorder=3)
        ax.annotate(
            f"{g:.1f}",
            xy=(g, y + (h + gap) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontfamily=MONO,
            color=INK,
            fontweight="bold",
        )

        if bench in ds:
            d = ds[bench]
            ax.barh(y - (h + gap) / 2, d, height=h, color=DEEPSEEK, zorder=3)
            ax.annotate(
                f"{d:.1f}",
                xy=(d, y - (h + gap) / 2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                fontfamily=MONO,
                color=INK,
                fontweight="bold",
            )
        else:
            # No bar, and say why — an empty row must never read as a zero score.
            ax.annotate(
                f"— {pending_note(bench)}",
                xy=(0, y - (h + gap) / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                fontfamily=MONO,
                color=DEEPSEEK,
                style="italic",
            )

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [DISPLAY[b] for b in order], fontsize=11.5, fontfamily=MONO, color=INK
    )
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(
        ["0", "20", "40", "60", "80", "100"],
        fontsize=9.5,
        fontfamily=MONO,
        color=INK_MUTED,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)  # solid hairline
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    # Separator under the comparable block — the rows below it are not a contest.
    if comparable and others:
        ax.axhline(len(order) - len(comparable) - 0.5, color=GRID, linewidth=1.2)

    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=GEMINI,
                label=f"{GEMINI_COL} — from repo CSV (team's runs)",
            ),
            plt.Rectangle(
                (0, 0), 1, 1, color=DEEPSEEK, label="DeepSeek-V4-Flash — MEASURED here"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.175),
        ncol=2,
        frameon=False,
        prop={"family": MONO, "size": 10},
        labelcolor=[GEMINI, DEEPSEEK],
    )

    fig.text(
        0.008,
        0.965,
        "Gemini-3.5-flash vs DeepSeek-V4-Flash",
        fontsize=15,
        fontfamily=MONO,
        color=INK,
    )
    fig.text(
        0.008,
        0.925,
        f"higher is better  ·  only the {len(comparable)} row(s) above the line are a like-for-like comparison",
        fontsize=10,
        fontfamily=MONO,
        color=INK_MUTED,
    )
    fig.text(
        0.008,
        0.035,
        "DeepSeek V4 Flash is text-only (probed: image -> HTTP 400 'does not support image inputs'; audio -> 400). Those rows have no bar, not a zero.\n"
        "Thinking settings differ and are NOT harmonised: Gemini GPQA/MMMLU were run thinking ON; our DeepSeek runs use reasoning_effort=none (its floor).\n"
        "Spider-2.0-Lite for DeepSeek is 27.4% after fixing an unclosed-markdown-fence bug in the SQL extractor; before the fix the same responses scored 15.6%.",
        fontsize=8.5,
        fontfamily=MONO,
        color=INK_MUTED,
        linespacing=1.7,
    )

    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.26)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}\n")
    print(f"{'benchmark':18} {GEMINI_COL:>18} {'deepseek-v4-flash':>18}")
    for b in order:
        d = f"{ds[b]:.2f}" if b in ds else f"n/a ({pending_note(b)})"
        print(f"  {DISPLAY[b]:16} {gem[b]:18.2f} {d:>18}")


if __name__ == "__main__":
    main()
