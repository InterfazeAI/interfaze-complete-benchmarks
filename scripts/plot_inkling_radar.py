"""
Radar chart of the full Inkling benchmark suite, with Inkling-Small alongside.

Provenance is the whole point of this chart's design, so it is encoded three ways
(colour, line style, and a per-point tag) and never by colour alone:

  MEASURED  — Inkling, run by this repo. Solid line, filled markers.
  pub       — Inkling-Small, Thinking Machines' published figure. Not run here.
  est       — Inkling-Small, ESTIMATED by inference from the published table.
              Not a measurement of any kind.

Inkling-Small is not on Fireworks serverless (dedicated deployment only), so
there is nothing measured to plot for it.

    uv run python scripts/plot_inkling_radar.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "inkling_benchmarks_radar.png"

# Validated palette (references/palette.md), light mode. Both slots pass every
# check vs this surface (worst adjacent CVD dE 24.7).
SURFACE = "#fcfcfb"
MEASURED = "#2a78d6"  # slot 1 — blue
CLAIMED = "#eb6834"  # slot 2 — orange
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
MONO = ["DejaVu Sans Mono", "Menlo", "monospace"]

# Thinking Machines' published table (thinkingmachines.ai/news/inkling-small/),
# reported at effort=0.99 — MAX thinking, not the floor our runs used.
PUBLISHED_SMALL = {"GPQA Diamond": 88.3, "MMMU-Pro": 73.1}

# Where nothing is published, an explicit estimate. Rationale from the published
# table's own shape: Inkling-Small holds parity on short-form reasoning and
# vision (GPQA +1.1, MMMU-Pro -0.4) but drops hard on factual recall (SimpleQA
# 20.9 vs 43.9) and long-horizon agentic work (Tau-3 Banking 13.6 vs 23.7,
# Terminal-Bench 52.7 vs 63.8).
#   MMMLU     — scaled by the published Global-MMLU-Lite ratio (86.8/88.7).
#   VoxPopuli — audio is near parity (VoiceBench 90.0 vs 91.4, MMAU 77.5/77.2).
#   OCR/RefCOCO — perception-bound like MMMU-Pro, so a small markdown.
#   Spider    — long unseen schema behaves like the agentic cluster: marked down.
ESTIMATE_RATIOS = {
    "MMMLU": 86.8 / 88.7,
    "VoxPopuli-AA": 0.994,
    "OCRBench V2": 0.97,
    "olmOCR": 0.97,
    "RefCOCO": 0.96,
    "Spider-2.0-Lite": 0.865,
}


def pct(x):
    return round(x * 100, 2)


def load_measured():
    """Every axis with a COMPLETE measured run, north then clockwise."""

    def j(name):
        p = RESULTS / name
        return json.loads(p.read_text()) if p.exists() else None

    ocr = j("ocrbench_v2_fireworks_inkling_metrics.json")
    gpqa = j("fireworks_inkling_thinkingoff_gpqa_diamond_metrics.json")
    mmmu = j(
        "mmmupro_standard_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    mmmlu = j(
        "mmmlulite_fireworks_accounts-fireworks-models-inkling_reasoningoff_metrics.json"
    )
    spider = j("spider2_lite_local_fireworks_inkling_metrics.json")
    asr = j("voxpopuli_aa_fireworks_accounts_fireworks_models_inkling_metrics.json")
    rc = j(
        "refcoco_val_fireworks_accounts_fireworks_models_inkling_oracle_metrics.json"
    )

    rows = [
        (
            "OCRBench V2",
            "Native OCR (EN)",
            pct(ocr["en_overall"]) if ocr else None,
            10000 if ocr else 0,
        ),
        # olmOCR-bench writes its score to stdout, not a metrics file — see
        # OLMOCR_SCORE below.
        ("olmOCR", "Complex document processing", OLMOCR_SCORE, 1403),
        (
            "RefCOCO",
            "Object detection (oracle)",
            pct(rc["accuracy"]) if rc else None,
            rc["total"] if rc else 0,
        ),
        (
            "Spider-2.0-Lite",
            "Text-to-SQL",
            pct(spider["accuracy_of_local_135"]),
            spider["total_local_subset"],
        ),
        (
            "VoxPopuli-AA",
            "Speech recognition (1-WER)",
            pct(1 - asr["corpus_wer"]),
            asr.get("samples") or asr.get("num_samples"),
        ),
        (
            "MMMLU",
            "Multilingual Q&A",
            pct(mmmlu["macro_accuracy"]),
            mmmlu["num_samples"],
        ),
        (
            "GPQA Diamond",
            "PhD-level problem solving",
            pct(gpqa["accuracy"]),
            gpqa["total"],
        ),
        (
            "MMMU-Pro",
            "Multimodal understanding (std)",
            pct(mmmu["accuracy"]),
            mmmu["num_samples"],
        ),
    ]
    return [r for r in rows if r[2] is not None]


# olmOCR-bench prints its headline to stdout rather than persisting a metrics
# file, so it is pinned here from the run log (1403/1403 pages, 8413 tests).
OLMOCR_SCORE = 74.9


def main():
    scores = load_measured()
    labels = [s[0] for s in scores]
    subs = [s[1] for s in scores]
    vals = [s[2] for s in scores]
    ns = [s[3] for s in scores]
    n = len(scores)

    small_vals, small_tags = [], []
    for name, v in zip(labels, vals):
        if name in PUBLISHED_SMALL:
            small_vals.append(PUBLISHED_SMALL[name])
            small_tags.append("pub")
        else:
            small_vals.append(round(v * ESTIMATE_RATIOS[name], 1))
            small_tags.append("est")

    # Guard: a tag per value. If these ever diverge, zip() below would silently
    # drop points rather than fail, which is how a mislabelled chart ships.
    assert len(small_tags) == len(small_vals) == len(vals), (
        len(small_tags),
        len(small_vals),
        len(vals),
    )

    theta = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    closed_theta = theta + [theta[0]]

    # No set_theta_zero_location — theta is computed absolutely above, so
    # matplotlib's default (0 = east, CCW) keeps index 0 due north.
    fig = plt.figure(figsize=(14.5, 10.5), facecolor=SURFACE)
    ax = fig.add_subplot(111, polar=True, facecolor=SURFACE)
    ax.set_ylim(0, 100)
    ax.set_axis_off()  # draw our own recessive web instead of matplotlib's

    # Web: solid hairline polygons, one shade off the surface (never dashed).
    for r in (20, 40, 60, 80, 100):
        ax.add_patch(
            Polygon(
                [(t, r) for t in theta],
                closed=True,
                fill=False,
                edgecolor=GRID,
                linewidth=0.8,
                zorder=1,
            )
        )
    for t in theta:
        ax.plot([t, t], [0, 100], color=GRID, linewidth=0.8, zorder=1)

    # Inkling-Small first so the measured series sits on top of it.
    ax.plot(
        closed_theta,
        small_vals + [small_vals[0]],
        color=CLAIMED,
        linewidth=2,
        linestyle=(0, (5, 3)),
        zorder=2,
    )
    ax.scatter(
        theta,
        small_vals,
        s=80,
        facecolors=SURFACE,
        edgecolors=CLAIMED,
        linewidths=2,
        zorder=3,
    )

    ax.fill(closed_theta, vals + [vals[0]], color=MEASURED, alpha=0.13, zorder=4)
    ax.plot(closed_theta, vals + [vals[0]], color=MEASURED, linewidth=2, zorder=5)
    ax.scatter(
        theta, vals, s=90, color=MEASURED, edgecolors=SURFACE, linewidths=2, zorder=6
    )

    # Direct value labels — every value readable without a tooltip. Offsets run
    # along the spoke (outward for the higher series) so the pair stays apart
    # even when the values nearly coincide.
    for t, v, sv, tag in zip(theta, vals, small_vals, small_tags):
        ux, uy = np.cos(t), np.sin(t)
        m_dir, s_dir = (16, -18) if v >= sv else (-18, 16)
        if abs(uy) >= abs(ux):
            m_off, s_off = (ux * m_dir, uy * m_dir), (ux * s_dir, uy * s_dir)
        else:
            # Horizontal spokes: these labels are wide, so stack them
            # perpendicular instead, nudged outward clear of the markers.
            m_off, s_off = (ux * 10, 13), (ux * 10, -13)
        ax.annotate(
            f"{v:.1f}%",
            xy=(t, v),
            xytext=m_off,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=MONO,
            color=MEASURED,
            fontweight="bold",
            zorder=7,
        )
        ax.annotate(
            f"{sv:.1f}% {tag}",
            xy=(t, sv),
            xytext=s_off,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.5,
            fontfamily=MONO,
            color=CLAIMED,
            zorder=7,
        )

    # One scale anchor on an empty diagonal — ring ticks along a spoke collide
    # with that axis's label block, and every value is direct-labelled anyway.
    ax.annotate(
        "100",
        xy=(np.pi / 2 - np.pi / n, 100),
        xytext=(8, 4),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        fontfamily=MONO,
        color=INK_MUTED,
        zorder=5,
    )

    # Axis labels outside the web; descriptor and n on one line.
    for t, name, sub, ncount in zip(theta, labels, subs, ns):
        x, y = np.cos(t), np.sin(t)
        ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
        ax.annotate(
            name,
            xy=(t, 100),
            xytext=(x * 26, y * 26 + 7),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=12.5,
            fontfamily=MONO,
            color=INK,
            zorder=5,
        )
        ax.annotate(
            f"{sub}  ·  n={ncount}",
            xy=(t, 100),
            xytext=(x * 26, y * 26 - 7),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9,
            fontfamily=MONO,
            color=INK_MUTED,
            zorder=5,
        )

    fig.text(
        0.5,
        0.973,
        "Inkling vs Inkling-Small — complete Fireworks benchmark suite",
        ha="center",
        fontsize=15,
        fontfamily=MONO,
        color=INK,
    )
    fig.text(
        0.5,
        0.945,
        "higher is better on every axis  ·  rings every 20 points  ·  n = measured run's sample count",
        ha="center",
        fontsize=9.5,
        fontfamily=MONO,
        color=INK_MUTED,
    )

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=MEASURED,
                lw=2,
                marker="o",
                markersize=8,
                markeredgecolor=SURFACE,
                label="Inkling — MEASURED here (reasoning_effort=none, the floor)",
            ),
            Line2D(
                [],
                [],
                color=CLAIMED,
                lw=2,
                linestyle=(0, (5, 3)),
                marker="o",
                markersize=8,
                markerfacecolor=SURFACE,
                markeredgecolor=CLAIMED,
                label="Inkling-Small — NOT measured: pub = published, est = estimated",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        frameon=False,
        prop={"family": MONO, "size": 10.5},
        labelcolor=[MEASURED, CLAIMED],
        handlelength=3,
        borderaxespad=0,
    )

    fig.text(
        0.5,
        0.045,
        "Inkling-Small is not on Fireworks serverless (dedicated deployment only) — nothing is measured for it here.\n"
        "pub = published at effort=0.99 (MAX thinking), NOT comparable to our floor runs · est = our inference, 6 of 8 axes, treat as hypotheses.\n"
        "RefCOCO is the format-tolerant oracle re-score (strict single-parser score is 32.4% — Inkling emits 4+ coordinate conventions).\n"
        "MMMU-Pro is the standard setting, the only one published for both (our vision run measured 68.1%; composite 70.7%).",
        ha="center",
        fontsize=8.5,
        fontfamily=MONO,
        color=INK_MUTED,
        linespacing=1.8,
    )

    fig.subplots_adjust(left=0.23, right=0.77, top=0.80, bottom=0.185)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}\n")
    print(f"{'benchmark':18} {'inkling':>10} {'small':>10}   n")
    for (name, _s, v, ncount), sv, tag in zip(scores, small_vals, small_tags):
        print(f"  {name:16} {v:9.2f}% {sv:9.1f}% {tag:4} n={ncount}")


if __name__ == "__main__":
    main()
