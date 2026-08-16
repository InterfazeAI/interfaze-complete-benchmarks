"""olmOCR-bench: PDF page -> markdown, scored by the external olmocr.bench scorer.

Special-cased: the scorer reads a directory of per-page .md files and prints its
table to stdout (no metrics.json), so `score` materializes the .md files from the
run records, invokes the scorer, tees stdout to logs/olmocr_<candidate>.log (what
report_scores reads), and parses the overall out of it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from bench_core.request import ImagePart, Message, ReasoningSpec, Request, TextPart
from bench_core.results import model_slug

NAME = "olmocr"
ID_KEY = "id"
PRIMARY_METRIC = "overall"
DEFAULTS = {"reasoning": "off", "rate_limit": 25, "max_in_flight": 8}

_HF_REPO = "allenai/olmOCR-bench"
_FULL_DATA_DIR = Path(__file__).resolve().parent / "bench" / "full_data"
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_SPLITS = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]
_MAX_TOKENS = 20000

PROMPT = (
    "Below is the image of one page of a PDF document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Turn equations into LaTeX using \\( \\) for inline math and \\[ \\] for display math. "
    "Never describe equations in words — always use LaTeX notation. "
    "Turn tables into markdown format.\n"
    "Remove the headers and footers completely — do not include any text "
    "that appears at the very top or very bottom of the page outside the main body content. "
    "This includes page numbers, journal names, author names in running headers, "
    "copyright lines, DOI lines, citation requests, institutional addresses in margins, "
    "and download dates. Keep references and footnotes that are part of the body.\n"
    "For multi-column layouts, read each column top to bottom before moving to the next.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve "
    "any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
    "If there is no text at all that you think you should read, you can output null.\n"
    "Do not hallucinate."
)

_NULL = ("null", "none", "n/a", "")


def _download() -> Path:
    from huggingface_hub import hf_hub_download

    data_dir = _FULL_DATA_DIR
    pdf_dir = data_dir / "pdfs"
    all_pdfs = set()
    for split in _SPLITS:
        dest = data_dir / f"{split}.jsonl"
        if not dest.exists():
            src = hf_hub_download(
                _HF_REPO, f"bench_data/{split}.jsonl", repo_type="dataset"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(Path(src).read_text())
        for line in dest.read_text().splitlines():
            if line.strip():
                all_pdfs.add(json.loads(line)["pdf"])
    for pdf_rel in sorted(all_pdfs):
        local = pdf_dir / pdf_rel
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(
            _HF_REPO, f"bench_data/pdfs/{pdf_rel}", repo_type="dataset"
        )
        os.symlink(src, str(local))
    return data_dir


def load_samples(sample_size: int | None = None) -> list[dict]:
    data_dir = _download()
    pairs = set()
    for jf in data_dir.glob("*.jsonl"):
        for line in jf.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                pairs.add((t["pdf"], t["page"]))
    samples = []
    for pdf_rel, page in sorted(pairs):
        pdf_path = data_dir / "pdfs" / pdf_rel
        if not pdf_path.exists():
            continue
        base = os.path.splitext(os.path.basename(pdf_rel))[0]
        parent = os.path.dirname(pdf_rel)
        out_rel = (
            f"{parent}/{base}_pg{page}_repeat1.md"
            if parent
            else f"{base}_pg{page}_repeat1.md"
        )
        samples.append(
            {
                "id": f"{pdf_rel}#{page}",
                "pdf_path": str(pdf_path),
                "page": page,
                "out_rel": out_rel,
            }
        )
    return samples[:sample_size] if sample_size else samples


def build_request(sample: dict, mode: str) -> Request:
    from benchmarks.olmocr.data.renderpdf import render_pdf_to_base64png

    b64 = render_pdf_to_base64png(
        sample["pdf_path"], page_num=sample["page"], target_longest_image_dim=2048
    )
    import base64

    png = base64.b64decode(b64)
    return Request(
        [Message("user", [TextPart(PROMPT), ImagePart(png, "image/png")])],
        reasoning=ReasoningSpec(mode),
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
    )


def parse(response, sample) -> str:
    raw = response.text or ""
    return "" if raw.strip().lower() in _NULL else raw


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def score(records: list[dict], samples: list[dict], target=None) -> dict:
    candidate = model_slug(target.name) if target is not None else "candidate"
    by_id = {s["id"]: s for s in samples}
    out_root = _FULL_DATA_DIR / candidate
    # materialize per-page .md from the run records (the scorer reads these)
    for r in records:
        s = by_id.get(r["id"])
        if s is None:
            continue
        md_path = out_root / s["out_rel"]
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(r.get("prediction") or "", encoding="utf-8")

    from olmocr.bench.benchmark import main as bench_main

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOGS_DIR / f"olmocr_{candidate}.log"
    import io

    buf = io.StringIO()
    argv, real = sys.argv, sys.stdout
    sys.argv = [
        "benchmark",
        "--dir",
        str(_FULL_DATA_DIR),
        "--candidate",
        candidate,
        "--force",
    ]
    with open(log_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(real, fh, buf)
        try:
            bench_main()
        finally:
            sys.stdout, sys.argv = real, argv

    return {
        "candidate": candidate,
        "log": str(log_path),
        **_parse_scores(buf.getvalue()),
    }


def _parse_scores(text: str) -> dict:
    text = text.replace("\r", "\n")
    head = re.search(
        r"^(\S+)\s*:\s*Average Score:\s*([\d.]+)%\s*±\s*([\d.]+)%", text, re.MULTILINE
    )
    overall = float(head.group(2)) if head else None
    splits = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"^\s+(\w+)\.jsonl\s*:\s*([\d.]+)%", text, re.MULTILINE)
    }
    base = re.search(r"^\s+baseline\s*:\s*([\d.]+)%", text, re.MULTILINE)
    if base:
        splits["baseline"] = float(base.group(1))
    return {"overall": overall, "splits": splits}
