"""
olmOCR-bench runner for Extend.ai's /parse_runs endpoint — max-accuracy config.

Per-page contract (matches run_reducto / run_gemini_pro_31):
    run_extend(pdf_path, page_num=1) -> markdown string for that one page.

Design:
  * Extract one PDF page locally (pymupdf) so the upload payload stays small.
  * Use `engine="parse_performance"` (Extend's high-accuracy engine).
  * Turn on agentic processing for text + tables, and chart extraction for
    figures, with `page_rotation_enabled` for sideways scans.
  * **Walk chunk.blocks[] instead of joining chunk.content** — gives us:
      - structural filtering of headers/footers/page numbers (block.type
        match) rather than a soft prompt instruction the agentic pass ignores;
      - real LaTeX from formula blocks via `block.details.latex`, wrapped
        in `\\(...\\)` so olmOCR's math tests can match it.

  This replaces the earlier prompt-based approach, which left
  `<page_number>` tags in the markdown and never produced LaTeX.
"""

import os
import tempfile

import pymupdf

from src.commons_extend import extend_client, record_usage


# Block types that olmOCR's `absent` tests expect to NOT be in the output.
# Filtered structurally before we serialize anything.
SKIP_BLOCK_TYPES = {"header", "footer", "page_number"}


# Config — only documented Extend.ai parameters (verified against
# docs.extend.ai/product/parsing/configuration-options):
#
#   target=markdown             best for LLM-style text scoring (olmOCR uses
#                               substring/fuzzy matches against markdown gold).
#   engine=parse_performance    high-accuracy engine (vs parse_light); required
#                               for cellBlocksEnabled, advancedChartExtraction,
#                               formattingDetection, etc.
#   chunking_strategy=page      we upload one page at a time, so a single
#                               page-chunk is all we need.
#   text.agentic.enabled=true   VLM second-pass on text blocks; ON because
#                               olmOCR includes degraded scans + handwriting.
#                               (custom_instructions removed — undocumented,
#                               doesn't filter content, gave us nothing.)
#   text.signature_detection_enabled=true  cheap helper for old-scans subset.
#   tables.enabled+agentic      table re-parsing for messy/multi-page tables;
#                               markdown target_format matches our scorer.
#   formulas.enabled=true       *** THIS IS THE BIG ONE ***. Without it,
#                               equations come back as plain `text` blocks and
#                               we never see `details.latex`. Set to true so
#                               formula blocks appear and we can emit LaTeX.
#   figures.enabled+adv-chart   charts -> structured data; cheap insurance.
#   page_rotation_enabled=true  handles sideways scans (old_scans subset).
PARSE_CONFIG = {
    "target": "markdown",
    "engine": "parse_performance",
    "chunking_strategy": {"type": "page"},
    "block_options": {
        "text": {
            "signature_detection_enabled": True,
            "agentic": {"enabled": True},
        },
        "tables": {
            "enabled": True,
            "target_format": "markdown",
            "table_header_continuation_enabled": True,
            "agentic": {"enabled": True},
        },
        "formulas": {"enabled": True},
        "figures": {
            "enabled": True,
            "advanced_chart_extraction_enabled": True,
        },
    },
    "advanced_options": {
        "page_rotation_enabled": True,
    },
}


def _extract_page_to_tempfile(pdf_path: str, page_num: int) -> str:
    """Extract a single 1-indexed page into a new temp PDF file. Returns path."""
    src = pymupdf.open(pdf_path)
    try:
        if page_num < 1 or page_num > src.page_count:
            raise ValueError(
                f"page_num {page_num} out of range for {pdf_path} (n_pages={src.page_count})"
            )
        out = pymupdf.open()
        try:
            out.insert_pdf(src, from_page=page_num - 1, to_page=page_num - 1)
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="extend_pg_")
            os.close(fd)
            out.save(tmp_path)
            return tmp_path
        finally:
            out.close()
    finally:
        src.close()


def _block_type(block) -> str:
    """Return the block.type as a plain string ('text' / 'formula' / ...).

    SDK exposes type as an enum-or-string union (UNKNOWN sentinel possible),
    so normalize via .value when available.
    """
    t = getattr(block, "type", None)
    if t is None:
        return ""
    return getattr(t, "value", str(t))


def _render_block(block) -> str:
    """Serialize a single block to markdown text.

    Formula blocks: emit `\\( latex \\)` from `block.details.latex` so olmOCR's
        math tests can substring-match the LaTeX. Falls back to `block.content`
        if `details.latex` is missing.
    All other (kept) types: use `block.content` as-is — Extend already
        formatted it per the parse config (markdown tables, etc.).
    """
    btype = _block_type(block)

    if btype == "formula":
        details = getattr(block, "details", None)
        latex = getattr(details, "latex", None) if details is not None else None
        if latex:
            latex = latex.strip()
            return f"\\( {latex} \\)"
        # fall through to content if no latex available

    return getattr(block, "content", None) or ""


def _serialize_run(run) -> str:
    """Walk chunks -> blocks, skip header/footer/page_number, return markdown."""
    output = getattr(run, "output", None)
    chunks = getattr(output, "chunks", None) if output is not None else None
    if not chunks:
        return ""

    pieces: list[str] = []
    for chunk in chunks:
        blocks = getattr(chunk, "blocks", None) or []
        for block in blocks:
            if _block_type(block) in SKIP_BLOCK_TYPES:
                continue
            piece = _render_block(block)
            if piece and piece.strip():
                pieces.append(piece.strip())
    return "\n\n".join(pieces).strip()


def run_extend(
    pdf_path: str,
    page_num: int = 1,
    timeout: float = 600.0,
) -> str:
    """Parse one PDF page through Extend.ai and return the markdown content."""
    single_page_path = _extract_page_to_tempfile(pdf_path, page_num)
    try:
        with open(single_page_path, "rb") as fh:
            upload = extend_client.files.upload(file=fh)

        run = extend_client.parse_runs.create_and_poll(
            file={"id": upload.id},
            config=PARSE_CONFIG,
        )

        record_usage("parse", getattr(run, "metrics", None))

        status = getattr(run.status, "value", str(run.status))
        if status != "PROCESSED":
            err = (
                getattr(run, "failure_message", None)
                or getattr(run, "failure_reason", None)
            )
            raise RuntimeError(f"Extend parse failed: status={status} error={err}")

        text = _serialize_run(run)
        if not text or text.strip().lower() in ("null", "none", "n/a"):
            return ""
        return text
    finally:
        try:
            os.unlink(single_page_path)
        except OSError:
            pass
