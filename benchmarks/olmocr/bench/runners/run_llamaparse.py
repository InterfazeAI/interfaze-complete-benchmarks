"""
olmOCR-bench runner for LlamaParse (llama-cloud).

Per-page contract (matches run_reducto / run_gemini_pro_31 / run_extend):
    run_llamaparse(pdf_path, page_num=1) -> markdown string for that one page.

We extract the requested 1-indexed page to a tiny temp PDF (via pymupdf) and
upload that. Same logic as run_extend — bulletproof, keeps the upload payload
small even on huge source PDFs, and removes any server-side page-range edge
cases.

Mode choice (verified against developers.llamaindex.ai/llamaparse):
  * tier="agentic_plus" — newer Tier API; "state-of-the-art models for
    maximum accuracy on the hardest documents". Supersedes the legacy
    `parse_mode` family (parse_page_with_agent et al).
  * version="latest"
  * expand=["markdown"] — populate result.markdown / result.markdown_full
    in the response. Without this expand, you get only job metadata.

Things intentionally NOT set in the initial config:
  * agentic_options.custom_prompt — soft natural-language steering, same
    family as Extend's `custom_instructions`. Burned us once already; we
    don't want to rely on it for header/footer removal. Start clean.
  * crop_box — geometric header/footer clipping. Real lever, but blindly
    applying e.g. top=0.08 can chop body text on pages without headers.
    Add only if a sample run shows headers leaking through.
  * processing_control / processing_options — left at server defaults.
"""

import os
import tempfile

import pymupdf

from src.commons_llamaparse import llamaparse_client, record_usage


PARSE_TIER = "agentic_plus"
PARSE_VERSION = "latest"


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
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="llamaparse_pg_")
            os.close(fd)
            out.save(tmp_path)
            return tmp_path
        finally:
            out.close()
    finally:
        src.close()


def _extract_markdown(result) -> str:
    """Pull the markdown text out of a ParsingGetResponse.

    Prefer `markdown_full` (flat string); fall back to joining
    `markdown.pages[i].markdown` so we still get something if the server
    only returns the structured form.
    """
    full = getattr(result, "markdown_full", None)
    if isinstance(full, str) and full.strip():
        return full.strip()

    md = getattr(result, "markdown", None)
    pages = getattr(md, "pages", None) if md is not None else None
    if not pages:
        return ""

    pieces: list[str] = []
    for page in pages:
        piece = getattr(page, "markdown", None)
        if isinstance(piece, str) and piece.strip():
            pieces.append(piece.strip())
    return "\n\n".join(pieces).strip()


def run_llamaparse(
    pdf_path: str,
    page_num: int = 1,
    timeout: float = 600.0,
) -> str:
    """Parse one PDF page through LlamaParse and return the markdown content."""
    single_page_path = _extract_page_to_tempfile(pdf_path, page_num)
    try:
        with open(single_page_path, "rb") as fh:
            uploaded = llamaparse_client.files.create(file=fh, purpose="parse")

        result = llamaparse_client.parsing.parse(
            file_id=uploaded.id,
            tier=PARSE_TIER,
            version=PARSE_VERSION,
            expand=["markdown"],
            timeout=timeout,
        )

        record_usage(PARSE_TIER, 1)

        job_status = getattr(getattr(result, "job", None), "status", None)
        if job_status != "COMPLETED":
            err = getattr(getattr(result, "job", None), "error_message", None)
            raise RuntimeError(
                f"LlamaParse job did not complete: status={job_status} error={err}"
            )

        text = _extract_markdown(result)
        if not text or text.strip().lower() in ("null", "none", "n/a"):
            return ""
        return text
    finally:
        try:
            os.unlink(single_page_path)
        except OSError:
            pass
