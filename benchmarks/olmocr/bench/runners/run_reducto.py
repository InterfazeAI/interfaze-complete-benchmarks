from src.providers.reducto import parse_r1, upload_file

# olmOCR pages are graded on their markdown; the headers_footers split rewards
# dropping running headers/footers, which r-1 does via retrieval.filter_blocks.
FILTER_BLOCKS = ["Header", "Footer", "Page Number"]


def run_reducto(
    pdf_path: str,
    page_num: int = 1,
    timeout: float = 600.0,
) -> str:
    """Parse a single PDF page through Reducto Parse r-1 and return the markdown."""
    file_id = upload_file(pdf_path, timeout=timeout)
    text, _job_id, _usage = parse_r1(
        file_id,
        page_range=[page_num],
        filter_blocks=FILTER_BLOCKS,
        timeout=timeout,
    )
    if not text or text.strip().lower() in ("null", "none", "n/a"):
        return ""
    return text
