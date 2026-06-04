"""
olmOCR-bench runner for Amazon Textract (AnalyzeDocument + LAYOUT + TABLES).

Per-page contract (matches the other runners):
    run_textract(pdf_path, page_num=1) -> markdown string for that one page.

Pipeline:
  1. Render the requested 1-indexed page to a PNG via olmOCR's own
     render_pdf_to_base64png (same helper the Gemini runner uses, so the
     rasterization matches what the dataset expects).
  2. Call the SYNC AnalyzeDocument API with FeatureTypes = LAYOUT + TABLES:
       - LAYOUT  -> reading order + element typing. Critically this tags
                    headers / footers / page numbers as distinct blocks, so
                    we can DROP them structurally (olmOCR's `absent` tests
                    want them gone). This is the real lever — not a prompt.
       - TABLES  -> table cell structure, linearized to markdown for the
                    table_tests subset.
  3. Linearize blocks -> markdown with MarkdownLinearizationConfig, overriding
     the hide_* flags so header/footer/page-number layout elements are
     excluded from the text.

Hard limitations of Textract on this benchmark (NOT config bugs):
  * No math / LaTeX recognition at all -> arxiv_math & old_scans_math ~0.
  * Only English/French/German/Italian/Portuguese/Spanish -> any CJK / Arabic
    / vertical-text page fails.
  * Sync AnalyzeDocument: 10 MB / single page / <=10000px per side — our
    rendered PNGs are well within this.
"""

import base64
import io

from PIL import Image

from olmocr.data.renderpdf import render_pdf_to_base64png
from textractor.data.constants import TextractFeatures
from textractor.data.markdown_linearization_config import (
    MarkdownLinearizationConfig,
)

from src.commons_textract import record_usage, textractor_client


# Features requested on every page. LAYOUT is what lets us identify and drop
# headers/footers/page-numbers; TABLES gives proper table structure.
FEATURES = [TextractFeatures.LAYOUT, TextractFeatures.TABLES]
_FEATURE_NAMES = ["LAYOUT", "TABLES"]

# Rendering resolution. olmOCR's VLM runners use 2048 on the longest edge;
# Textract needs >=15px character height, so a higher raster helps the
# tiny-text subset. 2048 keeps PNGs well under the 10 MB sync limit.
TARGET_LONGEST_DIM = 2048

# Markdown linearization with headers / footers / page numbers removed.
# MarkdownLinearizationConfig already sets markdown table format + '# ' / '## '
# heading prefixes; we only flip the hide_* flags to satisfy olmOCR's
# `absent` (header/footer must be gone) tests.
LINEARIZATION_CONFIG = MarkdownLinearizationConfig(
    hide_header_layout=True,
    hide_footer_layout=True,
    hide_page_num_layout=True,
)


def _render_page_image(pdf_path: str, page_num: int) -> Image.Image:
    """Render one 1-indexed PDF page to a PIL image for Textract."""
    b64 = render_pdf_to_base64png(
        pdf_path, page_num, target_longest_image_dim=TARGET_LONGEST_DIM
    )
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def run_textract(
    pdf_path: str,
    page_num: int = 1,
    timeout: float = 600.0,  # accepted for signature parity; unused (sync API)
) -> str:
    """OCR one PDF page through Amazon Textract and return markdown."""
    image = _render_page_image(pdf_path, page_num)

    document = textractor_client.analyze_document(
        file_source=image,
        features=FEATURES,
        save_image=False,
    )
    record_usage(_FEATURE_NAMES, pages=1)

    text = document.get_text(config=LINEARIZATION_CONFIG)
    if not text or text.strip().lower() in ("null", "none", "n/a"):
        return ""
    return text.strip()
