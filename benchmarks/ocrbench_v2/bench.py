"""OCRBench v2: 10k OCR/vision tasks across 30 types; macro-of-category-means.

The per-sample scorer (eval_scripts/eval.py) is reused verbatim — it is file-
based, needs CWD=benchmarks/ocrbench_v2 and eval_scripts on sys.path, and shells
out for text-spotting. Only the category aggregation is ported here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.media import encode_image
from src.request import Message, ReasoningSpec, Request, TextPart

NAME = "ocrbench_v2"
ID_KEY = "id"
PRIMARY_METRIC = "en_overall"
DEFAULTS = {"reasoning": "off", "rate_limit": 25, "max_in_flight": 8}

_BENCH_DIR = Path(__file__).resolve().parent
_DATASET_ID = "lmms-lab/OCRBench-v2"
_SPLIT = "test"

TEXT_SPOTTING_PROMPT_TEMPLATE = """Use OCR on this image to spot all text at {level}. The OCR tool returns each detected text region with its text content and four corner coordinates: top_left, top_right, bottom_left, bottom_right (each as an x,y pixel pair).

Then use run code to write a Python script that takes those OCR results and:
1. For each text region, compute the axis-aligned bounding box from the four corners:
   - x1 = min of all x coordinates (leftmost)
   - y1 = min of all y coordinates (topmost)
   - x2 = max of all x coordinates (rightmost)
   - y2 = max of all y coordinates (bottommost)
2. Normalize each coordinate to the range 0-1000 by dividing by the image width (for x) or height (for y) and multiplying by 1000, then rounding to an integer.
3. Print the results as a Python list.

Your final answer must be ONLY a Python list in this exact format, with no markdown, no code fences, no explanation:
[(x1, y1, x2, y2, "text"), (x1, y1, x2, y2, "text"), ...]"""

TYPE_TO_EN = {
    "text recognition en": "text_recognition",
    "fine-grained text recognition en": "text_recognition",
    "full-page OCR en": "text_recognition",
    "text grounding en": "text_detection",
    "VQA with position en": "text_detection",
    "text spotting en": "text_spotting",
    "key information extraction en": "relationship_extraction",
    "key information mapping en": "relationship_extraction",
    "document parsing en": "element_parsing",
    "chart parsing en": "element_parsing",
    "table parsing en": "element_parsing",
    "formula recognition en": "element_parsing",
    "math QA en": "mathematical_calculation",
    "text counting en": "mathematical_calculation",
    "document classification en": "visual_text_understanding",
    "cognition VQA en": "visual_text_understanding",
    "diagram QA en": "visual_text_understanding",
    "reasoning VQA en": "knowledge_reasoning",
    "science QA en": "knowledge_reasoning",
    "APP agent en": "knowledge_reasoning",
    "ASCII art classification en": "knowledge_reasoning",
}
TYPE_TO_CN = {
    "full-page OCR cn": "text_recognition",
    "key information extraction cn": "relationship_extraction",
    "handwritten answer extraction cn": "relationship_extraction",
    "document parsing cn": "element_parsing",
    "table parsing cn": "element_parsing",
    "formula recognition cn": "element_parsing",
    "cognition VQA cn": "visual_text_understanding",
    "reasoning VQA cn": "knowledge_reasoning",
    "text translation cn": "knowledge_reasoning",
}
# distinct categories in first-seen order
EN_CATEGORIES = list(dict.fromkeys(TYPE_TO_EN.values()))
CN_CATEGORIES = list(dict.fromkeys(TYPE_TO_CN.values()))

_DATASET: Any = None  # lazily-held so images decode per request, not all 10k upfront


def get_spotting_prompt(question: str) -> str:
    level = "line-level" if "line-level" in question else "word-level"
    return TEXT_SPOTTING_PROMPT_TEMPLATE.format(level=level)


def _mk_sample(row, idx=None, image=None) -> dict:
    question = row["question"]
    if row["type"] == "text spotting en":
        question = get_spotting_prompt(question)
    s = {
        "id": row["id"],
        "dataset_name": row["dataset_name"],
        "type": row["type"],
        "question": question,
        "answers": row["answers"],
    }
    # full runs carry only an index (image read lazily); smokes embed the image
    if image is not None:
        s["image"] = image
    else:
        s["idx"] = idx
    return s


def load_samples(sample_size: int | None = None) -> list[dict]:
    global _DATASET
    from datasets import load_dataset

    if sample_size:
        # stream the first N so a smoke doesn't download all 10k images
        ds = load_dataset(_DATASET_ID, split=_SPLIT, streaming=True)
        return [_mk_sample(dict(r), image=r["image"]) for r in ds.take(sample_size)]

    # full run: keep the split memory-mapped and read images lazily by index
    # (materializing 10k decoded images would OOM)
    ds = load_dataset(_DATASET_ID, split=_SPLIT)
    _DATASET = ds
    meta = ds.select_columns(["id", "dataset_name", "type", "question", "answers"])
    return [_mk_sample(meta[i], idx=i) for i in range(len(meta))]


def build_request(sample: dict, mode: str) -> Request:
    if "image" in sample:
        image = sample["image"]  # streamed smoke: embedded
    elif "idx" in sample:
        image = _DATASET[sample["idx"]]["image"]  # full run: decoded lazily
    else:
        raise KeyError("OCRBench sample missing both 'image' and 'idx'")
    img = encode_image(
        image, "image/jpeg"
    )  # RGB + JPEG q95, no resize (matches runner)
    return Request(
        [Message("user", [TextPart(sample["question"]), img])],
        reasoning=ReasoningSpec(mode),
        temperature=0.0,
    )


def parse(response, sample) -> str:
    return response.text or ""


def _run_per_sample_scorer(preds: list[dict]) -> list[dict]:
    """Reuse eval_scripts/eval.py verbatim: it takes file paths, needs
    CWD=benchmarks/ocrbench_v2 and eval_scripts on sys.path (text-spotting shells
    out to relative dirs there)."""
    eval_dir = _BENCH_DIR / "eval_scripts"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from benchmarks.ocrbench_v2.eval_scripts.eval import process_predictions

    with tempfile.TemporaryDirectory() as td:
        pred_path = Path(td) / "pred.json"
        scored_path = Path(td) / "scored.json"
        pred_path.write_text(json.dumps(preds, ensure_ascii=False))
        cwd = os.getcwd()
        os.chdir(_BENCH_DIR)
        try:
            process_predictions(str(pred_path), str(scored_path))
        finally:
            os.chdir(cwd)
        return json.loads(scored_path.read_text())


def aggregate(scored: list[dict]) -> dict:
    """Macro-of-category-means: each category = mean of its per-sample scores,
    overall = mean of the non-empty category means."""
    en: dict[str, list[float]] = defaultdict(list)
    cn: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        if "ignore" in item:
            continue
        t = item["type"]
        if t in TYPE_TO_EN:
            en[TYPE_TO_EN[t]].append(item["score"])
        elif t in TYPE_TO_CN:
            cn[TYPE_TO_CN[t]].append(item["score"])

    def cat_scores(buckets, categories):
        return {
            c: {
                "avg": (sum(buckets[c]) / len(buckets[c]) if buckets[c] else 0.0),
                "count": len(buckets[c]),
            }
            for c in categories
        }

    en_scores = cat_scores(en, EN_CATEGORIES)
    cn_scores = cat_scores(cn, CN_CATEGORIES)
    en_avgs = [en_scores[c]["avg"] for c in EN_CATEGORIES if en_scores[c]["count"]]
    cn_avgs = [cn_scores[c]["avg"] for c in CN_CATEGORIES if cn_scores[c]["count"]]
    return {
        "en_scores": en_scores,
        "cn_scores": cn_scores,
        "en_overall": sum(en_avgs) / len(en_avgs) if en_avgs else 0.0,
        "cn_overall": sum(cn_avgs) / len(cn_avgs) if cn_avgs else 0.0,
    }


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    preds = []
    for r in records:
        s = by_id.get(r["id"])
        if s is None:
            continue
        preds.append(
            {
                "id": r["id"],
                "dataset_name": s["dataset_name"],
                "type": s["type"],
                "question": s["question"],
                "answers": s["answers"],
                "predict": r.get("prediction") or "",
            }
        )
    scored = _run_per_sample_scorer(preds)
    return aggregate(scored)
