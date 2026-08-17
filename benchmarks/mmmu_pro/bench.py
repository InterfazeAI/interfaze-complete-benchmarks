"""MMMU-Pro: multimodal MCQ (A-J), standard (text+images) and vision settings."""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from bench_core.media import encode_image
from bench_core.request import Message, ReasoningSpec, Request, TextPart

NAME = "mmmu_pro"
ID_KEY = "id"
PRIMARY_METRIC = "accuracy"
VARIANTS = ["standard", "vision"]
DEFAULTS = {"reasoning": "off", "rate_limit": 25, "max_in_flight": 8}

_DATASET_REPO = "MMMU/MMMU_Pro"
_SPLIT = "test"
_CONFIGS = {"standard": "standard (10 options)", "vision": "vision"}
_MAX_IMAGE_SIDE = 1536
_LETTERS = list("ABCDEFGHIJ")

_PROMPT_STANDARD = (
    "Answer the following multiple-choice question. The question may reference "
    'images via tags like "<image 1>", "<image 2>". The corresponding images '
    "are attached in order.\n\n"
    "Respond with ONLY a single letter (A through J) corresponding to the correct "
    "option. Do not explain.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Answer:"
)
_PROMPT_VISION = (
    "The attached image renders a multiple-choice question with its options. "
    "Respond with ONLY a single letter (A through J) corresponding to the correct "
    "option. Do not explain.\n\n"
    "Answer:"
)

_LETTER_RE = re.compile(r"\b([A-J])\b")
_FALLBACK_RE = re.compile(r"[A-J]")


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if len(s) == 1 and s.upper() in _LETTERS:
        return s.upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    m = _FALLBACK_RE.search(s.upper())
    return m.group(0) if m else None


def _options(raw) -> list[str]:
    return list(raw) if isinstance(raw, list) else list(ast.literal_eval(raw))


def _options_block(options: list[str]) -> str:
    return "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(_LETTERS, options))


def load_samples(
    sample_size: int | None = None, variant: str = "standard"
) -> list[dict]:
    from bench_core.datautil import load_rows

    rows = load_rows(_DATASET_REPO, _SPLIT, sample_size, config=_CONFIGS[variant])
    samples = []
    for row in rows:
        row = dict(row)
        if variant == "vision":
            samples.append(
                {
                    "id": row["id"],
                    "setting": "vision",
                    "image": row["image"],
                    "options": _options(row["options"]),
                    "answer": str(row["answer"]).strip().upper(),
                    "subject": row.get("subject"),
                    "topic_difficulty": None,
                }
            )
        else:
            images = [row.get(f"image_{i}") for i in range(1, 8)]
            samples.append(
                {
                    "id": row["id"],
                    "setting": "standard",
                    "question": row["question"],
                    "options": _options(row["options"]),
                    "images": [im for im in images if im is not None],
                    "answer": str(row["answer"]).strip().upper(),
                    "subject": row.get("subject"),
                    "topic_difficulty": row.get("topic_difficulty"),
                }
            )
    return samples


def build_request(sample: dict, mode: str) -> Request:
    if sample["setting"] == "vision":
        prompt = _PROMPT_VISION
        pil_images = [sample["image"]]
    else:
        prompt = _PROMPT_STANDARD.format(
            question=sample["question"], options=_options_block(sample["options"])
        )
        pil_images = sample["images"]
    parts = [TextPart(prompt)]
    parts += [
        encode_image(im, "image/jpeg", max_side=_MAX_IMAGE_SIDE) for im in pil_images
    ]
    return Request([Message("user", parts)], ReasoningSpec(mode), temperature=0.0)


def parse(response, sample) -> str | None:
    return parse_answer(response.text or "")


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    latest = {r["id"]: r for r in records if r["id"] in by_id}

    total = correct = unparse = 0
    by_subject: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for rid, r in latest.items():
        s = by_id[rid]
        pred = r.get("prediction")
        ok = pred == s["answer"]
        total += 1
        correct += int(ok)
        unparse += int(pred is None)
        if s.get("subject") is not None:
            by_subject[s["subject"]].append(ok)
        if s.get("topic_difficulty") is not None:
            by_difficulty[s["topic_difficulty"]].append(ok)

    return {
        "accuracy": correct / total if total else 0.0,
        "num_samples": total,
        "unparseable": unparse,
        "per_subject": {
            k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in by_subject.items()
        },
        "per_difficulty": {
            k: {"n": len(v), "accuracy": sum(v) / len(v)}
            for k, v in by_difficulty.items()
        },
    }
