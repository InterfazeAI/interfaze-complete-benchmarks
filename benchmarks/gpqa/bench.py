"""GPQA Diamond: 198 expert MCQs; deterministic choice shuffle; letter-match accuracy."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from bench_core.request import Message, ReasoningSpec, Request, TextPart

NAME = "gpqa"
ID_KEY = "id"
PRIMARY_METRIC = "accuracy"
DEFAULTS = {"reasoning": "off", "rate_limit": 10, "max_in_flight": 8}

_DATASET_ID = "Idavidrein/gpqa"
_CONFIG = "gpqa_diamond"
_SPLIT = "train"  # GPQA Diamond ships as a single 'train' split (198 rows)

_PROMPT = (
    "The following is a multiple choice question (with answers). Respond with "
    "only the single letter (A, B, C, or D) corresponding to the correct "
    "answer. Do not show your work.\n\n"
    "Question: {question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)

_LETTER_RE = re.compile(r"\b([ABCD])\b")
_FIRST_LETTER_RE = re.compile(r"[ABCD]")


def _clean(s) -> str:
    return str(s).strip() if s is not None else ""


def _build_sample(row: dict) -> dict:
    # Shuffle choices deterministically by Record ID (avoids position bias and
    # eval-to-eval drift); track which letter holds the correct answer.
    record_id = str(row.get("Record ID") or row.get("Question", ""))
    correct = _clean(row["Correct Answer"])
    incorrect = [
        _clean(row["Incorrect Answer 1"]),
        _clean(row["Incorrect Answer 2"]),
        _clean(row["Incorrect Answer 3"]),
    ]
    choices = [(correct, True)] + [(x, False) for x in incorrect]
    random.Random(record_id).shuffle(choices)
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[next(i for i, (_, c) in enumerate(choices) if c)]
    return {
        "id": record_id,
        "question": _clean(row["Question"]),
        "a": choices[0][0],
        "b": choices[1][0],
        "c": choices[2][0],
        "d": choices[3][0],
        "correct_letter": correct_letter,
        "domain": _clean(row.get("High-level domain")),
    }


def load_samples(sample_size: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(_DATASET_ID, _CONFIG, split=_SPLIT)
    samples = [_build_sample(dict(row)) for row in ds]
    return samples[:sample_size] if sample_size else samples


def build_request(sample: dict, mode: str) -> Request:
    prompt = _PROMPT.format(
        question=sample["question"],
        a=sample["a"],
        b=sample["b"],
        c=sample["c"],
        d=sample["d"],
    )
    return Request(
        messages=[Message("user", [TextPart(prompt)])],
        reasoning=ReasoningSpec(mode),
        temperature=0.0,
    )


def parse(response, sample) -> str | None:
    text = (response.text or "").strip()
    if not text:
        return None
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    up = text.upper()
    m = _LETTER_RE.search(up)
    if m:
        return m.group(1)
    m = _FIRST_LETTER_RE.search(up)
    return m.group(0) if m else None


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    rows = [(r.get("prediction"), by_id[r["id"]]) for r in records if r["id"] in by_id]
    total = len(rows)
    correct = sum(1 for pred, s in rows if pred == s["correct_letter"])
    unparseable = sum(1 for pred, _ in rows if pred is None)

    by_domain: dict[str, list[bool]] = defaultdict(list)
    for pred, s in rows:
        by_domain[s["domain"] or "Unknown"].append(pred == s["correct_letter"])
    per_domain = {
        d: {"n": len(v), "accuracy": sum(v) / len(v)} for d, v in by_domain.items()
    }

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "unparseable": unparseable,
        "per_domain": per_domain,
    }
