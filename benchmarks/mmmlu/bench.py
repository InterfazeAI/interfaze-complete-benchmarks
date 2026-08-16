"""MMMLU-lite: 14-language MCQ; per-language accuracy, macro-averaged headline."""

from __future__ import annotations

import re
from collections import defaultdict

from bench_core.request import Message, ReasoningSpec, Request, TextPart

NAME = "mmmlu"
ID_KEY = "id"
PRIMARY_METRIC = "macro_accuracy"
DEFAULTS = {"reasoning": "off", "rate_limit": 50, "max_in_flight": 8}

_DATASET_LITE_ID = "opencompass/mmmlu_lite"
_SPLIT = "test"
LANGUAGES = [
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
]

# English meta-instruction held constant across all languages so the parser can
# rely on Latin A-D output.
_PROMPT = (
    "The following is a multiple choice question. Respond with only a single "
    "letter (A, B, C, or D) corresponding to the correct answer. Do not "
    "explain your reasoning.\n\n"
    "Question: {question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)

_LETTER_RE = re.compile(r"\b([ABCD])\b")
_FIRST_LETTER_RE = re.compile(r"[ABCD]")


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    m = _FIRST_LETTER_RE.search(s.upper())
    return m.group(0) if m else None


def load_samples(sample_size: int | None = None) -> list[dict]:
    from datasets import load_dataset

    samples = []
    for lang in LANGUAGES:
        ds = load_dataset(_DATASET_LITE_ID, lang, split=_SPLIT)
        for i, row in enumerate(ds):
            row = dict(row)
            samples.append(
                {
                    "id": f"{lang}:{i}",
                    "language": lang,
                    "subject": row["subject"],
                    "question": row["input"],
                    "a": row["A"],
                    "b": row["B"],
                    "c": row["C"],
                    "d": row["D"],
                    "answer": str(row["target"]).strip().upper(),
                }
            )
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
    return parse_answer(response.text or "")


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    # de-dupe records by id (last-wins), matching the old load_records
    latest = {r["id"]: r for r in records if r["id"] in by_id}

    by_lang: dict[str, list[bool]] = defaultdict(list)
    by_subject: dict[str, list[bool]] = defaultdict(list)
    unparse: dict[str, int] = defaultdict(int)
    for rid, r in latest.items():
        s = by_id[rid]
        pred = r.get("prediction")
        correct = pred == s["answer"]
        by_lang[s["language"]].append(correct)
        by_subject[s["subject"]].append(correct)
        if pred is None:
            unparse[s["language"]] += 1

    per_language = {
        lang: {
            "n": len(v),
            "accuracy": sum(v) / len(v) if v else 0.0,
            "unparseable": unparse[lang],
        }
        for lang, v in by_lang.items()
    }
    per_subject = {
        subj: {"n": len(v), "accuracy": sum(v) / len(v) if v else 0.0}
        for subj, v in by_subject.items()
    }

    lang_accs = [v["accuracy"] for v in per_language.values()]
    macro = sum(lang_accs) / len(lang_accs) if lang_accs else 0.0
    n_total = sum(v["n"] for v in per_language.values())
    # micro reconstructed the same lossy way the original did (float->int per lang)
    n_correct = sum(int(v["accuracy"] * v["n"]) for v in per_language.values())
    micro = n_correct / n_total if n_total else 0.0

    return {
        "macro_accuracy": macro,
        "micro_accuracy": micro,
        "num_samples": n_total,
        "per_language": per_language,
        "per_subject": per_subject,
    }
