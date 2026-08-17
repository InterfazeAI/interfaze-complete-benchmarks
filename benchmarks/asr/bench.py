# VoxPopuli-Cleaned-AA ASR: transcribe audio, score corpus + time-weighted WER.

from __future__ import annotations

import re
import unicodedata
from typing import Any

from jiwer import cer, wer

from bench_core.request import AudioPart, Message, ReasoningSpec, Request, TextPart

NAME = "voxpopuli_aa"
ID_KEY = "id"
PRIMARY_METRIC = "corpus_wer"
DEFAULTS = {"reasoning": "off", "rate_limit": 25, "max_in_flight": 8}

_DATASET_ID = "ArtificialAnalysis/VoxPopuli-Cleaned-AA"
_SPLIT = "test"

PROMPT = (
    "Transcribe the following audio. Fix anything that needs fixing — "
    "disfluencies, stutters, obvious misspeaks, garbled words, or misheard "
    "named entities — so the transcription reads as the speaker clearly "
    "intended. Output ONLY the cleaned transcription, no commentary, labels, "
    "speaker tags, or timestamps."
)

_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9' ]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    # Whisper-style: NFKC, lowercase, strip punctuation (keep apostrophes), collapse ws.
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_ALNUM_SPACE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def load_samples(sample_size: int | None = None) -> list[dict]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    ds = load_dataset(_DATASET_ID, split=_SPLIT)
    rows = list(ds)[:sample_size] if sample_size else list(ds)
    samples = []
    for row in tqdm(rows, desc="fetch audio"):
        row = dict(row)
        # Warm the HF cache; build_request reads the local file (fast, no memory held).
        path = hf_hub_download(
            repo_id=_DATASET_ID,
            repo_type="dataset",
            filename=f"audio/{row['file_name']}",
        )
        samples.append(
            {
                "id": str(row["id"]),
                "file_name": row["file_name"],
                "audio_path": path,
                "transcript": row["transcript"],
                "duration": row.get("duration"),
                "language": row.get("language"),
            }
        )
    return samples


def build_request(sample: dict, mode: str) -> Request:
    with open(sample["audio_path"], "rb") as f:
        audio = f.read()
    return Request(
        messages=[Message("user", [TextPart(PROMPT), AudioPart(audio, "audio/wav")])],
        reasoning=ReasoningSpec(mode),
        temperature=0.0,
    )


def parse(response, sample) -> str:
    return (response.text or "").strip()


def _sample_metric(fn, gt_norm: str, hyp_norm: str) -> float:
    if not gt_norm:
        return float("inf")
    try:
        return float(fn(gt_norm, hyp_norm))
    except (ValueError, ZeroDivisionError):
        return float("inf")


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    rows: list[dict[str, Any]] = []
    for r in records:
        s = by_id.get(r["id"])
        if s is None:
            continue
        gt = normalize_text(s["transcript"])
        hyp = normalize_text(r.get("prediction") or "")
        rows.append(
            {
                "gt": gt,
                "hyp": hyp,
                "wer": _sample_metric(wer, gt, hyp),
                "cer": _sample_metric(cer, gt, hyp),
                "duration": s.get("duration"),
            }
        )
    if not rows:
        return {}

    refs = [x["gt"] for x in rows if x["gt"]]
    hyps = [x["hyp"] for x in rows if x["gt"]]
    total_dur = sum(x["duration"] or 0 for x in rows)
    finite_wer = [x["wer"] for x in rows if x["wer"] != float("inf")]
    finite_cer = [x["cer"] for x in rows if x["cer"] != float("inf")]

    return {
        "corpus_wer": float(wer(refs, hyps)) if refs else float("inf"),
        "corpus_cer": float(cer(refs, hyps)) if refs else float("inf"),
        "mean_sample_wer": sum(finite_wer) / max(1, len(rows)),
        "mean_sample_cer": sum(finite_cer) / max(1, len(rows)),
        "time_weighted_wer": (
            sum(
                (x["wer"] if x["wer"] != float("inf") else 0) * (x["duration"] or 0)
                for x in rows
            )
            / total_dur
            if total_dur > 0
            else float("inf")
        ),
        "num_samples": len(rows),
        "total_duration_s": total_dur,
    }
