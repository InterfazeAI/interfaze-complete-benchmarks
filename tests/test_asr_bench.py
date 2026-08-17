"""ASR benchmark: normalize/parse units + offline scoring-parity replay.

The parity test replays each archived ASR run through the new score and asserts
corpus WER, time-weighted WER, CER, and counts match the archived metrics —
proving the port reproduces the numbers with no audio calls.
"""

import json
import pathlib
from types import SimpleNamespace as NS

import pytest

from benchmarks.asr import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHIVED_TAGS = [
    "voxpopuli_aa_fireworks_accounts_fireworks_models_inkling",
    "voxpopuli_aa_gemini_gemini-3.7-flash",
    "voxpopuli_aa_openrouter_thinkingmachines_inkling-small",
    "voxpopuli_aa_openrouter_google_gemini-3.7-flash",
]


def test_normalize_text_whisper_style():
    assert bench.normalize_text("Hello, WORLD!") == "hello world"
    assert bench.normalize_text("it's   fine\n") == "it's fine"
    assert bench.normalize_text("") == ""


def test_parse_strips_transcript():
    assert bench.parse(NS(text="  hi there \n"), None) == "hi there"


def test_score_corpus_wer_basic():
    samples = [
        {"id": "1", "transcript": "the cat sat", "duration": 1.0},
        {"id": "2", "transcript": "hello world", "duration": 1.0},
    ]
    records = [
        {"id": "1", "prediction": "the cat sat"},  # 0 errors
        {"id": "2", "prediction": "hello there"},  # 1/2 words wrong
    ]
    m = bench.score(records, samples)
    assert m["num_samples"] == 2
    assert m["corpus_wer"] == pytest.approx(1 / 5)  # 1 error over 5 ref words


@pytest.mark.parametrize("tag", ARCHIVED_TAGS)
def test_scoring_parity_with_archived_run(tag):
    resp_path = RESULTS / f"{tag}_responses.jsonl"
    metrics_path = RESULTS / f"{tag}_metrics.json"
    if not (resp_path.exists() and metrics_path.exists()):
        pytest.skip(f"archived run {tag} not present")

    archived = json.loads(metrics_path.read_text())
    samples, records = [], []
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        samples.append(
            {
                "id": rec["id"],
                "transcript": rec["transcript"],
                "duration": rec.get("duration"),
            }
        )
        records.append({"id": rec["id"], "prediction": rec["response"]})

    m = bench.score(records, samples)
    assert m["num_samples"] == archived["num_samples"]
    assert m["corpus_wer"] == pytest.approx(archived["corpus_wer"])
    assert m["corpus_cer"] == pytest.approx(archived["corpus_cer"])
    assert m["time_weighted_wer"] == pytest.approx(archived["time_weighted_wer"])
    assert m["mean_sample_wer"] == pytest.approx(archived["mean_sample_wer"])
