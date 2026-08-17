"""MMMU-Pro: parse units + offline scoring-parity replay (standard + vision)."""

import json
import pathlib

import pytest

from benchmarks.mmmu_pro import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHIVED_TAGS = [
    "mmmupro_standard_fireworks_accounts-fireworks-models-inkling_reasoningoff",
    "mmmupro_standard_gemini_gemini-3-7-flash_reasoningoff",
    "mmmupro_standard_openrouter_thinkingmachines-inkling-small_reasoningoff",
    "mmmupro_vision_fireworks_accounts-fireworks-models-inkling_reasoningoff",
    "mmmupro_vision_gemini_gemini-3-7-flash_reasoningoff",
    "mmmupro_vision_openrouter_thinkingmachines-inkling-small_reasoningoff",
]


def test_build_request_embedded_or_lazy_idx(monkeypatch):
    """Smokes embed images; full runs carry an index and read images lazily
    from _DATASET (so the samples list doesn't hold ~1730 rows of images)."""
    from PIL import Image

    img = Image.new("RGB", (16, 16))

    # standard, embedded (smoke): text + 2 image parts
    s = {
        "id": "x",
        "setting": "standard",
        "question": "q?",
        "options": ["a", "b"],
        "answer": "A",
        "images": [img, img],
    }
    assert len(bench.build_request(s, "off").messages[0].parts) == 3

    # standard, lazy idx: images read from the _DATASET row (image_2 is None)
    monkeypatch.setattr(bench, "_DATASET", {5: {"image_1": img, "image_2": None}})
    s2 = {
        "id": "y",
        "setting": "standard",
        "question": "q?",
        "options": ["a", "b"],
        "answer": "B",
        "idx": 5,
    }
    assert len(bench.build_request(s2, "off").messages[0].parts) == 2

    # vision, lazy idx
    monkeypatch.setattr(bench, "_DATASET", {7: {"image": img}})
    s3 = {
        "id": "z",
        "setting": "vision",
        "options": ["a", "b"],
        "answer": "C",
        "idx": 7,
    }
    assert len(bench.build_request(s3, "off").messages[0].parts) == 2


def test_parse_answer_a_to_j():
    assert bench.parse_answer("H") == "H"
    assert bench.parse_answer("(J)") == "J"
    assert bench.parse_answer("The answer is E.") == "E"
    assert bench.parse_answer("") is None


def test_options_block():
    assert bench._options_block(["x", "y", "z"]) == "A. x\nB. y\nC. z"


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
        if rec.get("response") is None:
            continue
        samples.append(
            {
                "id": rec["id"],
                "answer": rec["answer"],
                "subject": rec.get("subject"),
                "topic_difficulty": rec.get("topic_difficulty"),
            }
        )
        records.append(
            {"id": rec["id"], "prediction": bench.parse_answer(rec["response"])}
        )

    m = bench.score(records, samples)
    assert m["num_samples"] == archived["num_samples"]
    assert m["accuracy"] == pytest.approx(archived["accuracy"])
    assert m["unparseable"] == archived["unparseable"]
