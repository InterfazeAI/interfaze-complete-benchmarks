"""olmOCR: null-normalization, prompt and log parsing.

No offline score replay (the scorer needs the full dataset), so parity rests on
pinning the prompt, null handling and the log parser.
"""

import json
import pathlib
from types import SimpleNamespace as NS

from benchmarks.olmocr import harness as bench

LOGS = pathlib.Path(__file__).resolve().parent.parent / "logs"


def test_parse_null_becomes_empty():
    assert bench.parse(NS(text="null"), None) == ""
    assert bench.parse(NS(text="  N/A "), None) == ""
    assert bench.parse(NS(text="Real content."), None) == "Real content."


def test_prompt_is_stable():
    # the scorer's numbers depend on this prompt; guard against silent drift
    assert "Turn tables into markdown format." in bench.PROMPT
    assert "Do not hallucinate." in bench.PROMPT


def test_parse_scores_from_archived_log():
    log = LOGS / "olmocr_gemini-3-7-flash.log"
    if not log.exists():
        import pytest

        pytest.skip("archived olmOCR log not present")
    scores = bench.parse_scores(log.read_text(errors="ignore"))
    assert scores["overall"] == 77.3
    assert scores["splits"]["arxiv_math"] == 86.7
    assert scores["splits"]["old_scans"] == 44.9
    # the first `baseline:` line (summary block) wins
    assert scores["splits"]["baseline"] == 93.4


def test_parse_scores_reads_overall_ci_and_splits():
    text = (
        "gemini-3-7-flash: Average Score: 77.3% ± 1.1%\r"
        "    arxiv_math.jsonl: 86.7%\n"
        "    baseline: 93.4%\n"
        "    baseline: 12.0%\n"
    )
    assert bench.parse_scores(text) == {
        "candidate": "gemini-3-7-flash",
        "overall": 77.3,
        "ci": 1.1,
        "splits": {"arxiv_math": 86.7, "baseline": 93.4},
    }


def test_parse_scores_without_summary_line_is_empty():
    assert bench.parse_scores("no table here") == {
        "candidate": None,
        "overall": None,
        "ci": None,
        "splits": {},
    }


def test_smoke_data_dir_holds_only_sampled_pages(tmp_path, monkeypatch):
    full = tmp_path / "full"
    (full / "pdfs" / "split").mkdir(parents=True)
    for name in ("a.pdf", "b.pdf"):
        (full / "pdfs" / "split" / name).write_bytes(b"%PDF")
    tests = [{"pdf": "split/a.pdf", "page": 1}, {"pdf": "split/b.pdf", "page": 1}]
    (full / "split.jsonl").write_text("\n".join(json.dumps(t) for t in tests) + "\n")
    (full / "other.jsonl").write_text(json.dumps(tests[1]) + "\n")
    monkeypatch.setattr(bench, "_FULL_DATA_DIR", full)

    smoke = tmp_path / "smoke"
    smoke.mkdir()
    bench._smoke_data_dir(smoke, [{"id": "split/a.pdf#1", "page": 1}])

    assert sorted(p.name for p in smoke.glob("*.jsonl")) == ["split.jsonl"]
    assert json.loads((smoke / "split.jsonl").read_text()) == tests[0]
    assert [p.name for p in (smoke / "pdfs" / "split").iterdir()] == ["a.pdf"]
    assert (smoke / "pdfs" / "split" / "a.pdf").resolve() == (
        full / "pdfs" / "split" / "a.pdf"
    )
