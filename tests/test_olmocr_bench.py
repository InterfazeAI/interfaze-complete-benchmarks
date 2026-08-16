"""olmOCR: unit tests for the null-normalization and log parsing.

olmOCR has no offline score-replay (its scorer needs the full dataset + prints to
a log, no metrics.json). Parity is by-construction: identical PROMPT, PNG render,
and null handling. Here we pin those and the log parser against the archived log.
"""

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
    scores = bench._parse_scores(log.read_text(errors="ignore"))
    assert scores["overall"] == 77.3
    assert scores["splits"]["arxiv_math"] == 86.7
    assert scores["splits"]["old_scans"] == 44.9
    # matches report_scores.load_olmocr: the first `baseline:` line (summary block)
    assert scores["splits"]["baseline"] == 93.4
