"""Exercise retries through the CLI with a fake provider and real GPQA scorer."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.gpqa import bench
from src import cli
from src.capabilities import Capabilities
from src.errors import classify
from src.response import Response
from src.runner import Route
from tests.fakes import FakeAPIError


class ScriptedAdapter:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def encode(self, req, caps, model_id):
        return req

    def call(self, client, encoded):
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def decode(self, raw, caps):
        return Response(text=raw)

    def classify_error(self, exc):
        return classify(exc)


@pytest.mark.parametrize("failures", [1, 2])
def test_resumed_gpqa_scores_each_question_once(tmp_path, monkeypatch, failures):
    sample = {
        "id": "question-1",
        "question": "Question?",
        "a": "yes",
        "b": "no",
        "c": "other",
        "d": "unknown",
        "correct_letter": "A",
        "domain": "test",
    }
    adapter = ScriptedAdapter(
        [FakeAPIError(401, "bad key") for _ in range(failures)] + ["A"]
    )
    caps = Capabilities.from_dict(
        {"reasoning": {"style": "none"}, "media": {}, "response": "openai_chat"}
    )
    route = Route(provider="fake", model_id="fake", adapter=adapter, caps=caps)
    target = SimpleNamespace(name="fake", provider="fake", model_id="fake")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_target", lambda _: target)
    monkeypatch.setattr(cli, "_live_routes", lambda *_: [route])
    monkeypatch.setattr(bench, "load_samples", lambda _: [sample])
    args = SimpleNamespace(
        target="fake", benchmark="gpqa", reasoning=None, variant=None, sample=None
    )

    for _ in range(failures):
        cli.cmd_run(args)  # Failed response gets checkpointed.
    cli.cmd_run(args)  # Same question succeeds after resuming.

    result_dir = tmp_path / "results" / "gpqa" / "fake"
    records = [
        json.loads(line)
        for line in (result_dir / "responses.jsonl").read_text().splitlines()
    ]
    metrics = json.loads((result_dir / "metrics.json").read_text())
    assert len(records) == failures + 1  # Preserve every attempt on disk.
    assert metrics["total"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["unparseable"] == 0
