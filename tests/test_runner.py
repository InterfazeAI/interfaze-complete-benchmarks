"""The execution harness: resume, retry-with-backoff on retryable BenchErrors,
fail-fast on fatal, incremental checkpointing, bounded concurrency. Driven by a
scripted fake adapter through the real `execute`, so no network and no sleeps
(backoff_base=0).
"""

from bench_core.capabilities import Capabilities
from bench_core.request import Message, ReasoningSpec, Request, TextPart
from bench_core.response import Response
from bench_core.results import RunStore
from bench_core.runner import run_benchmark


class FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class MapAdapter:
    """Answers per-sample from a scripted map of id -> list of outcomes, where
    an outcome is either an Exception (raised) or a str (returned as text)."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = {}

    def encode(self, req, caps, model_id):
        return {"id": req.messages[0].parts[0].text}

    def call(self, client, encoded):
        sid = encoded["id"]
        self.calls[sid] = self.calls.get(sid, 0) + 1
        outcome = self.script[sid].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"text": outcome}

    def decode(self, raw, caps):
        return Response(text=raw["text"])

    def classify_error(self, exc):
        from bench_core.errors import classify

        return classify(exc)


CAPS = Capabilities.from_dict(
    {"reasoning": {"style": "none"}, "media": {}, "response": "openai_chat"}
)


def _build_request(sample):
    return Request([Message("user", [TextPart(sample["id"])])], ReasoningSpec("off"))


def _parse(resp, sample):
    return resp.text.upper()


async def _run(adapter, samples, store, **kw):
    return await run_benchmark(
        adapter=adapter,
        client=None,
        caps=CAPS,
        model_id="m",
        samples=samples,
        build_request=_build_request,
        parse=_parse,
        store=store,
        backoff_base=0.0,
        **kw,
    )


async def test_runs_all_samples_and_records_predictions(tmp_path):
    adapter = MapAdapter({"a": ["alpha"], "b": ["beta"]})
    store = RunStore("t", "m", root=tmp_path)
    result = await _run(adapter, [{"id": "a"}, {"id": "b"}], store)
    assert result.n_completed == 2
    recs = {r["id"]: r for r in store.load_responses()}
    assert recs["a"]["prediction"] == "ALPHA"
    assert recs["b"]["response"] == "beta"


async def test_resume_skips_completed(tmp_path):
    store = RunStore("t", "m", root=tmp_path)
    store.append_response({"id": "a", "response": "old", "prediction": "OLD"})
    adapter = MapAdapter(
        {"b": ["beta"]}
    )  # note: no script for "a" — must not be called
    result = await _run(adapter, [{"id": "a"}, {"id": "b"}], store)
    assert "a" not in adapter.calls
    assert result.n_completed == 1  # only the new one


async def test_retryable_error_is_retried_then_succeeds(tmp_path):
    adapter = MapAdapter({"a": [FakeAPIError(429, "slow"), "ok"]})
    store = RunStore("t", "m", root=tmp_path)
    result = await _run(adapter, [{"id": "a"}], store, max_retries=3)
    assert adapter.calls["a"] == 2
    assert result.n_completed == 1


async def test_fatal_error_records_failure_without_retry(tmp_path):
    adapter = MapAdapter({"a": [FakeAPIError(401, "bad key")]})
    store = RunStore("t", "m", root=tmp_path)
    result = await _run(adapter, [{"id": "a"}], store, max_retries=3)
    assert adapter.calls["a"] == 1  # not retried
    assert result.n_failed == 1
    assert store.load_responses()[0]["response"] is None


async def test_build_request_is_bounded_by_max_in_flight(tmp_path):
    """Requests must be built lazily as workers pick up samples — not all up
    front. Otherwise a full image benchmark decodes/encodes every image at once
    and OOMs CI. inflight is incremented in build_request and decremented in
    parse; all samples succeed here, so the pairing holds (parse runs only on
    success)."""
    ids = [str(i) for i in range(10)]
    adapter = MapAdapter({i: ["ok"] for i in ids})
    store = RunStore("t", "m", root=tmp_path)
    inflight = peak = 0

    def build_request(sample):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        return Request(
            [Message("user", [TextPart(sample["id"])])], ReasoningSpec("off")
        )

    def parse(resp, sample):
        nonlocal inflight
        inflight -= 1
        return resp.text

    await run_benchmark(
        adapter=adapter,
        client=None,
        caps=CAPS,
        model_id="m",
        samples=[{"id": i} for i in ids],
        build_request=build_request,
        parse=parse,
        store=store,
        backoff_base=0.0,
        rate_limit=0,
        max_in_flight=2,
    )
    assert peak <= 2  # not 10: at most max_in_flight requests built at a time


async def test_capability_hints_are_aggregated(tmp_path):
    adapter = MapAdapter(
        {"a": [Exception("Thinking level MINIMAL is not supported"), "ok"]}
    )
    store = RunStore("t", "m", root=tmp_path)
    # give the reasoning a floor so the ladder has something to raise
    caps = Capabilities.from_dict(
        {
            "reasoning": {
                "style": "thinking_level",
                "off_value": "minimal",
                "on_value": "high",
                "true_off": False,
            },
            "media": {},
            "response": "openai_chat",
        }
    )
    result = await run_benchmark(
        adapter=adapter,
        client=None,
        caps=caps,
        model_id="m",
        samples=[{"id": "a"}],
        build_request=_build_request,
        parse=_parse,
        store=store,
        backoff_base=0.0,
    )
    assert any("floor" in h.lower() for h in result.hints)
