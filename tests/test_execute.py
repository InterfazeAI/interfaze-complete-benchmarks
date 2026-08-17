"""The fallback ladder. On a PARAM_REJECTED 4xx it mutates the request/caps and
retries in-place (self-heal), recording a capability_hint to promote to YAML.
Transient/rate-limit/empty/fatal are surfaced as BenchError for the runner to
retry-or-fail. A scripted fake adapter drives the error sequence — no network.
"""

import pytest

from src.capabilities import Capabilities
from src.errors import ErrorKind
from src.execute import BenchError, execute
from src.request import Message, ReasoningSpec, Request, TextPart
from src.response import Response


class FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class ScriptedAdapter:
    """Records the encoded state on each call so tests can assert what the
    ladder adjusted; raises a scripted error sequence, then succeeds."""

    def __init__(self, errors, final_text="ok"):
        self.errors = list(errors)
        self.history = []
        self.final_text = final_text

    def encode(self, req, caps, model_id):
        return {
            "mode": req.reasoning.mode,
            "off": caps.reasoning.off_value,
            "toggle": caps.reasoning.extra.get("toggle"),
            "true_off": caps.reasoning.true_off,
            "temperature": req.temperature,
        }

    def call(self, client, encoded):
        self.history.append(encoded)
        if self.errors:
            raise self.errors.pop(0)
        return {"ok": True}

    def decode(self, raw, caps):
        return Response(text=self.final_text)

    def classify_error(self, exc):
        from src.errors import classify

        return classify(exc)


def _req(mode="off", temperature=0.0):
    return Request(
        [Message("user", [TextPart("q")])], ReasoningSpec(mode), temperature=temperature
    )


def _caps(**reasoning):
    base = {
        "style": "thinking_level",
        "off_value": "minimal",
        "on_value": "high",
        "true_off": False,
    }
    base.update(reasoning)
    return Capabilities.from_dict(
        {"reasoning": base, "media": {}, "response": "gemini_text"}
    )


def test_success_first_try_returns_response_no_hints():
    adapter = ScriptedAdapter(errors=[])
    resp, hints = execute(adapter, None, _req(), _caps(), "m")
    assert resp.text == "ok"
    assert hints == []
    assert len(adapter.history) == 1


def test_minimal_not_supported_raises_thinking_floor_then_succeeds():
    adapter = ScriptedAdapter(
        errors=[
            Exception(
                "400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported for this model."
            )
        ]
    )
    resp, hints = execute(adapter, None, _req(), _caps(off_value="minimal"), "m")
    assert resp.text == "ok"
    assert (
        adapter.history[0]["off"] == "minimal"
    )  # first attempt used the declared floor
    assert adapter.history[1]["off"] == "low"  # ladder raised it
    assert any("floor" in h.lower() for h in hints)


def test_temperature_rejected_drops_temperature_then_succeeds():
    adapter = ScriptedAdapter(
        errors=[FakeAPIError(400, "temperature is not supported with reasoning")]
    )
    _resp, hints = execute(adapter, None, _req(temperature=0.0), _caps(), "m")
    assert adapter.history[0]["temperature"] == 0.0
    assert adapter.history[1]["temperature"] is None
    assert any("temperature" in h.lower() for h in hints)


def test_reasoning_mandatory_forces_floor_on_reasoning_body():
    caps = Capabilities.from_dict(
        {
            "reasoning": {
                "style": "reasoning_body",
                "off_value": False,
                "on_value": True,
                "true_off": True,
                "extra": {"toggle": "enabled"},
            },
            "media": {},
            "response": "openai_chat",
        }
    )
    adapter = ScriptedAdapter(
        errors=[FakeAPIError(400, "Reasoning is mandatory and cannot be disabled")]
    )
    _resp, hints = execute(adapter, None, _req(), caps, "m")
    assert adapter.history[0]["toggle"] == "enabled"
    assert adapter.history[1]["toggle"] == "effort"  # switched away from enabled=false
    assert any("reasoning" in h.lower() for h in hints)


def test_fatal_error_propagates_as_bencherror():
    adapter = ScriptedAdapter(errors=[FakeAPIError(401, "bad key")])
    with pytest.raises(BenchError) as ei:
        execute(adapter, None, _req(), _caps(), "m")
    assert ei.value.classification.kind is ErrorKind.FATAL


def test_empty_content_surfaces_as_retryable_bencherror():
    adapter = ScriptedAdapter(errors=[], final_text="   ")
    with pytest.raises(BenchError) as ei:
        execute(adapter, None, _req(), _caps(), "m")
    assert ei.value.classification.kind is ErrorKind.EMPTY_CONTENT


def test_rate_limit_surfaces_as_retryable_bencherror():
    adapter = ScriptedAdapter(errors=[FakeAPIError(429, "slow down")])
    with pytest.raises(BenchError) as ei:
        execute(adapter, None, _req(), _caps(), "m")
    assert ei.value.classification.kind is ErrorKind.RATE_LIMITED


def test_param_retries_are_bounded():
    # A model that rejects MINIMAL forever must not loop indefinitely.
    adapter = ScriptedAdapter(
        errors=[Exception("Thinking level X is not supported")] * 10
    )
    with pytest.raises(BenchError):
        execute(
            adapter, None, _req(), _caps(off_value="minimal"), "m", max_param_retries=3
        )
    assert len(adapter.history) <= 4
