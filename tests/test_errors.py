"""Error classification — turns a provider exception into the signal the
fallback ladder acts on. Every PARAM_REJECTED case here is a real 4xx the audit
saw a model raise (Gemini "MINIMAL not supported", grok "Reasoning is
mandatory", GPT-5 temperature-under-reasoning).
"""

from bench_core.errors import ErrorKind, classify


class FakeAPIError(Exception):
    """Mimics an SDK error carrying a status code + message."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def test_429_is_rate_limited():
    assert (
        classify(FakeAPIError(429, "Too Many Requests")).kind is ErrorKind.RATE_LIMITED
    )


def test_503_is_transient():
    assert (
        classify(FakeAPIError(503, "Service Unavailable")).kind is ErrorKind.TRANSIENT
    )


def test_timeout_is_transient():
    assert classify(TimeoutError("request timed out")).kind is ErrorKind.TRANSIENT


def test_401_is_fatal():
    assert classify(FakeAPIError(401, "invalid api key")).kind is ErrorKind.FATAL


def test_404_model_not_found_is_fatal():
    assert classify(FakeAPIError(404, "model does not exist")).kind is ErrorKind.FATAL


def test_gemini_minimal_not_supported_raises_thinking_floor():
    # Raised with no status_code attr — parsed from the message, as google-genai does.
    c = classify(
        Exception(
            "400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported for this model."
        )
    )
    assert c.kind is ErrorKind.PARAM_REJECTED
    assert c.action == "raise_thinking_floor"


def test_reasoning_mandatory_forces_floor():
    c = classify(FakeAPIError(400, "Reasoning is mandatory and cannot be disabled"))
    assert c.kind is ErrorKind.PARAM_REJECTED
    assert c.action == "force_reasoning_floor"


def test_temperature_rejected_drops_temperature():
    c = classify(
        FakeAPIError(400, "temperature is not supported with reasoning enabled")
    )
    assert c.kind is ErrorKind.PARAM_REJECTED
    assert c.action == "drop_temperature"


def test_unknown_parameter_drops_that_param():
    c = classify(FakeAPIError(400, "Unknown parameter: 'top_k'"))
    assert c.kind is ErrorKind.PARAM_REJECTED
    assert c.action == "drop_param"
    assert c.param == "top_k"


def test_unrecognized_400_is_fatal():
    assert classify(FakeAPIError(400, "malformed request body")).kind is ErrorKind.FATAL
