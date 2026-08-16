"""Response normalization — unify the three response shapes and the
reasoning-token fallbacks the audit found (Fireworks reports reasoning tokens at
the top level of usage; Gemini as thoughts_token_count). Duck-typed fakes stand
in for the SDK objects so this stays a pure unit test.
"""

from types import SimpleNamespace as NS

from bench_core.capabilities import ResponseShape
from bench_core.decode import decode_response


def _openai_raw(
    content, reasoning_tokens=None, top_level_reasoning=None, finish="stop"
):
    details = NS(reasoning_tokens=reasoning_tokens)
    usage = NS(
        completion_tokens_details=details,
        reasoning_tokens=top_level_reasoning,
        prompt_tokens=11,
        completion_tokens=22,
    )
    msg = NS(content=content)
    choice = NS(message=msg, finish_reason=finish)
    return NS(choices=[choice], usage=usage, id="resp_1")


def test_openai_extracts_content_and_finish_reason():
    r = decode_response(_openai_raw("Answer: B"), ResponseShape.OPENAI_CHAT)
    assert r.text == "Answer: B"
    assert r.finish_reason == "stop"
    assert r.raw_id == "resp_1"


def test_openai_none_content_becomes_empty_string():
    r = decode_response(_openai_raw(None), ResponseShape.OPENAI_CHAT)
    assert r.text == ""


def test_openai_reasoning_tokens_prefers_details():
    r = decode_response(
        _openai_raw("x", reasoning_tokens=729), ResponseShape.OPENAI_CHAT
    )
    assert r.reasoning_tokens == 729


def test_openai_reasoning_tokens_falls_back_to_top_level_for_fireworks():
    r = decode_response(
        _openai_raw("x", reasoning_tokens=None, top_level_reasoning=512),
        ResponseShape.OPENAI_CHAT,
    )
    assert r.reasoning_tokens == 512


def test_anthropic_joins_text_blocks_and_skips_thinking():
    raw = NS(
        content=[
            NS(type="thinking", thinking="hmm"),
            NS(type="text", text="Hello "),
            NS(type="text", text="world"),
        ],
        usage=NS(input_tokens=5, output_tokens=7),
        id="msg_1",
    )
    r = decode_response(raw, ResponseShape.ANTHROPIC_BLOCKS)
    assert r.text == "Hello world"
    assert r.input_tokens == 5
    assert r.output_tokens == 7


def test_gemini_reads_text_and_thoughts_token_count():
    raw = NS(
        text="42",
        usage_metadata=NS(thoughts_token_count=88, prompt_token_count=3),
        response_id="gem_1",
    )
    r = decode_response(raw, ResponseShape.GEMINI_TEXT)
    assert r.text == "42"
    assert r.reasoning_tokens == 88
    assert r.raw_id == "gem_1"


def test_gemini_none_text_becomes_empty():
    raw = NS(text=None, usage_metadata=NS(thoughts_token_count=None), response_id="g")
    r = decode_response(raw, ResponseShape.GEMINI_TEXT)
    assert r.text == ""
