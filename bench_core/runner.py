"""The one execution harness. Owns resume, bounded concurrency, rate limiting,
retry-with-backoff, and incremental checkpointing so no benchmark reimplements
them (the audit found ~15 divergent copies). Benchmarks supply only
`build_request` and `parse`; inference goes through the fallback ladder.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from bench_core.errors import ErrorKind
from bench_core.execute import BenchError, execute
from bench_core.results import RunStore

_RETRYABLE = {ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT, ErrorKind.EMPTY_CONTENT}


@dataclass
class RunResult:
    n_completed: int = 0
    n_failed: int = 0
    hints: list[str] = field(default_factory=list)


class _RateLimiter:
    """Token bucket bounding how many requests *start* per second."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = float(rate)
        self.last = None
        self._lock = asyncio.Lock()

    async def acquire(self):
        if self.rate <= 0:
            return
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                if self.last is None:
                    self.last = now
                self.tokens = min(
                    self.rate, self.tokens + (now - self.last) * self.rate
                )
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            await asyncio.sleep(1.0 / self.rate)


async def run_benchmark(
    *,
    adapter,
    client,
    caps,
    model_id: str,
    samples,
    build_request,
    parse,
    store: RunStore,
    id_key: str = "id",
    done=None,
    rate_limit: float = 25.0,
    max_in_flight: int = 8,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> RunResult:
    if done is None:

        def done(r):
            return r.get("response") is not None

    completed = store.completed_ids(id_key, done)
    pending = [s for s in samples if s[id_key] not in completed]

    limiter = _RateLimiter(rate_limit)
    sem = asyncio.Semaphore(max_in_flight)
    result = RunResult()
    lock = asyncio.Lock()

    async def process(sample):
        req = build_request(sample)
        for attempt in range(max_retries + 1):
            await limiter.acquire()
            async with sem:
                try:
                    resp, hints = await asyncio.to_thread(
                        execute, adapter, client, req, caps, model_id
                    )
                except BenchError as be:
                    kind = be.classification.kind
                    if kind in _RETRYABLE and attempt < max_retries:
                        if backoff_base > 0:
                            await asyncio.sleep(backoff_base * (2**attempt))
                        continue
                    await _record_failure(sample, be, store, result, lock, id_key)
                    return
            await _record_success(
                sample, resp, hints, parse, store, result, lock, id_key
            )
            return

    await asyncio.gather(*(process(s) for s in pending))
    return result


async def _record_success(sample, resp, hints, parse, store, result, lock, id_key):
    prediction = parse(resp, sample)
    record = {
        id_key: sample[id_key],
        "response": resp.text,
        "prediction": prediction,
        "reasoning_tokens": resp.reasoning_tokens,
    }
    if hints:
        record["capability_hints"] = hints
    async with lock:
        store.append_response(record)
        result.n_completed += 1
        for h in hints:
            if h not in result.hints:
                result.hints.append(h)


async def _record_failure(sample, be, store, result, lock, id_key):
    record = {
        id_key: sample[id_key],
        "response": None,
        "prediction": None,
        "error": be.classification.kind.value,
        "error_message": be.classification.message,
    }
    async with lock:
        store.append_response(record)
        result.n_failed += 1
