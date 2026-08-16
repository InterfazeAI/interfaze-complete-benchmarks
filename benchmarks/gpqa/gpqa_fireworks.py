"""
GPQA Diamond benchmark for Fireworks-hosted models (default: Inkling).

Mirrors benchmarks.gpqa.gpqa_openai exactly — same dataset, same prompt, same
deterministic 4-choice shuffle, same parser, same metrics — but routes calls
through Fireworks' OpenAI-compatible endpoint.

Inkling always thinks. `--thinking off` maps to reasoning_effort="none", which
is its floor rather than a true disable, so even "off" runs emit reasoning
tokens. `--thinking on` maps to reasoning_effort="high" (or --effort).

Usage:
    uv run -m benchmarks.gpqa.gpqa_fireworks
    uv run -m benchmarks.gpqa.gpqa_fireworks --thinking on --effort high
    uv run -m benchmarks.gpqa.gpqa_fireworks --model accounts/fireworks/models/inkling-small
    uv run -m benchmarks.gpqa.gpqa_fireworks --limit 5
    uv run -m benchmarks.gpqa.gpqa_fireworks --evaluate-only

Env: FIREWORKS_API_KEY. The dataset (Idavidrein/gpqa) is gated — accept its
terms on HuggingFace and log in (`hf auth login`) before running.
"""

import re
import sys
import json
import time
import asyncio
import argparse
import traceback
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse all the dataset/parsing/metrics plumbing from the OpenAI runner.
from benchmarks.gpqa.gpqa_openai import (  # noqa: E402
    DATASET_ID,
    CONFIG,
    SPLIT,
    PROMPT_TEMPLATE,
    JsonlWriter,
    load_completed_ids,
    load_records,
    parse_letter,
    build_sample,
    compute_metrics,
)
from src.commons_fireworks import INKLING, fireworks_client  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL = INKLING
DEFAULT_THINKING = "off"  # 'on' or 'off' ('off' = reasoning_effort="none" floor)
DEFAULT_EFFORT: str | None = None  # None | low | medium | high | xhigh | max
TEMPERATURE = 0.0
CONCURRENCY = 10
MAX_RETRIES = 5
RETRY_BACKOFF_CAP_S = 30.0

MODEL = DEFAULT_MODEL
THINKING = DEFAULT_THINKING
EFFORT: str | None = DEFAULT_EFFORT


def model_slug(model: str) -> str:
    """accounts/fireworks/models/inkling-small -> inklingsmall"""
    return re.sub(r"[^a-z0-9]", "", model.rsplit("/", 1)[-1].lower())


def invoke_fireworks(messages: list[dict]):
    effort = "none" if THINKING == "off" else (EFFORT or "high")
    return fireworks_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        reasoning_effort=effort,
    )


def _reasoning_tokens(response) -> int | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    val = getattr(details, "reasoning_tokens", None) if details else None
    if val is None:
        # Fireworks reports it at the top level of `usage` for Inkling.
        val = getattr(u, "reasoning_tokens", None)
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


async def process_sample(
    sample: dict, semaphore: asyncio.Semaphore, writer: JsonlWriter, progress: dict
) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        a=sample["a"],
        b=sample["b"],
        c=sample["c"],
        d=sample["d"],
    )
    messages = [{"role": "user", "content": prompt}]
    last_error: str | None = None
    # Recorded per sample: stdout can be truncated by whatever captures it, and
    # "why did half the run retry" is only answerable from the checkpoint.
    retry_errors: list[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            async with semaphore:
                start = time.perf_counter()
                response = await asyncio.to_thread(invoke_fireworks, messages)
                latency_ms = int((time.perf_counter() - start) * 1000)
            content = (response.choices[0].message.content or "").strip()
            request_id = getattr(response, "id", None)
            if not content:
                last_error = "empty response content"
                raise RuntimeError(last_error)

            predicted = parse_letter(content)
            correct = predicted == sample["correct_letter"]
            reasoning_tokens = _reasoning_tokens(response)

            record = {
                "id": sample["id"],
                "domain": sample["domain"],
                "subdomain": sample["subdomain"],
                "correct_letter": sample["correct_letter"],
                "prediction": predicted,
                "correct": correct,
                "response": content,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "attempts": attempt,
                "reasoning_tokens": reasoning_tokens,
                "retry_errors": retry_errors,
            }
            await writer.append(record)

            progress["done"] += 1
            if correct:
                progress["correct"] += 1
            mark = "OK" if correct else "X "
            tqdm.write(
                f"[{progress['done']}/{progress['total']}] {mark} "
                f"id={sample['id']} domain={sample['domain']:10} "
                f"gold={sample['correct_letter']} pred={predicted or '?'} "
                f"latency={latency_ms}ms reasoning_tok={reasoning_tokens if reasoning_tokens is not None else '?'} "
                f"attempt={attempt}"
            )
            return record

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            retry_errors.append(last_error[:200])
            tqdm.write(
                f"[error] id={sample['id']} attempt={attempt}/{MAX_RETRIES} "
                f"latency={latency_ms}ms error={last_error}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(2 ** (attempt - 1), RETRY_BACKOFF_CAP_S))

    progress["failed"] += 1
    tqdm.write(f"[FAILED] id={sample['id']} after {MAX_RETRIES} attempts: {last_error}")
    return None


def print_summary(metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"GPQA Diamond — {DATASET_ID}/{CONFIG} ({MODEL}, thinking={THINKING})")
    print(f"{'=' * 60}")
    print(
        f"Accuracy   : {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"
    )
    print(f"Unparseable: {metrics['unparseable']}")
    print("\nPer high-level domain:")
    for d in sorted(metrics["per_domain"]):
        v = metrics["per_domain"][d]
        print(f"  {d:12} n={v['n']:>3} acc={v['accuracy']:.4f}")
    if metrics.get("latency"):
        lat = metrics["latency"]
        print(
            f"\nLatency    : mean={lat['mean_ms']:.0f}ms p50={lat['p50_ms']}ms "
            f"p90={lat['p90_ms']}ms p99={lat['p99_ms']}ms max={lat['max_ms']}ms"
        )


async def run_predictions(pred_path: Path, limit: int | None):
    print(f"Loading {DATASET_ID}/{CONFIG} (split={SPLIT})...")
    ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT)
    print(f"Loaded {len(ds)} rows")
    samples = [build_sample(dict(row)) for row in ds]

    done_ids = load_completed_ids(pred_path)
    pending = [s for s in samples if s["id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]
        print(f"--limit applied: will run at most {limit} sample(s)")
    print(
        f"Resume: {len(done_ids)} already completed, {len(pending)} remaining "
        f"(checkpoint: {pred_path})"
    )
    if not pending:
        return

    writer = JsonlWriter(pred_path)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress = {"total": len(pending), "done": 0, "correct": 0, "failed": 0}

    tasks = [process_sample(s, semaphore, writer, progress) for s in pending]
    try:
        await tqdm_asyncio.gather(*tasks, desc=f"GPQA Diamond / {MODEL}")
    except Exception:
        traceback.print_exc()
    acc = progress["correct"] / progress["done"] if progress["done"] else 0.0
    print(
        f"\nRun finished: {progress['done']}/{progress['total']} answered, "
        f"{progress['correct']} correct (acc={acc:.4f}), {progress['failed']} failed."
    )


def run_evaluation(pred_path: Path, metrics_path: Path):
    if not pred_path.exists():
        print(f"No predictions found at {pred_path}")
        sys.exit(1)
    results = load_records(pred_path)
    if not results:
        print(f"No records in {pred_path}")
        sys.exit(1)
    for r in results:
        if r.get("prediction") is None and r.get("response"):
            r["prediction"] = parse_letter(r["response"])
        if r.get("correct") is None and r.get("prediction") is not None:
            r["correct"] = r["prediction"] == r.get("correct_letter")

    metrics = compute_metrics(results)
    print_summary(metrics)
    reasoning_toks = [
        r["reasoning_tokens"] for r in results if r.get("reasoning_tokens") is not None
    ]
    output = {
        **metrics,
        "dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "model": MODEL,
        "thinking": THINKING,
        "effort": "none" if THINKING == "off" else (EFFORT or "high"),
        "temperature": TEMPERATURE,
        "concurrency": CONCURRENCY,
        "provider": "fireworks",
        "mean_reasoning_tokens": (
            sum(reasoning_toks) / len(reasoning_toks) if reasoning_toks else None
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    global MODEL, THINKING, EFFORT
    parser = argparse.ArgumentParser(
        description="GPQA Diamond benchmark for Fireworks-hosted models"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Fireworks model id (accounts/fireworks/...)",
    )
    parser.add_argument(
        "--thinking",
        default=DEFAULT_THINKING,
        choices=["on", "off"],
        help="off = reasoning_effort 'none' (Inkling's floor); on = 'high' or --effort",
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Explicit reasoning_effort. Only applied when thinking=on.",
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N unanswered samples",
    )
    args = parser.parse_args()

    MODEL = args.model
    THINKING = args.thinking
    EFFORT = args.effort

    effort_suffix = f"_effort{EFFORT}" if (THINKING == "on" and EFFORT) else ""
    tag = (
        f"fireworks_{model_slug(MODEL)}_thinking{THINKING}{effort_suffix}_gpqa_diamond"
    )
    pred_path = RESULTS_DIR / f"{tag}_responses.jsonl"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"

    if args.evaluate_only:
        run_evaluation(pred_path, metrics_path)
    elif args.predict_only:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
    else:
        asyncio.run(run_predictions(pred_path, limit=args.limit))
        run_evaluation(pred_path, metrics_path)


if __name__ == "__main__":
    main()
