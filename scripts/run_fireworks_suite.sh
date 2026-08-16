#!/usr/bin/env bash
# Run the whole benchmark suite against one Fireworks-hosted model.
#
# Serialized on purpose: every runner shares one Fireworks account rate limit,
# so running them concurrently just burns retries on 429s.
#
# Usage:
#   scripts/run_fireworks_suite.sh                                   # inkling (serverless)
#   MODEL=accounts/fireworks/models/inkling-small scripts/run_fireworks_suite.sh
#   ONLY=ocrbench,mmmlu scripts/run_fireworks_suite.sh               # subset
#   SKIP=ocrbench,olmocr scripts/run_fireworks_suite.sh              # all but these
#
# The two OCR benchmarks dominate the cost of a full pass (OCRBench v2 is 10,000
# image samples, olmOCR-bench 1,403 PDF pages), which is why they're separately
# skippable.
#
# Every runner checkpoints, so re-running skips completed samples. Logs land in
# logs/fireworks_<model-slug>/<benchmark>.log.

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-accounts/fireworks/models/inkling}"
# PROVIDER lets the same suite target another host (e.g. OpenRouter, which is
# the only place inkling-small is reachable). REASONING is the effort passed
# through; note "none" is a floor on Fireworks but a true off on OpenRouter.
PROVIDER="${PROVIDER:-fireworks}"
REASONING="${REASONING:-off}"
export BENCH_OLMOCR_PROVIDER="$PROVIDER"
export BENCH_OPENROUTER_EFFORT="${OPENROUTER_EFFORT:-none}"
export BENCH_OLMOCR_EFFORT="$BENCH_OPENROUTER_EFFORT"
# ASR effort is pinned separately: inkling wants "none" (at "high" it returns
# content=None after ~233s/sample), but some models REJECT "none" outright —
# gemini-3.7-flash 400s with "Reasoning is mandatory ... cannot be disabled".
ASR_EFFORT="${ASR_EFFORT:-none}"
SLUG="$(printf '%s' "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
LOG_DIR="logs/${PROVIDER}_${SLUG}"
mkdir -p "$LOG_DIR"

# RefCOCO split: val is what refcoco.py (the interfaze reference run) uses.
REFCOCO_SPLIT="${REFCOCO_SPLIT:-val}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

run_step() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then
    echo "[skip] $name (not in ONLY)"
    return 0
  fi
  if [ -n "$SKIP" ] && [[ ",$SKIP," == *",$name,"* ]]; then
    echo "[skip] $name (in SKIP)"
    return 0
  fi
  local log="$LOG_DIR/$name.log"
  echo "=== [$(date +%H:%M:%S)] $name -> $log"
  if "$@" >"$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK"
  else
    # Keep going — one benchmark failing shouldn't strand the other eight.
    echo "=== [$(date +%H:%M:%S)] $name FAILED (exit $?), see $log"
  fi
}

# Cheapest first so partial results show up early.
# ASR is pinned to effort=none even when the rest of the suite runs higher.
# Measured on inkling-small via OpenRouter: effort=high returns content=None
# after ~233s per sample (every sample would burn 3 retries and fail), while
# effort=none answers in ~1.3s. Transcription gains nothing from thinking.
if [ -n "${TARGET:-}" ]; then
  run_step asr uv run python -m bench_core run --benchmark asr --target "$TARGET"
else
  echo "  [skip] asr: set TARGET=<name> (python -m bench_core list-targets)"
fi

run_step spider2 uv run -m benchmarks.spider2_lite.spider2_lite \
  --provider "$PROVIDER" --model "$MODEL"

if [ -n "${TARGET:-}" ]; then
  R="$([ "$REASONING" = "high" ] && echo high || echo off)"
  run_step mmmupro_standard uv run python -m bench_core run --benchmark mmmu_pro \
    --target "$TARGET" --variant standard --reasoning "$R"
  run_step mmmupro_vision uv run python -m bench_core run --benchmark mmmu_pro \
    --target "$TARGET" --variant vision --reasoning "$R"
else
  echo "  [skip] mmmupro: set TARGET=<name> (python -m bench_core list-targets)"
fi

if [ -n "${TARGET:-}" ]; then
  run_step refcoco uv run python -m bench_core run --benchmark refcoco \
    --target "$TARGET" --variant "$REFCOCO_SPLIT"
else
  echo "  [skip] refcoco: set TARGET=<name> (python -m bench_core list-targets)"
fi

run_step olmocr uv run -m benchmarks.olmocr.olmocr_bench_fireworks --model "$MODEL"

if [ -n "${TARGET:-}" ]; then
  run_step mmmlu uv run python -m bench_core run --benchmark mmmlu --target "$TARGET" \
    --reasoning "$([ "$REASONING" = "high" ] && echo high || echo off)"
else
  echo "  [skip] mmmlu: set TARGET=<name> (python -m bench_core list-targets)"
fi

if [ -n "${TARGET:-}" ]; then
  run_step ocrbench uv run python -m bench_core run --benchmark ocrbench_v2 --target "$TARGET"
else
  echo "  [skip] ocrbench: set TARGET=<name> (python -m bench_core list-targets)"
fi

# Gated dataset — needs `hf auth login` plus accepted terms on the Hub.
# GPQA now runs through the unified CLI; provider/model/reasoning come from the
# target (bench_core/targets.yaml), so set TARGET=<name> (see `list-targets`).
if [ -n "${TARGET:-}" ]; then
  run_step gpqa uv run python -m bench_core run --benchmark gpqa --target "$TARGET" \
    --reasoning "$([ "$REASONING" = "high" ] && echo high || echo off)"
else
  echo "  [skip] gpqa: set TARGET=<name> (python -m bench_core list-targets)"
fi

# The format-tolerant oracle (upper bound) is now computed inline by the refcoco
# run and stored under submetrics.oracle in its metrics.json — no separate step.

echo "=== [$(date +%H:%M:%S)] suite finished for $MODEL"
