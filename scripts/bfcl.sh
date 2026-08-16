#!/usr/bin/env bash
# Thin wrapper around BFCL's CLI that fixes up the environment first.
#
#   scripts/bfcl.sh generate --model fireworks-qwen3p7-plus-FC --test-category simple_python
#   scripts/bfcl.sh evaluate --model fireworks-qwen3p7-plus-FC --test-category simple_python
#
# Two reasons this exists rather than calling `bfcl` directly:
#
#   1. BFCL derives DOTENV_PATH from BFCL_PROJECT_ROOT, which it reads from the
#      *process* environment — so BFCL_PROJECT_ROOT cannot be supplied by the
#      .env it is used to locate. It has to be exported before the CLI starts.
#   2. This project's .env spells the Fireworks key lowercase, while BFCL's
#      handler does a bare os.getenv("FIREWORKS_API_KEY"). We normalise here.
#
# Run scripts/setup_bfcl.sh once first.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

VENV="$ROOT/external/bfcl-venv"
RUNS="$ROOT/external/bfcl-runs"

if [ ! -x "$VENV/bin/bfcl" ]; then
  echo "bfcl not installed — run scripts/setup_bfcl.sh first" >&2
  exit 1
fi

export BFCL_PROJECT_ROOT="$RUNS"
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-$(
  grep -iE '^[[:space:]]*fireworks_api_key[[:space:]]*=' "$ROOT/.env" 2>/dev/null \
    | head -1 | cut -d= -f2- | tr -d ' "'"'" || true
)}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(
  grep -iE '^[[:space:]]*openrouter_api_key[[:space:]]*=' "$ROOT/.env" 2>/dev/null \
    | head -1 | cut -d= -f2- | tr -d ' "'"'" || true
)}"
# See setup_bfcl.sh — needed only so the parent handler's client can be built.
export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder-unused-by-fireworks-handler}"

if [ -z "$FIREWORKS_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
  echo "Neither FIREWORKS_API_KEY nor OPENROUTER_API_KEY found in $ROOT/.env" >&2
  exit 1
fi

cd "$RUNS"
exec "$VENV/bin/bfcl" "$@"
