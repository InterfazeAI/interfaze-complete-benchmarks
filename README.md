# Interfaze complete benchmark scripts

[Full break down blog](https://interfaze.ai/blog/interfaze-a-new-model-architecture-built-for-high-accuracy-at-scale) | [Leaderboard](https://interfaze.ai/leaderboards)

Runner scripts for the public benchmarks Interfaze is evaluated on. Each
benchmark lives in its own directory under `benchmarks/`.

## Setup

```bash
uv sync
```

Create a `.env` in the repo root with whichever provider keys you plan to use:

```
INTERFAZE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_KEY=...
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
```

Benchmarks run through one CLI: `uv run python -m bench_core run --target <name> --benchmark <name>` (see `python -m bench_core list-targets`). `--sample N` is a smoke test; re-running resumes from the per-run checkpoint (completed samples are skipped). The remaining bespoke runners (`olmocr_bench_reducto`, BFCL) keep their own `--limit`/`--evaluate-only` flags.

Two extra setup steps for specific benchmarks:

```bash
# olmOCR-bench scoring renders equations in headless chromium
uv run python -m playwright install chromium

# OCRBench v2 scoring needs NLTK corpora. Run it from OUTSIDE the repo root:
# nltk blocks imports resolving inside the CWD, and .venv lives in the project,
# so `import nltk` from here trips its guard.
(cd /tmp && uv run --project "$OLDPWD" python -c "
import nltk; [nltk.download(p, quiet=True) for p in ('wordnet','omw-1.4','punkt','punkt_tab')]")

# GPQA's dataset (Idavidrein/gpqa) is gated — accept the terms on the Hub, then
hf auth login
```

## BFCL (Berkeley Function Calling Leaderboard)

Links: [repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) · [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

BFCL lives in its own third-party checkout under `external/` (gitignored) with a
separate venv — its pins conflict with this repo's. `scripts/setup_bfcl.sh` is
the durable artifact: it clones, builds the venv, writes BFCL's `.env`, and
registers the Fireworks models idempotently. Re-run it any time.

```bash
scripts/setup_bfcl.sh                       # once; safe to re-run

# OpenRouter-hosted models (registered by the setup script)
scripts/bfcl.sh generate --model openrouter-qwen3.5-35b-a3b-FC \
    --test-category simple_python --num-threads 4
scripts/bfcl.sh evaluate --model openrouter-qwen3.5-35b-a3b-FC --test-category simple_python

scripts/bfcl.sh generate --model openrouter-gemma-4-31b-it-FC \
    --test-category simple_python --num-threads 4
scripts/bfcl.sh evaluate --model openrouter-gemma-4-31b-it-FC --test-category simple_python
```

`evaluate` re-scores from saved responses, so a registration fix (see
`underscore_to_dot` below) costs nothing to apply — no regeneration.

Use `scripts/bfcl.sh` rather than the `bfcl` CLI directly — it exports
`BFCL_PROJECT_ROOT` (which BFCL needs *before* start, since it derives its
`.env` path from it) and normalises the lowercase `fireworks_api_key` in this
repo's `.env` to the uppercase name BFCL's handler reads.

Three sharp edges worth knowing:

- This is BFCL **v4**: the category is `simple_python`, not `simple`.
- A Fireworks run needs a non-empty `OPENAI_API_KEY` even though it never calls
  OpenAI — `FireworksHandler.__init__` calls `super().__init__()`, and the parent
  `OpenAICompletionsHandler` builds an OpenAI client that raises without one.
  The setup script writes a placeholder.
- `soundfile` is an undeclared transitive dependency (`model_config.py` imports
  every handler eagerly → `qwen.py` → `qwen_agent` → `soundfile`). Without it,
  loading the model registry fails, so *no* model can run.
- **`underscore_to_dot` must match the provider, or correct answers score wrong.**
  BFCL's test set uses dotted function names (`math.factorial`), but OpenAI-style
  tool schemas forbid `.`, so OpenRouter sanitises them and the model replies
  `math_factorial`. With the flag set wrong, every such call fails as
  `wrong_func_name`: measured 55.0% vs 93.8% for the same responses. OpenRouter
  needs `True`; Fireworks passes dots through and needs `False`. The setup script
  encodes this per provider and *refreshes* existing entries on re-run, so a
  corrected flag actually takes effect.

Measured on `simple_python` (400 cases, function-calling mode):

| Model | Accuracy |
|---|---|
| Qwen3.5-35B-A3B (OpenRouter) | **93.75%** (375/400) |
| Qwen3.7-Plus (Fireworks) | **92.75%** (371/400) |
| Gemma-4-31B-IT (OpenRouter) | **92.25%** (369/400) |

### Fireworks / Inkling

`--provider fireworks` (or the `*_fireworks` runners) targets Fireworks-hosted
models over their OpenAI-compatible endpoint. Inkling takes text, image, and
audio input, so it runs on every benchmark here:

```bash
uv run python -m bench_core run --target inkling --benchmark ocrbench_v2
uv run python -m bench_core run --target inkling --benchmark olmocr
uv run python -m bench_core run --target inkling --benchmark gpqa
uv run python -m bench_core run --target inkling --benchmark mmmlu
uv run python -m bench_core run --target inkling --benchmark mmmu_pro --variant standard
uv run python -m bench_core run --target inkling --benchmark mmmu_pro --variant vision
uv run python -m bench_core run --target inkling --benchmark refcoco --variant val
uv run python -m bench_core run --target inkling --benchmark asr
uv run python -m bench_core run --target inkling --benchmark spider2
```

Two caveats when reading Fireworks numbers:

- **Reasoning can't be turned off.** Inkling's `reasoning_effort` accepts
  `none|low|medium|high|xhigh|max`; `none` is a floor, not a disable, and still
  emits reasoning tokens (measured: ~240 on MMMU-Pro standard, ~1100 on
  MMMU-Pro vision). Runs tagged `reasoningoff` here are therefore *not*
  like-for-like with providers whose thinking is genuinely disabled. Per-sample
  `reasoning_tokens` are recorded in the output JSONL so the real spend is
  auditable.
- **`inkling-small` is not on serverless.** Only
  `accounts/fireworks/models/inkling` is serverless; `inkling-small` needs an
  on-demand dedicated deployment first. Once deployed, pass it via `--model`
  and every runner writes to its own separate checkpoint.

---

## OCRBench v2

Links: [paper](https://arxiv.org/abs/2501.00321) · [repo](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2)

```bash
# Any target (10k tasks, macro-of-category en_overall)
uv run python -m bench_core run --target gpt-5.5          --benchmark ocrbench_v2
uv run python -m bench_core run --target gemini-3.7-flash --benchmark ocrbench_v2
uv run python -m bench_core run --target inkling          --benchmark ocrbench_v2
```

---

## olmOCR-Bench

Links: [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)

```bash
# Any target (renders each PDF page, scores via the external olmocr.bench scorer;
# results land in logs/olmocr_<target>.log, which the report reads)
uv run python -m bench_core run --target gemini-3.7-flash --benchmark olmocr
uv run python -m bench_core run --target inkling          --benchmark olmocr

# Reducto (a bespoke document-parsing API, not a chat model) keeps its own runner
uv run -m benchmarks.olmocr.olmocr_bench_reducto
```

---

## RefCOCO (Object Detection)

Links: [RefCOCO/RefCOCO+ paper](https://arxiv.org/abs/1608.00272) · [RefCOCOg paper](https://arxiv.org/abs/1511.02283) · [dataset](https://huggingface.co/datasets/lmms-lab/RefCOCO)

Metric: Acc@IoU=0.5 on the referring-expression bounding box.

```bash
# --variant selects dataset + split: base RefCOCO (val|testA|testB|test), or
# RefCOCO+ (plus-val|plus-testA|plus-testB) / RefCOCOg (g-val|g-test). Metrics
# include a strict Acc@0.5 plus a format-tolerant `oracle` upper bound.
uv run python -m bench_core run --target gpt-5.5          --benchmark refcoco --variant val
uv run python -m bench_core run --target gemini-3.7-flash --benchmark refcoco --variant testA
uv run python -m bench_core run --target inkling          --benchmark refcoco --variant plus-testB
uv run python -m bench_core run --target inkling          --benchmark refcoco --variant g-val
```

---

## VoxPopuli-Cleaned-AA (ASR)

Links: [dataset](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)

Metric: WER with Whisper-style text normalization.

```bash
# Any audio-capable target (the right audio wire-shape is chosen per provider)
uv run python -m bench_core run --target interfaze-beta  --benchmark asr
uv run python -m bench_core run --target gemini-3.7-flash --benchmark asr
uv run python -m bench_core run --target inkling          --benchmark asr
```

---

## MMMLU (Multilingual MMLU)

Links: [dataset](https://huggingface.co/datasets/openai/MMMLU)

14 languages, exact-match accuracy macro-averaged across languages.

```bash
# 14 languages, macro-averaged. --variant lite (opencompass/mmmlu_lite, ~20k,
# DEFAULT, matches the archived leaderboard numbers) or full (openai/MMMLU, ~196k).
uv run python -m bench_core run --target gpt-5.5          --benchmark mmmlu                 # lite
uv run python -m bench_core run --target gemini-3.7-flash --benchmark mmmlu --reasoning high
uv run python -m bench_core run --target inkling          --benchmark mmmlu --variant full  # full MMLU
```

---

## MMMU-Pro

Links: [paper](https://arxiv.org/abs/2409.02813) · [dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)

Two settings: `standard` (text + inline images) and `vision` (rendered question image).

```bash
# Any target, standard or vision
uv run python -m bench_core run --target gemini-3.7-flash --benchmark mmmu_pro --variant standard
uv run python -m bench_core run --target gemini-3.7-flash --benchmark mmmu_pro --variant vision
uv run python -m bench_core run --target inkling          --benchmark mmmu_pro --variant standard

# Run on Modal instead of locally
bash benchmarks/mmmu_pro/run_full.sh
bash benchmarks/mmmu_pro/run_smoke.sh
```

---

## GPQA Diamond

Links: [paper](https://arxiv.org/abs/2311.12022) · [dataset](https://huggingface.co/datasets/Idavidrein/gpqa) (config: `gpqa_diamond`)

```bash
# GPQA runs through the unified CLI. Provider, model, and reasoning defaults come
# from the target (bench_core/targets.yaml); pick any target by name.
uv run python -m bench_core list-targets
uv run python -m bench_core run --target gpt-5.5          --benchmark gpqa
uv run python -m bench_core run --target gemini-3.7-flash --benchmark gpqa --reasoning high
uv run python -m bench_core run --target grok-4.3         --benchmark gpqa
uv run python -m bench_core run --target inkling          --benchmark gpqa
```

---

## Spider 2.0-Lite (SQLite subset, N=135)

Links: [repo](https://github.com/xlang-ai/Spider2) · [paper](https://arxiv.org/abs/2411.07763)

Text-to-SQL with execution-accuracy scoring against per-example SQLite databases.

```bash
# One-time setup: clone Spider2 + download SQLite databases
uv run -m benchmarks.spider2_lite.fetch_data   # downloads the ~4GB data/ (required)

# Run any target (execution accuracy over the 135 local SQLite instances)
uv run python -m bench_core run --target gpt-5.5 --benchmark spider2
uv run python -m bench_core run --target inkling --benchmark spider2
```

## Automation (GitHub Actions)

Three workflows under `.github/workflows/`:

- **`tests.yml`** — the free gate on every push/PR: the offline `bench_core` suite
  + lint. `test_targets.py` validates every `bench_core/targets.yaml` entry, so a
  new model is checked here before it ever runs.
- **`benchmark.yml`** — runs benchmarks through `python -m bench_core run`.
  - *Manual* (`workflow_dispatch`): pick `target`, `benchmarks` (csv or `all`),
    `sample_size` (blank = full run), and an optional `reasoning` override.
  - *Nightly* (`schedule`): smoke-runs every target flagged `ci_regression: true`
    to catch a provider changing its API (an adapter breaking shows up as a crash,
    not a silent drift).
  - *On merge* (`push` to `main` touching `targets.yaml`): diffs the merge and runs
    a **full** benchmark for the target(s) that changed.
  - A `report` job diffs each run against the committed baseline
    (`scripts/ci_compare.py`) and writes the score tables to the job summary.
- **`add-target.yml`** — the interactive "add a model" form (Actions → add-target →
  Run workflow): enter `name` / `provider` / `model_id` / optional
  `capabilities_json` / `ci_regression`. It validates the entry, appends it to
  `targets.yaml`, and **opens a PR** — it does not run anything itself.

**Two ways to add a model, both gated on merge:**
1. *Guided:* run `add-target` from the UI → it opens a PR.
2. *By hand:* edit `bench_core/targets.yaml` in a PR.

Either way, `tests.yml` validates the entry on the PR, and **merging the PR** is
what triggers `benchmark.yml` to run the new target (full run). No code, and the
benchmark only runs once a human has merged.

**Secrets required:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`, `INTERFAZE_API_KEY`, `HF_TOKEN`.

**Not in CI:** `olmocr` (needs poppler + playwright + the full dataset to score)
and `spider2` (needs the ~4GB `data/`); run those locally or on a self-hosted runner.
