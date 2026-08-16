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
JIGSAWSTACK_API_KEY=...   # only for benchmarks/obj_detection/ob_det_api.py
```

Every runner accepts `--limit N` for a smoke test and `--evaluate-only` /
`--predict-only` to split prediction and scoring. Re-running a benchmark
resumes from its checkpoint file — already-completed samples are skipped.

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
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_fireworks
uv run -m benchmarks.olmocr.olmocr_bench_fireworks
uv run python -m bench_core run --target inkling --benchmark gpqa
uv run -m benchmarks.mmmlu.mmmlu_multi        --provider fireworks --model accounts/fireworks/models/inkling
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi  --provider fireworks --model accounts/fireworks/models/inkling --setting standard
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi  --provider fireworks --model accounts/fireworks/models/inkling --setting vision
uv run -m benchmarks.obj_detection.refcoco_multi --provider fireworks --model accounts/fireworks/models/inkling
uv run -m benchmarks.asr.voxpopuli_aa_multi   --provider fireworks --model accounts/fireworks/models/inkling
uv run -m benchmarks.spider2_lite.spider2_lite --provider fireworks
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
# Interfaze
uv run -m benchmarks.ocrbench_v2.ocrbench_v2

# Per-provider runners
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai_mini
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_anthropic
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_gemini
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_gemini_pro_31
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_grok
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_kimi          # via OpenRouter
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_fireworks     # Inkling

# Text-spotting EN subset only
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_text_spotting_en

# Evaluate without re-running predictions
uv run -m benchmarks.ocrbench_v2.ocrbench_v2 --evaluate-only
```

---

## olmOCR-Bench

Links: [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)

```bash
# Interfaze
uv run -m benchmarks.olmocr.olmocr_bench

# Per-provider runners
uv run -m benchmarks.olmocr.olmocr_bench_openai_mini
uv run -m benchmarks.olmocr.olmocr_bench_gemini_pro_31
uv run -m benchmarks.olmocr.olmocr_bench_grok
uv run -m benchmarks.olmocr.olmocr_bench_fireworks

# Useful flags
uv run -m benchmarks.olmocr.olmocr_bench --sample           # tiny sample dataset
uv run -m benchmarks.olmocr.olmocr_bench --generate-only    # predictions only
uv run -m benchmarks.olmocr.olmocr_bench --skip-generation  # evaluation only
```

---

## RefCOCO (Object Detection)

Links: [RefCOCO/RefCOCO+ paper](https://arxiv.org/abs/1608.00272) · [RefCOCOg paper](https://arxiv.org/abs/1511.02283) · [dataset](https://huggingface.co/datasets/lmms-lab/RefCOCO)

Metric: Acc@IoU=0.5 on the referring-expression bounding box.

```bash
# Interfaze (RefCOCO val by default)
uv run -m benchmarks.obj_detection.refcoco
uv run -m benchmarks.obj_detection.refcoco --split testA
uv run -m benchmarks.obj_detection.refcoco --dataset lmms-lab/RefCOCO+ --split testB

# Any provider via the multi runner
uv run -m benchmarks.obj_detection.refcoco_multi --provider openai    --model gpt-5.4
uv run -m benchmarks.obj_detection.refcoco_multi --provider anthropic --model claude-sonnet-4-6
uv run -m benchmarks.obj_detection.refcoco_multi --provider gemini    --model gemini-3-flash-preview

# JigsawStack object_detection API (instead of a VLM)
uv run -m benchmarks.obj_detection.ob_det_api

# Evaluate only
uv run -m benchmarks.obj_detection.refcoco --evaluate-only
```

---

## VoxPopuli-Cleaned-AA (ASR)

Links: [dataset](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)

Metric: WER with Whisper-style text normalization.

```bash
# Interfaze
uv run -m benchmarks.asr.voxpopuli_aa

# Other providers (audio-capable)
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider gemini    --model gemini-3-flash-preview
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider openai    --model gpt-4o-audio-preview
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider anthropic --model claude-sonnet-4-6

# Evaluate only
uv run -m benchmarks.asr.voxpopuli_aa --evaluate-only
```

---

## MMMLU (Multilingual MMLU)

Links: [dataset](https://huggingface.co/datasets/openai/MMMLU)

14 languages, exact-match accuracy macro-averaged across languages.

```bash
# Interfaze
uv run -m benchmarks.mmmlu.mmmlu
uv run -m benchmarks.mmmlu.mmmlu --languages DE_DE FR_FR     # subset of languages

# Any provider
uv run -m benchmarks.mmmlu.mmmlu_multi --provider openai    --model gpt-5.4-mini
uv run -m benchmarks.mmmlu.mmmlu_multi --provider gemini    --model gemini-3.1-pro-preview
uv run -m benchmarks.mmmlu.mmmlu_multi --provider anthropic --model claude-sonnet-4-6
uv run -m benchmarks.mmmlu.mmmlu_multi --provider interfaze --model interfaze-beta

# Evaluate only
uv run -m benchmarks.mmmlu.mmmlu --evaluate-only
```

---

## MMMU-Pro

Links: [paper](https://arxiv.org/abs/2409.02813) · [dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)

Two settings: `standard` (text + inline images) and `vision` (rendered question image).

```bash
# Any provider, standard or vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider gemini    --model gemini-3.1-pro-preview --setting standard
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider gemini    --model gemini-3.1-pro-preview --setting vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider openai    --model gpt-5.5               --setting standard
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider anthropic --model claude-sonnet-4-6     --setting vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider interfaze --model interfaze-beta        --setting standard

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
uv run -m benchmarks.spider2_lite.fetch_data

# Run
uv run -m benchmarks.spider2_lite.spider2_lite
uv run -m benchmarks.spider2_lite.spider2_lite --provider fireworks
uv run -m benchmarks.spider2_lite.spider2_lite --predict-only
uv run -m benchmarks.spider2_lite.spider2_lite --evaluate-only
```
