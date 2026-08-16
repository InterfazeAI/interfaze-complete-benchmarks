# Interfaze complete benchmark scripts

[Blog](https://interfaze.ai/blog/interfaze-a-new-model-architecture-built-for-high-accuracy-at-scale) · [Leaderboard](https://interfaze.ai/leaderboards)

Runner scripts for the public benchmarks Interfaze is evaluated on.

## Setup

```bash
uv sync
```

Add a `.env` with the provider keys you'll use:

```env
INTERFAZE_API_KEY=…   OPENAI_API_KEY=…   ANTHROPIC_API_KEY=…
GEMINI_KEY=…          OPENROUTER_API_KEY=…   FIREWORKS_API_KEY=…
```

## Run a benchmark

Everything runs through one CLI (`bench_core`). Models are defined in
`bench_core/targets.yaml`; add one there to benchmark a new model — no code.

```bash
uv run python -m bench_core list-targets                            # what's available
uv run python -m bench_core run --target inkling --benchmark gpqa   # a full run
uv run python -m bench_core run --target gpt-5.5 --benchmark ocrbench_v2 --sample 20   # smoke
uv run python scripts/report_scores.py                              # view scores
```

`--sample N` runs the first N samples (smoke); omit it for the full benchmark.
Re-running resumes from the checkpoint. Add `--reasoning off|low|medium|high` to
override the target's default.

| Benchmark | `--benchmark` | Notes | Links |
|---|---|---|---|
| GPQA Diamond | `gpqa` | dataset is gated → `hf auth login` first | [paper](https://arxiv.org/abs/2311.12022) · [data](https://huggingface.co/datasets/Idavidrein/gpqa) |
| MMMLU | `mmmlu` | `--variant lite` (default) or `full` | [data](https://huggingface.co/datasets/openai/MMMLU) |
| MMMU-Pro | `mmmu_pro` | `--variant standard` or `vision` | [paper](https://arxiv.org/abs/2409.02813) · [data](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| OCRBench v2 | `ocrbench_v2` | 10k tasks; needs NLTK corpora † | [paper](https://arxiv.org/abs/2501.00321) · [repo](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2) |
| olmOCR | `olmocr` | needs poppler + chromium † | [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [data](https://huggingface.co/datasets/allenai/olmOCR-bench) |
| RefCOCO | `refcoco` | `--variant val\|testA\|testB\|test\|plus-*\|g-*` | [paper](https://arxiv.org/abs/1608.00272) · [data](https://huggingface.co/datasets/lmms-lab/RefCOCO) |
| ASR (VoxPopuli) | `asr` | WER, any audio-capable target | [data](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA) |
| Spider2-Lite | `spider2` | run `fetch_data` first (~4GB) † | [repo](https://github.com/xlang-ai/Spider2) · [paper](https://arxiv.org/abs/2411.07763) |

<details>
<summary>† one-time setup for some benchmarks</summary>

```bash
# olmOCR — headless chromium for equation rendering
uv run python -m playwright install chromium

# OCRBench v2 — NLTK corpora. Run from OUTSIDE the repo (nltk trips on a CWD .venv):
(cd /tmp && uv run --project "$OLDPWD" python -c \
 "import nltk; [nltk.download(p, quiet=True) for p in ('wordnet','omw-1.4','punkt','punkt_tab')]")

# Spider2 — clone + download the SQLite databases
uv run -m benchmarks.spider2_lite.fetch_data
```
</details>

<details>
<summary>Reading Fireworks / Inkling numbers</summary>

Inkling's reasoning **can't be fully disabled** — `reasoning_effort` accepts
`none|low|…|max`, but `none` is a floor that still emits reasoning tokens, so
runs tagged `off` aren't like-for-like with providers that truly disable
thinking. Per-sample `reasoning_tokens` are recorded so the real spend is
auditable. `inkling-small` isn't on serverless (needs a dedicated deployment).
</details>

## BFCL (Berkeley Function Calling Leaderboard)

Separate third-party harness in its own venv ([repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) · [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)).

```bash
scripts/setup_bfcl.sh                                    # once; safe to re-run
scripts/bfcl.sh generate --model openrouter-qwen3.5-35b-a3b-FC --test-category simple_python --num-threads 4
scripts/bfcl.sh evaluate --model openrouter-qwen3.5-35b-a3b-FC --test-category simple_python
```

<details>
<summary>BFCL gotchas</summary>

- Use `scripts/bfcl.sh`, not `bfcl` directly (it sets `BFCL_PROJECT_ROOT` and the uppercase `FIREWORKS_API_KEY`).
- BFCL **v4**: category is `simple_python`, not `simple`.
- Fireworks runs need a non-empty `OPENAI_API_KEY` placeholder (parent handler builds an OpenAI client); `soundfile` is an undeclared dep the setup script installs.
- **`underscore_to_dot` must match the provider** or correct calls score as `wrong_func_name` (measured 55% vs 94%). OpenRouter → `True`, Fireworks → `False`; the setup script sets it per provider.

</details>

## Automation (GitHub Actions)

- **`tests.yml`** — offline suite + lint on every push/PR (validates new `targets.yaml` entries).
- **`benchmark.yml`** — runs a benchmark: manual dispatch (pick target/benchmarks), weekly smoke of `ci_regression` targets, and a **full run on merge** of a `targets.yaml` change.
- **`add-target.yml`** — a UI form to add a model that opens a PR (doesn't run anything).

Add a model via the form or by editing `targets.yaml` in a PR; the benchmark
runs **only once the PR is merged**. Needs the provider secrets + `HF_TOKEN` set
in the repo. `olmocr` and `spider2` aren't in CI (heavy deps / ~4GB data).
