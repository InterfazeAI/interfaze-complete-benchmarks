# Interfaze complete benchmark

[Blog](https://interfaze.ai/blog/interfaze-a-new-model-architecture-built-for-high-accuracy-at-scale) · [Leaderboard](https://interfaze.ai/leaderboards)

Runner scripts for the public benchmarks Interfaze is evaluated on.

## Setup

```bash
uv sync
```

Add a `.env` with the provider keys you'll use:

```env
INTERFAZE_API_KEY=…
OPENAI_API_KEY=…
ANTHROPIC_API_KEY=…
GEMINI_KEY=…
OPENROUTER_API_KEY=…
FIREWORKS_API_KEY=…
```

## Run a benchmark

Everything runs through one CLI (`python -m src`). Models are defined in
`src/targets.yaml`; add one there to benchmark a new model — no code.

```bash
uv run python -m src list-targets                            # what's available
uv run python -m src run --target inkling --benchmark gpqa   # a full run
uv run python -m src run --target gpt-5.5 --benchmark ocrbench_v2 --sample 20   # smoke
uv run python scripts/report_scores.py                              # view scores
```

`--sample N` runs the first N samples — a *smoke*. It **streams** only those N rows, so smoking a heavy image benchmark (e.g. `ocrbench_v2`, ~10k images) no longer downloads the whole split first. Smokes write to `results/_smoke/…`, isolated from full runs and hidden from `report_scores.py`, so they can't clobber a real score. Omit `--sample` for the full benchmark. Re-running resumes from the checkpoint. Add `--reasoning off|low|medium|high` to override the target's default.

> A **full** image-benchmark run loads the whole split into RAM (RefCOCO `val` ≈
> 8.8k images → several GB). Smoke with `--sample` first; if a full local run runs
> out of memory, see the lazy-by-index loader `ocrbench_v2` already uses.

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

## Automation (GitHub Actions)

- **`tests.yml`** — offline suite + lint on every push/PR (validates new `targets.yaml` entries).
- **`benchmark.yml`** — runs a benchmark: manual dispatch (pick target/benchmarks), weekly smoke of `ci_regression` targets, and a **full run on merge** of a `targets.yaml` change.
- **`add-target.yml`** — a UI form to add a model that opens a PR (doesn't run anything).

Add a model via the form or by editing `targets.yaml` in a PR; the benchmark runs **only once the PR is merged**. Needs the provider secrets + `HF_TOKEN` set in the repo. `olmocr` and `spider2` aren't in CI (heavy deps / ~4GB data).
