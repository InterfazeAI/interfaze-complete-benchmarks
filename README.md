# Interfaze complete benchmark

[Blog](https://interfaze.ai/blog/interfaze-a-new-model-architecture-built-for-high-accuracy-at-scale) · [Leaderboard](https://interfaze.ai/leaderboards)

The runners behind the public benchmarks Interfaze is evaluated on. Point any model from
`src/targets.yaml` at any benchmark with one command; adding a model is a YAML entry, not code.

## Quick start

**1. Install** — needs Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

**2. Add API keys** to a `.env` file in the repo root. You only need keys for the providers
of the models you run:

```env
OPENAI_API_KEY=…
ANTHROPIC_API_KEY=…
GEMINI_API_KEY=…        # GEMINI_KEY or GOOGLE_API_KEY also work
FIREWORKS_API_KEY=…
OPENROUTER_API_KEY=…
INTERFAZE_API_KEY=…
```

**3. Try a benchmark on a few samples first**, then run it in full:

```bash
uv run python -m src list-targets                                         # models you can run
uv run python -m src run --target gpt-5.5 --benchmark gpqa --sample 5     # quick check (5 samples)
uv run python -m src run --target gpt-5.5 --benchmark gpqa                # full run
```

**4. See the scores:**

```bash
uv run python scripts/report_scores.py
```

## Benchmarks

| Benchmark | `--benchmark` | `--variant` | Extra setup | Links |
| --- | --- | --- | --- | --- |
| GPQA Diamond | `gpqa` | — | [Hugging Face login](#gpqa) | [paper](https://arxiv.org/abs/2311.12022) · [data](https://huggingface.co/datasets/Idavidrein/gpqa) |
| MMMLU | `mmmlu` | `lite` (default), `full` | — | [data](https://huggingface.co/datasets/openai/MMMLU) |
| MMMU-Pro | `mmmu_pro` | `standard` (default), `vision` | — | [paper](https://arxiv.org/abs/2409.02813) · [data](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| OCRBench v2 | `ocrbench_v2` | — | [NLTK corpora](#ocrbench-v2) | [paper](https://arxiv.org/abs/2501.00321) · [repo](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2) |
| olmOCR | `olmocr` | — | [poppler + Chromium](#olmocr) | [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [data](https://huggingface.co/datasets/allenai/olmOCR-bench) |
| RefCOCO | `refcoco` | `val` (default), `testA`, `testB`, `test`, `plus-val`, `plus-testA`, `plus-testB`, `g-val`, `g-test` | — | [paper](https://arxiv.org/abs/1608.00272) · [data](https://huggingface.co/datasets/lmms-lab/RefCOCO) |
| ASR (VoxPopuli) | `asr` | — | needs an audio-capable model | [data](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA) |
| Spider2-Lite | `spider2` | — | [~4GB data download](#spider2-lite) | [repo](https://github.com/xlang-ai/Spider2) · [paper](https://arxiv.org/abs/2411.07763) |

### Extra setup

One-time steps, only for the benchmarks that need them.

#### GPQA

The dataset is gated: request access on its [Hugging Face page](https://huggingface.co/datasets/Idavidrein/gpqa), then log in with `hf auth login` (or set `HF_TOKEN`).

#### OCRBench v2

Download the NLTK corpora the scorer uses. Run it from **outside** the repo — NLTK refuses to load from a directory that contains the `.venv`:

```bash
(cd /tmp && uv run --project "$OLDPWD" python -c \
 "import nltk; [nltk.download(p, quiet=True) for p in ('wordnet','omw-1.4','punkt','punkt_tab')]")
```

#### olmOCR

Needs poppler to render PDF pages and headless Chromium to render equations:

```bash
brew install poppler                  # macOS; on Debian/Ubuntu: sudo apt-get install poppler-utils
uv run python -m playwright install chromium
```

The first run downloads the benchmark's ~1,400 PDFs.

#### Spider2-Lite

Clone the Spider2 repo and download its SQLite databases (~4GB) into `benchmarks/spider2_lite/data/`:

```bash
uv run -m benchmarks.spider2_lite.fetch_data
```

## Run options

| Option | What it does |
| --- | --- |
| `--sample N` | Runs only the first N samples — a quick check. Results go to `results/_smoke/`, so they never overwrite a full run and don't show up in `report_scores.py`. |
| `--variant V` | Picks a benchmark variant (see the table above). |
| `--reasoning off\|low\|medium\|high` | Overrides the reasoning effort (default `off`). Models that can't fully turn reasoning off use their lowest setting. |

- **Interrupted runs resume.** Every answer is saved as it arrives; re-run the same command to continue where it stopped.
- **Backup providers.** A model can list fallback providers in `targets.yaml`; if the main
  provider is down or out of quota, requests move to the next one. Providers without a key are skipped with a `skipping route …` message.

## Results

Each run writes to `results/<benchmark>/<model>/` (variants add a suffix, e.g. `mmmu_pro_vision`; ASR results are under `voxpopuli_aa`):

- `metrics.json` — the scores
- `responses.jsonl` — every model answer, one per line
- `run.json` — run details (providers used, failures, sample size)

`scripts/report_scores.py` prints every score as tables:

```bash
uv run python scripts/report_scores.py                       # pick a model interactively
uv run python scripts/report_scores.py --all                 # every model
uv run python scripts/report_scores.py --model gpt-5.5       # one model (repeatable)
uv run python scripts/report_scores.py --model gpt-5.5 --tsv # tab-separated, for pasting into a sheet
```

## Adding a model

Add an entry to [`src/targets.yaml`](src/targets.yaml). Most models need only a provider and the provider's model id:

```yaml
targets:
  my-model:
    provider: fireworks          # openai | anthropic | gemini | fireworks | openrouter | interfaze
    model_id: accounts/fireworks/models/my-model
```

Add `capabilities:` only where the model differs from its provider's defaults — the existing entries show the common cases (e.g. `grok-4.3` can't turn reasoning off, `gemini-2.5-pro` uses a thinking budget). Check the entry loads with `uv run pytest tests/test_targets.py`.

You can also add a model without editing YAML: run the **add-target** workflow from the
GitHub Actions tab (it opens a PR), or use `uv run python scripts/add_target.py --help` locally.

## CI (GitHub Actions)

| Workflow | When it runs | What it does |
| --- | --- | --- |
| `tests.yml` | every push and PR | offline tests and lint (also validates `targets.yaml`) |
| `benchmark.yml` | manually; weekly; when a `targets.yaml` change is merged | manual: the target and benchmarks you pick (5 samples by default). Weekly: a 5-sample check of every target marked `ci_regression: true`. On merge: a full run for each added or changed target. |
| `add-target.yml` | manually | a form that adds a model and opens a PR |

`olmocr` and `spider2` don't run in CI (heavy system dependencies and ~4GB of data). The
workflows need the provider keys and `HF_TOKEN` set as repository secrets.

**Regression check.** The benchmark summary compares each score with a saved baseline in `baselines/<benchmark>/<model>/metrics.json`. To set or update one, copy a full run's
`results/<benchmark>/<model>/metrics.json` there and commit it. Quick checks are shown next to the baseline but never flagged as regressions.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `no API key for any route of target …` | Add the key named in the `skipping route …` line above it to `.env`. |
| GPQA fails with 401 / gated dataset | [GPQA setup](#gpqa): request access and log in to Hugging Face. |
| `Resource wordnet not found` (OCRBench v2) | [Download the NLTK corpora](#ocrbench-v2). |
| `pdfinfo` / `pdftoppm` not found (olmOCR) | [Install poppler](#olmocr). |
| `Missing …/spider2-lite.jsonl` | [Fetch the Spider2 data](#spider2-lite). |
