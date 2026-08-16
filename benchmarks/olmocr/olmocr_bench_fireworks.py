"""
OlmOCR Benchmark for Fireworks-hosted models (default: Inkling).

Mirrors olmocr_bench.py (interfaze) but routes pages through Fireworks'
OpenAI-compatible endpoint.

Usage:
    uv run -m benchmarks.olmocr.olmocr_bench_fireworks
    uv run -m benchmarks.olmocr.olmocr_bench_fireworks --sample
    uv run -m benchmarks.olmocr.olmocr_bench_fireworks --skip-generation
    uv run -m benchmarks.olmocr.olmocr_bench_fireworks --generate-only
    uv run -m benchmarks.olmocr.olmocr_bench_fireworks --model accounts/fireworks/models/inkling-small
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "bench" / "sample_data"
FULL_DATA_DIR = Path(__file__).resolve().parent / "bench" / "full_data"

INKLING = "accounts/fireworks/models/inkling"
MODEL = INKLING
# Candidate name = the output subdirectory the olmOCR scorer reads. Derived from
# the model so inkling and inkling-small don't overwrite each other.
CANDIDATE_NAME = "inkling"
RATE_LIMIT = 25
MAX_RETRIES = 3

# Cap on requests in flight. RATE_LIMIT bounds how many requests *start* per
# second but not how many are outstanding — all ~1400 pages launched at once
# queue server-side and eventually 429 after minutes of latency. Override with
# BENCH_MAX_IN_FLIGHT.
MAX_IN_FLIGHT = int(os.getenv("BENCH_MAX_IN_FLIGHT", "8"))

HF_REPO = "allenai/olmOCR-bench"
SPLITS = [
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def model_slug(model: str) -> str:
    """accounts/fireworks/models/inkling-small -> inkling-small"""
    return re.sub(r"[^a-z0-9]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")


class RateLimiter:
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_refill = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                elapsed = now - self.last_refill
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                self.last_refill = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            await asyncio.sleep(1 / self.rate)


def download_full_dataset():
    data_dir = FULL_DATA_DIR
    pdf_dir = data_dir / "pdfs"
    all_pdfs = set()
    for split in SPLITS:
        jsonl_dest = data_dir / f"{split}.jsonl"
        if jsonl_dest.exists():
            with open(jsonl_dest) as f:
                tests = [json.loads(l) for l in f if l.strip()]
        else:
            print(f"  Downloading {split}.jsonl...")
            src = hf_hub_download(
                HF_REPO, f"bench_data/{split}.jsonl", repo_type="dataset"
            )
            with open(src) as f:
                tests = [json.loads(l) for l in f if l.strip()]
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(jsonl_dest, "w") as f:
                for t in tests:
                    f.write(json.dumps(t) + "\n")
        print(f"    {split}: {len(tests)} tests")
        for t in tests:
            all_pdfs.add(t["pdf"])

    print(f"\n  Total unique PDFs to download: {len(all_pdfs)}")
    downloaded = 0
    skipped = 0
    for pdf_rel in sorted(all_pdfs):
        local_path = pdf_dir / pdf_rel
        if local_path.exists():
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            src = hf_hub_download(
                HF_REPO, f"bench_data/pdfs/{pdf_rel}", repo_type="dataset"
            )
            os.symlink(src, str(local_path))
            downloaded += 1
        except Exception as e:
            print(f"    Failed to download {pdf_rel}: {e}")
    print(f"  PDFs: {downloaded} downloaded, {skipped} already existed")
    return data_dir


async def process_page(pdf_path, page_num, output_path, rate_limiter, sem):
    async with sem:
        return await _process_page(pdf_path, page_num, output_path, rate_limiter)


async def _process_page(pdf_path, page_num, output_path, rate_limiter):
    from olmocr.bench.runners.run_fireworks import run_fireworks

    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            result = await asyncio.to_thread(run_fireworks, pdf_path, page_num, MODEL)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
            else:
                print(
                    f"Failed after {MAX_RETRIES} attempts: {pdf_path} page {page_num}: {e}"
                )
                return False


async def generate_outputs(data_dir: Path):
    pdf_folder = data_dir / "pdfs"
    output_folder = data_dir / CANDIDATE_NAME

    pdf_pages = set()
    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                pdf_pages.add((t["pdf"], t["page"]))

    print(f"Found {len(pdf_pages)} unique (pdf, page) pairs to process")

    rate_limiter = RateLimiter(RATE_LIMIT)
    sem = asyncio.Semaphore(MAX_IN_FLIGHT)
    tasks = []
    for pdf_rel, page in sorted(pdf_pages):
        pdf_path = str(pdf_folder / pdf_rel)
        if not os.path.exists(pdf_path):
            continue
        base_name = os.path.splitext(os.path.basename(pdf_rel))[0]
        parent_dir = os.path.dirname(pdf_rel)
        md_filename = f"{base_name}_pg{page}_repeat1.md"
        if parent_dir:
            out_path = str(output_folder / parent_dir / md_filename)
        else:
            out_path = str(output_folder / md_filename)
        if os.path.exists(out_path):
            continue
        tasks.append(process_page(pdf_path, page, out_path, rate_limiter, sem))

    if not tasks:
        print("All outputs already exist, skipping generation.")
        return True
    print(f"Processing {len(tasks)} pages...")
    results = await tqdm_asyncio.gather(
        *tasks, desc=f"Generating {CANDIDATE_NAME} outputs"
    )
    num_success = sum(1 for r in results if r)
    num_failed = len(results) - num_success
    print(f"Done: {num_success} succeeded, {num_failed} failed")
    return num_failed == 0


class _Tee:
    """olmOCR-bench prints its scores and saves nothing, so mirror stdout to a
    log file — otherwise the numbers exist only in whatever terminal ran it and
    scripts/report_scores.py has nothing to read."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_evaluation(data_dir: Path):
    from olmocr.bench.benchmark import main as bench_main

    sys.argv = [
        "benchmark",
        "--dir",
        str(data_dir),
        "--candidate",
        CANDIDATE_NAME,
        "--force",
    ]
    log_path = PROJECT_ROOT / "logs" / f"olmocr_{CANDIDATE_NAME}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    real_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(real_stdout, fh)
        try:
            bench_main()
        finally:
            sys.stdout = real_stdout
    print(f"\nolmOCR scores also written to {log_path}")


async def main():
    global MODEL, CANDIDATE_NAME
    parser = argparse.ArgumentParser(
        description="Run OlmOCR benchmark against a Fireworks-hosted model"
    )
    parser.add_argument(
        "--model", default=INKLING, help="Fireworks model id (accounts/fireworks/...)"
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    MODEL = args.model
    CANDIDATE_NAME = model_slug(MODEL)

    if args.sample:
        data_dir = SAMPLE_DATA_DIR
        print("=== Using sample data ===")
    else:
        print("=== Downloading full olmOCR-bench dataset from HuggingFace ===")
        data_dir = download_full_dataset()

    if not args.skip_generation:
        print(f"\n=== Generating {CANDIDATE_NAME} outputs ===")
        await generate_outputs(data_dir)

    if not args.generate_only:
        print("\n=== Running OlmOCR Benchmark Evaluation ===")
        run_evaluation(data_dir)


if __name__ == "__main__":
    asyncio.run(main())
