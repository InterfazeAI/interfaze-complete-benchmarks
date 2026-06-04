"""
LlamaParse (llama-cloud) client + light usage tracking.

Mirrors src/commons_reducto.py / src/commons_extend.py so the olmOCR-bench
candidate is wired up the same way as the other doc-parsing providers.

Auth: LLAMA_CLOUD_API_KEY env var (keys start with `llx-`). The SDK reads
the env automatically when api_key is not passed, but we resolve it
explicitly so the import-time check fails loudly if it's missing.
"""

import copy
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud

load_dotenv()


if (LLAMA_CLOUD_API_KEY := os.getenv("LLAMA_CLOUD_API_KEY", None)) is None:
    raise ValueError(
        "LLAMA_CLOUD_API_KEY is not set — get it from https://cloud.llamaindex.ai"
    )


llamaparse_client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)


_usage_lock = threading.Lock()
_usage_state: dict = {
    "calls": 0,
    "pages": 0,
    "by_tier": {},
}


def record_usage(tier: str, pages: int) -> None:
    """Tally a single API response's usage. Safe to call from many threads.

    LlamaParse bills per-page; we record the per-call page count + tier so
    the usage snapshot reflects how much agentic_plus we burned.
    """
    pages = int(pages or 0)
    with _usage_lock:
        _usage_state["calls"] += 1
        _usage_state["pages"] += pages
        bucket = _usage_state["by_tier"].setdefault(
            tier or "unknown", {"calls": 0, "pages": 0}
        )
        bucket["calls"] += 1
        bucket["pages"] += pages


def get_usage_snapshot() -> dict:
    with _usage_lock:
        return copy.deepcopy(_usage_state)


def reset_usage() -> None:
    with _usage_lock:
        _usage_state["calls"] = 0
        _usage_state["pages"] = 0
        _usage_state["by_tier"] = {}


def write_usage_snapshot(path) -> None:
    """Atomically persist the current usage tally to a JSON file."""
    snap = get_usage_snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    tmp.replace(p)
