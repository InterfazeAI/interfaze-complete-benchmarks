"""
Extend.ai client + light usage tracking.

Mirrors src/commons_reducto.py so the olmOCR bench candidate looks the same
as the existing Reducto integration. Keeps the SDK construction in one place
and exposes a thread-safe page-count tally for the usage snapshot file.
"""

import copy
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from extend_ai import Extend

load_dotenv()


if (EXTEND_API_KEY := os.getenv("EXTEND_API_KEY", None)) is None:
    raise ValueError(
        "EXTEND_API_KEY is not set in environment variables — get it from https://www.extend.ai/"
    )


extend_client = Extend(token=EXTEND_API_KEY)


_usage_lock = threading.Lock()
_usage_state: dict = {
    "calls": 0,
    "pages": 0,
    "by_endpoint": {},
}


def record_usage(endpoint: str, metrics_obj) -> None:
    """Tally a single API response's usage. Safe to call from many threads.

    Extend's ParseRun returns a `metrics` field with `pageCount` (camelCase via
    SDK attribute access) and processing time. We only track pages here.
    """
    if metrics_obj is None:
        return
    pages = (
        getattr(metrics_obj, "page_count", None)
        or getattr(metrics_obj, "pageCount", None)
        or 0
    )
    with _usage_lock:
        _usage_state["calls"] += 1
        _usage_state["pages"] += pages
        bucket = _usage_state["by_endpoint"].setdefault(
            endpoint, {"calls": 0, "pages": 0}
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
        _usage_state["by_endpoint"] = {}


def write_usage_snapshot(path) -> None:
    """Atomically persist the current usage tally to a JSON file."""
    snap = get_usage_snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    tmp.replace(p)
