"""
Amazon Textract client (via amazon-textract-textractor) + usage tracking.

Mirrors the other provider commons modules. Textract is pure OCR (no LLM/VLM
reasoning, no LaTeX/math), so the value we add is in the *analysis features*
we request and how we linearize blocks back to markdown — see run_textract.py.

Auth: standard AWS credential chain. We require AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY in the env (.env is loaded), and a region (AWS_REGION
or AWS_DEFAULT_REGION, default us-east-1). boto3/textractor pick these up
automatically; we just resolve region explicitly so the client is pinned.
"""

import copy
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from textractor import Textractor

load_dotenv()


AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
    raise ValueError(
        "AWS credentials missing — set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
        "(and optionally AWS_REGION) in your environment / .env. "
        "Textract must be enabled for the account in that region."
    )


# Textractor wraps a boto3 'textract' client; credentials come from the
# default chain (env vars here). region_name pins the endpoint.
textractor_client = Textractor(region_name=AWS_REGION)


_usage_lock = threading.Lock()
_usage_state: dict = {
    "calls": 0,
    "pages": 0,
    "by_feature": {},
}


def record_usage(features: list[str], pages: int = 1) -> None:
    """Tally one AnalyzeDocument call. Textract bills per page per feature."""
    feat_key = "+".join(sorted(features)) if features else "none"
    with _usage_lock:
        _usage_state["calls"] += 1
        _usage_state["pages"] += int(pages or 0)
        bucket = _usage_state["by_feature"].setdefault(
            feat_key, {"calls": 0, "pages": 0}
        )
        bucket["calls"] += 1
        bucket["pages"] += int(pages or 0)


def get_usage_snapshot() -> dict:
    with _usage_lock:
        return copy.deepcopy(_usage_state)


def reset_usage() -> None:
    with _usage_lock:
        _usage_state["calls"] = 0
        _usage_state["pages"] = 0
        _usage_state["by_feature"] = {}


def write_usage_snapshot(path) -> None:
    """Atomically persist the current usage tally to a JSON file."""
    snap = get_usage_snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    tmp.replace(p)
