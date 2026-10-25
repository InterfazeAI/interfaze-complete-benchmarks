import copy
import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from reducto import Reducto

load_dotenv()


if (REDUCTO_API_KEY := os.getenv("REDUCTO_API_KEY", None)) is None:
    raise ValueError(
        "REDUCTO_API_KEY is not set in environment variables — get it from https://platform.reducto.ai"
    )


reducto_client: Any = Reducto(
    api_key=REDUCTO_API_KEY
)  # SDK stubs are strict; used dynamically


# ---------- Usage & budget tracking (thread-safe) -------------------------

# Parse r-1 is $10 / 1,000 pages, all-in (docs: reference/credit-usage; blog).
R1_USD_PER_PAGE = 0.01
BUDGET_CAP_USD = float(os.getenv("REDUCTO_BUDGET_USD", "20.0"))

_usage_lock = threading.Lock()
_usage_state: dict = {
    "calls": 0,
    "pages": 0,
    "credits": 0.0,
    "fields": 0,
    "usd": 0.0,
    "reserved": 0.0,
    "by_endpoint": {},
    "by_model": {},
}
_snapshot_path: Path | None = None
_ledger_path: Path | None = None


class BudgetExceeded(RuntimeError):
    """Raised before a call whose projected cost would cross the budget cap."""


def configure_usage(
    snapshot: str | Path | None = None,
    ledger: str | Path | None = None,
    cap_usd: float | None = None,
) -> None:
    """Set incremental-persistence paths and (optionally) the budget cap.

    With `snapshot`/`ledger` set, every recorded call is persisted immediately,
    so a crash or Ctrl-C mid-run never loses cost accounting.
    """
    global _snapshot_path, _ledger_path, BUDGET_CAP_USD
    if snapshot is not None:
        _snapshot_path = Path(snapshot)
    if ledger is not None:
        _ledger_path = Path(ledger)
        _ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if cap_usd is not None:
        BUDGET_CAP_USD = cap_usd


def _usage_get(usage, key, default=0):
    """Read a field from either an SDK usage object or a raw JSON dict."""
    if usage is None:
        return default
    val = (
        usage.get(key, default)
        if isinstance(usage, dict)
        else getattr(usage, key, default)
    )
    return default if val is None else val


def _parse_model(usage) -> str | None:
    """Model witness. parse_r1 normalizes the per-block model (result.chunks[].
    blocks[].extra.model) into usage['parse_model']; other API shapes may instead
    expose usage.usage_breakdown.parse_model.
    """
    pm = _usage_get(usage, "parse_model", None)
    if pm:
        return pm
    bd = _usage_get(usage, "usage_breakdown", None)
    if bd is None:
        return None
    return (
        bd.get("parse_model")
        if isinstance(bd, dict)
        else getattr(bd, "parse_model", None)
    )


def _model_from_body(body: dict) -> str | None:
    """The reliable r-1 engagement witness: the model stamped on each parsed block."""
    result = body.get("result") or {}
    for chunk in result.get("chunks") or []:
        blocks = chunk.get("blocks") or [] if isinstance(chunk, dict) else []
        for block in blocks:
            model = (block.get("extra") or {}).get("model")
            if model:
                return model
    return None


def _usd_from_usage(usage) -> float:
    """Authoritative dollar cost = pages × the r-1 all-in list rate ($0.01/page).

    The account's raw `credits` unit does not map 1:1 to dollars via the docs'
    example, so we bill on the published per-page rate (conservative) and keep
    raw credits separately for dashboard reconciliation.
    """
    return float(_usage_get(usage, "num_pages", 0) or 0) * R1_USD_PER_PAGE


def reserve_budget(est_usd: float) -> None:
    """Atomically reserve estimated spend; raise BudgetExceeded if over cap.

    Reserving before the call (and settling after) keeps concurrent callers from
    collectively overshooting the cap.
    """
    with _usage_lock:
        projected = _usage_state["usd"] + _usage_state["reserved"] + est_usd
        if projected > BUDGET_CAP_USD:
            raise BudgetExceeded(
                f"Reducto budget cap ${BUDGET_CAP_USD:.2f} would be exceeded: "
                f"spent ${_usage_state['usd']:.4f} + reserved ${_usage_state['reserved']:.4f} "
                f"+ est ${est_usd:.4f}"
            )
        _usage_state["reserved"] += est_usd


def release_reservation(est_usd: float) -> None:
    with _usage_lock:
        _usage_state["reserved"] = max(0.0, _usage_state["reserved"] - est_usd)


def record_usage(
    endpoint: str, usage, *, model: str | None = None, reserved: float = 0.0
) -> None:
    """Tally a single API response's usage and persist. Safe across threads.

    `reserved` releases the matching pre-call reservation atomically with the
    settle. Accepts an SDK usage object or a raw JSON dict.
    """
    if usage is None:
        if reserved:
            release_reservation(reserved)
        return
    pages = _usage_get(usage, "num_pages", 0) or 0
    credits = _usage_get(usage, "credits", 0) or 0
    fields = _usage_get(usage, "num_fields", 0) or 0
    usd = _usd_from_usage(usage)
    model = model or _parse_model(usage) or "unknown"
    with _usage_lock:
        _usage_state["reserved"] = max(0.0, _usage_state["reserved"] - reserved)
        _usage_state["calls"] += 1
        _usage_state["pages"] += pages
        _usage_state["credits"] += credits
        _usage_state["fields"] += fields
        _usage_state["usd"] += usd
        for group, name in (("by_endpoint", endpoint), ("by_model", model)):
            bucket = _usage_state[group].setdefault(
                name, {"calls": 0, "pages": 0, "credits": 0.0, "fields": 0, "usd": 0.0}
            )
            bucket["calls"] += 1
            bucket["pages"] += pages
            bucket["credits"] += credits
            bucket["fields"] += fields
            bucket["usd"] += usd
    _append_ledger(endpoint, model, pages, credits, usd, fields)
    if _snapshot_path is not None:
        write_usage_snapshot(_snapshot_path)


def _append_ledger(endpoint, model, pages, credits, usd, fields) -> None:
    if _ledger_path is None:
        return
    line = json.dumps(
        {
            "endpoint": endpoint,
            "model": model,
            "pages": pages,
            "credits": credits,
            "usd": usd,
            "fields": fields,
        }
    )
    with _usage_lock, open(_ledger_path, "a") as fh:
        fh.write(line + "\n")


def get_usage_snapshot() -> dict:
    with _usage_lock:
        return copy.deepcopy(_usage_state)


def reset_usage() -> None:
    with _usage_lock:
        _usage_state["calls"] = 0
        _usage_state["pages"] = 0
        _usage_state["credits"] = 0.0
        _usage_state["fields"] = 0
        _usage_state["usd"] = 0.0
        _usage_state["reserved"] = 0.0
        _usage_state["by_endpoint"] = {}
        _usage_state["by_model"] = {}


def write_usage_snapshot(path) -> None:
    """Atomically persist the current usage tally to a JSON file."""
    snap = get_usage_snapshot()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    tmp.replace(p)


def _join_chunks_from_url(url: str, timeout: float) -> str:
    data = httpx.get(url, timeout=timeout).json()
    chunks = data.get("chunks", [])
    return "\n\n".join((c.get("content") or "") for c in chunks).strip()


def _chunk_content(c) -> str:
    return (
        c.get("content") if isinstance(c, dict) else (getattr(c, "content", "") or "")
    ) or ""


def _text_from_result(result, timeout: float) -> str:
    """Join chunk content from a parse result (dict or SDK object; URL fallback)."""
    chunks = (
        result.get("chunks")
        if isinstance(result, dict)
        else getattr(result, "chunks", None)
    )
    if not chunks:
        url = (
            result.get("url")
            if isinstance(result, dict)
            else getattr(result, "url", None)
        )
        return _join_chunks_from_url(url, timeout) if url else ""
    return "\n\n".join(_chunk_content(c) for c in chunks).strip()


def upload_file(path: str, timeout: float = 600.0) -> str:
    """Upload a local file and return its reducto file_id."""
    with open(path, "rb") as fh:
        upload = reducto_client.upload(file=fh, timeout=timeout)
    return upload.file_id


def parse_r1(
    file_id: str,
    *,
    page_range: Iterable[int] | None = None,
    table_output_format: str = "html",
    merge_tables: bool = True,
    filter_blocks: Iterable[str] | None = None,
    timeout: float = 600.0,
    dump_raw_to: str | Path | None = None,
) -> tuple[str, str | None, dict]:
    """Parse via Reducto Parse **r-1** at max quality. Returns (markdown, job_id, usage).

    r-1 is a single-pass model: per Reducto's best-practices it handles layout,
    tables, figures, handwriting and scans natively, and *ignores* the legacy
    knobs (ocr_system, extraction_mode, intelligent_ordering, summarize_figures,
    agentic text passes). So max quality here is the minimal correct config —
    not the legacy enhance stack, which only adds cost/latency under r-1:
      - settings.model = "r-1"           : select the r-1 model
      - table_output_format = "html"     : docs' pick for complex/merged-cell tables
      - merge_tables = True              : rejoin tables split across page breaks
      - filter_blocks                    : drop Header/Footer/Page Number blocks

    Usage is read from the raw JSON body — the typed ParseUsage omits
    usage_breakdown.parse_model (the "R-1" engagement witness). Budget is
    reserved before the call and settled after.
    """
    npages = len(list(page_range)) if page_range is not None else 1
    est_usd = R1_USD_PER_PAGE * max(1, npages)
    reserve_budget(est_usd)
    settings: dict = {"model": "r-1"}
    if page_range is not None:
        settings["page_range"] = list(page_range)
    kwargs: dict = {
        "input": f"reducto://{file_id}",
        "timeout": timeout,
        "settings": settings,
        "formatting": {
            "table_output_format": table_output_format,
            "merge_tables": merge_tables,
        },
    }
    if filter_blocks is not None:
        kwargs["retrieval"] = {"filter_blocks": list(filter_blocks)}
    try:
        raw = reducto_client.parse.with_raw_response.run(**kwargs)
        body = raw.http_response.json()
    except Exception:
        release_reservation(est_usd)
        raise
    usage = body.get("usage") or {}
    model = _model_from_body(body) or _parse_model(usage)
    if model:
        usage["parse_model"] = model
    record_usage("parse", usage, model=model, reserved=est_usd)
    if dump_raw_to is not None:
        p = Path(dump_raw_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, indent=2))
    text = _text_from_result(body.get("result") or {}, timeout)
    return text, body.get("job_id"), usage


def parse_file(
    file_id: str,
    table_format: str = "md",
    timeout: float = 600.0,
) -> tuple[str, str | None]:
    """Parse a previously-uploaded file. Returns (joined_markdown, parse_job_id).

    parse_job_id is None when the response is large (URL result) — extract
    chaining still works against `reducto://<file_id>` in that case.
    """
    response = reducto_client.parse.run(
        input=f"reducto://{file_id}",
        timeout=timeout,
        enhance={
            "agentic": [{"scope": "text"}],
            "intelligent_ordering": True,
        },
        formatting={"table_output_format": table_format},
    )
    record_usage("parse", getattr(response, "usage", None))
    job_id = getattr(response, "job_id", None)
    result = response.result
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        url = getattr(result, "url", None)
        text = _join_chunks_from_url(url, timeout) if url else ""
    else:
        text = "\n\n".join((c.content or "") for c in chunks).strip()
    return text, job_id


def extract_with_schema(
    parse_job_id: str | None,
    file_id: str,
    schema: dict,
    system_prompt: str,
    timeout: float = 600.0,
) -> Any:
    """Run extract over an existing parse job (or a file_id fallback).

    Returns the extract.result (dict or list[dict]).
    """
    extract_input = (
        f"jobid://{parse_job_id}" if parse_job_id else f"reducto://{file_id}"
    )
    response = reducto_client.extract.run(
        input=extract_input,
        instructions={"schema": schema, "system_prompt": system_prompt},
        timeout=timeout,
    )
    record_usage("extract", getattr(response, "usage", None))
    return response.result


def parse_file_to_markdown(
    path: str,
    pages: Iterable[int] | None = None,
    timeout: float = 600.0,
) -> str:
    """Upload a local file (PDF / image) to Reducto, parse it, return joined markdown.

    pages: 1-indexed page numbers to include (None = all pages).

    Quality settings (per Reducto's best-practice docs):
      - enhance.agentic = [{"scope": "text"}]: VLM second pass; fixes math notation,
        handwriting, special characters.
      - enhance.intelligent_ordering = True: improves reading order on multi-column.
      - formatting.table_output_format = "md": markdown tables (matches our pipelines).
    """
    with open(path, "rb") as fh:
        upload = reducto_client.upload(file=fh, timeout=timeout)

    kwargs: dict = {
        "input": f"reducto://{upload.file_id}",
        "timeout": timeout,
        "enhance": {
            "agentic": [{"scope": "text"}, {"scope": "table"}],
            "intelligent_ordering": True,
        },
        "formatting": {"table_output_format": "md"},
    }
    if pages is not None:
        kwargs["settings"] = {"page_range": list(pages)}

    response = reducto_client.parse.run(**kwargs)
    record_usage("parse", getattr(response, "usage", None))

    result = response.result
    chunks = getattr(result, "chunks", None)
    if chunks is None:
        url = getattr(result, "url", None)
        return _join_chunks_from_url(url, timeout) if url else ""

    return "\n\n".join((c.content or "") for c in chunks).strip()
