"""Spider2-Lite (SQLite subset): text-to-SQL; execution accuracy over 135 local
instances. Scoring re-executes predicted SQL against the local .sqlite DBs and
compares result sets to gold — it needs the (gitignored) data/ directory.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sqlite3
import time
from pathlib import Path

from bench_core.request import Message, ReasoningSpec, Request, TextPart

NAME = "spider2_lite"
ID_KEY = "instance_id"
PRIMARY_METRIC = "accuracy_of_local_135"
DEFAULTS = {"reasoning": "off", "rate_limit": 8, "max_in_flight": 8}

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SPIDER2_LITE = _DATA_DIR / "Spider2" / "spider2-lite"
_SQLITE_DB_DIR = _SPIDER2_LITE / "resource" / "databases"
_SCHEMA_DIR = _SQLITE_DB_DIR / "sqlite"
_DOCUMENTS_DIR = _SPIDER2_LITE / "resource" / "documents"
_GOLD_DIR = _SPIDER2_LITE / "evaluation_suite" / "gold"
_EVAL_STANDARD = _GOLD_DIR / "spider2lite_eval.jsonl"
_GOLD_EXEC_DIR = _GOLD_DIR / "exec_result"
_ALL_EXAMPLES = _SPIDER2_LITE / "spider2-lite.jsonl"

QUERY_TIMEOUT_S = 120.0
MAX_DDL_CHARS = 80_000
MAX_EK_CHARS = 40_000

PROMPT_TEMPLATE = """You are an expert SQLite SQL developer. Write a SQL query that answers the user's question against the given database. Target dialect: SQLite.

### Database Schema
{schema}
{external_knowledge_section}
### Question
{question}

Return ONLY the final SQL query, wrapped in a fenced code block like:
```sql
SELECT ...
```
Do not include any explanation before or after the code block."""

_SQL_FENCE = re.compile(r"```sql\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)
# Unclosed fence: an opener with no terminator (some models never close the block).
_OPEN_FENCE = re.compile(r"```(?:sql)?[ \t]*\r?\n(.*)\Z", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    if not text:
        return ""
    for pat in (_SQL_FENCE, _ANY_FENCE, _OPEN_FENCE):
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return text.strip()


# --- prompt assembly (needs data/) ---
def load_schema(db_name: str) -> str:
    ddl_path = _SCHEMA_DIR / db_name / "DDL.csv"
    if not ddl_path.exists():
        return f"-- schema file missing: {ddl_path}"
    parts = []
    with open(ddl_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ddl = (row.get("DDL") or "").strip()
            if ddl:
                parts.append(ddl.rstrip(";") + ";")
    schema = "\n\n".join(parts)
    if len(schema) > MAX_DDL_CHARS:
        schema = schema[:MAX_DDL_CHARS] + "\n-- [schema truncated]"
    return schema


def load_external_knowledge(filename: str | None) -> str | None:
    if not filename:
        return None
    path = _DOCUMENTS_DIR / filename
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > MAX_EK_CHARS:
        text = text[:MAX_EK_CHARS] + "\n... [truncated]"
    return text


def build_prompt(example: dict) -> str:
    ek = load_external_knowledge(example.get("external_knowledge"))
    ek_section = f"\n### External Knowledge\n{ek}\n" if ek else "\n"
    return PROMPT_TEMPLATE.format(
        schema=load_schema(example["db"]),
        external_knowledge_section=ek_section,
        question=example["question"],
    )


def load_samples(sample_size: int | None = None) -> list[dict]:
    if not _ALL_EXAMPLES.exists():
        raise FileNotFoundError(
            f"Missing {_ALL_EXAMPLES}. The Spider2 data/ dir (gitignored, ~4GB) must "
            "be present. Run: uv run -m benchmarks.spider2_lite.fetch_data"
        )
    import json

    rows = [
        json.loads(line)
        for line in _ALL_EXAMPLES.read_text().splitlines()
        if line.strip()
    ]
    local = [r for r in rows if r["instance_id"].startswith("local")]
    samples = []
    for ex in local:
        samples.append(
            {
                "instance_id": ex["instance_id"],
                "db": ex["db"],
                "question": ex["question"],
                "external_knowledge": ex.get("external_knowledge"),
                "prompt": build_prompt(ex),
            }
        )
    return samples[:sample_size] if sample_size else samples


def build_request(sample: dict, mode: str) -> Request:
    return Request(
        [Message("user", [TextPart(sample["prompt"])])],
        reasoning=ReasoningSpec(mode),
        temperature=0.0,
    )


def parse(response, sample) -> str:
    return extract_sql(response.text or "")


# --- execution + comparison (verbatim port of the official evaluator's quirks) ---
def execute_sqlite(db_path: Path, sql: str):
    try:
        disk = sqlite3.connect(str(db_path))
        mem = sqlite3.connect(":memory:")
        try:
            import pandas as pd

            disk.backup(mem)
            deadline = time.monotonic() + QUERY_TIMEOUT_S
            # Non-zero from the handler aborts the query (unbounded joins otherwise
            # pin CPU forever and hang the whole scoring pass).
            mem.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0, 10_000
            )
            df = pd.read_sql_query(sql, mem)
            mem.set_progress_handler(None, 0)
            return True, df
        finally:
            mem.close()
            disk.close()
    except Exception as e:  # noqa: BLE001 — any failure = failed execution (score 0)
        return False, f"{type(e).__name__}: {e}"


def _normalize(v):
    import pandas as pd

    return 0 if pd.isna(v) else v


def _sort_key(x):
    return (x is None, str(x), isinstance(x, (int, float)))


def _vectors_match(v1, v2, ignore_order: bool, tol: float = 1e-2) -> bool:
    import pandas as pd

    v1 = [_normalize(x) for x in v1]
    v2 = [_normalize(x) for x in v2]
    if ignore_order:
        v1 = sorted(v1, key=_sort_key)
        v2 = sorted(v2, key=_sort_key)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def compare_table(pred, gold, condition_cols, ignore_order: bool) -> int:
    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold = gold.iloc[:, condition_cols]
    t_gold = gold.transpose().values.tolist()
    t_pred = pred.transpose().values.tolist()
    for gv in t_gold:
        if not any(_vectors_match(gv, pv, ignore_order) for pv in t_pred):
            return 0
    return 1


def compare_multi(pred, golds, multi_condition_cols, ignore_order: bool) -> int:
    if not golds:
        return 0
    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(s, list) for s in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in golds]
    for gold, cc in zip(golds, multi_condition_cols):
        if compare_table(pred, gold, cc, ignore_order):
            return 1
    return 0


def resolve_gold_paths(instance_id: str):
    base = _GOLD_EXEC_DIR / f"{instance_id}.csv"
    if base.exists():
        return [base], True
    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    matches = sorted(
        _GOLD_EXEC_DIR / name
        for name in os.listdir(_GOLD_EXEC_DIR)
        if pattern.match(name)
    )
    return matches, False


def load_eval_standard() -> dict:
    import json

    out = {}
    with open(_EVAL_STANDARD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["instance_id"]] = rec
    return out


def evaluate_record(instance_id: str, db: str, pred_sql: str, eval_std: dict) -> dict:
    import pandas as pd

    db_path = _SQLITE_DB_DIR / f"{db}.sqlite"
    if not db_path.exists():
        return {
            "instance_id": instance_id,
            "score": 0,
            "error": f"missing sqlite db: {db_path}",
        }
    if not (pred_sql or "").strip():
        return {"instance_id": instance_id, "score": 0, "error": "empty pred_sql"}
    ok, result = execute_sqlite(db_path, pred_sql)
    if not ok:
        return {"instance_id": instance_id, "score": 0, "error": f"sql error: {result}"}
    gold_paths, is_single = resolve_gold_paths(instance_id)
    if not gold_paths:
        return {"instance_id": instance_id, "score": 0, "error": "no gold file"}
    std = eval_std.get(instance_id, {})
    cc, ignore_order = std.get("condition_cols"), std.get("ignore_order", False)
    try:
        if is_single:
            score = compare_table(result, pd.read_csv(gold_paths[0]), cc, ignore_order)
        else:
            score = compare_multi(
                result, [pd.read_csv(p) for p in gold_paths], cc, ignore_order
            )
    except Exception as e:  # noqa: BLE001
        return {"instance_id": instance_id, "score": 0, "error": f"compare: {e}"}
    return {"instance_id": instance_id, "score": score, "error": None}


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["instance_id"]: s for s in samples}
    latest = {r["instance_id"]: r for r in records if r["instance_id"] in by_id}
    eval_std = load_eval_standard()
    per_example, correct = [], 0
    for iid, r in latest.items():
        res = evaluate_record(
            iid, by_id[iid]["db"], r.get("prediction") or "", eval_std
        )
        per_example.append(res)
        correct += res["score"]
    total = len(latest)
    total_local = len(samples)
    return {
        "accuracy_of_local_135": correct / total_local if total_local else 0.0,
        "accuracy_evaluated": correct / total if total else 0.0,
        "correct": correct,
        "total_evaluated": total,
        "total_local_subset": total_local,
        "per_example": per_example,
    }
