"""The single result contract: one slug, one layout, one content-based discovery."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

DEFAULT_ROOT = Path("results")


def model_slug(name: str) -> str:
    """The ONE slug rule. Strip any provider path, then collapse to
    lowercase-dash — so fireworks/openrouter/google spellings of the same model
    resolve to the same directory."""
    leaf = name.rsplit("/", 1)[-1]
    # Preserve dots/underscores/dashes (filesystem-safe and keeps version numbers
    # like "3.7" intact) so the slug matches the target's YAML `name` and no
    # un-mangling step is ever needed. Only collapse genuinely unsafe chars.
    return re.sub(r"[^a-z0-9._-]+", "-", leaf.lower()).strip("-")


class RunStore:
    def __init__(self, benchmark: str, target: str, root: Path | str = DEFAULT_ROOT):
        self.benchmark = benchmark
        self.target = target
        self.dir = Path(root) / benchmark / model_slug(target)

    @property
    def responses_path(self) -> Path:
        return self.dir / "responses.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.json"

    @property
    def run_path(self) -> Path:
        return self.dir / "run.json"

    def _ensure_dir(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def append_response(self, record: dict) -> None:
        """Append + fsync per row so a crash mid-run loses nothing already
        billed (the audit found several runners writing only at the end)."""
        self._ensure_dir()
        with open(self.responses_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_responses(self) -> list[dict]:
        if not self.responses_path.exists():
            return []
        out = []
        with open(self.responses_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def completed_ids(self, id_key: str, done: Callable[[dict], bool]) -> set:
        return {r[id_key] for r in self.load_responses() if done(r)}

    def write_metrics(self, metrics: dict) -> None:
        self._ensure_dir()
        self.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    def write_run(self, run: dict) -> None:
        self._ensure_dir()
        self.run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))


def discover(root: Path | str = DEFAULT_ROOT) -> list[dict]:
    """Every metrics.json under the layout — content-based, the single source
    the report reads."""
    root = Path(root)
    out = []
    for path in sorted(root.glob("*/*/metrics.json")):
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        out.append(data)
    return out


def load_ids(records: Iterable[dict], id_key: str) -> set:
    return {r[id_key] for r in records if id_key in r}
