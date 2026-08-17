"""Append a target to src/targets.yaml from UI/CLI inputs, then validate.

Used by the add-target GitHub workflow (which opens a PR) and locally:

    uv run python scripts/add_target.py --name my-model --provider fireworks \\
        --model-id accounts/fireworks/models/my-model \\
        --capabilities-json '{"reasoning":{"style":"effort","off_value":"none","on_value":"high","true_off":false}}' \\
        --ci-regression

The entry is appended as text (preserving the file's comments), then the whole
file is re-parsed and the new target must resolve to typed capabilities + build
its adapter — otherwise nothing is written and it exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    PROVIDERS,
    build_adapter,
    load_all_targets,
    resolve_capabilities,
)

TARGETS_FILE = ROOT / "src" / "targets.yaml"


def _entry_text(name, provider, model_id, capabilities, ci_regression) -> str:
    lines = [f"  {name}:", f"    provider: {provider}", f"    model_id: {model_id}"]
    if ci_regression:
        lines.append("    ci_regression: true")
    if capabilities:
        block = yaml.safe_dump(
            {"capabilities": capabilities}, sort_keys=False, default_flow_style=False
        )
        lines += ["    " + ln for ln in block.rstrip("\n").splitlines()]
    return "\n".join(lines) + "\n"


def add_target(
    name,
    provider,
    model_id,
    capabilities_json="",
    ci_regression=False,
    path=TARGETS_FILE,
) -> None:
    path = Path(path)
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
    existing = load_all_targets(path)
    if name in existing:
        raise SystemExit(f"target {name!r} already exists — edit it directly instead")
    capabilities = json.loads(capabilities_json) if capabilities_json.strip() else None

    current = path.read_text()
    new_text = (
        current.rstrip("\n")
        + "\n"
        + _entry_text(name, provider, model_id, capabilities, ci_regression)
    )

    # validate on a temp copy before touching the real file
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(new_text)
        tmp = Path(tf.name)
    try:
        targets = load_all_targets(tmp)
        if name not in targets:
            raise SystemExit(f"target {name!r} did not parse from the appended entry")
        resolve_capabilities(targets[name])  # merges + type-checks capabilities
        build_adapter(targets[name])  # provider resolves to an adapter
    finally:
        tmp.unlink(missing_ok=True)

    path.write_text(new_text)
    print(f"added target {name!r} ({provider} / {model_id}) to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--capabilities-json", default="")
    ap.add_argument("--ci-regression", action="store_true")
    args = ap.parse_args()
    add_target(
        args.name,
        args.provider,
        args.model_id,
        args.capabilities_json,
        args.ci_regression,
    )


if __name__ == "__main__":
    main()
