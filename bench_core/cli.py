"""bench CLI: run a benchmark against a target, or list targets."""

from __future__ import annotations

import argparse
import asyncio
import importlib

from bench_core.config import (
    build_adapter,
    load_all_targets,
    load_target,
    resolve_capabilities,
)
from bench_core.results import RunStore
from bench_core.runner import run_benchmark

# benchmark short-name -> module exposing NAME/load_samples/build_request/parse/score
BENCHMARKS = {
    "gpqa": "benchmarks.gpqa.bench",
}


def _import_benchmark(name: str):
    if name not in BENCHMARKS:
        raise SystemExit(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return importlib.import_module(BENCHMARKS[name])


def cmd_run(args) -> None:
    target = load_target(args.target)
    bench = _import_benchmark(args.benchmark)
    mode = args.reasoning or bench.DEFAULTS.get("reasoning", "off")
    caps = resolve_capabilities(target, benchmark=args.benchmark)
    adapter = build_adapter(target)
    if not adapter.resolve_key():
        raise SystemExit(
            f"no API key for provider {target.provider!r} (looked for {adapter.key_spec})"
        )
    client = adapter.build_client()

    print(f"Loading {args.benchmark} samples...")
    samples = bench.load_samples(args.sample)
    store = RunStore(bench.NAME, target.name)
    d = bench.DEFAULTS

    result = asyncio.run(
        run_benchmark(
            adapter=adapter,
            client=client,
            caps=caps,
            model_id=target.model_id,
            samples=samples,
            build_request=lambda s: bench.build_request(s, mode),
            parse=bench.parse,
            store=store,
            id_key=bench.ID_KEY,
            rate_limit=d.get("rate_limit", 25),
            max_in_flight=d.get("max_in_flight", 8),
        )
    )

    metrics = bench.score(store.load_responses(), samples)
    metrics.update(
        {
            "benchmark": bench.NAME,
            "target": target.name,
            "provider": target.provider,
            "model_id": target.model_id,
            "n": metrics.get("total"),
            "reasoning": {
                "mode": mode,
                "style": caps.reasoning.style.value,
                "true_off": caps.reasoning.true_off,
            },
            "capability_hints": result.hints,
        }
    )
    store.write_metrics(metrics)
    store.write_run(
        {
            "provider": target.provider,
            "model_id": target.model_id,
            "reasoning_mode": mode,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "sample_size": args.sample,
        }
    )

    acc = metrics.get("accuracy")
    print(
        f"\n{bench.NAME} / {target.name}: "
        + (f"accuracy={acc:.4f} " if acc is not None else "")
        + f"({metrics.get('correct')}/{metrics.get('total')})  failed={result.n_failed}"
    )
    print(f"written to {store.metrics_path}")
    if result.hints:
        print("capability hints (promote to targets.yaml):")
        for h in result.hints:
            print("  -", h)


def cmd_list_targets(args) -> None:
    for name, t in sorted(load_all_targets().items()):
        print(f"{name:22} {t.provider:11} {t.model_id}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark against a target")
    r.add_argument("--target", required=True, help="target name (see list-targets)")
    r.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    r.add_argument(
        "--reasoning", default=None, help="off|low|medium|high (default: benchmark's)"
    )
    r.add_argument(
        "--sample", type=int, default=None, help="run only the first N samples"
    )
    r.set_defaults(func=cmd_run)

    lt = sub.add_parser("list-targets", help="list configured targets")
    lt.set_defaults(func=cmd_list_targets)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
