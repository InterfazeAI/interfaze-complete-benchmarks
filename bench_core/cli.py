"""bench CLI: run a benchmark against a target, or list targets."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect

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
    "asr": "benchmarks.asr.bench",
    "mmmlu": "benchmarks.mmmlu.bench",
    "mmmu_pro": "benchmarks.mmmu_pro.bench",
    "refcoco": "benchmarks.obj_detection.bench",
    "ocrbench_v2": "benchmarks.ocrbench_v2.bench",
    "olmocr": "benchmarks.olmocr.harness",
    "spider2": "benchmarks.spider2_lite.bench",
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

    variants = getattr(bench, "VARIANTS", None)
    if variants:
        variant = args.variant or variants[0]
        if variant not in variants:
            raise SystemExit(
                f"--variant must be one of {variants} for {args.benchmark}"
            )
        samples = bench.load_samples(args.sample, variant=variant)
        result_name = f"{bench.NAME}_{variant}"
    elif args.variant:
        raise SystemExit(f"{args.benchmark} has no variants")
    else:
        samples = bench.load_samples(args.sample)
        result_name = bench.NAME

    print(f"Loading {result_name} samples...")
    store = RunStore(result_name, target.name)
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

    score_kwargs = {}
    if "target" in inspect.signature(bench.score).parameters:
        score_kwargs["target"] = target
    metrics = bench.score(store.load_responses(), samples, **score_kwargs)
    metrics.update(
        {
            "benchmark": result_name,
            "target": target.name,
            "provider": target.provider,
            "model_id": target.model_id,
            "n": metrics.get("total") or metrics.get("num_samples"),
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

    primary = getattr(bench, "PRIMARY_METRIC", None)
    pv = metrics.get(primary) if primary else None
    head = f"{primary}={pv:.4f}  " if isinstance(pv, (int, float)) else ""
    print(
        f"\n{bench.NAME} / {target.name}: {head}"
        f"n={metrics.get('n')}  failed={result.n_failed}"
    )
    print(f"written to {store.metrics_path}")
    if result.hints:
        print("capability hints (promote to targets.yaml):")
        for h in result.hints:
            print("  -", h)


def cmd_list_targets(args) -> None:
    for name, t in sorted(load_all_targets().items()):
        print(f"{name:22} {t.provider:11} {t.model_id}")


def cmd_migrate_results(args) -> None:
    from bench_core.migrate import migrate_results

    log = migrate_results(dry_run=args.dry_run)
    for line in log:
        print(line)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(log)} action(s)")


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
        "--variant",
        default=None,
        help="benchmark variant, e.g. mmmu_pro: standard|vision",
    )
    r.add_argument(
        "--sample", type=int, default=None, help="run only the first N samples"
    )
    r.set_defaults(func=cmd_run)

    lt = sub.add_parser("list-targets", help="list configured targets")
    lt.set_defaults(func=cmd_list_targets)

    mr = sub.add_parser(
        "migrate-results", help="backfill legacy flat results into the new contract"
    )
    mr.add_argument(
        "--dry-run", action="store_true", help="show actions without writing"
    )
    mr.set_defaults(func=cmd_migrate_results)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
