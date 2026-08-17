from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect

from bench_core.config import (
    build_routes,
    load_all_targets,
    load_target,
)
from bench_core.results import RunStore
from bench_core.runner import run_benchmark

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
    # primary provider + ordered fallbacks; keep only routes whose key is present
    routes = build_routes(target, benchmark=args.benchmark)
    live = []
    for rt in routes:
        if rt.adapter.resolve_key():
            rt.client = rt.adapter.build_client()
            live.append(rt)
        else:
            print(
                f"skipping route {rt.provider} (no API key; looked for {rt.adapter.key_spec})"
            )
    if not live:
        raise SystemExit(f"no API key for any route of target {target.name!r}")

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

    root = "results" if args.sample is None else "results/_smoke" # sampled runs are smokes, so isolated
    store = RunStore(result_name, target.name, root=root)
    d = bench.DEFAULTS

    result = asyncio.run(
        run_benchmark(
            routes=live,
            samples=samples,
            build_request=lambda s: bench.build_request(s, mode),
            parse=bench.parse,
            store=store,
            id_key=bench.ID_KEY,
            rate_limit=d.get("rate_limit", 25),
            max_in_flight=d.get("max_in_flight", 8),
        )
    )

    responses = store.load_responses()
    hosts: dict = {}
    for r in responses:
        h = r.get("host")
        if h:
            hosts[h] = hosts.get(h, 0) + 1

    score_kwargs = {}
    if "target" in inspect.signature(bench.score).parameters:
        score_kwargs["target"] = target
    caps = live[0].caps  # primary (first available) route's resolved caps
    metrics = bench.score(responses, samples, **score_kwargs)
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
            "hosts": hosts,  # which provider(s) actually served the rows
            "capability_hints": result.hints,
        }
    )
    store.write_metrics(metrics)
    store.write_run(
        {
            "provider": target.provider,
            "model_id": target.model_id,
            "routes": [rt.provider for rt in live],  # providers available this run
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


def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
