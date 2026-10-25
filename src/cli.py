from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
from collections import Counter

from dotenv import load_dotenv

from src.config import (
    build_routes,
    load_all_targets,
    load_target,
)
from src.results import RunStore
from src.runner import run_benchmark

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


def _live_routes(target, benchmark: str) -> list:
    """Primary route + ordered fallbacks, keeping only those whose API key is present."""
    live = []
    for rt in build_routes(target, benchmark=benchmark):
        if not rt.adapter.resolve_key():
            print(
                f"skipping route {rt.provider} (no API key; looked for {rt.adapter.key_spec})"
            )
            continue
        rt.client = rt.adapter.build_client()
        live.append(rt)
    if not live:
        raise SystemExit(f"no API key for any route of target {target.name!r}")
    return live


def _resolve_variant(bench, benchmark: str, variant: str | None) -> str | None:
    variants = getattr(bench, "VARIANTS", None)
    if not variants:
        if variant:
            raise SystemExit(f"{benchmark} has no variants")
        return None
    variant = variant or variants[0]
    if variant not in variants:
        raise SystemExit(f"--variant must be one of {variants} for {benchmark}")
    return variant


def cmd_run(args) -> None:
    target = load_target(args.target)
    bench = _import_benchmark(args.benchmark)
    mode = args.reasoning or bench.DEFAULTS.get("reasoning", "off")
    live = _live_routes(target, args.benchmark)

    variant = _resolve_variant(bench, args.benchmark, args.variant)
    result_name = f"{bench.NAME}_{variant}" if variant else bench.NAME
    print(f"Loading {result_name} samples...")
    if variant:
        samples = bench.load_samples(args.sample, variant=variant)
    else:
        samples = bench.load_samples(args.sample)

    smoke = args.sample is not None
    root = "results/_smoke" if smoke else "results"  # isolated so smokes can't clobber
    store = RunStore(result_name, target.name, root=root)
    defaults = bench.DEFAULTS

    result = asyncio.run(
        run_benchmark(
            routes=live,
            samples=samples,
            build_request=lambda s: bench.build_request(s, mode),
            parse=bench.parse,
            store=store,
            id_key=bench.ID_KEY,
            rate_limit=defaults.get("rate_limit", 25),
            max_in_flight=defaults.get("max_in_flight", 8),
            desc=f"{result_name} / {target.name}",
        )
    )

    responses = store.load_responses()
    hosts = dict(Counter(r["host"] for r in responses if r.get("host")))

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
    primary_value = metrics.get(primary) if primary else None
    headline = (
        f"{primary}={primary_value:.4f}  "
        if isinstance(primary_value, (int, float))
        else ""
    )
    print(
        f"\n{bench.NAME} / {target.name}: {headline}"
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
    load_dotenv()

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
