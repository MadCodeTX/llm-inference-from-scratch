#!/usr/bin/env python3
"""CPU timing: sequential vs chunkwise gated-delta prefill.

Times ``GatedDeltaNet.forward_prefill(method="sequential")`` against
``method="chunkwise"`` at tiny and medium dims on CPU and prints one
JSON object per (config, method) plus a comparison line. The numbers
are honest wall-clock medians from ``time.perf_counter`` on whatever
machine runs this. No winner is promised: on CPU the triangular solve
has fixed overhead while the sequential form pays a Python per-token
loop, so either can come out ahead at these sizes; the production GPU
advantage (kernel fusion / parallelism) is not measured here. This
script reports whichever way the timing falls.

Also reports the max abs output difference between the two methods for
the timed input, so a timing line can never silently trade correctness.

Usage::

    .venv/bin/python benchmarks/microbench_linear_attn.py
    .venv/bin/python benchmarks/microbench_linear_attn.py --repeats 3 --chunk-sizes 16 64

No GPU is touched; everything runs on CPU tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from inference_from_scratch.linear_attention import GatedDeltaNet  # noqa: E402

CONFIGS = [
    {
        "label": "tiny",
        "batch": 1, "seq_len": 128,
        "d_model": 32, "num_key_heads": 2, "key_dim": 8,
        "num_value_heads": 4, "value_dim": 8, "conv_kernel": 4,
    },
    {
        "label": "medium",
        "batch": 1, "seq_len": 256,
        "d_model": 128, "num_key_heads": 4, "key_dim": 32,
        "num_value_heads": 8, "value_dim": 32, "conv_kernel": 4,
    },
]


def _median(times: list[float]) -> float:
    ordered = sorted(times)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _time(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup non-negative")
    if any(c <= 0 for c in args.chunk_sizes):
        parser.error("--chunk-sizes must be positive")

    torch.manual_seed(args.seed)
    print(json.dumps({
        "event": "start", "benchmark": "linear_attn_prefill", "device": "cpu",
        "torch_version": torch.__version__, "repeats": args.repeats,
        "warmup": args.warmup, "note": "CPU wall-clock medians; no winner promised, measured values reported as-is",
    }), flush=True)

    with torch.no_grad():
        for cfg in CONFIGS:
            model_kwargs = {k: cfg[k] for k in (
                "d_model", "num_key_heads", "key_dim",
                "num_value_heads", "value_dim", "conv_kernel",
            )}
            model = GatedDeltaNet(**model_kwargs).eval()
            x = torch.randn(cfg["batch"], cfg["seq_len"], cfg["d_model"])
            seq_out, _ = model.forward_prefill(x, method="sequential")
            results: dict[str, float] = {}

            def run_sequential():
                return model.forward_prefill(x, method="sequential")

            times = _time(run_sequential, args.warmup, args.repeats)
            med = _median(times)
            results["sequential"] = med
            print(json.dumps({
                "event": "result", "label": cfg["label"], "method": "sequential",
                "batch": cfg["batch"], "seq_len": cfg["seq_len"], **model_kwargs,
                "seconds_median": med, "seconds_all": times,
                "tokens_per_second": cfg["batch"] * cfg["seq_len"] / med if med else None,
            }), flush=True)

            for chunk_size in args.chunk_sizes:
                chunk_out, _ = model.forward_prefill(
                    x, method="chunkwise", chunk_size=chunk_size
                )
                max_diff = (seq_out - chunk_out).abs().max().item()

                def run_chunkwise(cs=chunk_size):
                    return model.forward_prefill(x, method="chunkwise", chunk_size=cs)

                times = _time(run_chunkwise, args.warmup, args.repeats)
                med = _median(times)
                results[f"chunkwise_{chunk_size}"] = med
                print(json.dumps({
                    "event": "result", "label": cfg["label"], "method": "chunkwise",
                    "chunk_size": chunk_size,
                    "batch": cfg["batch"], "seq_len": cfg["seq_len"], **model_kwargs,
                    "seconds_median": med, "seconds_all": times,
                    "tokens_per_second": cfg["batch"] * cfg["seq_len"] / med if med else None,
                    "max_abs_diff_vs_sequential": max_diff,
                }), flush=True)

            comparison = {
                "event": "comparison", "label": cfg["label"],
                "sequential_seconds_median": results["sequential"],
            }
            for chunk_size in args.chunk_sizes:
                key = f"chunkwise_{chunk_size}"
                comparison[f"speedup_chunkwise_{chunk_size}_vs_sequential"] = (
                    results["sequential"] / results[key] if results[key] else None
                )
            print(json.dumps(comparison), flush=True)

    print(json.dumps({"event": "done", "benchmark": "linear_attn_prefill"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
