#!/usr/bin/env python3
"""CPU timing: static (preallocated) vs legacy (immutable) hybrid decode.

Times ``TinyHybridLM.decode_token_static`` over a
:class:`StaticHybridState` against the legacy ``decode_step`` path
(which concatenates KV tensors and allocates a fresh recurrent state
per token) at tiny dims on CPU, and prints one JSON object per method
plus a comparison line. The numbers are honest wall-clock medians from
``time.perf_counter`` on whatever machine runs this. No winner is
promised: on CPU at tiny dims Python overhead dominates both paths,
and the static path's real payoff (fixed addresses / shapes for CUDA
graph capture, see docs/25) is not a CPU wall-clock claim. This script
reports whichever way the timing falls, plus the max abs logit
difference so a timing line can never silently trade correctness.

Usage::

    .venv/bin/python benchmarks/microbench_hybrid_decode.py
    .venv/bin/python benchmarks/microbench_hybrid_decode.py --repeats 3 --seq-len 64

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

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402


def _tiny_config():
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=256,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
    )


def _median(times):
    ordered = sorted(times)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _time(fn, warmup, repeats):
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
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.warmup < 0 or args.seq_len <= 0 or args.batch <= 0:
        parser.error("--repeats/--seq-len/--batch must be positive and --warmup non-negative")

    torch.manual_seed(args.seed)
    print(json.dumps({
        "event": "start", "benchmark": "hybrid_decode_static_vs_legacy",
        "device": "cpu", "torch_version": torch.__version__,
        "repeats": args.repeats, "warmup": args.warmup,
        "seq_len": args.seq_len, "batch": args.batch,
        "note": "CPU wall-clock medians; no winner promised, measured values reported as-is",
    }), flush=True)

    with torch.no_grad():
        model = TinyHybridLM(_tiny_config()).eval()
        tokens = torch.randint(0, 97, (args.batch, args.seq_len))

        # Correctness anchor (untimed): same tokens, both paths.
        legacy_state = model.initial_state(args.batch)
        legacy_logits = []
        for t in range(args.seq_len):
            logits, legacy_state = model.decode_step(tokens[:, t], legacy_state)
            legacy_logits.append(logits[:, 0, :])
        static_state = model.static_state(batch_size=args.batch,
                                          max_seq_len=args.seq_len)
        static_logits = [model.decode_token_static(tokens[:, t], static_state)
                         for t in range(args.seq_len)]
        max_diff = max(
            (a - b).abs().max().item()
            for a, b in zip(legacy_logits, static_logits))

        def run_legacy():
            state = model.initial_state(args.batch)
            out = None
            for t in range(args.seq_len):
                out, state = model.decode_step(tokens[:, t], state)
            return out

        def run_static():
            static_state.reset()
            out = None
            for t in range(args.seq_len):
                out = model.decode_token_static(tokens[:, t], static_state)
            return out

        results = {}
        for label, fn in (("legacy", run_legacy), ("static", run_static)):
            times = _time(fn, args.warmup, args.repeats)
            med = _median(times)
            results[label] = med
            print(json.dumps({
                "event": "result", "label": "tiny", "method": label,
                "batch": args.batch, "seq_len": args.seq_len,
                "seconds_median": med, "seconds_all": times,
                "tokens_per_second": args.batch * args.seq_len / med if med else None,
                "max_abs_diff_vs_legacy": max_diff if label == "static" else 0.0,
            }), flush=True)

        print(json.dumps({
            "event": "comparison", "label": "tiny",
            "legacy_seconds_median": results["legacy"],
            "static_seconds_median": results["static"],
            "speedup_static_vs_legacy": (
                results["legacy"] / results["static"] if results["static"] else None),
            "max_abs_diff_vs_legacy": max_diff,
        }), flush=True)

    print(json.dumps({"event": "done", "benchmark": "hybrid_decode_static_vs_legacy"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
