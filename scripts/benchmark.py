"""Benchmark (Ch 00/08): prefill+decode timing, cache on/off, vs seq len.

Prefill and decode are different workloads (Ch 00): prefill is one
parallel pass over the prompt, decode is one memory-bound step per
token. This script times both phases separately and shows how
decode throughput (tokens/sec) changes with sequence length, with
the KV cache on and off. The no-cache path re-runs the whole
sequence per token, so its per-token cost grows linearly with
length while the cached path's stays flat-ish -- that gap *is*
the reason the cache exists.

Numbers are CPU-tiny-model numbers: read the *ratios and trends*,
not the absolutes. Run: .venv/bin/python scripts/benchmark.py
"""

from __future__ import annotations

import time

import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.generate import generate
from inference_from_scratch.kv_cache import KVCache
from inference_from_scratch.metrics import measure_generate, summarize
from inference_from_scratch.model import DecoderLM


def _bench(fn, repeats: int = 3) -> float:
    """Best-of-`repeats` wall time (seconds) for fn()."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64, d_model=64, n_layer=2, n_head=4,
        n_kv_head=2, d_ff=128, max_seq_len=512,
    )
    model = DecoderLM(config).eval()
    print(f"model: params={model.num_params()} d_model={config.d_model} "
          f"n_layer={config.n_layer} (random weights; timing only)")

    # --- Prefill: one forward over T prompt tokens -------------------
    print("\nprefill (single forward over the prompt):")
    print(f"{'prompt_len':>10} {'ms':>8} {'tokens/s':>10}")
    for t in (16, 64, 128, 256):
        idx = torch.randint(0, 64, (1, t))
        with torch.no_grad():
            ms = _bench(lambda: model(idx)) * 1e3
        print(f"{t:>10} {ms:>8.2f} {t / (ms / 1e3):>10.0f}")

    # --- Decode: greedy tokens/sec vs prompt length, cache on/off ----
    print("\ndecode (greedy, 32 new tokens):")
    print(f"{'prompt_len':>10} {'cache tok/s':>12} {'no-cache tok/s':>15} {'speedup':>8}")
    for t in (8, 32, 64, 128):
        idx = torch.randint(0, 64, (1, t))
        with torch.no_grad():
            s_c = _bench(lambda: generate(
                model, idx, max_new_tokens=32, temperature=0, use_cache=True))
            s_n = _bench(lambda: generate(
                model, idx, max_new_tokens=32, temperature=0, use_cache=False))
        tps_c, tps_n = 32 / s_c, 32 / s_n
        print(f"{t:>10} {tps_c:>12.1f} {tps_n:>15.1f} {tps_c / tps_n:>7.1f}x")

    # --- Percentiles (Ch 13 / sec. 1.4): TTFT and TPS distributions --
    # Means hide the tail; report p50/p90 over repeated end-to-end runs.
    print("\npercentiles (10 runs, prompt_len=32, 16 new tokens, greedy):")
    idx = torch.randint(0, 64, (1, 32))
    ttfts_ms, tps_vals = [], []
    for _ in range(10):
        r = measure_generate(model, idx, 16, temperature=0)
        ttfts_ms.append(r["ttft_s"] * 1e3)
        tps_vals.append(r["tps"])
    ttft_sum, tps_sum = summarize(ttfts_ms), summarize(tps_vals)
    print(f"{'metric':>10} {'p50':>10} {'p90':>10} {'p99':>10} {'mean':>10}")
    print(f"{'TTFT ms':>10} {ttft_sum['p50']:>10.2f} {ttft_sum['p90']:>10.2f} "
          f"{ttft_sum['p99']:>10.2f} {ttft_sum['mean']:>10.2f}")
    print(f"{'TPS':>10} {tps_sum['p50']:>10.1f} {tps_sum['p90']:>10.1f} "
          f"{tps_sum['p99']:>10.1f} {tps_sum['mean']:>10.1f}")


if __name__ == "__main__":
    main()
