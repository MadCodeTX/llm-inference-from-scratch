"""Bench reference dequant vs Triton fused FP8 vs scaled_mm-if-valid.

Run with the GPU env:
  .venv-gpu/bin/python benchmarks/fp8_linear_bench.py

Reports per (shape, M): ms, effective GB/s (FP8 weight bytes + scale
bytes + activation/output bytes, divided by time), and max/rel diff
of Triton vs the dequant reference. ``torch._scaled_mm`` is only run
when the checkpoint scale layout passes
``scaled_mm_block_supported``; otherwise its rejection reason and the
probe evidence are printed (see src/inference_from_scratch/fp8_triton.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.fp8_linear import scaled_mm_block_supported
from inference_from_scratch.fp8_triton import fp8_triton_linear, triton_available
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8

SHAPES = [  # (label, N out_features, K in_features) from Qwen3.8-27B
    ("gate_up", 17408, 5120),
    ("down", 5120, 17408),
    ("qkv", 10240, 5120),
]
MS = [1, 4, 16, 128]


def _time(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps  # ms


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable; bench requires GPU")
        return
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
          f"triton_available={triton_available()}")
    supported, reason = scaled_mm_block_supported(
        torch.ones(2, 2, dtype=torch.bfloat16)
    )
    print(f"scaled_mm checkpoint-layout supported={supported}: {reason}")
    if not supported:
        print("scaled_mm: NOT RUN (invalid for checkpoint [128,128] BF16 "
              "scale grid on SM89; probe raised 'Invalid scaling "
              "configuration' for plain/float/transposed/padded layouts)")

    for label, n, k in SHAPES:
        torch.manual_seed(0)
        w = torch.randn(n, k) * 0.05
        wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
        wq, ws_bf = wq.cuda(), ws.cuda().to(torch.bfloat16)
        weight_bytes = wq.numel() + ws_bf.numel() * 2
        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

            def ref_fn():
                return F.linear(
                    x, dequantize_block_fp8(wq, ws_bf, out_dtype=torch.bfloat16)
                )

            ref = ref_fn()
            ref_ms = _time(ref_fn)
            total_bytes = weight_bytes + x.numel() * 2 + m * n * 2

            line = (
                f"{label} N={n} K={k} M={m}: ref {ref_ms:.4f} ms "
                f"{total_bytes / ref_ms / 1e6:.1f} GB/s"
            )
            if triton_available():
                got = fp8_triton_linear(x, wq, ws_bf)
                diff = (got.float() - ref.float()).abs().max().item()
                rel = diff / ref.float().abs().max().item()
                tri_ms = _time(lambda: fp8_triton_linear(x, wq, ws_bf))
                line += (
                    f" | triton {tri_ms:.4f} ms "
                    f"{total_bytes / tri_ms / 1e6:.1f} GB/s "
                    f"speedup {ref_ms / tri_ms:.2f}x "
                    f"maxdiff {diff:.5f} rel {rel:.2e}"
                )
            print(line, flush=True)
        del w, wq, ws, ws_bf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
