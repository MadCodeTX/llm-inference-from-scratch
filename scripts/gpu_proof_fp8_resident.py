#!/usr/bin/env python3
"""GPU proof: resident-FP8 layer-3 q_proj vs CPU dequant reference.

Loads the real checkpoint tensor
``model.language_model.layers.3.self_attn.q_proj.weight`` (block FP8,
[128, 128] scales) from the local Qwen3.8-27B-FP8 snapshot into a
:class:`BlockFP8Linear` on ``cuda:0``, runs one forward, and compares
against the same module on CPU (the dequantize-then-matmul reference)
on an identical input. Prints the measured max-abs diff and peak GPU
memory; every number printed comes from this run.

Run inside an approved GPU window (incumbent stopped first via
``/home/nick/llm/vllm/launch.sh --stop``)::

    .venv-gpu/bin/python scripts/gpu_proof_fp8_resident.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.fp8_linear import BlockFP8Linear
from inference_from_scratch.hybrid_checkpoint import (
    _load_one, build_manifest, load_tensor,
)
from inference_from_scratch.quantization import dequantize_block_fp8

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)
CHECKPOINT_NAME = "model.language_model.layers.3.self_attn.q_proj.weight"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    snapshot = Path(args.snapshot)
    manifest = build_manifest(snapshot)
    entry = manifest.by_name()[CHECKPOINT_NAME]
    out_features, in_features = (int(entry.shape[0]), int(entry.shape[1]))
    print(f"checkpoint tensor: {CHECKPOINT_NAME} shape={entry.shape} "
          f"dtype={entry.dtype} shard={entry.shard}")

    raw = load_tensor(snapshot, CHECKPOINT_NAME, dequantize=False, manifest=manifest)
    scale = _load_one(snapshot, CHECKPOINT_NAME + "_scale_inv", manifest=manifest)
    print(f"raw weight: dtype={raw.dtype} shape={tuple(raw.shape)}; "
          f"scale: dtype={scale.dtype} shape={tuple(scale.shape)}")

    def build(device: str) -> BlockFP8Linear:
        module = BlockFP8Linear(in_features, out_features, bias=False, device=device)
        with torch.no_grad():
            module.weight.copy_(raw.to(device))
            module.weight_scale_inv.copy_(
                scale.to(dtype=torch.bfloat16, device=device))
        return module.eval()

    gen = torch.Generator().manual_seed(args.seed)
    x_cpu = torch.randn(args.rows, in_features, generator=gen).to(torch.bfloat16)

    cpu_mod = build("cpu")
    with torch.no_grad():
        ref = cpu_mod(x_cpu)
        dense_w = dequantize_block_fp8(
            raw, scale, out_dtype=torch.bfloat16)
        dense_ref = x_cpu @ dense_w.T
    print(f"cpu module vs direct dequant matmul max_abs="
          f"{(ref - dense_ref).abs().max().item():.6g}")

    dev_idx = torch.device(args.device).index or 0
    gpu_mod = build(args.device)
    x_gpu = x_cpu.to(args.device)
    with torch.no_grad():
        gpu_mod(x_gpu)  # warmup: initialises the CUDA context
        torch.cuda.synchronize(dev_idx)
        torch.cuda.reset_peak_memory_stats(dev_idx)
        got_gpu = gpu_mod(x_gpu)
        torch.cuda.synchronize(dev_idx)
    got = got_gpu.cpu()
    diff = (got.float() - ref.float()).abs()
    peak = torch.cuda.max_memory_allocated(dev_idx)
    fp8_bytes = raw.numel() + scale.numel() * 2
    bf16_bytes = raw.numel() * 2
    print(f"device={args.device} ({torch.cuda.get_device_name(0)}) "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"input: shape={tuple(x_cpu.shape)} dtype=bf16 seed={args.seed}")
    print(f"output: shape={tuple(got.shape)} finite={bool(torch.isfinite(got).all())}")
    print(f"cuda vs cpu-dequant max_abs={diff.max().item():.6g} "
          f"mean_abs={diff.mean().item():.6g}")
    print(f"peak_memory_allocated={peak} B ({peak / 2**20:.2f} MiB)")
    print(f"resident weight bytes: fp8+scale={fp8_bytes} vs bf16={bf16_bytes}")
    ok = bool(torch.isfinite(got).all()) and diff.max().item() < 0.05
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
