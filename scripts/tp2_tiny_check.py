#!/usr/bin/env python
"""Real two-process TP check: distributed shards vs dense reference.

Run under torchrun with 2 processes, GPU (NCCL) or CPU (gloo):

  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_tiny_check.py
  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_tiny_check.py --backend gloo --device cpu

Each process holds ONLY its rank's shards (via
``src/inference_from_scratch/tp_distributed.py``) and compares against
a dense single-device reference built from the same seed/weights.
fp32 is asserted tight; bf16 max error is reported and asserted loose.
Prints per-rank shard shapes and peak memory. Exits non-zero on mismatch.
"""
from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import GatedFullAttention, HybridConfig  # noqa: E402
from inference_from_scratch.linear_attention import GatedDeltaNet  # noqa: E402
from inference_from_scratch.mlp import SwiGLU  # noqa: E402
from inference_from_scratch.tensor_parallel import (  # noqa: E402
    merge_linear_state, shard_linear_state)
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedFullAttention, DistributedGatedDeltaNet, DistributedSwiGLU,
    DistributedTPContext)


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, rank: int,
           atol: float, rtol: float) -> float:
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    print(f"[rank {rank}] {name}: max_abs_err={err:.3e} allclose(atol={atol})={ok}", flush=True)
    if not ok:
        raise AssertionError(f"{name} mismatch on rank {rank}: max_abs_err={err}")
    return err


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=None, choices=["nccl", "gloo"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = p.parse_args()
    backend = args.backend or ("nccl" if args.device != "cpu" else "gloo")
    device_type = args.device or ("cuda" if backend == "nccl" else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5

    ctx = DistributedTPContext.from_env(backend=backend, device_type=device_type)
    rank, dev = ctx.rank, ctx.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    print(f"[rank {rank}] backend={backend} device={dev} world={ctx.world_size} dtype={args.dtype}", flush=True)

    # --- tiny SwiGLU MLP ---
    torch.manual_seed(101)
    mlp_dense = SwiGLU(32, 64).to(dev, dtype)
    mlp_tp = DistributedSwiGLU.from_dense(mlp_dense, ctx).to(dtype)
    torch.manual_seed(102)
    x_mlp = torch.randn(2, 4, 32, device=dev, dtype=dtype)
    with torch.no_grad():
        _check("mlp", mlp_tp(x_mlp), mlp_dense(x_mlp), rank, atol, rtol)
    print(f"[rank {rank}] mlp shards: gate={tuple(mlp_tp.gate_proj.weight.shape)} "
          f"up={tuple(mlp_tp.up_proj.weight.shape)} down={tuple(mlp_tp.down_proj.weight.shape)}", flush=True)

    # --- tiny GatedFullAttention ---
    cfg = _tiny_cfg()
    torch.manual_seed(201)
    attn_dense = GatedFullAttention(cfg).to(dev, dtype)
    attn_dense.eval()
    attn_tp = DistributedFullAttention(ctx, attn_dense, config=cfg).to(dtype)
    torch.manual_seed(202)
    x_attn = torch.randn(2, 4, 32, device=dev, dtype=dtype)
    with torch.no_grad():
        _check("attention", attn_tp(x_attn), attn_dense(x_attn), rank, atol, rtol)
    print(f"[rank {rank}] attn shards: q={tuple(attn_tp.q_weight.shape)} k={tuple(attn_tp.k_weight.shape)} "
          f"v={tuple(attn_tp.v_weight.shape)} o={tuple(attn_tp.o_weight.shape)} "
          f"q_heads={attn_tp.q_heads_local} kv_heads={attn_tp.kv_heads_local}", flush=True)

    # --- GatedDeltaNet decode step (lifted per-rank step) ---
    torch.manual_seed(301)
    delta_dense = GatedDeltaNet(d_model=32, num_key_heads=2, key_dim=8,
                                num_value_heads=4, value_dim=8, conv_kernel=4).to(dev, dtype)
    delta_dense.eval()
    delta_tp = DistributedGatedDeltaNet(ctx, delta_dense, config=cfg).to(dtype)
    torch.manual_seed(302)
    x0 = torch.randn(2, 32, device=dev, dtype=dtype)
    x1 = torch.randn(2, 32, device=dev, dtype=dtype)
    with torch.no_grad():
        _, state = delta_dense.step(x0, None)
        shard = shard_linear_state(state, rank, ctx.world_size)
        shard = (shard[0].to(dev), shard[1].to(dev))
        exp_out, exp_state = delta_dense.step(x1, state)
        tp_out, tp_state = delta_tp.step(x1, shard)
        _check("delta_step", tp_out, exp_out, rank, atol, rtol)
        # Gather per-rank states and compare merged state with dense.
        s_parts = ctx.all_gather(tp_state[0].contiguous())
        conv_parts = [t for t in ctx.all_gather(tp_state[1].contiguous())]
        merged = merge_linear_state(list(zip(s_parts, conv_parts)))
        _check("delta_state_S", merged[0], exp_state[0], rank, atol, rtol)
        _check("delta_state_conv", merged[1], exp_state[1], rank, atol, rtol)
    print(f"[rank {rank}] delta shards: {delta_tp.shard_shapes} "
          f"S_r={tuple(tp_state[0].shape)} conv_r={tuple(tp_state[1].shape)}", flush=True)

    # --- collective sanity: all_gather_concat_dim0 ---
    t = torch.full((rank + 1,), float(rank), device=dev, dtype=dtype)
    gathered = ctx.all_gather_concat_dim0(t)
    assert gathered.shape[0] == sum(range(1, ctx.world_size + 1)), gathered.shape
    print(f"[rank {rank}] all_gather_concat_dim0: in={tuple(t.shape)} out={tuple(gathered.shape)}", flush=True)

    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated(dev)
        print(f"[rank {rank}] peak_memory_bytes={peak} ({peak / 2**20:.1f} MiB)", flush=True)
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(f"[rank {rank}] peak_memory_bytes={peak} (ru_maxrss)", flush=True)
    ctx.barrier()
    if rank == 0:
        print("TP2_TINY_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
