#!/usr/bin/env python
"""Real two-process distributed hybrid LM check vs dense reference.

Run under torchrun with 2 processes, GPU (NCCL) or CPU (gloo):

  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_model_check.py
  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_model_check.py --backend gloo --device cpu

Each process holds ONLY its rank's shards of a tiny 3-linear+1-full
``TinyHybridLM`` (via ``DistributedHybridLM.from_dense``) and compares
prefill + 6 decode-step logits against the dense model built from the
same seed. fp32 is asserted tight; greedy tokens must match exactly.
Prints per-rank parameter bytes proving sharding (< dense).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, DistributedTPContext)


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, rank: int,
           atol: float, rtol: float) -> float:
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    print(f"[rank {rank}] {name}: max_abs_err={err:.3e} allclose={ok}", flush=True)
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
    print(f"[rank {rank}] backend={backend} device={dev} world={ctx.world_size} dtype={args.dtype}", flush=True)

    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev, dtype)
    dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg).to(dtype)
    tp.eval()

    local_bytes = tp.local_parameter_bytes()
    dense_bytes = DistributedHybridLM.dense_parameter_bytes(dense)
    print(f"[rank {rank}] param_bytes local={local_bytes} dense={dense_bytes} "
          f"sharded_lt_dense={local_bytes < dense_bytes}", flush=True)
    if not local_bytes < dense_bytes:
        raise AssertionError(f"rank {rank} holds {local_bytes} bytes, not < dense {dense_bytes}")
    for i, blk in enumerate(tp.layers):
        if blk.layer_type == "full_attention":
            print(f"[rank {rank}] layer{i} full q={tuple(blk.self_attn.q_weight.shape)} "
                  f"o={tuple(blk.self_attn.o_weight.shape)} mlp_gate={tuple(blk.mlp.gate_proj.weight.shape)}", flush=True)
        else:
            print(f"[rank {rank}] layer{i} linear qkv={tuple(blk.linear_attn.in_proj_qkv_w.shape)} "
                  f"out={tuple(blk.linear_attn.out_proj_w.shape)} mlp_gate={tuple(blk.mlp.gate_proj.weight.shape)}", flush=True)

    torch.manual_seed(77)
    idx = torch.randint(0, cfg.vocab_size, (2, 5), device=dev)
    with torch.no_grad():
        dense_logits, dense_state = dense.prefill(idx)
        tp_logits, tp_state = tp.prefill(idx)
        _check("prefill_logits", tp_logits, dense_logits, rank, atol, rtol)
        # TPState local shapes: full KV sharded by KV heads, linear S by value heads.
        print(f"[rank {rank}] tp_state seq_len={tp_state.seq_len} entries="
              f"{[(t, None if p is None else (tuple(p[0].shape), tuple(p[1].shape))) for t, p in tp_state.entries]}", flush=True)
        # Explicit text position_ids must reproduce the default path.
        pos = torch.arange(idx.shape[1], device=dev)
        d_pos_logits, _ = dense.prefill(idx, position_ids=pos)
        t_pos_logits, _ = tp.prefill(idx, position_ids=pos)
        _check("prefill_position_ids_logits", t_pos_logits, d_pos_logits, rank, atol, rtol)
        dense_greedy = dense_logits[:, -1].argmax(dim=-1)
        tp_greedy = tp_logits[:, -1].argmax(dim=-1)
        print(f"[rank {rank}] prefill greedy dense={dense_greedy.tolist()} tp={tp_greedy.tolist()}", flush=True)
        if not torch.equal(dense_greedy, tp_greedy):
            raise AssertionError("prefill greedy tokens differ")
        next_token = dense_greedy
        for step_i in range(6):
            d_logits, dense_state = dense.step(next_token, dense_state)
            t_logits, tp_state = tp.step(next_token, tp_state)
            _check(f"decode_step{step_i}_logits", t_logits, d_logits, rank, atol, rtol)
            d_greedy = d_logits[:, -1].argmax(dim=-1)
            t_greedy = t_logits[:, -1].argmax(dim=-1)
            print(f"[rank {rank}] step{step_i} greedy dense={d_greedy.tolist()} tp={t_greedy.tolist()}", flush=True)
            if not torch.equal(d_greedy, t_greedy):
                raise AssertionError(f"decode step {step_i} greedy tokens differ")
            next_token = d_greedy

    ctx.barrier()
    if rank == 0:
        print("TP2_MODEL_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
