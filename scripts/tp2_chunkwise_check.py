#!/usr/bin/env python
"""Real two-process chunkwise-prefill TP check (gloo CPU / NCCL GPU).

Run under torchrun with 2 processes:

  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_chunkwise_check.py --backend gloo --device cpu
  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_chunkwise_check.py --backend nccl --device cuda

Proves ``prefill_chunkwise_static`` (chunk 4) against ``prefill_static``
(token loop) and the dense model on a tiny 3-linear+1-full model:
last logits fp32-close, state contents (KV filled region, S, conv)
close, greedy continuation (6 tokens via ``decode_token_static``)
identical, and non-empty-state continuation equal to one-shot.
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
    args = p.parse_args()
    backend = args.backend or ("nccl" if args.device == "cuda" else "gloo")
    device_type = args.device or ("cuda" if backend == "nccl" else "cpu")
    atol = rtol = 1e-5

    ctx = DistributedTPContext.from_env(backend=backend, device_type=device_type)
    rank, dev = ctx.rank, ctx.device
    print(f"[rank {rank}] backend={backend} device={dev} world={ctx.world_size}", flush=True)

    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev)
    dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg)
    tp.eval()

    torch.manual_seed(77)
    idx = torch.randint(0, cfg.vocab_size, (2, 11), device=dev)
    with torch.no_grad():
        dense_logits, dense_state = dense.prefill(idx)
        static_state = tp.static_state(batch_size=2, max_seq_len=32)
        static_logits = tp.prefill_static(idx, static_state)

        state = tp.static_state(batch_size=2, max_seq_len=32)
        ptrs0 = state.data_ptrs()
        chunk_logits = tp.prefill_chunkwise_static(idx, state, chunk_size=4)
        _check("chunkwise_vs_static", chunk_logits, static_logits[:, -1], rank, atol, rtol)
        _check("chunkwise_vs_dense", chunk_logits, dense_logits[:, -1], rank, atol, rtol)

        filled = int(idx.shape[1])
        for i in range(len(cfg.layer_types)):
            if state.kv_k[i] is not None:
                _check(f"state_kv_k_layer{i}", state.kv_k[i][:, :, :filled],
                       static_state.kv_k[i][:, :, :filled], rank, atol, rtol)
                _check(f"state_kv_v_layer{i}", state.kv_v[i][:, :, :filled],
                       static_state.kv_v[i][:, :, :filled], rank, atol, rtol)
            if state.linear_S[i] is not None:
                _check(f"state_S_layer{i}", state.linear_S[i],
                       static_state.linear_S[i], rank, atol, rtol)
                _check(f"state_conv_layer{i}", state.linear_conv_buffers[i],
                       static_state.linear_conv_buffers[i], rank, atol, rtol)
        print(f"[rank {rank}] state_contents: allclose=True filled={filled}", flush=True)
        if state.data_ptrs() != ptrs0:
            raise AssertionError("state data_ptrs moved across chunkwise prefill")
        print(f"[rank {rank}] ptrs_stable=True seq_len={state.seq_len}", flush=True)

        # Greedy continuation: 6 decode_token_static tokens, both states.
        nxt = dense_logits[:, -1].argmax(dim=-1)
        if not torch.equal(chunk_logits.argmax(dim=-1), nxt):
            raise AssertionError("chunkwise greedy first token differs from dense")
        for step_i in range(6):
            d_logits, dense_state = dense.step(nxt, dense_state)
            s_out = tp.decode_token_static(nxt, static_state)
            c_out = tp.decode_token_static(nxt, state)
            _check(f"greedy_step{step_i}_chunk_vs_static", c_out, s_out, rank, atol, rtol)
            if not torch.equal(c_out.argmax(dim=-1), d_logits[:, 0].argmax(dim=-1)):
                raise AssertionError(f"greedy step {step_i} token differs from dense")
            nxt = d_logits[:, 0].argmax(dim=-1)
        print(f"[rank {rank}] greedy_continuation: tokens_identical=True steps=6", flush=True)

        # Non-empty state: static prefix (5) + chunkwise rest == one-shot.
        split_state = tp.static_state(batch_size=2, max_seq_len=32)
        tp.prefill_static(idx[:, :5], split_state)
        cont = tp.prefill_chunkwise_static(idx[:, 5:], split_state, chunk_size=4)
        one_state = tp.static_state(batch_size=2, max_seq_len=32)
        one = tp.prefill_chunkwise_static(idx, one_state, chunk_size=4)
        _check("nonempty_continuation", cont, one, rank, atol, rtol)
        _check("nonempty_S", split_state.linear_S[0], one_state.linear_S[0], rank, atol, rtol)
        _check("nonempty_kv", split_state.kv_k[3][:, :, :filled],
               one_state.kv_k[3][:, :, :filled], rank, atol, rtol)

    ctx.barrier()
    if rank == 0:
        print("TP2_CHUNKWISE_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
