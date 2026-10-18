#!/usr/bin/env python
"""Real two-process static-decode TP check vs legacy step + dense.

Run under torchrun with 2 processes, GPU (NCCL) or CPU (gloo):

  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_static_check.py
  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_static_check.py --backend gloo --device cpu

Each process holds ONLY its rank's shards of a tiny 3-linear+1-full
``TinyHybridLM`` (via ``DistributedHybridLM.from_dense``) and proves
``TPStaticState`` + ``decode_token_static`` / ``prefill_static``
against the existing distributed ``prefill``/``step`` path and the
dense model: fp32 tight, greedy tokens exact, state pointers stable
across steps and reset, and overflow raising before any mutation.
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

    torch.manual_seed(77)
    idx = torch.randint(0, cfg.vocab_size, (2, 5), device=dev)
    with torch.no_grad():
        dense_logits, dense_state = dense.prefill(idx)
        tp_logits, tp_state = tp.prefill(idx)
        _check("prefill_logits", tp_logits, dense_logits, rank, atol, rtol)

        state = tp.static_state(batch_size=2, max_seq_len=16)
        print(f"[rank {rank}] static kv={tuple(state.kv_k[3].shape)} "
              f"S={tuple(state.linear_S[0].shape)} conv={tuple(state.linear_conv_buffers[0].shape)}", flush=True)
        static_logits = tp.prefill_static(idx, state)
        _check("prefill_static_vs_tp_prefill", static_logits, tp_logits, rank, atol, rtol)
        _check("prefill_static_vs_dense", static_logits, dense_logits, rank, atol, rtol)
        ptrs0 = state.data_ptrs()

        out_buf = torch.empty(2, cfg.vocab_size, device=dev, dtype=dtype)
        next_token = dense_logits[:, -1].argmax(dim=-1)
        first_run: list[torch.Tensor] = []
        for step_i in range(6):
            d_logits, dense_state = dense.step(next_token, dense_state)
            t_logits, tp_state = tp.step(next_token, tp_state)
            s_logits = tp.decode_token_static(next_token, state, out=out_buf)
            if s_logits is not out_buf:
                raise AssertionError("decode_token_static did not return the out buffer")
            _check(f"static_step{step_i}_vs_tp_step", s_logits, t_logits[:, 0], rank, atol, rtol)
            _check(f"static_step{step_i}_vs_dense", s_logits, d_logits[:, 0], rank, atol, rtol)
            first_run.append(s_logits.clone())
            d_greedy = d_logits[:, 0].argmax(dim=-1)
            s_greedy = s_logits.argmax(dim=-1)
            print(f"[rank {rank}] step{step_i} greedy dense={d_greedy.tolist()} static={s_greedy.tolist()}", flush=True)
            if not torch.equal(d_greedy, s_greedy):
                raise AssertionError(f"static decode step {step_i} greedy tokens differ")
            next_token = d_greedy
        if state.data_ptrs() != ptrs0:
            raise AssertionError("state data_ptrs moved across decode steps")
        print(f"[rank {rank}] ptrs_stable_across_steps=True seq_len={state.seq_len}", flush=True)

        # Reset zeroes in place: rerun the same greedy chain identically.
        state.reset()
        if state.data_ptrs() != ptrs0 or state.seq_len != 0:
            raise AssertionError("reset moved pointers or left position nonzero")
        tp.prefill_static(idx, state)
        next_token = dense_logits[:, -1].argmax(dim=-1)
        # Recompute the dense/legacy chain is unnecessary: static rerun
        # must reproduce the first static run token-for-token.
        dense_logits2, dense_state2 = dense.prefill(idx)
        for step_i in range(6):
            d_logits, dense_state2 = dense.step(next_token, dense_state2)
            s_logits = tp.decode_token_static(next_token, state)
            _check(f"static_rerun_step{step_i}", s_logits, first_run[step_i], rank, atol, rtol)
            next_token = d_logits[:, 0].argmax(dim=-1)
        print(f"[rank {rank}] reset_rerun_equal=True", flush=True)

        # Overflow raises BEFORE mutation (position + buffers unchanged).
        small = tp.static_state(batch_size=2, max_seq_len=3)
        tp.prefill_static(idx[:, :3], small)
        pos_before = int(small.position.item())
        kv_before = small.kv_k[3].clone()
        s_before = small.linear_S[0].clone()
        try:
            tp.decode_token_static(next_token, small)
        except ValueError as exc:
            print(f"[rank {rank}] overflow_raised=True msg={exc}", flush=True)
        else:
            raise AssertionError("decode past max_seq_len did not raise")
        if int(small.position.item()) != pos_before:
            raise AssertionError("overflow mutated position")
        if not (torch.equal(small.kv_k[3], kv_before) and torch.equal(small.linear_S[0], s_before)):
            raise AssertionError("overflow mutated state buffers")
        print(f"[rank {rank}] overflow_no_mutation=True position={pos_before}", flush=True)

    ctx.barrier()
    if rank == 0:
        print("TP2_STATIC_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
