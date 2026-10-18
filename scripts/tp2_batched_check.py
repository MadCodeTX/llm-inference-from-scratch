#!/usr/bin/env python
"""Real two-process batched static-decode TP check (gloo/CPU or NCCL/GPU).

Run under torchrun with 2 processes:

  .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/tp2_batched_check.py --backend gloo --device cpu

Proves, on each rank's shards of a tiny 3-linear+1-full model:
(1) B=3 ``generate_static_batched`` over different prompts equals
    three independent B=1 runs (tokens exact, prefill logits close);
(2) staggered selective reset: ``reset(rows=[1])`` mid-decode leaves
    rows 0/2 continuing exactly as an uninterrupted batched run,
    while row 1 restarted from its prompt equals a fresh B=1 run.
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


def _check(name, out, ref, rank, atol=1e-5, rtol=1e-5):
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    print(f"[rank {rank}] {name}: max_abs_err={err:.3e} allclose={ok}", flush=True)
    if not ok:
        raise AssertionError(f"{name} mismatch on rank {rank}: {err}")
    return err


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=None, choices=["nccl", "gloo"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = p.parse_args()
    backend = args.backend or ("nccl" if args.device != "cpu" else "gloo")
    device_type = args.device or ("cuda" if backend == "nccl" else "cpu")
    ctx = DistributedTPContext.from_env(backend=backend, device_type=device_type)
    rank, dev = ctx.rank, ctx.device
    print(f"[rank {rank}] backend={backend} device={dev} world={ctx.world_size}", flush=True)

    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev)
    dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg)
    tp.eval()

    prompts = torch.tensor(
        [[3, 11, 42, 7, 90], [55, 1, 2, 3, 4], [97 - 3, 20, 20, 8, 13]],
        dtype=torch.long, device=dev) % cfg.vocab_size
    max_new = 6
    with torch.no_grad():
        # (1) Batched vs three B=1 runs.
        state = tp.static_state(batch_size=3, max_seq_len=32)
        if tuple(state.position.shape) != (3,):
            raise AssertionError(f"position shape {tuple(state.position.shape)} != (3,)")
        batched = tp.generate_static_batched(prompts, max_new, state, chunk_size=2)
        print(f"[rank {rank}] batched_tokens={batched.tolist()} positions={state.positions}", flush=True)
        for r in range(3):
            s1 = tp.static_state(batch_size=1, max_seq_len=32)
            single = tp.generate_static_batched(prompts[r:r + 1], max_new, s1, chunk_size=2)
            if not torch.equal(batched[r:r + 1], single):
                raise AssertionError(f"row {r}: batched {batched[r].tolist()} != single {single[0].tolist()}")
            print(f"[rank {rank}] row{r}_batched_eq_single=True", flush=True)

        # (2) Staggered selective reset.
        total_ref_steps = 3 + int(prompts.shape[1]) + 3
        ref_state = tp.static_state(batch_size=3, max_seq_len=64)
        ref_logits0 = tp.prefill_chunkwise_static(prompts, ref_state, chunk_size=2)
        ref_logits, nxt = [], ref_logits0.argmax(dim=-1)
        for _ in range(total_ref_steps):
            lg = tp.decode_token_static(nxt, ref_state)
            ref_logits.append(lg)
            nxt = lg.argmax(dim=-1)

        st = tp.static_state(batch_size=3, max_seq_len=64)
        lg0 = tp.prefill_chunkwise_static(prompts, st, chunk_size=2)
        _check("staggered_prefill_vs_ref", lg0, ref_logits0, rank)
        nxt = lg0.argmax(dim=-1)
        for i in range(3):
            lg = tp.decode_token_static(nxt, st)
            _check(f"staggered_pre_reset_step{i}_rows02", lg[[0, 2]], ref_logits[i][[0, 2]], rank)
            nxt = lg.argmax(dim=-1)
        # Snapshot rows 0/2, reset only row 1.
        kv0_before = st.kv_k[3][[0, 2]].clone()
        s0_before = st.linear_S[0][[0, 2]].clone()
        st.reset(rows=[1])
        if st.positions[1] != 0 or st.positions[0] == 0:
            raise AssertionError(f"selective reset positions wrong: {st.positions}")
        if not (torch.equal(st.kv_k[3][[0, 2]], kv0_before)
                and torch.equal(st.linear_S[0][[0, 2]], s0_before)):
            raise AssertionError("selective reset touched rows 0/2")
        if bool(st.kv_k[3][1].abs().max().item() != 0 or st.linear_S[0][1].abs().max().item() != 0):
            raise AssertionError("selective reset did not zero row 1")
        print(f"[rank {rank}] selective_reset_rows02_untouched=True positions={st.positions}", flush=True)

        # Row 1 restarts by re-feeding its prompt token-by-token; rows 0/2 continue greedy.
        fresh = tp.static_state(batch_size=1, max_seq_len=64)
        fresh_prefill_logits = tp.prefill_static(prompts[1:2], fresh)  # [1, T, V]
        step = 3
        last_row1_logits = None
        for t in range(int(prompts.shape[1])):
            inp = torch.stack([nxt[0], prompts[1, t], nxt[2]])
            lg = tp.decode_token_static(inp, st)
            _check(f"staggered_restart_step{t}_rows02", lg[[0, 2]], ref_logits[step][[0, 2]], rank)
            last_row1_logits = lg[1:2]
            nxt = lg.argmax(dim=-1)
            step += 1
        _check("row1_restarted_vs_fresh_prefill", last_row1_logits,
               fresh_prefill_logits[:, -1], rank)
        fresh_nxt = fresh_prefill_logits[:, -1].argmax(dim=-1)
        for i in range(3):
            lg = tp.decode_token_static(nxt, st)
            fresh_lg = tp.decode_token_static(fresh_nxt, fresh)
            _check(f"staggered_continue{i}_rows02", lg[[0, 2]], ref_logits[step][[0, 2]], rank)
            _check(f"row1_continue{i}_vs_fresh", lg[1:2], fresh_lg, rank)
            nxt = lg.argmax(dim=-1)
            fresh_nxt = fresh_lg.argmax(dim=-1)
            step += 1
        print(f"[rank {rank}] staggered_reset_proof=True positions={st.positions}", flush=True)

    ctx.barrier()
    if rank == 0:
        print("TP2_BATCHED_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
