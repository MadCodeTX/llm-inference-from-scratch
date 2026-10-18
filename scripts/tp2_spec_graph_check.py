#!/usr/bin/env python
"""Tiny TP2 NCCL check: graph speculative == eager speculative == greedy.

Run (always timeout-wrapped so a capture hang cannot wedge the box):
  timeout 180 .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_spec_graph_check.py

Checks, on a tiny 3-linear+1-full DistributedHybridLM + TPMTP draft:
(1) captured 2-token chunk logits vs eager chunk logits on identically
prefilled states; (2) captured draft-step logits vs the functional
draft; (3) full speculative_generate_tp with graph_verify+graph_draft
emits exactly the plain greedy ids (Kiely sec. 5.2 invariant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.mtp import MTPConfig, MTPHead, TPMTPDraftModule  # noqa: E402
from inference_from_scratch.speculation import (  # noqa: E402
    _ChunkGraph, _DraftGraph, speculative_generate_tp)
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
        partial_rotary_factor=0.25, rope_theta=10000.0,
        mtp_num_hidden_layers=1)


def main() -> int:
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev).eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg).eval()
    torch.manual_seed(99)
    head = MTPHead(MTPConfig.from_hybrid_config(cfg)).to(dev).eval()
    mtp = TPMTPDraftModule.from_dense(
        ctx, head, embedding=tp.embed_tokens, lm_head=tp.lm_head).eval()
    idx = torch.tensor([[3, 11, 5, 7, 9]], device=dev)
    ok = True
    with torch.no_grad():
        # (1) chunk graph vs eager chunk.
        s_a = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s_a)
        eager_logits, _ = tp.decode_chunk_static(
            torch.tensor([[4, 9]], device=dev), s_a, return_hidden=True)
        s_b = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s_b)
        chunk_graph = _ChunkGraph(tp, s_b)
        try:
            g_logits, _g_hidden = chunk_graph.run(4, 9)
            chunk_diff = (g_logits.float() - eager_logits.float()).abs().max().item()
        finally:
            chunk_graph.close()
        print(f"[rank {rank}] chunk_graph_max_abs_diff={chunk_diff:.3e}", flush=True)
        ok &= chunk_diff < 1e-4

        # (2) draft graph vs functional draft.
        _lg, hidden_pre = tp.prefill_static(
            idx, tp.static_state(batch_size=1, max_seq_len=32),
            return_hidden=True)
        hidden_t = tp.norm(hidden_pre[0, -1])
        want_id, want_logits, _st = mtp.draft_token(
            hidden_t, 7, None, position=4,
            embedding=tp.embed_tokens, lm_head=tp.lm_head)
        mtp_static = mtp.initial_static_state(batch_size=1, max_seq_len=32)
        draft_graph = _DraftGraph(tp, mtp, mtp_static,
                                  tp.embed_tokens, tp.lm_head)
        try:
            d_logits = draft_graph.run(hidden_t, 7, 4)
            draft_diff = (d_logits.float() - want_logits.float()).abs().max().item()
            draft_id_ok = int(d_logits[0].argmax().item()) == int(want_id)
        finally:
            draft_graph.close()
        print(f"[rank {rank}] draft_graph_max_abs_diff={draft_diff:.3e} "
              f"draft_id_ok={draft_id_ok}", flush=True)
        ok &= draft_diff < 1e-4 and draft_id_ok

        # (3) full graph speculative == plain greedy == eager speculative.
        logits, state = tp.prefill(idx)
        ref = [int(logits[0, -1].argmax().item())]
        for _ in range(11):
            logits, state = tp.step(torch.tensor([ref[-1]], device=dev), state)
            ref.append(int(logits[0, 0].argmax().item()))
        out_e, stats_e = speculative_generate_tp(tp, mtp, idx, 12)
        out_g, stats_g = speculative_generate_tp(
            tp, mtp, idx, 12, graph_verify=True, graph_draft=True)
        got_e, got_g = out_e[0, 5:].tolist(), out_g[0, 5:].tolist()
        print(f"[rank {rank}] ref={ref} eager={got_e} graph={got_g}", flush=True)
        print(f"[rank {rank}] stats_eager={stats_e} stats_graph={stats_g}", flush=True)
        ok &= (got_e == ref and got_g == ref)
    ctx.barrier()
    if rank == 0:
        print("TP2_SPEC_GRAPH_CHECK " + ("PASS" if ok else "FAIL"), flush=True)
    ctx.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
