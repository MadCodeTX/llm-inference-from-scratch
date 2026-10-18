#!/usr/bin/env python
"""Tiny TP2 NCCL check: graph2 == eager spec == plain greedy.

Run: timeout 180 .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_spec_graph2_check.py
Random tiny MTP gives frequent rejects, so rollback+refeed cycles run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.mtp import MTPConfig, MTPHead, TPMTPDraftModule  # noqa: E402
from inference_from_scratch.speculation import speculative_generate_tp  # noqa: E402
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
        logits, state = tp.prefill(idx)
        ref = [int(logits[0, -1].argmax().item())]
        for _ in range(11):
            logits, state = tp.step(torch.tensor([ref[-1]], device=dev), state)
            ref.append(int(logits[0, 0].argmax().item()))
        out_e, stats_e = speculative_generate_tp(tp, mtp, idx, 12)
        out_g, stats_g = speculative_generate_tp(
            tp, mtp, idx, 12, graph_verify=True, graph_draft=True)
        out_2, stats_2 = speculative_generate_tp(tp, mtp, idx, 12, graph2=True)
        got_e, got_g, got_2 = (out_e[0, 5:].tolist(), out_g[0, 5:].tolist(),
                               out_2[0, 5:].tolist())
        print(f"[rank {rank}] ref={ref} eager={got_e} graph={got_g} graph2={got_2}", flush=True)
        print(f"[rank {rank}] stats_eager={stats_e} stats_graph={stats_g} stats_graph2={stats_2}", flush=True)
        ok &= (got_e == ref and got_g == ref and got_2 == ref)
        # graph2 invariant: exactly one host sync recorded per cycle.
        ok &= int(stats_2.get("host_syncs", -1)) == int(stats_2["cycles"])
    ctx.barrier()
    if rank == 0:
        print("TP2_SPEC_GRAPH2_CHECK " + ("PASS" if ok else "FAIL"), flush=True)
    ctx.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
