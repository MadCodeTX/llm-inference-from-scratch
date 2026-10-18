#!/usr/bin/env python
"""Two-process speculative TP check: MTP gamma=1 vs plain TP greedy.

Run under torchrun with 2 processes, CPU (gloo) or GPU (NCCL):

  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_speculative_check.py --backend gloo --device cpu

Each rank holds only its shards of a tiny 3-linear+1-full model plus
a TPMTPDraftModule; speculative ids must EXACTLY equal the plain
distributed greedy ids on every rank (Kiely sec. 5.2 invariant).
"""
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=None, choices=["nccl", "gloo"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = p.parse_args()
    backend = args.backend or ("nccl" if args.device != "cpu" else "gloo")
    device_type = args.device or ("cuda" if backend == "nccl" else "cpu")
    ctx = DistributedTPContext.from_env(backend=backend, device_type=device_type)
    rank, dev = ctx.rank, ctx.device
    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev)
    dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg)
    tp.eval()
    torch.manual_seed(99)
    head = MTPHead(MTPConfig.from_hybrid_config(cfg)).to(dev).eval()
    mtp = TPMTPDraftModule.from_dense(
        ctx, head, embedding=tp.embed_tokens, lm_head=tp.lm_head).eval()
    idx = torch.tensor([[3, 11, 5, 7, 9]], device=dev)
    with torch.no_grad():
        logits, state = tp.prefill(idx)
        ref = [int(logits[0, -1].argmax().item())]
        for _ in range(11):
            logits, state = tp.step(torch.tensor([ref[-1]], device=dev), state)
            ref.append(int(logits[0, 0].argmax().item()))
        out, stats = speculative_generate_tp(tp, mtp, idx, 12)
    got = out[0, 5:].tolist()
    print(f"[rank {rank}] ref={ref} got={got} stats={stats}", flush=True)
    if got != ref:
        raise AssertionError(f"speculative ids differ on rank {rank}")
    print(f"[rank {rank}] TP2_SPECULATIVE_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
