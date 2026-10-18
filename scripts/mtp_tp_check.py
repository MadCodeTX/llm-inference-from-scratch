#!/usr/bin/env python
"""Tiny 2-process TP MTP draft check: TP draft logits vs dense.

Run under torchrun with 2 processes, GPU (NCCL) or CPU (gloo):

  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/mtp_tp_check.py
  .venv/bin/torchrun --nproc_per_node=2 scripts/mtp_tp_check.py --backend gloo --device cpu

Each rank builds TPMTPDraftModule from the same seeded dense MTPHead
(fc replicated, MTP block sharded like a normal block) and compares two
sequential draft steps (state threaded, commit-on-accept) against the
dense MTPDraftModule on the same inputs. fp32 tight, bf16 loose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.mtp import (  # noqa: E402
    LEADING_WIRING, MTPConfig, MTPDraftModule, MTPHead, TPMTPDraftModule)
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402

D, VOCAB = 32, 97


def _cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
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
    print(f"[rank {rank}] backend={backend} device={dev} world={ctx.world_size}",
          flush=True)

    torch.manual_seed(11)
    head = MTPHead(MTPConfig.from_hybrid_config(_cfg()),
                   fc_concat_order=LEADING_WIRING["fc_concat_order"],
                   norm_pairing=LEADING_WIRING["norm_pairing"],
                   lm_head_input=LEADING_WIRING["lm_head_input"]).eval()
    emb = torch.nn.Embedding(VOCAB, D)
    lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    with torch.no_grad():
        for mod in (head, emb, lm_head):
            for par in mod.parameters():
                par.copy_(torch.randn_like(par.float()) * 0.3)
    dense = MTPDraftModule(head=head, embedding=emb, lm_head=lm_head).eval()
    head_dev = MTPHead(MTPConfig.from_hybrid_config(_cfg()),
                       fc_concat_order=LEADING_WIRING["fc_concat_order"],
                       norm_pairing=LEADING_WIRING["norm_pairing"],
                       lm_head_input=LEADING_WIRING["lm_head_input"]).eval()
    head_dev.load_state_dict(head.state_dict())
    head_dev = head_dev.to(dev)
    emb_dev = torch.nn.Embedding(VOCAB, D).to(dev)
    emb_dev.load_state_dict(emb.state_dict())
    lm_dev = torch.nn.Linear(D, VOCAB, bias=False).to(dev)
    lm_dev.load_state_dict(lm_head.state_dict())
    dense_dev = MTPDraftModule(head=head_dev, embedding=emb_dev, lm_head=lm_dev).eval()
    tp = TPMTPDraftModule.from_dense(ctx, head_dev, embedding=emb_dev,
                                     lm_head=lm_dev).eval()
    print(f"[rank {rank}] fc shape={tuple(tp.fc.weight.shape)} (replicated) "
          f"q shard={tuple(tp.layers[0].self_attn.q_weight.shape)}", flush=True)

    torch.manual_seed(21)
    hidden = torch.randn(2, D)
    tokens = [torch.tensor([3, 4]), torch.tensor([5, 6])]
    with torch.no_grad():
        d_state, t_state = dense_dev.initial_state(), tp.initial_state()
        for step, tok in enumerate(tokens):
            d_id, d_logits, d_state = dense_dev.draft_token(
                hidden.to(dev), tok.to(dev), d_state, position=step)
            t_id, t_logits, t_state = tp.draft_token(
                hidden.to(dev), tok.to(dev), t_state, position=step)
            err = (t_logits.float() - d_logits.float()).abs().max().item()
            ok = torch.allclose(t_logits, d_logits, atol=1e-5, rtol=1e-5)
            print(f"[rank {rank}] step {step}: max_abs_err={err:.3e} "
                  f"allclose={ok} ids_equal={bool((t_id == d_id).all())}", flush=True)
            if not ok or not bool((t_id == d_id).all()):
                raise AssertionError(f"TP MTP draft mismatch step {step}: {err}")
            hidden = torch.randn(2, D)
    ctx.barrier()
    if rank == 0:
        print("MTP_TP_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
