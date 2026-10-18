#!/usr/bin/env python
"""TPEngineCore smoke: 5 staggered requests on 2 slots, TP2.

Tiny model, real collectives. Default is NCCL/GPU (GPU gate must be
clear); ``--backend gloo --device cpu`` runs the same proof on CPU
(the durable test drives it that way under torchrun).

  .venv-gpu/bin/torchrun --standalone --nproc_per_node=2 scripts/tp_engine_smoke.py
  .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/tp_engine_smoke.py --backend gloo --device cpu

Proves per-request ids EXACTLY equal independent B=1 greedy runs,
slot reuse, and that every engine step respects the token budget.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedHybridLM, DistributedTPContext  # noqa: E402
from inference_from_scratch.tp_engine import TPEngineCore  # noqa: E402

PROMPTS = [[3, 11, 42, 7, 90], [55, 1, 2, 3, 4], [94, 20, 20, 8, 13],
           [10, 10, 10, 10, 10], [1, 96, 5, 5, 77]]


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

    def b1(prompt, max_new):
        s1 = tp.static_state(batch_size=1, max_seq_len=64)
        return tp.generate_static_batched(
            torch.tensor([prompt], device=dev), max_new, s1, chunk_size=2)[0].tolist()

    with torch.no_grad():
        refs = [b1(p, 6) for p in PROMPTS]
        eng = TPEngineCore(tp, ctx, max_slots=2, max_seq_len=64, token_budget=8)
        ids = [eng.submit(PROMPTS[0], 6), eng.submit(PROMPTS[1], 6)]
        eng.step(); eng.step()
        ids.append(eng.submit(PROMPTS[2], 6))
        eng.step()
        ids.append(eng.submit(PROMPTS[3], 6)); ids.append(eng.submit(PROMPTS[4], 6))
        eng.run()
        exact = all(eng.poll(r)["token_ids"] == ref and eng.poll(r)["status"] == "finished"
                    for r, ref in zip(ids, refs))
        print(f"[rank {rank}] staggered_exact={exact} "
              f"tokens={[eng.poll(r)['token_ids'] for r in ids]}", flush=True)
        if not exact:
            raise AssertionError(f"engine != B=1 refs: {[eng.poll(r) for r in ids]} vs {refs}")
        budget_ok = all(r.total_tokens <= 8 for r in eng.history)
        print(f"[rank {rank}] budget_respected={budget_ok} steps={len(eng.history)} "
              f"totals={[r.total_tokens for r in eng.history]}", flush=True)
        if not budget_ok:
            raise AssertionError("token budget exceeded")
        # Slot reuse is implied: 5 requests finished through 2 slots.
        print(f"[rank {rank}] slot_reuse=True slots=2 requests=5", flush=True)

    ctx.barrier()
    if rank == 0:
        print("TP_ENGINE_SMOKE PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
