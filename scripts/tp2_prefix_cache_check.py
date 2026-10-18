#!/usr/bin/env python
"""2-process prefix-cache TP check (gloo CPU / NCCL GPU).
Run: torchrun --nproc_per_node=2 scripts/tp2_prefix_cache_check.py --backend gloo --device cpu
Two prompts sharing 32-token prefix: warm run logits + greedy exactly equal fresh cold runs.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedHybridLM, DistributedTPContext  # noqa: E402
from inference_from_scratch.tp_prefix_cache import TPBlockPrefixCache, prefill_with_prefix_cache  # noqa: E402

def _tiny_cfg():
    return HybridConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention","linear_attention","linear_attention","full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=64,
        max_position_embeddings=128, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8, partial_rotary_factor=0.25, rope_theta=10000.0)

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--backend", default=None, choices=["nccl","gloo"]); p.add_argument("--device", default=None, choices=["cuda","cpu"])
    a = p.parse_args(); backend = a.backend or ("nccl" if a.device=="cuda" else "gloo"); dt = a.device or ("cuda" if backend=="nccl" else "cpu")
    ctx = DistributedTPContext.from_env(backend=backend, device_type=dt); rank, dev = ctx.rank, ctx.device
    cfg = _tiny_cfg(); torch.manual_seed(1234); dense = TinyHybridLM(cfg).to(dev); dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg); tp.eval()
    torch.manual_seed(99)
    shared = torch.randint(0,97,(1,32)); sufA = torch.randint(0,97,(1,13)); sufB = torch.randint(0,97,(1,13))
    promptA = torch.cat([shared,sufA],dim=1).to(dev); promptB = torch.cat([shared,sufB],dim=1).to(dev)
    with torch.no_grad():
        cache = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=16)
        sA = tp.static_state(batch_size=1, max_seq_len=64); logitsA, nA, dA = prefill_with_prefix_cache(tp, promptA, sA, cache, chunk_size=4)
        assert nA == 0, nA
        # fresh cold B
        cache2 = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=16)
        sBf = tp.static_state(batch_size=1, max_seq_len=64); logitsBf, nBf, _ = prefill_with_prefix_cache(tp, promptB, sBf, cache2, chunk_size=4)
        # warm B
        sB = tp.static_state(batch_size=1, max_seq_len=64); logitsB, nB, dB = prefill_with_prefix_cache(tp, promptB, sB, cache, chunk_size=4)
        print(f"[rank {rank}] prefix_reuse: n_reused={nB} delta={dB}", flush=True)
        assert nB == 32, nB
        if not torch.equal(logitsB, logitsBf): raise AssertionError(f"logits not exact max={(logitsB-logitsBf).abs().max()}")
        print(f"[rank {rank}] warm_vs_fresh_logits: exact=True", flush=True)
        nxtB, nxtF = logitsB.argmax(-1), logitsBf.argmax(-1); assert torch.equal(nxtB,nxtF)
        for i in range(4):
            oB = tp.decode_token_static(nxtB, sB); oF = tp.decode_token_static(nxtF, sBf)
            if not torch.equal(oB,oF): raise AssertionError(f"greedy {i} not exact")
            nxtB, nxtF = oB.argmax(-1), oF.argmax(-1)
        print(f"[rank {rank}] greedy_continuation: exact=True steps=4", flush=True)
    ctx.barrier()
    if rank==0: print("TP2_PREFIX_CACHE_CHECK PASS", flush=True)
    ctx.destroy(); return 0
if __name__ == "__main__": raise SystemExit(main())
