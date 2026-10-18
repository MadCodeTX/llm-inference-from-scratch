#!/usr/bin/env python3
"""TP2 prefix-cache TTFT micro-bench (real 27B, scaled agent shape).
torchrun --nproc_per_node=2 scripts/prefix_ttft_bench.py
Shape labelled: 2048 shared + 256 unique = 2304 tokens (scaled down from
8192+512 agent shape). Cold = fresh cache; warm = cache populated by a
prompt sharing the 2048 prefix. Median of 3. Ours-vs-ours only; Arm A
agent TTFT 1.81s at 8704 tokens noted as context, no win claim.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"benchmarks"))
from inference_from_scratch.hybrid import HybridConfig, QwenRMSNorm  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state, resident_memory_report  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402
from inference_from_scratch.tp_prefix_cache import TPBlockPrefixCache, prefill_with_prefix_cache  # noqa: E402
from run_bench import build_token_ids, load_tokenizer  # noqa: E402
DEFAULT_SNAPSHOT = "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"

def _run(model, idx, cache, dev):
    state = model.static_state(batch_size=1, max_seq_len=int(idx.shape[1])+8)
    torch.cuda.synchronize(dev); t0=time.perf_counter()
    logits, n_reused, delta = prefill_with_prefix_cache(model, idx, state, cache, chunk_size=128)
    torch.cuda.synchronize(dev); return time.perf_counter()-t0, int(logits.float().argmax(-1).item()), n_reused

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="benchmarks/baseline/ours_prefix_ttft.json")
    ap.add_argument("--shared", type=int, default=2048); ap.add_argument("--unique", type=int, default=256)
    a = ap.parse_args()
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda"); rank, dev = ctx.rank, ctx.device
    snap = Path(a.snapshot); config = HybridConfig.from_text_config(snap); manifest = build_manifest(snap)
    tok = load_tokenizer(snap)
    model, report = load_rank_state(snap, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=True, manifest=manifest, lm_head_mode="vocab", include_mtp=False, use_scale_cache=False)
    for m in model.modules():
        if isinstance(m, QwenRMSNorm): m.use_fused_rmsnorm = True
    shared = build_token_ids(a.shared, tok)
    # Unique suffixes: offset token ids deterministically distinct from shared tail
    uniqA = build_token_ids(a.unique, tok); uniqB = [(t+12345) % int(config.vocab_size) for t in uniqA]
    promptA = torch.tensor([shared+uniqA], dtype=torch.long, device=dev)
    promptB = torch.tensor([shared+uniqB], dtype=torch.long, device=dev)
    results = {}
    with torch.inference_mode():
        _run(model, promptA[:, :128], TPBlockPrefixCache(block_size=16), dev)  # warmup
        cold, warm = [], []
        arg_cold = arg_warm = None; n_warm = 0
        for _ in range(3):
            s, arg_cold, _ = _run(model, promptB, TPBlockPrefixCache(block_size=16), dev); cold.append(s)
        for _ in range(3):
            cache = TPBlockPrefixCache(block_size=16)
            _run(model, promptA, cache, dev)  # populate (untimed)
            s, arg_warm, n_warm = _run(model, promptB, cache, dev); warm.append(s)
        results = {"shape_label": f"scaled agent: {a.shared} shared + {a.unique} unique = {a.shared+a.unique} tokens",
            "shared_tokens": a.shared, "unique_tokens": a.unique, "total_tokens": a.shared+a.unique,
            "block_size": 16, "chunk_size": 128, "reps": 3,
            "cold_ttft_s_median": statistics.median(cold), "cold_ttft_s": cold,
            "warm_ttft_s_median": statistics.median(warm), "warm_ttft_s": warm,
            "warm_n_reused": n_warm, "last_argmax_cold": arg_cold, "last_argmax_warm": arg_warm,
            "speedup_median": statistics.median(cold)/statistics.median(warm) if statistics.median(warm) else None,
            "context_only": "Arm A agent-shared-prefix TTFT 1.81s at 8704 tokens (different shape; no win claim)"}
        if rank == 0:
            print(json.dumps(results, indent=2), flush=True)
            out = ROOT/a.out; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(results, indent=2)+"\n")
    ctx.barrier(); ctx.destroy(); return 0
if __name__ == "__main__": raise SystemExit(main())
