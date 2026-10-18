#!/usr/bin/env python3
"""TP2 batched static-decode bench (real 27B snapshot, eager mode).

Run (GPU gate must be clear first):

  .venv-gpu/bin/torchrun --standalone --nproc_per_node=2 scripts/batched_decode_bench.py

For each B in {1,2,4,8,16}: short prompts built from the Arm A
quality probe "capital" (benchmarks/baseline/arm_a_quality_probes.json,
token ids [760, 6511, 314, 9338, 369]) replicated per row with a
distinct suffix token per row (so rows are genuinely different
prompts), then a fixed 64 greedy decode tokens, ignore-EOS, via
``DistributedHybridLM.generate_static_batched`` (chunkwise prefill of
the whole batch + GPU-argmax decode loop, no per-token host sync).

Timing: the timed loop replicates ``generate_static_batched``
step-for-step (prefill_chunkwise_static + decode_token_static with
on-device argmax) so prefill and decode phases are timed separately;
its tokens are checked token-for-token against the helper's own run.
Reports step ms (decode loop / decode steps), aggregate tok/s and
tok/min (B * 64 / generate time), per-seq tok/s, peak allocated
memory per rank (all-gathered), and for B=1 that row 0's ids equal a
single-row (B=1 state, batch-1 prompt) run's ids.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")
CAPITAL_IDS = [760, 6511, 314, 9338, 369]  # "The capital of France is"
DECODE_TOKENS = 64


def _prompts(batch: int, dev) -> torch.Tensor:
    rows = [CAPITAL_IDS + [1000 + 17 * r] for r in range(batch)]
    return torch.tensor(rows, dtype=torch.long, device=dev)


def _timed_generate(model, idx, max_new, dev):
    """Manual twin of generate_static_batched with phase timing."""
    B = int(idx.shape[0])
    state = model.static_state(batch_size=B, max_seq_len=int(idx.shape[1]) + max_new + 4)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    logits = model.prefill_chunkwise_static(idx, state, chunk_size=128)
    torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    nxt = logits.float().argmax(dim=-1)
    gen = [nxt]
    for _ in range(max_new - 1):
        logits = model.decode_token_static(nxt, state)
        nxt = logits.float().argmax(dim=-1)
        gen.append(nxt)
    out = torch.stack(gen, dim=1)
    torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    return out, (t1 - t0), (t2 - t1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="benchmarks/baseline/ours_tp2_batched_decode.json")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    args = ap.parse_args()
    max_new = int(args.decode_tokens)

    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    t0 = time.perf_counter()
    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=True,
        manifest=manifest, lm_head_mode="vocab", include_mtp=False,
        use_scale_cache=False)
    torch.cuda.synchronize(dev)
    load_s = time.perf_counter() - t0
    from inference_from_scratch.hybrid import QwenRMSNorm
    for mod in model.modules():
        if isinstance(mod, QwenRMSNorm):
            mod.use_fused_rmsnorm = True
    print(f"[rank {rank}] load {load_s:.1f}s resident={report['resident_bytes'] / 1024**3:.3f} GiB", flush=True)

    batches = [int(v) for v in args.batches.split(",") if v.strip()]
    rows = []
    single_ids: list[int] | None = None
    with torch.inference_mode():
        # Warmup: one small batched generate (Triton/kernel warmup).
        warm = _prompts(2, dev)
        ws = model.static_state(batch_size=2, max_seq_len=int(warm.shape[1]) + 8)
        model.generate_static_batched(warm, 4, ws, chunk_size=128)
        torch.cuda.synchronize(dev)
        for B in batches:
            idx = _prompts(B, dev)
            # Correctness: helper run.
            hs = model.static_state(batch_size=B, max_seq_len=int(idx.shape[1]) + max_new + 4)
            helper_ids = model.generate_static_batched(idx, max_new, hs, chunk_size=128)
            torch.cuda.synchronize(dev)
            # Timed twin (best of 2 after the helper run has warmed this B).
            best = None
            for _ in range(2):
                ids, pre_s, dec_s = _timed_generate(model, idx, max_new, dev)
                if not torch.equal(ids, helper_ids):
                    raise AssertionError(f"B={B}: timed twin tokens differ from helper")
                tot = pre_s + dec_s
                if best is None or tot < best[0]:
                    best = (tot, pre_s, dec_s, ids)
            tot, pre_s, dec_s, ids = best
            steps = max(max_new - 1, 1)
            agg_tok_s = B * max_new / tot if tot > 0 else None
            mem = torch.tensor([float(torch.cuda.max_memory_allocated(dev))],
                               dtype=torch.float64, device=dev)
            mem_all = ctx.all_gather(mem)
            peak_gib = [float(m.item()) / 1024**3 for m in mem_all]
            row = {
                "batch": B, "prompt_tokens_per_row": int(idx.shape[1]),
                "decode_tokens": max_new, "ignore_eos": True, "greedy": True,
                "prefill_s": pre_s, "decode_s": dec_s, "generate_s": tot,
                "decode_steps": steps,
                "step_ms": (dec_s / steps * 1000.0) if dec_s else None,
                "aggregate_tok_per_s": agg_tok_s,
                "aggregate_tok_per_min": (agg_tok_s * 60.0 if agg_tok_s else None),
                "per_seq_tok_per_s": (max_new / tot if tot > 0 else None),
                "peak_mem_gib_per_rank": peak_gib,
                "row0_ids": ids[0].tolist(),
            }
            if B == 1:
                single_ids = ids[0].tolist()
                row["b1_row0_eq_single_run"] = True  # this B=1 run IS the single run; checked below vs helper
                s1 = model.static_state(batch_size=1, max_seq_len=int(idx.shape[1]) + max_new + 4)
                one = model.generate_static_batched(idx, max_new, s1, chunk_size=128)
                row["b1_row0_eq_single_run"] = bool(one[0].tolist() == single_ids)
                if not row["b1_row0_eq_single_run"]:
                    raise AssertionError("B=1 row0 ids differ from single run")
            rows.append(row)
            print(f"[rank {rank}] B={B} step_ms={row['step_ms']:.3f} agg_tok_s={agg_tok_s:.1f} "
                  f"agg_tok_min={agg_tok_s * 60:.0f} per_seq={max_new / tot:.2f}/s "
                  f"peak_gib={peak_gib}", flush=True)
            torch.cuda.reset_peak_memory_stats(dev)

    ctx.barrier()
    if rank == 0:
        out = {"script": "scripts/batched_decode_bench.py", "snapshot": str(snapshot),
               "world": ctx.world_size, "mode": "eager (no CUDA graph)",
               "prompt": "capital quality probe [760,6511,314,9338,369] + distinct suffix token 1000+17*r per row",
               "decode_tokens": max_new, "torch": torch.__version__,
               "gpus": [torch.cuda.get_device_name(i) for i in range(ctx.world_size)],
               "load_s_rank0": load_s, "rows": rows,
               "note": "step_ms = decode-loop seconds / (decode_tokens-1) decode steps; aggregate uses full generate time (prefill+decode)."}
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
        print("BATCHED_DECODE_BENCH PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
