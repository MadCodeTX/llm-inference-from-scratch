#!/usr/bin/env python3
"""Profile TP2 chunkwise prefill (prompt 2048, chunk 128) with torch.profiler.

Run: .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/profile_prefill_tp2.py
Writes benchmarks/profile/tp2_prefill_profile.json (rank 0): wall time,
tok/s, and GPU-time split by category bucketed from kernel names, plus
the raw top-kernel table so the bucketing is auditable.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
from inference_from_scratch.hybrid import HybridConfig, QwenRMSNorm  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402
from run_bench import build_token_ids, load_tokenizer  # noqa: E402
DEFAULT_SNAPSHOT = ("/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
                    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")

def _cat(name: str) -> str:
    n = name.lower()
    if "fp8_dequant" in n or "_fp8_" in n: return "triton_fp8"
    if "nccl" in n or "allreduce" in n or "all_reduce" in n or "reduce_scatter" in n or "allgather" in n: return "collectives"
    if "rmsnorm" in n or "rms_norm" in n or "layernorm" in n or "layer_norm" in n: return "norms"
    if "gated" in n or "delta" in n or "conv1d" in n or "conv_depthwise" in n or "trsm" in n or "solve_triangular" in n: return "gated_delta_chunk_core"
    if "attn" in n or "attention" in n or "sdpa" in n or "flash" in n or "softmax" in n: return "full_attn"
    return "other_gpu_elementwise_matmul"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="benchmarks/profile/tp2_prefill_profile.json")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    config = HybridConfig.from_text_config(Path(args.snapshot))
    manifest = build_manifest(Path(args.snapshot))
    tok = load_tokenizer(Path(args.snapshot))
    model, _rep = load_rank_state(Path(args.snapshot), config=config, rank=rank,
                                  world=ctx.world_size, device=dev, fp8_mode="resident",
                                  ctx=ctx, use_triton=True, manifest=manifest,
                                  lm_head_mode="vocab", include_mtp=False, use_scale_cache=False)
    for mod in model.modules():
        if isinstance(mod, QwenRMSNorm): mod.use_fused_rmsnorm = True
    ids = torch.tensor([build_token_ids(args.prompt, tok)], dtype=torch.long, device=dev)
    with torch.inference_mode():
        warm = torch.tensor([build_token_ids(128, tok)], dtype=torch.long, device=dev)
        st = model.static_state(batch_size=1, max_seq_len=136)
        model.prefill_chunkwise_static(warm, st, chunk_size=128)
        torch.cuda.synchronize(dev)
        state = model.static_state(batch_size=1, max_seq_len=args.prompt + 8)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False) as prof:
            last = model.prefill_chunkwise_static(ids, state, chunk_size=args.chunk)
            torch.cuda.synchronize(dev)
        wall = time.perf_counter() - t0
    ctx.barrier()
    if rank == 0:
        cats: dict[str, float] = {}
        top = []
        total_cuda = 0.0
        for evt in prof.key_averages():
            # Leaf CUDA kernels only: wrapper ops (aten::/c10d::/nccl:/
            # record_*) nest the same device time and would double-count.
            if evt.key.startswith(("aten::", "c10d::", "nccl:", "record_")): continue
            ct = float(getattr(evt, "device_time_total", 0) or getattr(evt, "cuda_time_total", 0) or 0)
            if ct <= 0: continue
            total_cuda += ct
            cats[_cat(evt.key)] = cats.get(_cat(evt.key), 0.0) + ct
            top.append({"kernel": evt.key, "category": _cat(evt.key), "cuda_us": ct, "count": evt.count})
        top.sort(key=lambda r: -r["cuda_us"])
        out = {"script": "scripts/profile_prefill_tp2.py", "prompt_tokens": args.prompt,
               "chunk_size": args.chunk, "world": ctx.world_size,
               "wall_s": wall, "tok_per_s": args.prompt / wall,
               "last_argmax": int(last.float().argmax(dim=-1).item()),
               "total_cuda_us_rank0": total_cuda,
               "category_cuda_us": cats,
               "category_share_of_cuda": {k: (v / total_cuda if total_cuda else 0.0) for k, v in cats.items()},
               "host_plus_gap_s_est": max(0.0, wall - total_cuda / 1e6),
               "note": ("Categories bucketed from kernel names (see _cat); "
                        "'other_gpu_elementwise_matmul' holds remaining GEMM/elementwise "
                        "(dense BF16 linears, gated-delta chunk core ops, MLP activations) "
                        "that kernel names do not separate. Raw top kernels included."),
               "top_kernels": top[:40]}
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in ("wall_s", "tok_per_s", "category_share_of_cuda")}, indent=2), flush=True)
        print(f"wrote {args.out}"); print("PREFILL_PROFILE_TP2 PASS", flush=True)
    ctx.destroy()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
