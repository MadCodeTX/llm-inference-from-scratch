#!/usr/bin/env python3
"""Profile graph-mode (CUDA-graph, non-speculative) TP2 decode.

Loads the model exactly as ``scripts/run_qwen38_tp2.py --decode-mode
graph`` (norm fused, lm_head vocab), prefills the capital probe,
captures ONE decode graph, then:

1. Times clean graph replays (headline TPOT for this profile).
2. Runs replays under ``torch.profiler`` (CUDA) and ranks kernels
   via ``key_averages``: Triton FP8 total, fused gated-delta (when
   ``--delta fused``), SDPA/attention, norms, NCCL, lm_head-shaped
   FP8, and the residual elementwise/copy bucket that the fused
   gated-delta kernel replaces.

Outputs ``benchmarks/profile/tp2_graph_profile.json`` plus a summary
section appended to ``benchmarks/profile/SUMMARY_tp2.md``-style file
``benchmarks/profile/SUMMARY_graph.md`` (written fresh by rank 0).

Run:

    .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/profile_tp2_graph.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_qwen38_tp2 as runner  # noqa: E402
from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    load_rank_state, resident_memory_report)
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402

ROOFLINE_TPOT_FLOOR_MS = 19.17


def categorize(name: str) -> str:
    n = name.lower()
    if "gated_delta_kernel" in n:
        return "fused_gated_delta"
    if "fp8_dequant" in n or "_fp8_" in n:
        return "triton_fp8"
    if "nccl" in n:
        return "nccl"
    if any(s in n for s in ("flash", "sdpa", "fmha", "attention_kernel",
                            "mem_efficient", "cutlass")):
        return "full_attn_sdpa"
    if "rms_norm" in n or "rmsnorm" in n or "layer_norm_kernel" in n:
        return "rms_norm"
    if "gemvx" in n or "gemv" in n:
        return "dense_gemv_incl_lm_head"
    if "rotary" in n or "rope" in n:
        return "rope"
    return "elementwise_copy_reduce_other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=runner.DEFAULT_SNAPSHOT)
    ap.add_argument("--tokens", type=int, default=32, help="generated tokens (capital)")
    ap.add_argument("--profile-replays", type=int, default=16)
    ap.add_argument("--delta", choices=list(runner.DELTA_MODES), default="ref")
    ap.add_argument("--out", default="benchmarks/profile/tp2_graph_profile.json")
    ap.add_argument("--summary", default="benchmarks/profile/SUMMARY_graph.md")
    args = ap.parse_args()

    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    probes = json.loads(Path("benchmarks/baseline/arm_a_quality_probes.json").read_text())["probes"]
    probe = next(p for p in probes if p["id"] == "capital")
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    mem = resident_memory_report(snapshot, config=config, world=ctx.world_size,
                                 manifest=manifest, lm_head_mode="vocab")
    if float(mem["per_rank_total_gib"]) > runner.MEMORY_GUARD_GIB:
        print("FAIL: resident estimate exceeds guard", flush=True)
        ctx.destroy()
        return 2

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = tok.encode(probe["prompt"])
    assert prompt_ids == probe["prompt_token_ids"]

    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=True, manifest=manifest,
        lm_head_mode="vocab")
    torch.cuda.synchronize(dev)
    runner.apply_norm_mode(model, "fused")
    n_delta = runner.apply_delta_mode(model, args.delta)
    print(f"[rank {rank}] loaded delta={args.delta} layers={n_delta}", flush=True)

    gen_ids: list[int] = []
    with torch.inference_mode():
        plan = runner.graph_decode_plan(len(prompt_ids), args.tokens)
        state = model.static_state(batch_size=1, max_seq_len=plan["max_seq_len"])
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        logits = model.prefill_static(idx, state)
        nxt = logits[0, -1].float().argmax()
        gen_ids.append(int(nxt))
        # Warmup on scratch (also JIT-compiles the fused kernel).
        scratch = model.static_state(batch_size=1, max_seq_len=plan["max_seq_len"])
        warm_in = nxt.reshape(1).clone()
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(3):
                model.decode_token_static(warm_in, scratch)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch
        ctx.barrier()
        static_input = nxt.reshape(1).clone()
        static_out = torch.empty(1, int(config.vocab_size), device=dev, dtype=logits.dtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.decode_token_static(static_input, state, out=static_out)
        torch.cuda.synchronize(dev)
        ctx.barrier()

        # Clean timed replays (also produce the exact ids).
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(plan["replays"]):
            graph.replay()
            nxt = static_out.float().argmax(dim=-1)
            gen_ids.append(int(nxt.reshape(())))
            static_input.copy_(nxt)
        torch.cuda.synchronize(dev)
        clean_s = time.perf_counter() - t0

        # Profiled replays: rewind position so replays stay inside the
        # KV buffers (kernel shapes/timing unaffected; ids discarded).
        state.position.fill_(len(prompt_ids))
        # Profiled replays (state advances; ids discarded).
        from torch.profiler import ProfilerActivity, profile
        prof = None
        if rank == 0:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=True)
        ctx_obj = prof if prof is not None else _Null()
        with ctx_obj:
            for _ in range(args.profile_replays):
                graph.replay()
                nxt = static_out.float().argmax(dim=-1)
                static_input.copy_(nxt)
            torch.cuda.synchronize(dev)
        del graph
        torch.cuda.synchronize(dev)

    ref = json.loads(Path("benchmarks/baseline/ours_tp2_graph.json").read_text())
    ref_ids = next(p["generated_token_ids"] for p in ref["probes"] if p["id"] == "capital")
    exact = gen_ids[:32] == ref_ids[:32]
    clean_tpot = clean_s / plan["replays"] * 1e3 if plan["replays"] else None

    if rank == 0:
        rows = [{"name": e.key, "self_cuda_us": float(e.self_device_time_total),
                 "count": int(e.count)} for e in prof.key_averages()]
        rows = [r for r in rows if r["self_cuda_us"] > 0]
        rows.sort(key=lambda r: -r["self_cuda_us"])
        cats: dict[str, dict] = {}
        for r in rows:
            c = cats.setdefault(categorize(r["name"]),
                                {"ms_per_token": 0.0, "kernels_per_token": 0.0})
            c["ms_per_token"] += r["self_cuda_us"] / 1e3 / args.profile_replays
            c["kernels_per_token"] += r["count"] / args.profile_replays
        total_kernel_ms = sum(r["self_cuda_us"] for r in rows) / 1e3 / args.profile_replays
        total_kernels = sum(r["count"] for r in rows) / args.profile_replays
        out = {
            "script": "scripts/profile_tp2_graph.py",
            "snapshot": str(snapshot), "world": ctx.world_size,
            "probe": "capital", "tokens": args.tokens, "delta_mode": args.delta,
            "profile_replays": args.profile_replays,
            "generated_ids": gen_ids, "exact_vs_graph_reference": bool(exact),
            "clean_tpot_ms": clean_tpot, "clean_tok_per_s": (1e3 / clean_tpot if clean_tpot else None),
            "roofline_tpot_floor_ms": ROOFLINE_TPOT_FLOOR_MS,
            "profiler_rank0": {
                "total_kernel_ms_per_token": total_kernel_ms,
                "total_kernels_per_token": total_kernels,
                "categories_ms_per_token": dict(sorted(
                    cats.items(), key=lambda kv: -kv[1]["ms_per_token"])),
                "top_kernels": [
                    {"name": r["name"],
                     "ms_per_token": r["self_cuda_us"] / 1e3 / args.profile_replays,
                     "count_per_token": r["count"] / args.profile_replays}
                    for r in rows[:60]],
            },
            "torch": torch.__version__, "cuda": torch.version.cuda,
        }
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        lines = [
            "# TP2 graph-mode decode profile — measured",
            "",
            f"Capital x{args.tokens}, graph mode, delta={args.delta}. "
            f"Clean TPOT {clean_tpot:.2f} ms/token; exact vs graph reference: {exact}.",
            "",
            "## Kernel categories (rank 0, per token)",
            "",
        ]
        for name, c in out["profiler_rank0"]["categories_ms_per_token"].items():
            lines.append(f"- {name}: {c['ms_per_token']:.3f} ms/token, "
                         f"{c['kernels_per_token']:.1f} kernels/token")
        lines += ["", "## Top kernels (rank 0, per token)", ""]
        for r in out["profiler_rank0"]["top_kernels"][:15]:
            lines.append(f"- {r['name']}: {r['ms_per_token']:.3f} ms/token "
                         f"({r['count_per_token']:.1f}/token)")
        Path(args.summary).write_text("\n".join(lines) + "\n")
        print(json.dumps({"clean_tpot_ms": clean_tpot, "exact": exact,
                          "categories": out["profiler_rank0"]["categories_ms_per_token"]},
                         indent=2), flush=True)
        print(f"wrote {args.out} and {args.summary}", flush=True)
    ctx.barrier()
    if rank == 0:
        print("PASS" if exact else "FAIL", flush=True)
    ctx.destroy()
    return 0 if exact else 1


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
