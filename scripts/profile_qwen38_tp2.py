#!/usr/bin/env python3
"""Profile the TP2 decode step of the full Qwen3.8-27B-FP8 model.

Loads the model exactly as ``scripts/run_qwen38_tp2.py`` (true TP2,
resident FP8, Triton by default), warms up on the capital probe, then
times 16 decode tokens post-warmup two ways:

1. A clean wall-clock segment (instrumentation recording disabled) --
   the headline TPOT for this profile.
2. An instrumented segment: CUDA-event timers on BOTH ranks via
   process-local monkey-patches (restored in ``finally``; no engine
   change) for Triton FP8 linears total, full-attention, gated-delta
   step, MLP, block norms and the whole decode step, plus a
   ``torch.profiler`` (CUDA+CPU, record_shapes) run on rank 0 only.
   Collectives are counted by a patched ``all_reduce_sum`` and timed
   from the profiler's NCCL kernels. Host gap = profiled wall minus
   profiler GPU-busy time, per token.

Outputs ``benchmarks/profile/tp2_profile.json`` and
``benchmarks/profile/SUMMARY_tp2.md`` comparing against the pipeline
bridge profile and the 19.17 ms TP2 roofline floor.

Run (approved GPU window, incumbent stopped):

    INFERENCE_FP8_TRITON=1 .venv-gpu/bin/torchrun --nproc_per_node=2 \
        scripts/profile_qwen38_tp2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_qwen38_tp2 as runner  # noqa: E402  (load/compare helpers)
from inference_from_scratch.fp8_linear import BlockFP8Linear  # noqa: E402
from inference_from_scratch.hybrid import HybridConfig, QwenRMSNorm  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    load_rank_state, resident_memory_report)
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedFullAttention, DistributedGatedDeltaNet, DistributedHybridBlock,
    DistributedHybridLM, DistributedSwiGLU, DistributedTPContext)

ROOFLINE_TPOT_FLOOR_MS = 19.17
TIMERS: dict[str, list[tuple]] = {}
COUNTERS = {"all_reduce_count": 0, "all_reduce_bytes": 0, "fp8_calls": 0}
ENABLED = {"on": False}


def per_token_ms(totals_ms: dict, n_tokens: int) -> dict:
    """Per-token milliseconds from segment totals."""
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    return {k: float(v) / n_tokens for k, v in totals_ms.items()}


def host_gap_ms(wall_ms_per_token: float, gpu_busy_ms_per_token: float) -> float:
    """Host gap per token: wall time the GPU was not busy."""
    return float(wall_ms_per_token) - float(gpu_busy_ms_per_token)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=runner.DEFAULT_SNAPSHOT)
    ap.add_argument("--tokens", type=int, default=16,
                    help="profiled decode tokens per segment (post-warmup)")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--probes", default="benchmarks/baseline/arm_a_quality_probes.json")
    ap.add_argument("--bridge", default="benchmarks/baseline/ours_pipeline_fastfp8.json")
    ap.add_argument("--out", default="benchmarks/profile/tp2_profile.json")
    ap.add_argument("--summary", default="benchmarks/profile/SUMMARY_tp2.md")
    ap.add_argument("--fp8-path", choices=list(runner.FP8_PATH_CHOICES), default="triton")
    ap.add_argument("--norm", choices=list(runner.NORM_MODES), default="manual",
                    help="RMSNorm (A/B): manual (profile baseline, default) or fused")
    ap.add_argument("--memory-guard-gib", type=float, default=runner.MEMORY_GUARD_GIB)
    return ap


def _time(cat, dev, fn, *a, **k):
    if not ENABLED["on"]:
        return fn(*a, **k)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(dev)
    s.record(stream)
    r = fn(*a, **k)
    e.record(stream)
    TIMERS.setdefault(cat, []).append((s, e))
    return r


def _totals() -> dict:
    return {c: sum(s.elapsed_time(e) for s, e in evs) for c, evs in TIMERS.items()}


def install_patches():
    """Install process-local timers; returns a restore callable."""
    orig = {
        "fp8": BlockFP8Linear.forward,
        "attn": DistributedFullAttention.forward_with_state,
        "delta": DistributedGatedDeltaNet.step,
        "mlp": DistributedSwiGLU.forward,
        "block": DistributedHybridBlock.step,
        "norm": QwenRMSNorm.forward,
        "all_reduce": DistributedTPContext.all_reduce_sum,
    }

    def fp8_fwd(self, x):
        if ENABLED["on"]:
            COUNTERS["fp8_calls"] += 1
        return _time("fp8_linear", x.device, orig["fp8"], self, x)

    def all_reduce(self, tensor):
        if ENABLED["on"]:
            COUNTERS["all_reduce_count"] += 1
            COUNTERS["all_reduce_bytes"] += tensor.numel() * tensor.element_size()
        return orig["all_reduce"](self, tensor)

    BlockFP8Linear.forward = fp8_fwd
    DistributedFullAttention.forward_with_state = lambda s, x, *a, **k: _time(
        "full_attention", x.device, orig["attn"], s, x, *a, **k)
    DistributedGatedDeltaNet.step = lambda s, x, *a, **k: _time(
        "gated_delta_step", x.device, orig["delta"], s, x, *a, **k)
    DistributedSwiGLU.forward = lambda s, x, *a, **k: _time(
        "mlp", x.device, orig["mlp"], s, x, *a, **k)
    DistributedHybridBlock.step = lambda s, x, *a, **k: _time(
        "block_step_memo", x.device, orig["block"], s, x, *a, **k)
    QwenRMSNorm.forward = lambda s, x, *a, **k: _time(
        "norms", x.device, orig["norm"], s, x, *a, **k)
    DistributedTPContext.all_reduce_sum = all_reduce

    def restore():
        BlockFP8Linear.forward = orig["fp8"]
        DistributedFullAttention.forward_with_state = orig["attn"]
        DistributedGatedDeltaNet.step = orig["delta"]
        DistributedSwiGLU.forward = orig["mlp"]
        DistributedHybridBlock.step = orig["block"]
        QwenRMSNorm.forward = orig["norm"]
        DistributedTPContext.all_reduce_sum = orig["all_reduce"]

    return restore


def main() -> int:
    args = build_parser().parse_args()
    if args.tokens < 1 or args.warmup < 0:
        print("FAIL: --tokens >= 1 and --warmup >= 0 required", flush=True)
        return 2
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    probes = json.loads(Path(args.probes).read_text())["probes"]
    probe = probes[0]
    bridge = {p["id"]: p for p in json.loads(Path(args.bridge).read_text())["probes"]}
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    mem = resident_memory_report(snapshot, config=config, world=ctx.world_size,
                                 manifest=manifest)
    est_gib = float(mem["per_rank_total_gib"])
    if rank == 0:
        print(f"profile TP2: world={ctx.world_size} fp8_path={args.fp8_path} "
              f"tokens={args.tokens} warmup={args.warmup} est={est_gib:.3f} GiB/rank",
              flush=True)
    if est_gib > args.memory_guard_gib:
        print(f"FAIL: estimate {est_gib:.3f} GiB exceeds guard", flush=True)
        ctx.destroy()
        return 2

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = tok.encode(probe["prompt"])
    assert prompt_ids == probe["prompt_token_ids"]

    t0 = time.perf_counter()
    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx,
        use_triton=runner.use_triton_for_path(args.fp8_path), manifest=manifest)
    torch.cuda.synchronize(dev)
    load_s = time.perf_counter() - t0
    runner.apply_norm_mode(model, args.norm)
    print(f"[rank {rank}] load {load_s:.1f}s resident="
          f"{report['resident_bytes'] / 1024 ** 3:.3f} GiB", flush=True)

    restore = install_patches()
    try:
        with torch.inference_mode():
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
            logits, state = model.prefill(idx)
            nxt = logits[0, -1].float().argmax()
            gen_t = [nxt]
            for _ in range(args.warmup):  # warmup decode (Triton JIT), unprofiled
                logits, state = model.step(nxt.reshape(1, 1), state)
                nxt = logits[0, -1].float().argmax()
                gen_t.append(nxt)
            torch.cuda.synchronize(dev)

            def run_segment(n_tokens, timed_step: bool):
                nonlocal nxt, state
                seg_gen = []
                torch.cuda.synchronize(dev)
                t_seg = time.perf_counter()
                for _ in range(n_tokens):
                    if timed_step:
                        s_ev = torch.cuda.Event(enable_timing=True)
                        e_ev = torch.cuda.Event(enable_timing=True)
                        s_ev.record()
                        logits, state = model.step(nxt.reshape(1, 1), state)
                        nxt = logits[0, -1].float().argmax()
                        e_ev.record()
                        TIMERS.setdefault("decode_step_total", []).append((s_ev, e_ev))
                    else:
                        logits, state = model.step(nxt.reshape(1, 1), state)
                        nxt = logits[0, -1].float().argmax()
                    seg_gen.append(nxt)
                torch.cuda.synchronize(dev)
                return time.perf_counter() - t_seg, seg_gen

            # Clean segment: patches installed but recording disabled.
            ENABLED["on"] = False
            clean_s, clean_gen = run_segment(args.tokens, timed_step=False)
            gen_t.extend(clean_gen)

            # Instrumented segment: event timers both ranks + profiler rank 0.
            TIMERS.clear()
            for k in COUNTERS:
                COUNTERS[k] = 0
            ENABLED["on"] = True
            prof_ctx = nullcontext()
            prof = None
            if rank == 0:
                from torch.profiler import ProfilerActivity, profile
                prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                               record_shapes=True)
                prof_ctx = prof
            with prof_ctx:
                prof_s, prof_gen = run_segment(args.tokens, timed_step=True)
            ENABLED["on"] = False
            gen_t.extend(prof_gen)
            torch.cuda.synchronize(dev)
            ev_totals = _totals()
            counters = dict(COUNTERS)
    finally:
        restore()

    gen_ids = [int(v) for v in torch.stack(gen_t).cpu().tolist()]
    local = {
        "rank": rank, "load_s": load_s,
        "event_totals_ms": ev_totals,
        "event_per_token_ms": per_token_ms(ev_totals, args.tokens),
        "counters": counters,
        "counters_per_token": per_token_ms(
            {k: float(v) for k, v in counters.items()}, args.tokens),
        "generated_ids": gen_ids,
        "clean_wall_ms_per_token": clean_s / args.tokens * 1e3,
        "profiled_wall_ms_per_token": prof_s / args.tokens * 1e3,
    }
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, local)
    ok = all(g["generated_ids"] == gathered[0]["generated_ids"] for g in gathered)

    if rank == 0:
        ev = []
        ev_shapes = []
        gpu_busy_ms = nccl_ms = 0.0
        nccl_count = 0
        if prof is not None:
            ev = [{"name": e.key, "self_cuda_time_us": e.self_device_time_total,
                   "self_cpu_time_us": e.self_cpu_time_total, "count": e.count}
                  for e in prof.key_averages()]
            # GPU busy counts actual kernels only: aten wrapper rows
            # (aten::mm, nccl:all_reduce, ...) carry their kernels' time
            # again and would double-count.
            def _is_kernel(name: str) -> bool:
                n = name.lower()
                return "kernel" in n or "gemv" in n or "gemm" in n
            gpu_busy_ms = sum(r["self_cuda_time_us"] for r in ev
                              if _is_kernel(r["name"])) / 1e3
            nccl_rows = [r for r in ev if "nccldevkernel" in r["name"].lower()]
            nccl_ms = sum(r["self_cuda_time_us"] for r in nccl_rows) / 1e3
            nccl_count = sum(r["count"] for r in nccl_rows)
            ev_shapes = [{"name": e.key,
                          "input_shapes": [list(s) for s in e.input_shapes],
                          "self_cuda_time_us": e.self_device_time_total,
                          "count": e.count}
                         for e in prof.key_averages(group_by_input_shape=True)]
        busy_pt = gpu_busy_ms / args.tokens
        wall_pt = float(gathered[0]["profiled_wall_ms_per_token"])
        clean_pt = float(gathered[0]["clean_wall_ms_per_token"])
        ref_bridge = bridge[probe["id"]]["generated_token_ids"]
        cmp_bridge = runner.compare_ids(gen_ids, ref_bridge)
        pipe_tok_s = [p.get("decode_tok_per_s") for p in
                      json.loads(Path(args.bridge).read_text())["probes"]]
        pipe_ms = 1e3 / (sum(pipe_tok_s) / len(pipe_tok_s))
        out = {
            "script": "scripts/profile_qwen38_tp2.py",
            "runner_script": "scripts/run_qwen38_tp2.py",
            "snapshot": str(snapshot), "world": ctx.world_size,
            "fp8_path": args.fp8_path, "norm_mode": args.norm,
            "probe": probe["id"],
            "warmup_tokens": args.warmup, "profiled_tokens": args.tokens,
            "generated_ids": gen_ids, "vs_bridge": cmp_bridge,
            "cross_rank_ids_equal": bool(ok),
            "load_s_by_rank": {g["rank"]: g["load_s"] for g in gathered},
            "clean_wall_ms_per_token": clean_pt,
            "clean_tok_per_s": 1e3 / clean_pt,
            "profiled_wall_ms_per_token": wall_pt,
            "event_per_token_ms_by_rank": {g["rank"]: g["event_per_token_ms"]
                                           for g in gathered},
            "event_totals_ms_by_rank": {g["rank"]: g["event_totals_ms"]
                                        for g in gathered},
            "counters_by_rank": {g["rank"]: g["counters"] for g in gathered},
            "counters_per_token_by_rank": {g["rank"]: g["counters_per_token"]
                                           for g in gathered},
            "profiler_rank0": {
                "gpu_busy_ms_per_token": busy_pt,
                "host_gap_ms_per_token": host_gap_ms(wall_pt, busy_pt),
                "nccl_ms_per_token": nccl_ms / args.tokens,
                "nccl_kernel_count": nccl_count,
                "nccl_kernels_per_token": nccl_count / args.tokens,
                "top_ops": sorted(ev, key=lambda r: -r["self_cuda_time_us"])[:25],
                "top_ops_by_shape": sorted(
                    ev_shapes, key=lambda r: -r["self_cuda_time_us"])[:25],
            },
            "comparisons": {
                "roofline_tpot_floor_ms": ROOFLINE_TPOT_FLOOR_MS,
                "clean_vs_floor_x": clean_pt / ROOFLINE_TPOT_FLOOR_MS,
                "pipeline_fastfp8_ms_per_token": pipe_ms,
                "pipeline_bridge_profile_ms_per_token": 551.5,
                "arm_a_tpot_ms": 20.3,
            },
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(ctx.world_size)],
            "notes": [
                "Event categories overlap: fp8_linear is a subset of "
                "full_attention/gated_delta_step/mlp; block_step_memo is the "
                "sum of per-block steps; norms is a subset of block_step_memo "
                "plus the final norm. decode_step_total covers the whole step.",
                "Collective count comes from the patched all_reduce_sum; "
                "collective GPU time comes from rank-0 profiler NCCL kernels.",
                "Host gap = profiled wall - profiler GPU busy (rank 0). "
                "Event recording inflates profiled wall vs the clean segment.",
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

        r0 = gathered[0]["event_per_token_ms"]
        cats = [
            ("Triton FP8 linears (subset of mixer+MLP)", r0.get("fp8_linear", 0.0)),
            ("Full attention (16 layers, memo)", r0.get("full_attention", 0.0)),
            ("Gated-delta step (48 layers, memo)", r0.get("gated_delta_step", 0.0)),
            ("MLP (64 layers, memo)", r0.get("mlp", 0.0)),
            ("Norms (block + final, memo)", r0.get("norms", 0.0)),
            ("Collectives (NCCL, profiler rank 0)", nccl_ms / args.tokens),
            ("Decode step total (event, rank 0)", r0.get("decode_step_total", 0.0)),
        ]
        ranked = sorted(cats, key=lambda kv: -kv[1])
        lines = [
            "# TP2 decode profile — measured",
            "",
            f"Source: [profile script](/home/nick/muse/llm-inference-from-scratch/scripts/profile_qwen38_tp2.py) → [tp2_profile.json](/home/nick/muse/llm-inference-from-scratch/benchmarks/profile/tp2_profile.json). Probe `{probe['id']}`, {args.warmup} warmup + {args.tokens} clean + {args.tokens} instrumented decode tokens, both RTX 4090, TP2 resident FP8 (Triton). Generated ids cross-rank equal: {bool(ok)}; vs pipeline bridge record: {cmp_bridge['matching_positions']}/{cmp_bridge['compared_tokens']} exact (prefix compared over the recorded 32; profile generates {len(gen_ids)}).",
            "",
            f"Headline: clean wall **{clean_pt:.2f} ms/token ({1e3 / clean_pt:.2f} tok/s)**; instrumented wall {wall_pt:.2f} ms/token (event recording perturbs it). Roofline TP2 floor {ROOFLINE_TPOT_FLOOR_MS:.2f} ms/token → clean is {clean_pt / ROOFLINE_TPOT_FLOOR_MS:.2f}× the floor. Pipeline bridge (fastfp8) is {pipe_ms:.1f} ms/token and the old dequant bridge profile was 551.5 ms/token; vLLM Arm A c1 TPOT is ~20.3 ms.",
            "",
            "## Ranked costs (per decode token, rank 0 events; profiler for collectives)",
            "",
            "| Rank | Cost | ms/token | Notes |",
            "|---|---|---|---|",
        ]
        notes = {
            "Triton FP8 linears (subset of mixer+MLP)": "subset of the mixer/MLP rows; never added to them",
            "Full attention (16 layers, memo)": "includes its FP8 linears + 1 all-reduce per layer",
            "Gated-delta step (48 layers, memo)": "includes its FP8 linears + 1 all-reduce per layer",
            "MLP (64 layers, memo)": "includes its FP8 linears + 1 all-reduce per layer",
            "Norms (block + final, memo)": "QwenRMSNorm only; gated norm sits inside gated-delta",
            "Collectives (NCCL, profiler rank 0)": f"{gathered[0]['counters']['all_reduce_count'] / args.tokens:.0f} all-reduces/token counted on rank 0",
            "Decode step total (event, rank 0)": "whole model.step + argmax, event-timed",
        }
        for i, (name, ms) in enumerate(ranked, 1):
            lines.append(f"| {i} | {name} | {ms:.3f} | {notes[name]} |")
        lines += [
            "",
            f"Profiler (rank 0): GPU busy {busy_pt:.2f} ms/token, host gap (profiled wall − GPU busy) **{host_gap_ms(wall_pt, busy_pt):.2f} ms/token**. Collective count {gathered[0]['counters']['all_reduce_count']} over {args.tokens} tokens ({gathered[0]['counters']['all_reduce_count'] / args.tokens:.1f}/token) on rank 0; FP8 linear calls {gathered[0]['counters']['fp8_calls'] / args.tokens:.1f}/token.",
            "",
            "## Reading",
            "",
            "Overlapping categories are marked memo and never summed. The non-overlapping check is: decode-step event total vs clean wall vs GPU busy. Next lever is whichever of FP8 linears / attention / gated-delta / collectives / host gap dominates above; see the fix section appended after the A/B run.",
            "",
        ]
        Path(args.summary).write_text("\n".join(lines))
        print(json.dumps({"clean_ms_per_token": clean_pt,
                          "profiled_ms_per_token": wall_pt,
                          "event_per_token_rank0": r0,
                          "gpu_busy_ms_per_token": busy_pt,
                          "host_gap_ms_per_token": host_gap_ms(wall_pt, busy_pt),
                          "nccl_ms_per_token": nccl_ms / args.tokens,
                          "all_reduce_per_token": gathered[0]["counters"]["all_reduce_count"] / args.tokens,
                          "cross_rank_equal": bool(ok),
                          "vs_bridge": cmp_bridge}, indent=2), flush=True)
        print(f"wrote {args.out} and {args.summary}", flush=True)
    ctx.barrier()
    if rank == 0:
        print("PASS" if ok else "FAIL", flush=True)
    ctx.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
