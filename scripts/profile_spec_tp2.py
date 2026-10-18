#!/usr/bin/env python3
"""Host-overhead diagnosis for TP2 mtp1-graph speculation (torchrun x2).

Runs the capital probe (32 tokens, mtp1-graph) once warm and once under
``torch.profiler`` (CPU+CUDA), with the off-by-default segment timing in
``speculative_generate_tp`` enabled. Writes
``benchmarks/profile/spec_host_diagnosis.json`` (rank 0 only).

Method for separating the two kinds of "host overhead":
- A segment that launches work and returns (draft/verify replay launch,
  snapshot launches, input fill) is host CPU work: if it is large while
  the GPU stream is drained, the GPU idles behind the host.
- A segment that ends in ``.item()``/``.tolist()`` (draft readback,
  argmax-pair readback) is a synchronisation: its wall time is mostly
  the host *waiting* for already-queued GPU work, not CPU work. The
  profiler corroborates: ``cudaStreamSynchronize``/``_local_scalar_dense``
  CPU time is wait, while ``aten::clone``/``aten::copy_``/
  ``cudaLaunchKernel`` CPU time is real host work.

Run:
    .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/profile_spec_tp2.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "run_qwen38_tp2", ROOT / "scripts" / "run_qwen38_tp2.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def _top_rows(prof, cycles: int, limit: int = 30):
    rows = []
    for e in prof.key_averages():
        rows.append({
            "key": e.key, "count": int(e.count),
            "cpu_ms_total": float(e.cpu_time_total) / 1000.0,
            "cuda_ms_total": float(e.device_time_total) / 1000.0
            if hasattr(e, "device_time_total") else float(e.cuda_time_total) / 1000.0,
        })
    cpu = sorted(rows, key=lambda r: -r["cpu_ms_total"])[:limit]
    cuda = sorted(rows, key=lambda r: -r["cuda_ms_total"])[:limit]
    for group in (cpu, cuda):
        for r in group:
            r["per_cycle_count"] = r["count"] / max(cycles, 1)
    return cpu, cuda


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=_runner.DEFAULT_SNAPSHOT)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--probe", default="capital")
    ap.add_argument("--out", default="benchmarks/profile/spec_host_diagnosis.json")
    args = ap.parse_args()

    _runner.validate_fp8_config("new")
    from inference_from_scratch.hybrid import HybridConfig
    from inference_from_scratch.hybrid_checkpoint import build_manifest
    from inference_from_scratch.tp_checkpoint import load_rank_state
    from inference_from_scratch.tp_distributed import DistributedTPContext
    from inference_from_scratch.speculation import (
        speculative_generate_tp, summarize_segments)

    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    probes = json.loads(
        (ROOT / "benchmarks/baseline/arm_a_quality_probes.json").read_text())["probes"]
    probe = [p for p in probes if p["id"] == args.probe][0]
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = tok.encode(probe["prompt"])
    assert prompt_ids == probe["prompt_token_ids"]

    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx,
        use_triton=_runner.use_triton_for_path("triton"),
        manifest=manifest, lm_head_mode="vocab", include_mtp=True,
        use_scale_cache=False)
    _runner.apply_norm_mode(model, "fused")
    mtp = model.mtp_draft
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    def _run(segment_profile=None, profile=None):
        return speculative_generate_tp(
            model, mtp, idx, int(args.max_new_tokens),
            embedding=model.embed_tokens, lm_head=model._lm_head_logits,
            graph_verify=True, graph_draft=True,
            profile=profile, segment_profile=segment_profile)

    with torch.inference_mode():
        _run()  # warmup: allocator, kernels, capture machinery
        torch.cuda.synchronize(dev)
        seg: dict = {}
        phase: dict = {}
        from torch.profiler import ProfilerActivity, profile as tprofile
        t0 = time.perf_counter()
        with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                      record_shapes=False, with_stack=False) as prof:
            out, stats = _run(segment_profile=seg, profile=phase)
            torch.cuda.synchronize(dev)
        wall_s = time.perf_counter() - t0
    cycles = int(stats["cycles"])
    cpu_top, cuda_top = _top_rows(prof, cycles)
    # Fixed-cost baseline: 1 token => prefill + graph capture, 0 cycles.
    # Subtracting it from the 32-token profiler totals isolates the
    # marginal per-cycle loop cost from prefill/capture contamination.
    with torch.inference_mode():
        from torch.profiler import ProfilerActivity as _PA, profile as _tp
        with _tp(activities=[_PA.CPU, _PA.CUDA],
                 record_shapes=False, with_stack=False) as prof_base:
            _out1, stats1 = speculative_generate_tp(
                model, mtp, idx, 1,
                embedding=model.embed_tokens, lm_head=model._lm_head_logits,
                graph_verify=True, graph_draft=True)
            torch.cuda.synchronize(dev)
    base_cpu, base_cuda = _top_rows(prof_base, 1)
    base_by_key = {r["key"]: r for r in base_cpu}
    marginal = []
    for r in cpu_top:
        b = base_by_key.get(r["key"], {"cpu_ms_total": 0.0, "count": 0})
        marginal.append({
            "key": r["key"],
            "loop_cpu_ms_total": r["cpu_ms_total"] - b["cpu_ms_total"],
            "loop_count": r["count"] - b["count"],
            "per_cycle_count": (r["count"] - b["count"]) / max(cycles, 1),
            "per_cycle_cpu_ms": (r["cpu_ms_total"] - b["cpu_ms_total"]) / max(cycles, 1),
        })
    marginal = [m for m in marginal if m["loop_count"] > 0]
    marginal.sort(key=lambda m: -m["per_cycle_cpu_ms"])
    gen_ids = [int(v) for v in out[0, len(prompt_ids):].tolist()]
    matrix = json.loads(
        (ROOT / "benchmarks/baseline/ours_tp2_mtp_matrix.json").read_text())
    ref = [r for r in matrix["modes_32"]["mtp1-graph"]
           if r["id"] == args.probe][0]["generated_token_ids"]
    seg_pc = seg.get("segment_per_cycle_ms", {})
    derived = {
        "loop_wall_ms_per_cycle": seg.get("segment_loop_wall_ms", 0.0) / max(cycles, 1),
        "sync_wait_segments_ms_per_cycle": (
            seg_pc.get("draft_token_readback_ms", 0.0)
            + seg_pc.get("argmax_pair_readback_ms", 0.0)),
        "host_launch_segments_ms_per_cycle": (
            seg_pc.get("draft_replay_launch_ms", 0.0)
            + seg_pc.get("verify_input_fill_ms", 0.0)
            + seg_pc.get("verify_replay_ms", 0.0)
            + seg_pc.get("snapshot_ms", 0.0)
            + seg_pc.get("loop_bookkeeping_ms", 0.0)),
        "rollback_refeed_ms_per_rollback": (
            seg.get("segment_totals_ms", {}).get("rollback_refeed_ms", 0.0)
            / max(int(stats["rollbacks"]), 1)),
        "note": ("sync-wait segments end in .item()/.tolist() and mostly "
                 "wait for draft/verify GPU work; host-launch segments "
                 "return after queueing work, so they are host CPU time "
                 "while the GPU stream is drained/idle. rollback_refeed "
                 "is an eager (non-graph) token decode: launch-bound."),
    }
    result = {
        "probe": args.probe, "max_new_tokens": int(args.max_new_tokens),
        "mode": "mtp1-graph", "world": ctx.world_size,
        "generated_token_ids": gen_ids,
        "exact_vs_matrix_mtp1_graph": gen_ids == ref,
        "stats": stats, "cycles": cycles,
        "profiled_wall_per_cycle_ms": wall_s * 1000.0 / max(cycles, 1),
        "phase_profile": phase,
        "phase_per_cycle_ms": {k: v / max(cycles, 1)
                               for k, v in phase.items() if k != "wall_ms"},
        "segments": seg,
        "derived": derived,
        "segment_ranking": summarize_segments(seg_pc) if seg_pc else [],
        "profiler_cpu_top30": cpu_top,
        "profiler_cuda_top30": cuda_top,
        "profiler_baseline_1tok_cpu_top30": base_cpu,
        "profiler_loop_marginal_cpu": marginal,
        "baseline_1tok_stats": stats1,
        "method": (
            "sync-wait = wall of segments ending in .item()/.tolist() plus "
            "profiler cudaStreamSynchronize/_local_scalar_dense CPU time; "
            "host CPU work = launch-only segments (replay launch, input "
            "fill, snapshot launches) plus profiler aten::clone/copy_/"
            "cudaLaunchKernel CPU time. _PhaseTimer host_overhead_ms is "
            "wall minus CUDA-event GPU phases, so it mixes both."),
    }
    if rank == 0:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in
                          ("exact_vs_matrix_mtp1_graph", "stats",
                           "phase_per_cycle_ms", "segment_ranking")},
                         indent=2), flush=True)
        print(f"wrote {args.out}", flush=True)
    ctx.barrier()
    ctx.destroy()
    return 0 if gen_ids == ref else 1


if __name__ == "__main__":
    raise SystemExit(main())
