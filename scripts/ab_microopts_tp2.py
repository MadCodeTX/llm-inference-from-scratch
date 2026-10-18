#!/usr/bin/env python3
"""Paired interleaved A/B for TP2 micro-opts (same process, ABBA segments).

Loads each lm_head mode once; inside a model, scale-cache off/on
alternate per 8-token segment (order reversed on pass 2) so clock
drift cancels. All passes must generate identical capital ids.

    .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/ab_microopts_tp2.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_spec = importlib.util.spec_from_file_location(
    "runner", Path(__file__).with_name("run_qwen38_tp2.py"))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402


def _gen_segmented(model, prompt_ids, scale_modes, vocab):
    """Generate 32 tokens; scale mode switches per segment (list per token)."""
    dev = model.ctx.device
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    gen, finite = [], True
    with torch.inference_mode():
        # segment boundaries: mode changes at token indices in scale_modes keys
        def set_mode(m):
            runner.apply_scale_cache_mode(model, m)
            torch.cuda.synchronize(dev)
        set_mode(scale_modes[0])
        if vocab:
            nxt, val, state = model.prefill_greedy(idx)
            finite &= bool(torch.isfinite(val).all())
            gen.append(int(nxt.item()))
        else:
            logits, state = model.prefill(idx)
            finite &= bool(torch.isfinite(logits[0, -1]).all())
            nxt = logits[0, -1].argmax().reshape(1)
            gen.append(int(nxt.item()))
        seg_times, cur_mode, seg_start = [], scale_modes[0], time.perf_counter()
        seg_counts = []
        count_in_seg = 1
        for i in range(1, 32):
            want = scale_modes[i] if i < len(scale_modes) else cur_mode
            if want != cur_mode:
                torch.cuda.synchronize(dev)
                seg_times.append(time.perf_counter() - seg_start)
                seg_counts.append(count_in_seg)
                set_mode(want)
                cur_mode, seg_start, count_in_seg = want, time.perf_counter(), 0
            if vocab:
                nxt, val, state = model.step_greedy(nxt.reshape(1, 1), state)
                finite &= bool(torch.isfinite(val).all())
                gen.append(int(nxt.item()))
            else:
                logits, state = model.step(nxt.reshape(1, 1), state)
                finite &= bool(torch.isfinite(logits[0, -1]).all())
                nxt = logits[0, -1].argmax().reshape(1)
                gen.append(int(nxt.item()))
            count_in_seg += 1
        torch.cuda.synchronize(dev)
        seg_times.append(time.perf_counter() - seg_start)
        seg_counts.append(count_in_seg)
    return gen, finite, seg_times, seg_counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=runner.DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="benchmarks/baseline/ours_tp2_microopts_paired.json")
    args = ap.parse_args()
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    probes = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/baseline/arm_a_quality_probes.json").read_text())["probes"]
    probe = next(p for p in probes if p["id"] == "capital")
    bridge = {p["id"]: p for p in json.loads((Path(__file__).resolve().parents[1] / "benchmarks/baseline/ours_pipeline_fastfp8.json").read_text())["probes"]}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = tok.encode(probe["prompt"])
    assert prompt_ids == probe["prompt_token_ids"]
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)

    # token-index -> scale mode schedules (ABBA): pass0 off*8 on*8 off*8 on*8(prefill counts in seg0), pass1 reversed
    sched0 = ["off"] * 8 + ["on"] * 8 + ["off"] * 8 + ["on"] * 8
    sched1 = ["on"] * 8 + ["off"] * 8 + ["on"] * 8 + ["off"] * 8
    results = {}
    all_ids = {}
    for lm_head in ("replicated", "vocab"):
        model, report = load_rank_state(
            snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
            fp8_mode="resident", ctx=ctx, use_triton=True, manifest=manifest,
            lm_head_mode=lm_head, use_scale_cache=False)
        runner.apply_norm_mode(model, "fused")
        per_mode = {"off": [], "on": []}  # ms/token samples (decode segments only approximated by segment incl prefill seg0 excluded below)
        ids_ref = None
        for pi, sched in enumerate((sched0, sched1)):
            gen, finite, seg_times, seg_counts = _gen_segmented(model, prompt_ids, sched, lm_head == "vocab")
            assert finite
            if ids_ref is None:
                ids_ref = gen
            assert gen == ids_ref, f"{lm_head} pass{pi} ids differ"
            # attribute segments: segment k mode = sched at its first token; skip segment 0 (prefill)
            tok_seen = 0
            for st, cnt in zip(seg_times, seg_counts):
                mode = sched[min(tok_seen, 31)]
                if tok_seen > 0:  # decode-only segments
                    per_mode[mode].append(1000.0 * st / cnt)
                tok_seen += cnt
            print(f"[rank {rank}] {lm_head} pass{pi} ids_match=True seg_ms={ [round(s,3) for s in seg_times] }", flush=True)
        all_ids[lm_head] = ids_ref
        results[lm_head] = {
            mode: {"samples_ms_per_token": vals,
                   "mean_ms_per_token": sum(vals) / len(vals) if vals else None}
            for mode, vals in per_mode.items()}
        del model
        torch.cuda.empty_cache()
    assert all_ids["replicated"] == all_ids["vocab"]
    cmp_bridge = runner.compare_ids(all_ids["replicated"], bridge["capital"]["generated_token_ids"])
    assert cmp_bridge["exact_match"], cmp_bridge
    if rank == 0:
        out = {
            "method": "paired interleaved ABBA segments in one process per lm_head mode "
                      "(8-token segments, scale toggled between segments, order reversed "
                      "pass 2); segment 0 (prefill) excluded from timing",
            "probe": "capital", "max_new_tokens": 32,
            "generated_token_ids": all_ids["replicated"],
            "vs_bridge": cmp_bridge,
            "vs_arm_a": runner.compare_ids(all_ids["replicated"], probe["output_token_ids"]),
            "results_ms_per_token_decode_segments": results,
            "torch": torch.__version__,
        }
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
        print("AB_MICROOPTS PASS", flush=True)
    ctx.barrier()
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
