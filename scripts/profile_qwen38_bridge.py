#!/usr/bin/env python3
"""Attribute the pipeline bridge's ~548 ms/token decode cost (no optimization).

Reuses scripts/run_qwen38_gpu.py loading (meta build, resident FP8 convert,
layer-by-layer load, pipeline split 32), then profiles ONE capital-probe
generation (prefill + 8 decode tokens) two ways:
  1. torch.profiler (CUDA+CPU, record_shapes) -> key_averages grouped by
     aten op, rolled up via inference_from_scratch.profiling (aten names
     cannot separate dequant elementwise ops from other elementwise work;
     that split comes from the event timers below, stated honestly).
  2. Coarse CUDA-event timers via reversible monkey-patches active only
     in this process: BlockFP8Linear dequant vs F.linear, GatedDeltaNet
     prefill (used for T=1 decode by the runner), GatedFullAttention,
     SwiGLU (memo: subset of linear time, never added), cross-device
     hidden transfer in runner loop, argmax/.item host sync.
Also micro-times BlockFP8Linear at M=1 for real projection shapes.
Outputs benchmarks/profile/bridge_profile.json (+ optional chrome trace).
Run: .venv-gpu/bin/python scripts/profile_qwen38_bridge.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.fp8_linear import BlockFP8Linear
from inference_from_scratch.hybrid import DeviceMappedHybridRunner, HybridConfig
from inference_from_scratch.profiling import rollup, roofline_ms
from inference_from_scratch.quantization import dequantize_block_fp8

SNAP = ("/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
        "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")
TIMERS: dict[str, list[tuple]] = {}
ENABLED = {"on": True}

def _time(cat, dev, fn, *a, **k):
    if not ENABLED["on"]:
        return fn(*a, **k)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(dev)
    s.record(stream); r = fn(*a, **k); e.record(stream)
    TIMERS.setdefault(cat, []).append((s, e)); return r

def _totals():
    out = {}
    for c, evs in TIMERS.items():
        if c == "sampling_host_sync_wall": continue
        out[c] = sum(s.elapsed_time(e) for s, e in evs)
    return out

def microbench(device="cuda:0"):
    shapes = [(5120, 17408), (17408, 5120), (12288, 5120), (5120, 10240)]  # (in,out); 3rd = q_proj stored [12288,5120] transposed view
    rows = []
    for inf, outf in shapes:
        m = BlockFP8Linear(inf, outf, bias=False).to(device)
        with torch.no_grad():
            m.weight.copy_(torch.randn(outf, inf, device=device).to(torch.float8_e4m3fn))
        x = torch.randn(1, inf, device=device, dtype=torch.bfloat16)
        for _ in range(3): m(x)
        torch.cuda.synchronize(device)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        N = 20
        with torch.no_grad():
            s.record()
            for _ in range(N): m(x)
            e.record()
        torch.cuda.synchronize(device)
        ms = s.elapsed_time(e) / N
        moved = outf * inf * (1 + 2 + 2) + (inf + outf) * 2  # fp8 read + bf16 write + bf16 read + act
        gbps = moved / (ms / 1e3) / 1e9
        rows.append({"in": inf, "out": outf, "ms_per_call": ms, "bytes_model": moved,
                     "eff_GBps": gbps, "roofline_ms_at_920GBps": roofline_ms(moved),
                     "slowdown_vs_roofline": ms / roofline_ms(moved) if roofline_ms(moved) else None})
        del m, x; torch.cuda.empty_cache()
    return rows

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--snapshot", default=SNAP)
    ap.add_argument("--split", type=int, default=32); ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--trace", action="store_true"); a = ap.parse_args()
    if torch.cuda.device_count() < 2: print("FAIL need 2 GPUs"); return 1
    micro = microbench("cuda:0")
    # --- load exactly as run_qwen38_gpu.py ---
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_qwen38_gpu as R
    from transformers import AutoTokenizer
    from inference_from_scratch.hybrid_checkpoint import build_manifest, load_into_model
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    config = HybridConfig.from_text_config(a.snapshot); manifest = build_manifest(Path(a.snapshot))
    fp8_modules = {e.target_name[:-len(".weight")] for e in manifest.entries if e.kind == "fp8_weight" and e.target_name and not e.target_name.startswith("mtp.")}
    from torch import nn
    with torch.device("meta"):
        from inference_from_scratch.hybrid import HybridLM
        model = HybridLM(config)
        for name, mod in list(model.named_modules()):
            if name in fp8_modules and isinstance(mod, nn.Linear) and not isinstance(mod, BlockFP8Linear):
                repl = BlockFP8Linear(mod.in_features, mod.out_features, bias=False)
                p, _, at = name.rpartition("."); setattr(model.get_submodule(p) if p else model, at, repl)
    plan_dev = ("cuda:0", "cuda:1")
    from inference_from_scratch.hybrid import pipeline_device_plan
    plan = pipeline_device_plan(config.num_hidden_layers, a.split, *plan_dev)
    R._place(model.embed_tokens, plan["embed"])
    for i, b in enumerate(model.layers): R._place(b, plan["layers"][i])
    R._place(model.norm, plan["norm"]); R._place(model.lm_head, plan["lm_head"]); model.eval()
    t0 = time.perf_counter(); rep = load_into_model(model, Path(a.snapshot), fp8_mode="resident", strict=True, manifest=manifest)
    load_s = time.perf_counter() - t0
    if not rep["ok"]: print("FAIL load", rep); return 1
    runner = DeviceMappedHybridRunner(model, a.split, *plan_dev)
    probe = json.loads(Path("benchmarks/baseline/arm_a_quality_probes.json").read_text())["probes"][0]
    prompt_ids = tok.encode(probe["prompt"]); ref = probe["output_token_ids"]
    # --- reversible instrumentation ---
    orig_fwd = BlockFP8Linear.forward; orig_gdn = None; orig_gfa = None; orig_mlp = None
    from inference_from_scratch.linear_attention import GatedDeltaNet
    from inference_from_scratch.hybrid import GatedFullAttention
    from inference_from_scratch.mlp import SwiGLU
    def fp8_fwd(self, x):
        if self.use_fast_path:
            f = self._try_scaled_mm(x)
            if f is not None: return f
        dt = x.dtype if x.is_floating_point() else torch.bfloat16
        w = _time(f"dequant@{x.device}", x.device, dequantize_block_fp8, self.weight, self.weight_scale_inv, block=self.block, out_dtype=dt)
        return _time(f"matmul_linear_fp8@{x.device}", x.device, F.linear, x, w, self.bias.to(dt) if self.bias is not None else None)
    BlockFP8Linear.forward = fp8_fwd
    orig_gdn = GatedDeltaNet.forward_prefill; orig_gfa = GatedFullAttention.forward; orig_mlp = SwiGLU.forward
    GatedDeltaNet.forward_prefill = lambda s, x, **k: _time("linear_attn", x.device, orig_gdn, s, x, **k)
    GatedFullAttention.forward = lambda s, x, *ar, **k: _time("full_attn", x.device, orig_gfa, s, x, *ar, **k)
    SwiGLU.forward = lambda s, x, *ar, **k: _time("mlp_memo_subset_of_linear", x.device, orig_mlp, s, x, *ar, **k)
    try:
        with torch.no_grad():
            idx = torch.tensor([prompt_ids])
            logits, state = runner.run(idx)  # warmup/prefill unprofiled
            nxt = int(logits[0, -1].argmax().item()); gen = [nxt]
            TIMERS.clear()
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                t_dec = time.perf_counter()
                for _ in range(a.tokens - 1):
                    step = torch.tensor([[nxt]], dtype=torch.long, device="cuda:1")
                    logits, state = runner.run(step, state=state)
                    t_s = time.perf_counter(); nxt = int(logits[0, -1].argmax().item()); TIMERS.setdefault("sampling_host_sync_wall", []).append((t_s, time.perf_counter()))
                    gen.append(nxt)
                torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
                decode_s = time.perf_counter() - t_dec
            if a.trace: prof.export_chrome_trace("benchmarks/profile/bridge_trace.json")
    finally:
        BlockFP8Linear.forward = orig_fwd; GatedDeltaNet.forward_prefill = orig_gdn
        GatedFullAttention.forward = orig_gfa; SwiGLU.forward = orig_mlp
    ev = [{"name": e.key, "self_cuda_time_us": e.self_device_time_total, "count": e.count} for e in prof.key_averages()]
    timers = _totals()
    timers["sampling_host_sync_wall"] = sum((e - s) * 1e3 for s, e in TIMERS.get("sampling_host_sync_wall", []) if isinstance(s, float))
    per_token = {k: v / max(a.tokens - 1, 1) for k, v in timers.items() if k != "sampling_host_sync_wall"}
    out = {"probe": probe["id"], "generated": gen, "ref": ref[:len(gen)], "exact_prefix_match": gen == ref[:len(gen)],
           "load_s": load_s, "decode_s_profiled": decode_s, "decode_tokens": a.tokens - 1,
           "ms_per_token_wall": decode_s / max(a.tokens - 1, 1) * 1e3,
           "profiler_rollup": rollup(ev), "profiler_top_ops": sorted(ev, key=lambda r: -r["self_cuda_time_us"])[:25],
           "event_timers_total_ms": timers, "event_timers_per_token_ms": per_token, "microbench_M1": micro}
    Path("benchmarks/profile").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/profile/bridge_profile.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("exact_prefix_match", "ms_per_token_wall", "event_timers_per_token_ms")}, indent=2))
    print("PASS" if out["exact_prefix_match"] else "CHECK tokens")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
