"""Sweep Triton FP8 tile configs at prefill M (128/512) + dequant cuBLAS.

Run: .venv-gpu/bin/python benchmarks/fp8_prefill_sweep.py --out benchmarks/profile/fp8_prefill_sweep.json
Sweeps BLOCK_M {64,128} x BLOCK_N {32,64,128} x num_warps {4,8}
(BLOCK_K=128, stages=3) at M in {64,128,512} on real TP2 shard shapes,
plus the legacy config and the dequant+cuBLAS reference path.
Each row: CUDA-event median ms, GB/s (weight bytes / t), max/rel diff
vs dequant reference. Winners are hand-copied into the per-bucket map
in src/inference_from_scratch/fp8_triton.py (no runtime autotune).
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.fp8_triton import fp8_triton_linear, triton_available
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8

# TP2 per-rank shards + full shapes (weight [N,K] as stored)
SHAPES = [
    ("tp_down_2560x8704", 2560, 8704),
    ("tp_gate_8704x5120", 8704, 5120),
    ("full_down_5120x17408", 5120, 17408),
    ("full_gate_17408x5120", 17408, 5120),
]
MS = [64, 128, 512]
CONFIGS = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128, "num_warps": w, "num_stages": 3}
    for bm in (64, 128) for bn in (32, 64, 128) for w in (4, 8)
]
LEGACY = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}

def _time(fn, reps=20):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/profile/fp8_prefill_sweep.json")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    if not triton_available():
        print("CUDA/Triton unavailable"); return 1
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}", flush=True)
    rows = []
    for label, n, k in SHAPES:
        torch.manual_seed(0)
        w = torch.randn(n, k) * 0.05
        wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
        wq, ws_bf = wq.cuda(), ws.cuda().to(torch.bfloat16)
        del w
        wbytes = wq.numel()
        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            wdeq = dequantize_block_fp8(wq, ws_bf, out_dtype=torch.bfloat16)
            ref = F.linear(x, wdeq)
            rmax = ref.float().abs().max().item() or 1.0
            # dequant+cuBLAS candidate: dequant materialised per call + F.linear
            def _cublas():
                wd = dequantize_block_fp8(wq, ws_bf, out_dtype=torch.bfloat16)
                return F.linear(x, wd)
            for _ in range(5): _cublas()
            torch.cuda.synchronize()
            ctimes = [_time(_cublas, reps=args.reps) for _ in range(args.rounds)]
            cms = statistics.median(ctimes)
            rows.append({"label": label, "N": n, "K": k, "M": m, "path": "dequant_cublas",
                         "config": None, "ms": cms, "weight_gb_s": wbytes / cms / 1e6,
                         "max_abs_diff": 0.0, "rel_diff": 0.0})
            print(f"{label} M={m} dequant+cuBLAS: {cms:.4f} ms", flush=True)
            cand = list(CONFIGS) + [dict(LEGACY)]
            ok = []
            for cfg in cand:
                try:
                    for _ in range(5): fp8_triton_linear(x, wq, ws_bf, config=cfg)
                    torch.cuda.synchronize(); ok.append(cfg)
                except Exception as exc:
                    rows.append({"label": label, "N": n, "K": k, "M": m, "path": "triton",
                                 "config": dict(cfg), "ms": None, "weight_gb_s": None,
                                 "max_abs_diff": None, "rel_diff": None,
                                 "is_legacy": cfg == LEGACY, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"{label} M={m} BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} W={cfg['num_warps']}: SKIP {type(exc).__name__}", flush=True)
            torch.cuda.synchronize()
            times = {id(c): [] for c in ok}
            for _ in range(args.rounds):
                for cfg in ok:
                    times[id(cfg)].append(_time(lambda c=cfg: fp8_triton_linear(x, wq, ws_bf, config=c), reps=args.reps))
            for cfg in ok:
                got = fp8_triton_linear(x, wq, ws_bf, config=cfg)
                diff = (got.float() - ref.float()).abs().max().item()
                ms = statistics.median(times[id(cfg)])
                rows.append({"label": label, "N": n, "K": k, "M": m, "path": "triton",
                             "config": dict(cfg), "ms": ms, "weight_gb_s": wbytes / ms / 1e6,
                             "max_abs_diff": diff, "rel_diff": diff / rmax,
                             "is_legacy": cfg == LEGACY})
                print(f"{label} M={m} BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} W={cfg['num_warps']}: {ms:.4f} ms rel={diff/rmax:.2e}", flush=True)
            del x, ref, wdeq
            torch.cuda.empty_cache()
        del wq, ws, ws_bf
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
