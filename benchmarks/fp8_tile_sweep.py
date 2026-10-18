"""Sweep Triton FP8 tile configs at M=1 real shapes (post-CUDA-graph retune).

Run: .venv-gpu/bin/python benchmarks/fp8_tile_sweep.py --out benchmarks/profile/fp8_tile_sweep.json
Sweeps BLOCK_N {32,64,128} x num_warps {4,8} x BLOCK_M {16,32} at M=1.
Each config: ms (CUDA-event median of interleaved rounds), GB/s, rel diff
vs dequant reference. No autotune is shipped; winners are hand-copied
into TUNED_CONFIGS in src/inference_from_scratch/fp8_triton.py.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.fp8_triton import fp8_triton_linear, triton_available
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8

# (label, N, K) as stored weight orientations [N,K]
SHAPES = [
    ("down_5120x17408", 5120, 17408),
    ("gate_17408x5120", 17408, 5120),
    ("shape_5120x12288", 5120, 12288),
    ("shape_5120x10240", 5120, 10240),
    # TP2 per-rank shard families (extra, justifies family generalisation)
    ("tp_gate_8704x5120", 8704, 5120),
    ("tp_down_5120x8704", 5120, 8704),
]
CONFIGS = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128, "num_warps": w, "num_stages": 3}
    for bm in (16, 32) for bn in (32, 64, 128) for w in (4, 8)
]

def _time(fn, reps=100):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/profile/fp8_tile_sweep.json")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    if not triton_available():
        print("CUDA/Triton unavailable"); return 1
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    rows = []
    for label, n, k in SHAPES:
        torch.manual_seed(0)
        w = torch.randn(n, k) * 0.05
        wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
        wq, ws_bf = wq.cuda(), ws.cuda().to(torch.bfloat16)
        del w
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        ref = F.linear(x, dequantize_block_fp8(wq, ws_bf, out_dtype=torch.bfloat16))
        total_bytes = wq.numel() + ws_bf.numel()*2 + x.numel()*2 + n*2
        # warmup/compile each config
        for cfg in CONFIGS:
            for _ in range(10): fp8_triton_linear(x, wq, ws_bf, config=cfg)
        torch.cuda.synchronize()
        times = {id(c): [] for c in CONFIGS}
        for _ in range(args.rounds):
            for cfg in CONFIGS:
                times[id(cfg)].append(_time(lambda c=cfg: fp8_triton_linear(x, wq, ws_bf, config=c), reps=args.reps))
        for cfg in CONFIGS:
            got = fp8_triton_linear(x, wq, ws_bf, config=cfg)
            diff = (got.float()-ref.float()).abs().max().item()
            rel = diff / ref.float().abs().max().item() if ref.float().abs().max().item() else 0.0
            ms = statistics.median(times[id(cfg)])
            row = {"label": label, "N": n, "K": k, "M": 1, "config": cfg,
                   "ms": ms, "gb_s": total_bytes/ms/1e6, "max_abs_diff": diff, "rel_diff": rel,
                   "is_default": cfg == {"BLOCK_M":16,"BLOCK_N":128,"BLOCK_K":128,"num_warps":4,"num_stages":3}}
            rows.append(row)
            print(f"{label} N={n} K={k} BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} W={cfg['num_warps']}: {ms:.4f} ms {row['gb_s']:.1f} GB/s rel={rel:.2e}{' DEFAULT' if row['is_default'] else ''}", flush=True)
        del wq, ws, ws_bf, x, ref
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2)+"\n")
    print(f"wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
