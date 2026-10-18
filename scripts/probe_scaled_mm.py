"""Probe torch._scaled_mm vs the checkpoint [128,128] block scales.

Run: .venv-gpu/bin/python scripts/probe_scaled_mm.py
Tries plain / float32 / transposed / padded scale layouts at two
sizes; every 128x128 BlockWise attempt raises
'Invalid scaling configuration' on torch 2.11.0+cu128 / RTX 4090
(SM 8.9), and _scaled_mm would also need lossy FP8 activations.
Conclusion: no exact _scaled_mm path; use fp8_triton instead.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_from_scratch.quantization import quantize_block_fp8


def attempt(name, sa, sb, xq, wq):
    try:
        torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16)
        print(name, "OK")
    except Exception as exc:
        print(name, "FAIL", str(exc).splitlines()[-1][:220])


def main():
    print(torch.__version__, torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
    for m, k, n in [(128, 256, 256), (512, 512, 512)]:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda")
        w = torch.randn(n, k, device="cuda")
        xq, xs = quantize_block_fp8(x.float(), block=(128, 128))
        wq, ws = quantize_block_fp8(w.float(), block=(128, 128))
        xq, wq = xq.cuda(), wq.cuda()
        xs, ws = xs.cuda(), ws.cuda()
        print(f"-- M={m} K={k} N={n} scale_a={tuple(xs.shape)} scale_b={tuple(ws.shape)}")
        attempt("plain", xs, ws, xq, wq)
        attempt("float32", xs.float(), ws.float(), xq, wq)
        attempt("b_transposed", xs.float(), ws.float().t().contiguous().t(), xq, wq)
        attempt("both_transposed", xs.float().t().contiguous().t(),
                ws.float().t().contiguous().t(), xq, wq)


if __name__ == "__main__":
    main()
