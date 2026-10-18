#!/usr/bin/env python3
"""GPU roofline microbenchmarks -- run ONLY inside an approved GPU window.

This harness measures the three hardware ceilings the roofline model
in ``src/inference_from_scratch/bottlenecks.py`` reasons about, at
shapes taken from the real 27B model served on this box
(hidden=5120, intermediate=17408, vocab=248320; see WIN.md):

* ``bandwidth`` -- large ``copy_`` timing (achieved HBM bytes/s);
* ``gemm`` -- bf16 matmul plus fp8 ``torch._scaled_mm`` when
  available, at 27B-ish shapes (5120x17408 MLP, 5120x12288 QKV,
  5120x248320 LM head, and their transposes);
* ``allreduce`` -- ``torch.distributed`` NCCL all-reduce bytes sweep.

Every subcommand prints JSON Lines (one JSON object per line) so
results can be diffed and archived without parsing prose.

SAFETY GUARD
------------
Actually touching a GPU requires the explicit flag::

    --i-understand-gpu-window

Without it each subcommand prints only a JSON ``plan`` line
describing what *would* run and exits 0. It allocates nothing,
initialises no CUDA context, and starts no process group. This
matters on this box: vLLM normally serves production traffic on
both GPUs, so these benchmarks must never run outside an approved
window (see ``benchmarks/WIN.md``).

AllReduce (2 ranks, NCCL) -- run via torchrun inside the window::

    torchrun --nproc_per_node=2 benchmarks/microbench_roofline.py \
        allreduce --i-understand-gpu-window

``torchrun`` supplies RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
MASTER_PORT in the environment; this script reads them through
``torch.distributed`` env:// initialisation. Running ``allreduce``
directly (without torchrun) will report world_size=1 and still time
the sweep, which is useful only as a smoke of the code path.

Examples (GPU window only)::

    .venv-gpu/bin/python benchmarks/microbench_roofline.py bandwidth --i-understand-gpu-window
    .venv-gpu/bin/python benchmarks/microbench_roofline.py gemm --i-understand-gpu-window
    torchrun --nproc_per_node=2 benchmarks/microbench_roofline.py allreduce --i-understand-gpu-window

Plan-only (safe anywhere, touches no GPU)::

    .venv-gpu/bin/python benchmarks/microbench_roofline.py bandwidth
    .venv-gpu/bin/python benchmarks/microbench_roofline.py gemm
    .venv-gpu/bin/python benchmarks/microbench_roofline.py allreduce
"""

from __future__ import annotations

import argparse
import json
import sys

GUARD_FLAG = "--i-understand-gpu-window"

# 27B-ish GEMM shapes: (label, M, K, N). K/N come from the served
# model's config (hidden 5120, intermediate 17408, QKV out 12288,
# vocab 248320); M sweeps decode (1) through prefill (1024+).
DEFAULT_GEMM_SHAPES: list[tuple[str, int, int, int]] = [
    ("mlp_gate_up_decode", 1, 5120, 17408),
    ("mlp_gate_up_small", 16, 5120, 17408),
    ("mlp_gate_up_prefill", 128, 5120, 17408),
    ("mlp_gate_up_large", 1024, 5120, 17408),
    ("mlp_down_prefill", 128, 17408, 5120),
    ("qkv_prefill", 128, 5120, 12288),
    ("lm_head_decode", 1, 5120, 248320),
    ("lm_head_prefill", 128, 5120, 248320),
]

DEFAULT_BANDWIDTH_MIB = [64, 256, 1024]
DEFAULT_ALLREDUCE_MIB = [1, 16, 64, 256]


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _plan(subcommand: str, detail: dict) -> int:
    _emit({
        "event": "plan",
        "subcommand": subcommand,
        "will_touch_gpu": False,
        "requires_flag": GUARD_FLAG,
        "note": (
            "plan only: pass --i-understand-gpu-window inside an "
            "approved GPU window to actually run"
        ),
        **detail,
    })
    return 0


def _require_torch_cuda(subcommand: str):
    """Import torch and confirm CUDA; emit a JSON error on failure."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - reported as JSON
        _emit({"event": "error", "subcommand": subcommand,
               "error": f"torch import failed: {type(exc).__name__}: {exc}"})
        return None
    if not torch.cuda.is_available():
        _emit({"event": "error", "subcommand": subcommand,
               "error": "torch.cuda.is_available() is False"})
        return None
    return torch


def _time_cuda(fn, warmup: int, repeats: int) -> float:
    """Best-of-``repeats`` seconds for ``fn`` using CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


# ---------------------------------------------------------------------------
# bandwidth
# ---------------------------------------------------------------------------

def run_bandwidth(args) -> int:
    if not args.i_understand_gpu_window:
        return _plan("bandwidth", {
            "sizes_mib": args.sizes_mib,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "device": args.device,
            "method": "copy_ between two device buffers, CUDA-event timed",
        })
    torch = _require_torch_cuda("bandwidth")
    if torch is None:
        return 1
    dev = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(dev)
    _emit({"event": "start", "subcommand": "bandwidth",
           "device": str(dev),
           "device_name": torch.cuda.get_device_name(args.device)})
    for mib in args.sizes_mib:
        nbytes = int(mib) * 1024 * 1024
        # float32 buffer: element count chosen so bytes == nbytes.
        src = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
        dst = torch.empty_like(src)
        secs = _time_cuda(lambda: dst.copy_(src), args.warmup, args.repeats)
        del src, dst
        torch.cuda.empty_cache()
        _emit({
            "event": "result", "subcommand": "bandwidth",
            "size_mib": mib, "bytes": nbytes, "seconds": secs,
            "gbytes_per_s": (nbytes / secs) / 1e9 if secs else None,
            # copy_ reads src and writes dst: 2x traffic.
            "effective_gbytes_per_s": (2 * nbytes / secs) / 1e9 if secs else None,
        })
    _emit({"event": "done", "subcommand": "bandwidth"})
    return 0


# ---------------------------------------------------------------------------
# gemm
# ---------------------------------------------------------------------------

def run_gemm(args) -> int:
    shapes = DEFAULT_GEMM_SHAPES
    if args.ms:
        wanted = set(args.ms)
        shapes = [s for s in shapes if s[1] in wanted] or [
            (f"m{m}", m, 5120, 17408) for m in args.ms
        ]
    if not args.i_understand_gpu_window:
        return _plan("gemm", {
            "shapes": [
                {"label": label, "m": m, "k": k, "n": n}
                for label, m, k, n in shapes
            ],
            "dtypes": ["bf16", "fp8_scaled_mm_if_available"],
            "repeats": args.repeats, "warmup": args.warmup,
            "device": args.device,
        })
    torch = _require_torch_cuda("gemm")
    if torch is None:
        return 1
    dev = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(dev)
    _emit({"event": "start", "subcommand": "gemm", "device": str(dev),
           "device_name": torch.cuda.get_device_name(args.device),
           "scaled_mm_available": hasattr(torch, "_scaled_mm")})
    for label, m, k, n in shapes:
        flops = 2.0 * m * k * n
        # --- bf16 ---
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
        secs = _time_cuda(lambda: torch.matmul(a, b), args.warmup, args.repeats)
        _emit({
            "event": "result", "subcommand": "gemm", "label": label,
            "dtype": "bf16", "m": m, "k": k, "n": n, "seconds": secs,
            "tflops": (flops / secs) / 1e12 if secs else None,
        })
        del a, b
        # --- fp8 scaled_mm (best effort) ---
        fp8_row: dict = {
            "event": "result", "subcommand": "gemm", "label": label,
            "dtype": "fp8", "m": m, "k": k, "n": n,
        }
        try:
            a8 = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
            b8 = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn).t()
            sa = torch.tensor(1.0, device=dev)
            sb = torch.tensor(1.0, device=dev)
            secs8 = _time_cuda(
                lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb,
                                         out_dtype=torch.bfloat16),
                args.warmup, args.repeats,
            )
            fp8_row.update({
                "available": True, "seconds": secs8,
                "tflops": (flops / secs8) / 1e12 if secs8 else None,
            })
            del a8, b8
        except Exception as exc:  # noqa: BLE001 - availability is data
            fp8_row.update({
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        _emit(fp8_row)
        torch.cuda.empty_cache()
    _emit({"event": "done", "subcommand": "gemm"})
    return 0


# ---------------------------------------------------------------------------
# allreduce
# ---------------------------------------------------------------------------

def run_allreduce(args) -> int:
    if not args.i_understand_gpu_window:
        return _plan("allreduce", {
            "sizes_mib": args.sizes_mib,
            "repeats": args.repeats, "warmup": args.warmup,
            "backend": "nccl",
            "launch": (
                "torchrun --nproc_per_node=2 benchmarks/"
                "microbench_roofline.py allreduce "
                "--i-understand-gpu-window"
            ),
        })
    torch = _require_torch_cuda("allreduce")
    if torch is None:
        return 1
    import os

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", args.device))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        _emit({"event": "start", "subcommand": "allreduce",
               "backend": "nccl", "world_size": world,
               "device_name": torch.cuda.get_device_name(local_rank)})
    for mib in args.sizes_mib:
        nbytes = int(mib) * 1024 * 1024
        buf = torch.ones(nbytes // 4, dtype=torch.float32,
                         device=f"cuda:{local_rank}")
        secs = _time_cuda(lambda: dist.all_reduce(buf),
                          args.warmup, args.repeats)
        # Bus bandwidth: all-reduce moves 2*(world-1)/world x data.
        bus = (nbytes * 2 * (world - 1) / world / secs) / 1e9 if secs and world > 1 else None
        if rank == 0:
            _emit({
                "event": "result", "subcommand": "allreduce",
                "size_mib": mib, "bytes": nbytes, "seconds": secs,
                "world_size": world,
                "gbytes_per_s": (nbytes / secs) / 1e9 if secs else None,
                "bus_gbytes_per_s": bus,
            })
        del buf
    if rank == 0:
        _emit({"event": "done", "subcommand": "allreduce",
               "world_size": world})
    dist.barrier()
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(GUARD_FLAG, dest="i_understand_gpu_window",
                    action="store_true",
                    help="required to actually touch a GPU; without "
                         "it only a JSON plan is printed")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_bw = sub.add_parser("bandwidth", help="large copy_ timing")
    _add_common(p_bw)
    p_bw.add_argument("--sizes-mib", type=int, nargs="+",
                      default=DEFAULT_BANDWIDTH_MIB)
    p_bw.set_defaults(fn=run_bandwidth)

    p_gemm = sub.add_parser("gemm", help="bf16 + fp8 GEMM at 27B shapes")
    _add_common(p_gemm)
    p_gemm.add_argument("--ms", type=int, nargs="+", default=None,
                        help="restrict to these M values")
    p_gemm.set_defaults(fn=run_gemm)

    p_ar = sub.add_parser("allreduce", help="NCCL all-reduce sweep")
    _add_common(p_ar)
    p_ar.add_argument("--sizes-mib", type=int, nargs="+",
                      default=DEFAULT_ALLREDUCE_MIB)
    p_ar.set_defaults(fn=run_allreduce)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
