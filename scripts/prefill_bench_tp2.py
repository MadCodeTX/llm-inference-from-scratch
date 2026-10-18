#!/usr/bin/env python3
"""TP2 prefill bench: token-loop static vs chunkwise prefill (real 27B).

Run under torchrun with 2 ranks on the real snapshot:

  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/prefill_bench_tp2.py

Compares, at prompt lengths 512 / 2048 (suite filler token ids,
``benchmarks/run_bench.build_token_ids``):

* ``prefill_static`` -- token loop, one weight read per token. Timed
  only at 512 and labelled capped there: at 2048 the loop is not run
  (it is the slow path this bench exists to replace).
* ``prefill_chunkwise_static`` with chunk 128 and chunk 256 -- every
  weight read once per chunk.

Reports seconds + tok/s per method and last-token argmax equality
between methods. This is an ours-vs-ours prefill comparison ONLY: it
is not a TTFT win/loss vs vLLM (no server, no queueing, no sampling).

FP8 dispatch note (measured, see docs/26_prefill.md): resident FP8
uses the Triton kernel only while flattened rows ``B*M <=
TRITON_MAX_M (128)``; chunk 256 (B=1) therefore takes the reference
dequant path, chunk 128 stays on Triton.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from inference_from_scratch.fp8_linear import TRITON_MAX_M  # noqa: E402
from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    load_rank_state, resident_memory_report)
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402
from run_bench import build_token_ids, load_tokenizer  # noqa: E402

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")


def _time_prefill(model, idx, chunk_size, dev):
    """(seconds, last_argmax) for one fresh-state prefill."""
    state = model.static_state(batch_size=1, max_seq_len=int(idx.shape[1]) + 8)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    if chunk_size is None:
        logits = model.prefill_static(idx, state)
        last = logits[:, -1]
    else:
        last = model.prefill_chunkwise_static(idx, state, chunk_size=int(chunk_size))
    torch.cuda.synchronize(dev)
    secs = time.perf_counter() - t0
    return secs, int(last.float().argmax(dim=-1).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", default="benchmarks/baseline/ours_tp2_prefill_bench.json")
    ap.add_argument("--fp8-path", choices=["triton", "reference"], default="triton")
    ap.add_argument("--lengths", default="512,2048")
    args = ap.parse_args()

    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    mem = resident_memory_report(snapshot, config=config, world=ctx.world_size,
                                 manifest=manifest, lm_head_mode="vocab")
    if rank == 0:
        print(f"resident estimate: {float(mem['per_rank_total_gib']):.3f} GiB/rank", flush=True)
    tok = load_tokenizer(snapshot)
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]

    t0 = time.perf_counter()
    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=(args.fp8_path == "triton"),
        manifest=manifest, lm_head_mode="vocab", include_mtp=False,
        use_scale_cache=False)
    torch.cuda.synchronize(dev)
    load_s = time.perf_counter() - t0
    # Fused RMSNorm, as in the measured TP2 runner default.
    from inference_from_scratch.hybrid import QwenRMSNorm
    for mod in model.modules():
        if isinstance(mod, QwenRMSNorm):
            mod.use_fused_rmsnorm = True
    print(f"[rank {rank}] load {load_s:.1f}s resident={report['resident_bytes'] / 1024**3:.3f} GiB "
          f"TRITON_MAX_M={TRITON_MAX_M}", flush=True)

    rows = []
    with torch.inference_mode():
        # Warmup (untimed): one 128-token chunkwise prefill warms Triton.
        warm = torch.tensor([build_token_ids(128, tok)], dtype=torch.long, device=dev)
        _time_prefill(model, warm, 128, dev)
        for n in lengths:
            ids = build_token_ids(n, tok)
            idx = torch.tensor([ids], dtype=torch.long, device=dev)
            argmax_by_method: dict[str, int] = {}
            for label, chunk in (("prefill_static", None),
                                 ("chunkwise128", 128), ("chunkwise256", 256)):
                if chunk is None and n > 512:
                    rows.append({"prompt_tokens": n, "method": label, "chunk_size": None,
                                 "status": "capped_at_512_not_run", "seconds": None,
                                 "tok_per_s": None, "last_argmax": None})
                    print(f"[rank {rank}] len={n} {label}: capped_at_512_not_run", flush=True)
                    continue
                secs, amax = _time_prefill(model, idx, chunk, dev)
                argmax_by_method[label] = amax
                dispatch = ("n/a-dense-or-token-loop" if chunk is None else
                            ("triton" if 1 * int(chunk) <= TRITON_MAX_M else "reference-M>128"))
                rows.append({"prompt_tokens": n, "method": label, "chunk_size": chunk,
                             "status": "measured", "seconds": secs,
                             "tok_per_s": n / secs if secs > 0 else None,
                             "last_argmax": amax, "fp8_dispatch_B1": dispatch})
                print(f"[rank {rank}] len={n} {label}: {secs:.4f}s "
                      f"{n / secs:.1f} tok/s argmax={amax} fp8={dispatch}", flush=True)
            vals = list(argmax_by_method.values())
            equal = len(set(vals)) == 1 if vals else None
            rows.append({"prompt_tokens": n, "method": "argmax_equality",
                         "chunk_size": None, "status": "measured",
                         "seconds": None, "tok_per_s": None,
                         "last_argmax": vals[0] if vals else None,
                         "argmax_equal_across_measured_methods": equal,
                         "argmax_by_method": argmax_by_method})
            print(f"[rank {rank}] len={n} argmax_equal={equal} by_method={argmax_by_method}",
                  flush=True)

    ctx.barrier()
    if rank == 0:
        out = {"script": "scripts/prefill_bench_tp2.py", "snapshot": str(snapshot),
               "world": ctx.world_size, "fp8_path": args.fp8_path,
               "lm_head_mode": "vocab", "triton_max_m": int(TRITON_MAX_M),
               "note": ("ours-vs-ours prefill only; NOT a TTFT comparison vs vLLM. "
                        "prefill_static capped at 512 (labelled, not run at 2048). "
                        "B=1: chunk128 rows<=TRITON_MAX_M -> Triton FP8; "
                        "chunk256 -> reference dequant FP8."),
               "torch": torch.__version__,
               "gpus": [torch.cuda.get_device_name(i) for i in range(ctx.world_size)],
               "load_s_rank0": load_s, "rows": rows}
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
        print("PREFILL_BENCH_TP2 PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
