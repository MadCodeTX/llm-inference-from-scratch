#!/usr/bin/env python3
"""Qwen3.8-27B-FP8 OpenAI server under TP2 (torchrun, 2 ranks).

Only rank 0 serves HTTP; rank 1 runs ``participate`` -- it mirrors
every submit/step via ``broadcast_object_list`` and never opens a
socket. All ranks build the same TPEngineCore in the same order, so
collectives inside the batched decode stay in lockstep.

Adopted local decode defaults preserved (explicit flags, no library
default changed): --decode-mode graph, scale-cache on (graph-scoped
resolver), --fp8-config gemv-bk512, gated-norm fused, norm fused,
delta fused, lm_head vocab.

Run (approved GPU window, incumbent stopped):
  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/serve_qwen38_tp2.py \
      --port 8000 --model Qwen/Qwen3.8-27B-FP8
Then (same harness as vLLM):
  python benchmarks/run_bench.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen3.8-27B-FP8 --tokenizer-dir <snapshot>
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import run_qwen38_tp2 as runner  # noqa: E402
from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.openai_server import (  # noqa: E402
    BatchedEngineBackend, encode_lockstep_command,
    make_server, participate, participate_compact)
from inference_from_scratch.tp_checkpoint import load_rank_state, resident_memory_report  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402
from inference_from_scratch.tp_engine import TPEngineCore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=runner.DEFAULT_SNAPSHOT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    # Static KV state is slots*seq_len sized. Measured on 24 GiB/rank
    # (serving_readiness_round1): 8x16384 OOMs at graph capture; 8x8192
    # is the largest proven sizing. Larger shapes are explicit opt-ins.
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--tp-sync", choices=("object", "compact"), default="object",
                    help="lockstep transport: object=broadcast_object_list "
                         "(default), compact=flat-int tensor broadcast")
    ap.add_argument("--memory-guard-gib", type=float, default=runner.MEMORY_GUARD_GIB)
    ap.add_argument("--graph-capture-mode", choices=("real-state", "scratch"),
                    default="real-state",
                    help="graph warmup: real-state=snapshot/restore occupied "
                         "rows only (default, no second full state), "
                         "scratch=legacy full scratch state")
    ap.add_argument("--no-graph-capture-guard", action="store_true",
                    help="disable the fail-closed capture sizing guard "
                         "(capture may OOM instead of falling back to eager)")
    a = ap.parse_args()
    # Adopted defaults via existing explicit config helpers only.
    runner.validate_fp8_config("gemv-bk512")
    scale_cache = runner.resolve_scale_cache_mode(None, "graph")  # -> "on"
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(a.snapshot)
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    mem = resident_memory_report(snapshot, config=config, world=ctx.world_size,
                                 manifest=manifest, lm_head_mode="vocab")
    if float(mem["per_rank_total_gib"]) > a.memory_guard_gib:
        if rank == 0:
            print(f"FAIL: resident {mem['per_rank_total_gib']:.3f} GiB/rank > guard", flush=True)
        ctx.destroy(); return 2
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    t0 = time.perf_counter()
    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=True, manifest=manifest,
        lm_head_mode="vocab", include_mtp=False,
        use_scale_cache=(scale_cache == "on"))
    torch.cuda.synchronize(dev)
    runner.apply_norm_mode(model, "fused")
    runner.apply_gated_norm_mode(model, "fused")
    runner.apply_delta_mode(model, "fused")
    if rank == 0:
        print(f"loaded in {time.perf_counter()-t0:.1f}s resident="
              f"{report['resident_bytes']/(1024**3):.2f} GiB/rank decode=graph "
              f"scale_cache={scale_cache} fp8=gemv-bk512", flush=True)
    eos = int(tok.eos_token_id) if tok.eos_token_id is not None else 248044
    engine = TPEngineCore(model, ctx, max_slots=a.max_slots,
                          max_seq_len=a.max_seq_len, token_budget=a.token_budget,
                          eos_id=eos, decode_mode="graph",
                          graph_capture_mode=a.graph_capture_mode,
                          graph_capture_guard=not a.no_graph_capture_guard)

    def encode(text):
        try: ids = tok.encode(text, add_special_tokens=False)
        except TypeError: ids = tok.encode(text)
        return [int(t) for t in ids]

    def decode(ids):
        try: return tok.decode([int(t) for t in ids], skip_special_tokens=False)
        except TypeError: return tok.decode([int(t) for t in ids])

    if rank != 0:
        if a.tp_sync == "compact":
            def recv_command():
                hdr = torch.empty(1, dtype=torch.int64, device=dev)
                dist.broadcast(hdr, src=0)
                buf = torch.empty(int(hdr.item()), dtype=torch.int64, device=dev)
                dist.broadcast(buf, src=0)
                return buf.tolist()
            try: participate_compact(engine, recv_command)
            finally: ctx.destroy()
            return 0

        def recv():
            box = [None]
            dist.broadcast_object_list(box, src=0)
            m = box[0]
            return (m.get("submits", []), bool(m.get("step")), bool(m.get("shutdown")))
        try: participate(engine, recv, eos_id=eos)
        finally: ctx.destroy()
        return 0

    if a.tp_sync == "compact":
        def sync_hook(submits, do_step, shutdown=False):
            payload = encode_lockstep_command(submits, do_step, shutdown)
            hdr = torch.tensor([len(payload)], dtype=torch.int64, device=dev)
            dist.broadcast(hdr, src=0)
            dist.broadcast(torch.tensor(payload, dtype=torch.int64, device=dev),
                           src=0)
    else:
        def sync_hook(submits, do_step, shutdown=False):
            packed = [(list(s[0]), int(s[1]),
                       bool(s[2]) if len(s) > 2 else False,
                       s[3] if len(s) > 3 else None) for s in submits]
            dist.broadcast_object_list(
                [{"submits": packed, "step": bool(do_step),
                  "shutdown": bool(shutdown)}], src=0)

    backend = BatchedEngineBackend(a.model, engine, encode, decode,
                                   eos_id=eos, sync_hook=sync_hook)
    srv = make_server(backend, a.host, a.port, a.request_timeout)
    print(f"serving {a.model} on http://{a.host}:{a.port} rank0-only HTTP, "
          f"slots={a.max_slots} max_seq_len={a.max_seq_len} budget={a.token_budget}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try: backend.close()  # shutdown broadcast frees rank-1 participate()
        except Exception: pass
        srv.server_close()
        try: engine.close()
        except Exception: pass
        ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
