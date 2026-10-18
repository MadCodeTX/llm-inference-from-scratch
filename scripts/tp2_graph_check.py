#!/usr/bin/env python
"""TP2 CUDA-graph check: capture ONE decode_token_static step per rank.

Run (always wrap in timeout so a capture deadlock cannot hang forever):
  timeout 180 .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_graph_check.py
  timeout 180 .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_graph_check.py --mode collective

Model mode: tiny DistributedHybridLM (as tp2_static_check) on NCCL/cuda.
Warmup eager static steps on a side stream, dist.barrier, capture
decode_token_static into a per-rank torch.cuda.CUDAGraph (static input
token tensor copied between replays, static logits out), replay greedy
tokens (argmax OUTSIDE graph on static logits; token copy_ into static
input), compare vs eager static tokens/logits, report eager/graph
us/token (CUDA events around loop, excluding capture).

Collective mode: minimal 1-collective capture probe (single all_reduce)
to bisect an NCCL-capture blocker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedHybridLM, DistributedTPContext  # noqa: E402


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def collective_probe(ctx, dev) -> int:
    rank = ctx.rank
    print(f"[rank {rank}] collective probe: warmup all_reduce", flush=True)
    x = torch.ones(4, device=dev)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        for _ in range(3):
            y = torch.ones(4, device=dev)
            dist.all_reduce(y)
    torch.cuda.current_stream(dev).wait_stream(side)
    ctx.barrier()
    print(f"[rank {rank}] collective probe: capturing one all_reduce", flush=True)
    static_in = torch.ones(4, device=dev)
    static_out = torch.empty(4, device=dev)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            static_out.copy_(static_in)
            dist.all_reduce(static_out)
    except Exception as exc:
        print(f"[rank {rank}] COLLECTIVE_CAPTURE_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(f"[rank {rank}] collective probe: capture OK, replaying", flush=True)
    static_in.fill_(2.0)
    g.replay()
    torch.cuda.synchronize(dev)
    expected = 2.0 * ctx.world_size
    ok = bool(torch.allclose(static_out, torch.full_like(static_out, expected)))
    print(f"[rank {rank}] collective_out={static_out.tolist()} expected={expected} ok={ok}", flush=True)
    ctx.barrier()
    if rank == 0:
        print("TP2_COLLECTIVE_GRAPH_CHECK " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="model", choices=["model", "collective"])
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=32)
    p.add_argument("--seed-token", type=int, default=7)
    p.add_argument("--lm-head", default="replicated", choices=["replicated", "vocab"])
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    if args.tokens < 1 or args.tokens > args.max_seq_len:
        p.error("--tokens must be in [1, --max-seq-len] (capture cannot check overflow)")
    if not torch.cuda.is_available():
        print("CUDA unavailable", flush=True)
        return 1
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    print(f"[rank {rank}] torch={torch.__version__} device={dev} world={ctx.world_size} mode={args.mode}", flush=True)
    if args.mode == "collective":
        rc = collective_probe(ctx, dev)
        ctx.destroy()
        return rc

    cfg = _tiny_cfg()
    torch.manual_seed(1234)
    dense = TinyHybridLM(cfg).to(dev)
    dense.eval()
    tp = DistributedHybridLM.from_dense(dense, ctx, config=cfg, lm_head_mode=args.lm_head)
    tp.eval()
    vocab = cfg.vocab_size
    seed = torch.tensor([args.seed_token], dtype=torch.long, device=dev)

    with torch.no_grad():
        # Eager static reference.
        state_e = tp.static_state(batch_size=1, max_seq_len=args.max_seq_len)
        tok = seed.clone()
        eager_tokens, eager_logits = [], []
        for _ in range(args.tokens):
            lg = tp.decode_token_static(tok, state_e)
            eager_logits.append(lg.clone())
            tok = lg.argmax(dim=-1)
            eager_tokens.append(int(tok.item()))
        eager_logits_t = torch.stack(eager_logits)
        print(f"[rank {rank}] eager_tokens={eager_tokens}", flush=True)

        # Warmup eager static steps on a side stream (scratch state).
        state_g = tp.static_state(batch_size=1, max_seq_len=args.max_seq_len)
        static_input = seed.clone()
        static_out = torch.zeros(1, vocab, device=dev, dtype=eager_logits_t.dtype)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(3):
                tp.decode_token_static(static_input, state_g)
        torch.cuda.current_stream(dev).wait_stream(side)
        state_g.reset()
        static_input.copy_(seed)
        ctx.barrier()
        print(f"[rank {rank}] capturing one decode_token_static step (lm_head={args.lm_head})", flush=True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                tp.decode_token_static(static_input, state_g, out=static_out)
        except Exception as exc:
            print(f"[rank {rank}] CAPTURE_FAIL {type(exc).__name__}: {exc}", flush=True)
            ctx.barrier()
            ctx.destroy()
            return 1
        print(f"[rank {rank}] capture OK", flush=True)
        ctx.barrier()

        # Replay.
        state_g.reset()
        static_input.copy_(seed)
        torch.cuda.synchronize(dev)
        graph_tokens, graph_logits = [], []
        for _ in range(args.tokens):
            graph.replay()
            graph_logits.append(static_out.clone())
            nxt = static_out.argmax(dim=-1)
            graph_tokens.append(int(nxt.item()))
            static_input.copy_(nxt)
        graph_logits_t = torch.stack(graph_logits)
        max_diff = (graph_logits_t.float() - eager_logits_t.float()).abs().max().item()
        tokens_equal = graph_tokens == eager_tokens
        pos = state_g.seq_len
        print(f"[rank {rank}] graph_tokens={graph_tokens} tokens_equal={tokens_equal} "
              f"logits_max_abs_diff={max_diff:.3e} position={pos}", flush=True)

        # Timing: CUDA events around loops, excluding capture.
        def time_eager():
            state_e.reset()
            t = seed.clone()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.tokens):
                lg = tp.decode_token_static(t, state_e)
                t = lg.argmax(dim=-1)
            e.record(); e.synchronize()
            return s.elapsed_time(e) * 1000.0 / args.tokens

        def time_graph():
            state_g.reset()
            static_input.copy_(seed)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.tokens):
                graph.replay()
                static_input.copy_(static_out.argmax(dim=-1))
            e.record(); e.synchronize()
            return s.elapsed_time(e) * 1000.0 / args.tokens

        time_eager(); time_graph()
        eager_us = _median([time_eager() for _ in range(args.repeats)])
        graph_us = _median([time_graph() for _ in range(args.repeats)])
        if rank == 0:
            print(json.dumps({"event": "timing", "eager_us_per_token": eager_us,
                              "graph_us_per_token": graph_us,
                              "speedup": eager_us / graph_us if graph_us else None,
                              "tokens": args.tokens, "repeats": args.repeats}), flush=True)
        print(f"[rank {rank}] timing eager={eager_us:.1f} us/token graph={graph_us:.1f} us/token", flush=True)
        ok = tokens_equal and max_diff < 1e-4 and pos == args.tokens
        # graph faster not required for correctness PASS on tiny TP (NCCL in graph may differ); report only.
    ctx.barrier()
    if rank == 0:
        print("TP2_GRAPH_CHECK " + ("PASS" if ok else "FAIL"), flush=True)
    print(f"[rank {rank}] cleanup: sync+delete graph", flush=True)
    torch.cuda.synchronize(dev)
    del graph
    torch.cuda.synchronize(dev)
    print(f"[rank {rank}] cleanup: barrier", flush=True)
    ctx.barrier()
    print(f"[rank {rank}] cleanup: destroy process group", flush=True)
    ctx.destroy()
    print(f"[rank {rank}] cleanup: done", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
