"""CUDA-graph smoke for ONE captured static hybrid decode step.

Tiny hybrid (3 linear + 1 full, D=32, vocab 97, seed 3) on ``cuda:0``
in fp32 over a :class:`StaticHybridState`:

* warmup on a side stream, then capture ONE
  ``TinyHybridLM.decode_token_static`` step into a
  ``torch.cuda.CUDAGraph`` with a static input-token tensor and a
  static output-logits buffer (``static_output.copy_(logits)`` inside
  the graph; ``decode_token_static`` itself returns fresh logits);
* greedy feedback stays OUTSIDE the graph: between replays the next
  token is ``argmax`` on the static output, fed back with ``copy_``
  into the static input. Sampling is not captured (v1);
* 16 graph-replayed tokens/logits are compared against the eager
  static path (same greedy feedback), and eager vs graph us/token is
  timed with CUDA events. The tiny model isolates launch overhead:
  at these dims nearly all eager time is kernel launches + the
  per-step host sync, not maths.

Graph-safety note: during capture ``decode_token_static`` keeps the
position as the state's 0-dim tensor (no ``.item()``), gathers RoPE
rows with ``index_select``, writes KV with ``index_copy_`` at the
tensor position, and masks the fixed ``max_seq_len`` KV buffer
on-device. Python overflow/equality checks cannot run during
capture, so this script guarantees ``tokens <= max_seq_len`` upfront.

Usage: .venv-gpu/bin/python scripts/cuda_graph_smoke.py
Exit 1 on CUDA absence, token mismatch, or logit diff past 1e-5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402

D, VOCAB = 32, 97
SEED = 3
TOL = 1e-5


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _median(xs):
    ordered = sorted(xs)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seed-token", type=int, default=7)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--max-seq-len", type=int, default=32)
    args = ap.parse_args()
    if args.tokens <= 0 or args.repeats <= 0:
        ap.error("--tokens/--repeats must be positive")
    if args.tokens > args.max_seq_len:
        ap.error("--tokens must be <= --max-seq-len (capture cannot check overflow)")
    if not torch.cuda.is_available():
        print("CUDA unavailable; nothing to smoke-test")
        return 1
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = TinyHybridLM(_tiny_config()).eval().to(dev)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={dev} ({torch.cuda.get_device_name(dev)}) seed={args.seed}")
    print(f"tiny config: D={D} VOCAB={VOCAB} layers=3xlinear+1xfull fp32 "
          f"tokens={args.tokens} max_seq_len={args.max_seq_len} tol={TOL}")
    seed_token = torch.tensor([args.seed_token], dtype=torch.long, device=dev)
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"  -> {'PASS' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)

    with torch.no_grad():
        # ---- eager static path, greedy feedback outside the step ----
        state_e = model.static_state(batch_size=1, max_seq_len=args.max_seq_len)
        tok = seed_token.clone()
        eager_tokens, eager_logits = [], []
        for _ in range(args.tokens):
            logits = model.decode_token_static(tok, state_e)
            eager_logits.append(logits.clone())
            tok = logits.argmax(dim=-1)
            eager_tokens.append(int(tok.item()))
        eager_logits_t = torch.stack(eager_logits)

        # ---- capture ONE decode step --------------------------------
        state_g = model.static_state(batch_size=1, max_seq_len=args.max_seq_len)
        static_input = seed_token.clone()
        static_output = torch.zeros(1, VOCAB, device=dev)
        # Warmup on a side stream (PyTorch CUDA-graph requirement);
        # tensor position exercises the same graph-safe kernels.
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(3):
                model.decode_token_static(
                    static_input, state_g, position=state_g.position)
        torch.cuda.current_stream(dev).wait_stream(side)
        state_g.reset()
        static_input.copy_(seed_token)
        torch.cuda.reset_peak_memory_stats(dev)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_output.copy_(
                    model.decode_token_static(static_input, state_g))
        except Exception as exc:  # capture blocker: report exactly, exit 1
            print(f"CAPTURE FAIL: {type(exc).__name__}: {exc}")
            return 1
        print("capture: OK (one decode_token_static step, "
              "static input + static output buffers)")

        # ---- 16 graph replays, greedy feedback OUTSIDE the graph ----
        state_g.reset()
        static_input.copy_(seed_token)
        graph_tokens, graph_logits = [], []
        for _ in range(args.tokens):
            graph.replay()
            graph_logits.append(static_output.clone())
            nxt = static_output.argmax(dim=-1)
            graph_tokens.append(int(nxt.item()))
            static_input.copy_(nxt)
        graph_logits_t = torch.stack(graph_logits)
        max_diff = (graph_logits_t - eager_logits_t).abs().max().item()
        tokens_equal = graph_tokens == eager_tokens
        print(f"correctness: tokens_equal={tokens_equal} "
              f"eager_tokens={eager_tokens} graph_tokens={graph_tokens}")
        print(f"correctness: logits max_abs_diff={max_diff:.3e} "
              f"state_position={state_g.seq_len} (expected {args.tokens})")
        check("graph tokens == eager tokens", tokens_equal)
        check("graph logits == eager logits", max_diff < TOL)
        check("position advanced per replay", state_g.seq_len == args.tokens)

        # ---- timing: CUDA events, median us/token -------------------
        def time_eager() -> float:
            state_e.reset()
            t = seed_token.clone()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.tokens):
                lg = model.decode_token_static(t, state_e)
                t = lg.argmax(dim=-1)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000.0 / args.tokens

        def time_graph() -> float:
            state_g.reset()
            static_input.copy_(seed_token)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.tokens):
                graph.replay()
                static_input.copy_(static_output.argmax(dim=-1))
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000.0 / args.tokens

        # One untimed pass each so timing starts from a warm allocator.
        time_eager(); time_graph()
        eager_us = _median([time_eager() for _ in range(args.repeats)])
        graph_us = _median([time_graph() for _ in range(args.repeats)])
        peak = torch.cuda.max_memory_allocated(dev)
        print(json.dumps({
            "event": "timing", "device": str(dev),
            "eager_us_per_token_median": eager_us,
            "graph_us_per_token_median": graph_us,
            "speedup_eager_over_graph": eager_us / graph_us if graph_us else None,
            "tokens_per_run": args.tokens, "repeats": args.repeats,
            "peak_memory_bytes": peak,
            "note": "tiny model: isolates launch overhead; greedy argmax "
                    "+ copy_ feedback is outside the graph in both paths",
        }))
        print(f"timing: eager={eager_us:.1f} us/token graph={graph_us:.1f} us/token "
              f"speedup={eager_us / graph_us:.2f}x peak_memory={peak} B "
              f"({peak / 2**20:.2f} MiB)")
        check("graph faster than eager (launch overhead removed)", graph_us < eager_us)

    print("RESULT:", "FAIL " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
