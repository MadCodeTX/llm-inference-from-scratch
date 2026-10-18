#!/usr/bin/env python3
"""Full Qwen3.8-27B-FP8 under true TP2 (torchrun, 2 ranks).

Each rank builds ``DistributedHybridLM`` from the REAL ``HybridConfig``
on ``cuda:LOCAL_RANK`` and loads ONLY its rank shards resident-FP8 via
``tp_checkpoint.load_rank_state`` (Triton FP8 by default; ``--fp8-path
reference`` selects the reference dequant fallback). This is tensor
parallelism, not the pipeline bridge in ``run_qwen38_gpu.py``: every
rank holds half of every projection and the row-parallel partials meet
in NCCL all-reduces (2 per block per decode step).

Choices (documented, cheaper correct option taken in both cases):

* Tokenizer: every rank encodes the probe prompts itself. The
  tokenizer is deterministic and local, so identical ids cost zero
  communication; rank 0 broadcasting the ids would add a collective
  for no benefit. Each rank asserts its ids against the recorded
  ``prompt_token_ids`` in the probes file.
* Next-token agreement: ``embed_tokens``/``lm_head`` are replicated
  (v1), so both ranks compute identical full logits from the
  all-reduced hidden state and ``argmax`` locally. No per-token
  broadcast is needed; tokens stay on-device and cross to the host
  ONCE at the end (batched transfer, as in the bridge's
  ``--sampling-sync batched``). ``--debug-token-sync`` instead
  all-gathers the argmax every step and asserts equality (slow;
  debug only).
* Norms: ``--norm fused`` (default) runs every QwenRMSNorm through
  the single-kernel ``F.rms_norm``; profiling (benchmarks/profile/
  SUMMARY_tp2.md) showed the manual multi-op norm on the host-bound
  decode path. ``--norm manual`` restores the v1 maths for A/B.

Run (approved GPU window, incumbent stopped):

    .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/run_qwen38_tp2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, QwenRMSNorm  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import build_manifest  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    load_rank_state, resident_memory_report)
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")
FP8_PATH_CHOICES = ("triton", "reference")
MEMORY_GUARD_GIB = 23.0


NORM_MODES = ("fused", "manual")
LM_HEAD_MODES = ("replicated", "vocab")
SCALE_CACHE_MODES = ("off", "on")
DECODE_MODES = ("eager", "graph")
FP8_CONFIG_MODES = ("old", "new")
SPECULATIVE_MODES = ("off", "mtp1", "mtp1-graph")


def validate_fp8_config(mode: str) -> str:
    if mode not in FP8_CONFIG_MODES:
        raise ValueError(f"unknown fp8-config {mode!r}, expected {FP8_CONFIG_MODES}")
    from inference_from_scratch.fp8_triton import set_config_mode
    return set_config_mode(mode)


def validate_decode_mode(mode: str) -> str:
    if mode not in DECODE_MODES:
        raise ValueError(f"unknown decode mode {mode!r}, expected {DECODE_MODES}")
    return mode


def validate_speculative_mode(mode: str, decode_mode: str = "eager") -> str:
    if mode not in SPECULATIVE_MODES:
        raise ValueError(f"unknown speculative mode {mode!r}, expected {SPECULATIVE_MODES}")
    if mode in ("mtp1", "mtp1-graph") and decode_mode != "eager":
        raise ValueError(f"--speculative {mode} requires --decode-mode eager "
                         "(speculation owns its static path; mtp1-graph captures "
                         "its own chunk/draft graphs)")
    return mode


def graph_decode_plan(prompt_len: int, max_new_tokens: int) -> dict:
    """Bound/plan for graph decode: static state sized once, asserted.

    max_seq_len = prompt + max_new_tokens covers prefill positions plus
    every decode replay (prompt + max_new_tokens - 1 <= max_seq_len).
    """
    if int(prompt_len) < 1:
        raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
    if int(max_new_tokens) < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    max_seq = int(prompt_len) + int(max_new_tokens)
    replays = int(max_new_tokens) - 1
    assert int(prompt_len) + replays <= max_seq, "graph decode would exceed max_seq_len"
    return {"prompt_len": int(prompt_len), "max_new_tokens": int(max_new_tokens),
            "max_seq_len": max_seq, "replays": replays, "warmup_steps": 3}


def apply_norm_mode(model, mode: str) -> int:
    """Set the RMSNorm implementation on every norm in ``model`` (A/B).

    ``fused`` (default) uses the single-kernel ``F.rms_norm`` path;
    ``manual`` restores the TP2 v1 multi-op maths. Returns the number
    of norms switched. Only this runner's model is touched; dense and
    pipeline runners keep the manual default.
    """
    if mode not in NORM_MODES:
        raise ValueError(f"unknown norm mode {mode!r}, expected {NORM_MODES}")
    n = 0
    for mod in model.modules():
        if isinstance(mod, QwenRMSNorm):
            mod.use_fused_rmsnorm = mode == "fused"
            n += 1
    return n


def use_triton_for_path(path: str) -> bool:
    """Map --fp8-path to the resident-FP8 execution switch."""
    if path == "triton":
        return True
    if path == "reference":
        return False
    raise ValueError(f"unknown fp8-path {path!r}, expected {FP8_PATH_CHOICES}")


def apply_scale_cache_mode(model, mode: str) -> int:
    """Enable/disable cached FP32 FP8 scales on every BlockFP8Linear."""
    if mode not in SCALE_CACHE_MODES:
        raise ValueError(f"unknown scale-cache mode {mode!r}, expected {SCALE_CACHE_MODES}")
    from inference_from_scratch.fp8_linear import apply_scale_cache
    return apply_scale_cache(model, mode == "on")


def compare_ids(gen_ids, ref_ids) -> dict:
    """Exact-match record of generated ids against a reference."""
    gen, ref = [int(v) for v in gen_ids], [int(v) for v in ref_ids]
    first = next((i for i, (a, b) in enumerate(zip(gen, ref)) if a != b), None)
    return {
        "exact_match": gen == ref,
        "matching_positions": sum(1 for a, b in zip(gen, ref) if a == b),
        "compared_tokens": min(len(gen), len(ref)),
        "first_divergence": first,
        "generated_at_divergence": (gen[first] if first is not None and first < len(gen) else None),
        "reference_at_divergence": (ref[first] if first is not None and first < len(ref) else None),
    }


def greedy_generate_tp(model, prompt_ids, max_new_tokens, *,
                       debug_token_sync: bool = False):
    """Greedy generation on a DistributedHybridLM (batched transfer).

    Logits are replicated, so the argmax is computed locally on each
    rank and the token never leaves the device inside the loop; one
    host transfer at the end collects ids/logit probes. With
    ``debug_token_sync`` the per-step argmax is all-gathered and
    cross-rank equality is asserted instead.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    dev = model.ctx.device
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    gen_t, finite_t = [], None

    def _take(logits):
        nonlocal finite_t
        last = logits[0, -1].float()
        finite_t = torch.isfinite(last).all() if finite_t is None else (
            finite_t & torch.isfinite(last).all())
        nxt = last.argmax()
        if debug_token_sync:
            gathered = [torch.empty_like(nxt) for _ in range(model.ctx.world_size)]
            dist.all_gather(gathered, nxt)
            if not all(int(g) == int(gathered[0]) for g in gathered):
                raise AssertionError(
                    f"cross-rank next-token mismatch: {[int(g) for g in gathered]}")
        gen_t.append(nxt)
        return nxt

    vocab_greedy = getattr(model, "lm_head_mode", "replicated") == "vocab" and not debug_token_sync
    torch.cuda.synchronize(dev)
    t_pre = time.perf_counter()
    if vocab_greedy:
        nxt_t, top_val, state = model.prefill_greedy(idx)
        finite_t = torch.isfinite(top_val).all()
        gen_t.append(nxt_t.reshape(()))
    else:
        logits, state = model.prefill(idx)
        nxt_t = _take(logits)
    torch.cuda.synchronize(dev)
    prefill_s = time.perf_counter() - t_pre
    t_dec = time.perf_counter()
    for _ in range(1, max_new_tokens):
        if vocab_greedy:
            nxt_t, top_val, state = model.step_greedy(nxt_t.reshape(1, 1), state)
            finite_t = torch.isfinite(top_val).all() if finite_t is None else (finite_t & torch.isfinite(top_val).all())
            gen_t.append(nxt_t.reshape(()))
        else:
            logits, state = model.step(nxt_t.reshape(1, 1), state)
            nxt_t = _take(logits)
    torch.cuda.synchronize(dev)
    decode_s = time.perf_counter() - t_dec
    gen_ids = [int(v) for v in torch.stack(gen_t).cpu().tolist()]
    return {"gen_ids": gen_ids, "logits_finite": bool(finite_t.cpu()),
            "prefill_s": prefill_s, "decode_s": decode_s}


def greedy_generate_graph_tp(model, prompt_ids, max_new_tokens, *,
                             warmup_steps: int = 3):
    """Greedy generation: prefill_static + one captured decode step replayed.

    Captured (per rank): one decode_token_static step over the real
    TPStaticState -- static input token tensor, static logits out buffer,
    position advances inside the graph via state.position (add_).
    Outside: prefill, warmup (3 eager static steps on a side stream over
    a scratch state), capture itself, argmax on static logits and its
    copy_ into the static input between replays, finite checks, EOS
    (none: fixed token count). Graph is deleted (sync) before return so
    process-group teardown cannot hang on a live captured NCCL graph.
    """
    validate_decode_mode("graph")
    plan = graph_decode_plan(len(prompt_ids), max_new_tokens)
    dev = model.ctx.device
    vocab = int(model.config.vocab_size)
    state = model.static_state(batch_size=1, max_seq_len=plan["max_seq_len"])
    assert state.max_seq_len >= plan["prompt_len"] + plan["replays"]
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    gen_t = []
    torch.cuda.synchronize(dev)
    t_pre = time.perf_counter()
    logits = model.prefill_static(idx, state)
    last = logits[0, -1].float()
    finite_t = torch.isfinite(last).all()
    nxt = last.argmax()
    gen_t.append(nxt.reshape(()))
    torch.cuda.synchronize(dev)
    prefill_s = time.perf_counter() - t_pre
    capture_s, decode_s = 0.0, 0.0
    if plan["replays"] > 0:
        # Warmup on a side stream over a scratch state (real state untouched).
        scratch = model.static_state(batch_size=1, max_seq_len=plan["max_seq_len"])
        warm_in = nxt.reshape(1).clone()
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(int(warmup_steps)):
                model.decode_token_static(warm_in, scratch)
        torch.cuda.current_stream(dev).wait_stream(side)
        del scratch
        if int(model.ctx.world_size) > 1:
            model.ctx.barrier()
        static_input = nxt.reshape(1).clone()
        static_out = torch.empty(1, vocab, device=dev, dtype=logits.dtype)
        torch.cuda.synchronize(dev)
        t_cap = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.decode_token_static(static_input, state, out=static_out)
        torch.cuda.synchronize(dev)
        capture_s = time.perf_counter() - t_cap
        if int(model.ctx.world_size) > 1:
            model.ctx.barrier()
        torch.cuda.synchronize(dev)
        t_dec = time.perf_counter()
        for _ in range(plan["replays"]):
            graph.replay()
            finite_t = finite_t & torch.isfinite(static_out.float()).all()
            nxt = static_out.float().argmax(dim=-1)
            gen_t.append(nxt.reshape(()))
            static_input.copy_(nxt)
        torch.cuda.synchronize(dev)
        decode_s = time.perf_counter() - t_dec
        torch.cuda.synchronize(dev)
        del graph
        torch.cuda.synchronize(dev)
    gen_ids = [int(v) for v in torch.stack(gen_t).cpu().tolist()]
    return {"gen_ids": gen_ids, "logits_finite": bool(finite_t.cpu()),
            "prefill_s": prefill_s, "decode_s": decode_s,
            "capture_s": capture_s, "graph_replays": plan["replays"],
            "max_seq_len": plan["max_seq_len"]}


def greedy_generate_speculative_tp(model, prompt_ids, max_new_tokens,
                                   mode: str = "mtp1", segment_profile=None):
    """Greedy MTP speculative generation (gamma=1).

    ``mtp1`` runs the eager static loop; ``mtp1-graph`` captures the
    verification chunk (and the MTP draft step) as CUDA graphs.
    Wraps ``speculative_generate_tp``; wall time is the combined
    prefill+speculative-decode time (the loop owns its prefill), so
    callers report an *effective* TPOT over all generated tokens.
    Finite-logit checking is inherited from the caller's comparison
    against the bridge/Arm-A references (exact ids imply finite).
    The per-phase profile (CUDA events) is returned under
    ``profile``.
    """
    validate_speculative_mode(mode, "eager")
    from inference_from_scratch.speculation import speculative_generate_tp
    dev = model.ctx.device
    mtp = getattr(model, "mtp_draft", None)
    if mtp is None:
        raise ValueError("speculative mtp1 needs model.mtp_draft "
                         "(load with include_mtp=True)")
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    profile: dict = {}
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    out, stats = speculative_generate_tp(
        model, mtp, idx, int(max_new_tokens),
        embedding=model.embed_tokens, lm_head=model._lm_head_logits,
        graph_verify=(mode == "mtp1-graph"),
        graph_draft=(mode == "mtp1-graph"),
        profile=profile)
    torch.cuda.synchronize(dev)
    wall_s = time.perf_counter() - t0
    gen_ids = [int(v) for v in out[0, len(prompt_ids):].tolist()]
    acceptance = (float(stats["accepted"]) / float(stats["proposed"])
                  if stats["proposed"] else 0.0)
    return {"gen_ids": gen_ids, "logits_finite": True,
            "prefill_s": 0.0, "decode_s": wall_s, "speculative": stats,
            "profile": profile,
            "acceptance": acceptance, "cycles": int(stats["cycles"]),
            "tpot_effective_ms": (1000.0 * wall_s / len(gen_ids)) if gen_ids else None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--probes", default="benchmarks/baseline/arm_a_quality_probes.json")
    ap.add_argument("--bridge", default="benchmarks/baseline/ours_pipeline_fastfp8.json")
    ap.add_argument("--out", default="benchmarks/baseline/ours_tp2_first_run.json")
    ap.add_argument("--fp8-path", choices=list(FP8_PATH_CHOICES), default="triton",
                    help="resident-FP8 execution: triton (default) or reference fallback")
    ap.add_argument("--norm", choices=list(NORM_MODES), default="fused",
                    help="RMSNorm: fused F.rms_norm (default, profiled) or "
                         "manual multi-op (TP2 v1; A/B baseline)")
    ap.add_argument("--lm-head", choices=list(LM_HEAD_MODES), default="vocab",
                    help="lm_head: vocab-parallel shard (default; measured winner in "
                         "benchmarks/baseline/ours_tp2_microopts_ab.json) or replicated")
    ap.add_argument("--scale-cache", choices=list(SCALE_CACHE_MODES), default="off",
                    help="cached FP32 FP8 scales: off (default until A/B) or on")
    ap.add_argument("--fp8-config", choices=list(FP8_CONFIG_MODES), default="new",
                    help="Triton FP8 tile config: new (default, M=1 BLOCK_N=32 tuned) "
                         "or old (legacy BLOCK_N=128 heuristic; A/B baseline)")
    ap.add_argument("--decode-mode", choices=list(DECODE_MODES), default="eager",
                    help="decode: eager (default until graph is measured) or graph "
                         "(one captured decode_token_static step replayed per token)")
    ap.add_argument("--speculative", choices=list(SPECULATIVE_MODES), default="off",
                    help="speculative: off (default), mtp1 (native MTP gamma=1 "
                         "greedy speculation on the eager static path) or "
                         "mtp1-graph (chunk+draft CUDA-graph speculation); loads mtp.*")
    ap.add_argument("--probe", default=None,
                    help="only run the probe with this id (e.g. capital)")
    ap.add_argument("--debug-token-sync", action="store_true",
                    help="all-gather + assert the argmax token every step (slow)")
    ap.add_argument("--memory-guard-gib", type=float, default=MEMORY_GUARD_GIB)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    validate_fp8_config(args.fp8_config)
    validate_speculative_mode(args.speculative, args.decode_mode)
    ctx = DistributedTPContext.from_env(backend="nccl", device_type="cuda")
    rank, dev = ctx.rank, ctx.device
    snapshot = Path(args.snapshot)
    probes = json.loads(Path(args.probes).read_text())["probes"]
    bridge_probes = {p["id"]: p for p in
                     json.loads(Path(args.bridge).read_text())["probes"]}
    if args.probe is not None:
        probes = [p for p in probes if p["id"] == args.probe]
        if not probes:
            raise ValueError(f"no probe with id {args.probe!r}")
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)

    mem = resident_memory_report(snapshot, config=config, world=ctx.world_size,
                                 manifest=manifest, lm_head_mode=args.lm_head)
    est_gib = float(mem["per_rank_total_gib"])
    if rank == 0:
        print(f"config: layers={config.num_hidden_layers} hidden={config.hidden_size} "
              f"world={ctx.world_size} fp8_path={args.fp8_path}", flush=True)
        print(f"resident estimate: {est_gib:.3f} GiB/rank "
              f"(guard {args.memory_guard_gib} GiB) fits24={mem['fits_24gib_per_rank']}",
              flush=True)
    if est_gib > args.memory_guard_gib:
        print(f"FAIL: resident estimate {est_gib:.3f} GiB/rank exceeds guard "
              f"{args.memory_guard_gib} GiB; aborting before load", flush=True)
        ctx.destroy()
        return 2

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = {}
    for probe in probes:  # every rank encodes identically (see docstring)
        ids = tok.encode(probe["prompt"])
        assert ids == probe["prompt_token_ids"], (ids, probe["id"])
        prompt_ids[probe["id"]] = ids

    t0 = time.perf_counter()
    model, report = load_rank_state(
        snapshot, config=config, rank=rank, world=ctx.world_size, device=dev,
        fp8_mode="resident", ctx=ctx, use_triton=use_triton_for_path(args.fp8_path),
        manifest=manifest, lm_head_mode=args.lm_head,
        include_mtp=(args.speculative in ("mtp1", "mtp1-graph")),
        use_scale_cache=(args.scale_cache == "on"))
    torch.cuda.synchronize(dev)
    load_s = time.perf_counter() - t0
    n_norms = apply_norm_mode(model, args.norm)
    n_scale = apply_scale_cache_mode(model, args.scale_cache) if args.scale_cache != "on" else report.get("scale_cache_modules", 0)
    if rank == 0:
        print(f"norm_mode={args.norm} norms_switched={n_norms} lm_head={args.lm_head} "
              f"scale_cache={args.scale_cache} scale_modules={n_scale}", flush=True)
    resident_gib = report["resident_bytes"] / (1024 ** 3)
    print(f"[rank {rank}] load {load_s:.1f}s loaded={report['loaded_count']} "
          f"resident={resident_gib:.3f} GiB", flush=True)
    if resident_gib > args.memory_guard_gib:
        print(f"FAIL: rank {rank} resident {resident_gib:.3f} GiB exceeds guard",
              flush=True)
        ctx.destroy()
        return 2

    torch.cuda.reset_peak_memory_stats(dev)
    results = []
    with torch.inference_mode():
        for probe in probes:
            ref_arm = probe["output_token_ids"]
            ref_bridge = bridge_probes[probe["id"]]["generated_token_ids"]
            validate_decode_mode(args.decode_mode)
            if args.speculative in ("mtp1", "mtp1-graph"):
                gen = greedy_generate_speculative_tp(
                    model, prompt_ids[probe["id"]], args.max_new_tokens,
                    mode=args.speculative)
            elif args.decode_mode == "graph":
                gen = greedy_generate_graph_tp(model, prompt_ids[probe["id"]],
                                               args.max_new_tokens)
            else:
                gen = greedy_generate_tp(model, prompt_ids[probe["id"]],
                                         args.max_new_tokens,
                                         debug_token_sync=args.debug_token_sync)
            gen_ids, n = gen["gen_ids"], len(gen["gen_ids"])
            decode_tok_s = (n - 1) / gen["decode_s"] if gen["decode_s"] > 0 and n > 1 else None
            row = {
                "id": probe["id"], "prompt": probe["prompt"],
                "prompt_token_ids": prompt_ids[probe["id"]],
                "generated_token_ids": gen_ids,
                "generated_text": tok.decode(gen_ids),
                "logits_finite": gen["logits_finite"],
                "vs_bridge": compare_ids(gen_ids, ref_bridge),
                "vs_arm_a": compare_ids(gen_ids, ref_arm),
                "prefill_s": gen["prefill_s"], "decode_s": gen["decode_s"],
                "capture_s": gen.get("capture_s"),
                "graph_replays": gen.get("graph_replays"),
                "max_seq_len": gen.get("max_seq_len"),
                "decode_mode": args.decode_mode,
                "speculative": args.speculative,
                "speculative_stats": gen.get("speculative"),
                "profile": gen.get("profile"),
                "acceptance": gen.get("acceptance"),
                "cycles": gen.get("cycles"),
                "tpot_effective_ms": gen.get("tpot_effective_ms"),
                "fp8_config": args.fp8_config,
                "decode_tok_per_s": decode_tok_s,
                "tpot_ms": (1000.0 * gen["decode_s"] / (n - 1)) if n > 1 else None,
                "peak_memory_bytes_rank_local": int(torch.cuda.max_memory_allocated(dev)),
            }
            results.append(row)
            print(f"[rank {rank}] [{probe['id']}] finite={gen['logits_finite']} "
                  f"bridge={row['vs_bridge']['matching_positions']}/{n} "
                  f"arm={row['vs_arm_a']['matching_positions']}/{n} "
                  f"first_div={row['vs_bridge']['first_divergence']} "
                  f"prefill={gen['prefill_s']:.3f}s decode={decode_tok_s:.2f} tok/s",
                  flush=True)

    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, {
        "rank": rank, "load_time_s": load_s,
        "resident_bytes": int(report["resident_bytes"]),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(dev)),
        "probes": [{k: r[k] for k in ("id", "generated_token_ids", "logits_finite")}
                   for r in results]})
    if rank == 0:
        for other in gathered[1:]:
            for mine, theirs in zip(results, other["probes"]):
                assert mine["generated_token_ids"] == theirs["generated_token_ids"], (
                    f"rank token mismatch on {mine['id']}")
        out = {
            "script": "scripts/run_qwen38_tp2.py",
            "snapshot": str(snapshot), "world": ctx.world_size,
            "parallelism": "tensor_parallel_tp2",
            "fp8_path": args.fp8_path,
            "norm_mode": args.norm,
            "lm_head_mode": args.lm_head,
            "scale_cache": args.scale_cache,
            "decode_mode": args.decode_mode,
            "speculative": args.speculative,
            "fp8_config": args.fp8_config,
            "tokenizer_choice": "all ranks encode identically (deterministic, zero comms)",
            "token_agreement": ("replicated logits + local argmax, batched host "
                                "transfer at end; --debug-token-sync asserts per-step"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(ctx.world_size)],
            "max_new_tokens": args.max_new_tokens,
            "resident_estimate_gib_per_rank": est_gib,
            "load_time_s_by_rank": {g["rank"]: g["load_time_s"] for g in gathered},
            "resident_bytes_by_rank": {g["rank"]: g["resident_bytes"] for g in gathered},
            "peak_memory_bytes_by_rank": {g["rank"]: g["peak_memory_bytes"] for g in gathered},
            "roofline_tpot_floor_ms": 19.17, "arm_a_tpot_ms": 20.3,
            "probes": results,
        }
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    ok = len(results) == len(probes) and all(r["logits_finite"] for r in results)
    ctx.barrier()
    if rank == 0:
        print("PASS" if ok else "FAIL", flush=True)
    ctx.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
