#!/usr/bin/env python3
"""Full Qwen3.8-27B-FP8 end-to-end across both GPUs (pipeline-by-layers).

Builds HybridLM from the REAL HybridConfig on meta, converts checkpoint
FP8 linears to resident BlockFP8Linear, places embed + layers
0..split-1 on cuda:0 and layers split..63 + norm + lm_head on cuda:1,
streams tensors layer-by-layer via hybrid_checkpoint (never a full CPU
state dict), and greedy-generates for the baseline quality probes.
Hidden states cross devices only at the split boundary, inside
DeviceMappedHybridRunner. Run only in an approved GPU window
(incumbent stopped first via /home/nick/llm/vllm/launch.sh --stop):

    .venv-gpu/bin/python scripts/run_qwen38_gpu.py --split 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.fp8_linear import BlockFP8Linear, scaled_mm_block_supported
from inference_from_scratch.hybrid import (
    DeviceMappedHybridRunner, HybridConfig, HybridLM, pipeline_device_plan,
)
from inference_from_scratch.hybrid_checkpoint import build_manifest, load_into_model

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")

# Fastest correct measured path is Triton (bench: 6-27x vs reference at
# M=1..128, rel diff <=8e-3). scaled_mm is probed-but-unusable for the
# checkpoint layout, so it stays an explicit opt-in that falls back.
FP8_PATH_CHOICES = ("reference", "triton", "scaled-mm")


def fp8_kwargs_for_path(path: str) -> dict:
    """Map --fp8-path to BlockFP8Linear kwargs."""
    if path == "reference":
        return {"use_triton": False, "use_fast_path": False}
    if path == "triton":
        return {"use_triton": True, "use_fast_path": False}
    if path == "scaled-mm":
        return {"use_triton": False, "use_fast_path": True}
    raise ValueError(f"unknown fp8-path {path!r}, expected {FP8_PATH_CHOICES}")


def summarize_divergence(gen_ids, ref_ids, got_logits=None, ref_logits=None,
                         top1_logits=None, top2_logits=None):
    """First-divergence record with logit quantification (or None)."""
    first = next((i for i, (a, b) in enumerate(zip(gen_ids, ref_ids)) if a != b), None)
    if first is None:
        return None
    out = {"step": int(first), "generated_token": int(gen_ids[first]),
           "reference_token": int(ref_ids[first])}
    if got_logits is not None and ref_logits is not None:
        out["generated_logit"] = float(got_logits[first])
        out["reference_logit"] = float(ref_logits[first])
        out["logit_diff_got_minus_ref"] = float(got_logits[first] - ref_logits[first])
    if top1_logits is not None and top2_logits is not None:
        out["top1_logit"] = float(top1_logits[first])
        out["top2_logit"] = float(top2_logits[first])
        out["top1_top2_margin"] = float(top1_logits[first] - top2_logits[first])
    return out


def greedy_generate_bridge(runner, prompt_ids, max_new_tokens, device1,
                             sampling_sync="batched", ref_ids=None):
    """Greedy generation via the pipeline runner.

    sampling_sync="per-token" reproduces the original loop (argmax
    .item() + isfinite sync every token). "batched" keeps argmax,
    finite-flag and per-token logit probes on device1 and does ONE
    host transfer at the end (profile: sampling_host_sync_wall was a
    measured per-token stall; the token itself never needs the host
    because the runner moves idx to device0 internally).
    Returns dict with gen_ids/finite/prefill_s/decode_s/divergence.
    """
    if sampling_sync not in ("per-token", "batched"):
        raise ValueError(f"sampling_sync must be per-token|batched, got {sampling_sync!r}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    dev1 = torch.device(device1)
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long)
    ref = list(ref_ids) if ref_ids is not None else []
    if sampling_sync == "per-token":
        gen_ids: list[int] = []
        finite = True
        got_l, ref_l, t1_l, t2_l = [], [], [], []
        _sync(str(runner.device0), str(runner.device1))
        t_pre = time.perf_counter()
        logits, state = runner.run(idx)
        last = logits[0, -1]
        finite = finite and bool(torch.isfinite(last).all())
        topv, _ = last.float().topk(2)
        nxt = int(last.argmax().item())
        got_l.append(float(last[nxt]))
        if ref:
            ref_l.append(float(last[ref[0]]))
        t1_l.append(float(topv[0])); t2_l.append(float(topv[1]))
        _sync(str(runner.device0), str(runner.device1))
        prefill_s = time.perf_counter() - t_pre
        gen_ids.append(nxt)
        t_dec = time.perf_counter()
        for step_i in range(1, max_new_tokens):
            step = torch.tensor([[nxt]], dtype=torch.long, device=dev1)
            logits, state = runner.run(step, state=state)
            last = logits[0, -1]
            finite = finite and bool(torch.isfinite(last).all())
            topv, _ = last.float().topk(2)
            nxt = int(last.argmax().item())
            if step_i < len(ref):
                ref_l.append(float(last[ref[step_i]]))
            got_l.append(float(last[nxt])); t1_l.append(float(topv[0])); t2_l.append(float(topv[1]))
            gen_ids.append(nxt)
        _sync(str(runner.device0), str(runner.device1))
        decode_s = time.perf_counter() - t_dec
        div = summarize_divergence(gen_ids, ref, got_l if ref else None,
                                     ref_l if ref else None, t1_l, t2_l) if ref else None
        return {"gen_ids": gen_ids, "logits_finite": bool(finite),
                "prefill_s": prefill_s, "decode_s": decode_s,
                "divergence_logits": div}
    # batched: no per-token host sync
    gen_t, got_t, ref_t, t1_t, t2_t = [], [], [], [], []
    finite_t = None
    _sync(str(runner.device0), str(runner.device1))
    t_pre = time.perf_counter()
    logits, state = runner.run(idx)
    last = logits[0, -1].float()
    finite_t = torch.isfinite(last).all()
    topv, _ = last.topk(2)
    nxt_t = last.argmax()
    gen_t.append(nxt_t); got_t.append(last[nxt_t])
    if ref:
        ref_t.append(last[ref[0]])
    t1_t.append(topv[0]); t2_t.append(topv[1])
    _sync(str(runner.device0), str(runner.device1))
    prefill_s = time.perf_counter() - t_pre
    t_dec = time.perf_counter()
    for step_i in range(1, max_new_tokens):
        step = nxt_t.reshape(1, 1).to(dtype=torch.long)
        logits, state = runner.run(step, state=state)
        last = logits[0, -1].float()
        finite_t = finite_t & torch.isfinite(last).all()
        topv, _ = last.topk(2)
        nxt_t = last.argmax()
        gen_t.append(nxt_t); got_t.append(last[nxt_t])
        if step_i < len(ref):
            ref_t.append(last[ref[step_i]])
        t1_t.append(topv[0]); t2_t.append(topv[1])
    _sync(str(runner.device0), str(runner.device1))
    decode_s = time.perf_counter() - t_dec
    # single host transfer for everything collected on GPU
    gen_ids = torch.stack(gen_t).cpu().tolist()
    finite = bool(finite_t.cpu()) if finite_t is not None else True
    got_l = torch.stack(got_t).cpu().tolist() if got_t else []
    ref_l = torch.stack(ref_t).cpu().tolist() if ref_t else []
    t1_l = torch.stack(t1_t).cpu().tolist() if t1_t else []
    t2_l = torch.stack(t2_t).cpu().tolist() if t2_t else []
    div = summarize_divergence(gen_ids, ref, got_l if ref else None,
                                 ref_l if ref else None, t1_l, t2_l) if ref else None
    return {"gen_ids": [int(v) for v in gen_ids], "logits_finite": bool(finite),
            "prefill_s": prefill_s, "decode_s": decode_s,
            "divergence_logits": div}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--device0", default="cuda:0")
    ap.add_argument("--device1", default="cuda:1")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--probes", default="benchmarks/baseline/arm_a_quality_probes.json")
    ap.add_argument("--out", default="benchmarks/baseline/ours_pipeline_first_run.json")
    ap.add_argument("--log-rms", action="store_true")
    ap.add_argument("--fp8-path", choices=list(FP8_PATH_CHOICES), default="triton",
                    help="FP8 linear path (default triton: fastest correct measured; "
                         "reference keeps the exact dequant path reproducible; "
                         "scaled-mm is probed-unusable and falls back to reference)")
    ap.add_argument("--sampling-sync", choices=["per-token", "batched"], default="batched",
                    help="per-token reproduces the original .item()-per-token loop; "
                         "batched (default) does one host transfer at the end")
    ap.add_argument("--grad-mode", choices=["inference", "no-grad"], default="inference",
                    help="inference (default) uses torch.inference_mode; no-grad reproduces torch.no_grad")
    ap.add_argument("--dump-hidden", default=None, metavar="PATH",
                    help="also save per-probe {token_ids, hidden_pre_norm, "
                         "hidden_post_norm} (BF16, hidden rows aligned with "
                         "token_ids[:-1]) to this .pt path; default off")
    return ap


def _place(module: nn.Module, device: str) -> None:
    module.to_empty(device=device)  # meta -> device (load overwrites data)
    for p in module.parameters():
        p.data = p.data.to(dtype=torch.bfloat16)


def _sync(*devices: str) -> None:
    for d in devices:
        if torch.device(d).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(d)


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("FAIL: need 2 CUDA devices")
        return 1
    snapshot = Path(args.snapshot)
    probes_doc = json.loads(Path(args.probes).read_text())
    probes = probes_doc["probes"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    config = HybridConfig.from_text_config(snapshot)
    manifest = build_manifest(snapshot)
    fp8_modules = {
        e.target_name[: -len(".weight")]
        for e in manifest.entries
        if e.kind == "fp8_weight" and e.target_name is not None
        and not e.target_name.startswith("mtp.")
    }
    fp8_kwargs = fp8_kwargs_for_path(args.fp8_path)
    if args.fp8_path == "scaled-mm":
        supported, reason = scaled_mm_block_supported(
            torch.ones(1, 1, dtype=torch.bfloat16))
        print(f"WARNING: --fp8-path scaled-mm requested but not usable "
              f"({reason}); BlockFP8Linear will fall back to reference", flush=True)
    print(f"config: layers={config.num_hidden_layers} hidden={config.hidden_size} "
          f"fp8_modules={len(fp8_modules)} fp8_path={args.fp8_path} "
          f"sampling_sync={args.sampling_sync} grad_mode={args.grad_mode}", flush=True)

    t_build = time.perf_counter()
    with torch.device("meta"):
        model = HybridLM(config)
        converted = 0
        for name, module in list(model.named_modules()):
            if name in fp8_modules and isinstance(module, nn.Linear) \
                    and not isinstance(module, BlockFP8Linear):
                repl = BlockFP8Linear(module.in_features, module.out_features,
                                      bias=False, **fp8_kwargs)
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, repl)
                converted += 1
    print(f"meta build + fp8 convert ({converted}) in "
          f"{time.perf_counter() - t_build:.1f}s", flush=True)

    plan = pipeline_device_plan(config.num_hidden_layers, args.split,
                                args.device0, args.device1)
    _place(model.embed_tokens, plan["embed"])
    for i, block in enumerate(model.layers):
        _place(block, plan["layers"][i])
    _place(model.norm, plan["norm"])
    _place(model.lm_head, plan["lm_head"])
    model.eval()

    t0 = time.perf_counter()
    report = load_into_model(model, snapshot, fp8_mode="resident",
                             strict=True, manifest=manifest)
    load_s = time.perf_counter() - t0
    print(f"load: ok={report['ok']} loaded_count={report['loaded_count']} "
          f"resident={len(report['resident_targets'])} in {load_s:.1f}s", flush=True)
    if not report["ok"]:
        print(f"FAIL validation: {report}")
        return 1

    rms_log: list[dict] = []

    def hook(layer_idx: int, h: torch.Tensor) -> None:
        if args.log_rms:
            rms_log.append({"layer": layer_idx,
                            "rms": float(h.float().pow(2).mean().sqrt())})

    dump_chunks: dict[str, list] = {"pre": [], "post": []}

    def hidden_hook(pre, post):
        dump_chunks["pre"].append(pre.detach().to("cpu", torch.bfloat16).clone())
        dump_chunks["post"].append(post.detach().to("cpu", torch.bfloat16).clone())

    runner = DeviceMappedHybridRunner(
        model, args.split, args.device0, args.device1, rms_hook=hook,
        hidden_hook=hidden_hook if args.dump_hidden else None)
    dump_payload: dict = {"prompts": {}}
    for d in (args.device0, args.device1):
        torch.cuda.reset_peak_memory_stats(d)

    grad_ctx = torch.inference_mode if args.grad_mode == "inference" else torch.no_grad
    results = []
    for probe in probes:
        prompt_ids = tok.encode(probe["prompt"])
        assert prompt_ids == probe["prompt_token_ids"], (prompt_ids, probe)
        ref = probe["output_token_ids"]
        dump_chunks["pre"].clear(); dump_chunks["post"].clear()
        with grad_ctx():
            gen = greedy_generate_bridge(
                runner, prompt_ids, args.max_new_tokens, args.device1,
                sampling_sync=args.sampling_sync, ref_ids=ref)
        gen_ids = gen["gen_ids"]; finite = gen["logits_finite"]
        prefill_s = gen["prefill_s"]; decode_s = gen["decode_s"]
        text = tok.decode(gen_ids)
        matching = sum(1 for a, b in zip(gen_ids, ref) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(gen_ids, ref)) if a != b), None)
        peak = {d: torch.cuda.max_memory_allocated(d) for d in (args.device0, args.device1)}
        row = {
            "id": probe["id"], "prompt": probe["prompt"],
            "prompt_token_ids": prompt_ids, "generated_token_ids": gen_ids,
            "generated_text": text, "logits_finite": finite,
            "vllm_token_ids": ref, "vllm_text": probe["text"],
            "exact_match": gen_ids == ref, "matching_positions": matching,
            "compared_tokens": min(len(gen_ids), len(ref)),
            "first_divergence": first_div,
            "divergence_logits": gen["divergence_logits"],
            "prefill_s": prefill_s, "decode_s": decode_s,
            "decode_tok_per_s": (len(gen_ids) - 1) / decode_s if decode_s > 0 else None,
            "peak_memory_bytes": peak,
        }
        results.append(row)
        if args.dump_hidden:
            dump_payload["prompts"][probe["id"]] = {
                "prompt": probe["prompt"], "prompt_len": len(prompt_ids),
                "token_ids": list(prompt_ids) + [int(v) for v in gen_ids],
                "hidden_pre_norm": torch.cat(dump_chunks["pre"], dim=1)[0],
                "hidden_post_norm": torch.cat(dump_chunks["post"], dim=1)[0],
            }
        print(f"[{probe['id']}] finite={finite} match={matching}/{len(ref)} "
              f"first_div={first_div} prefill={prefill_s:.2f}s "
              f"decode={(len(gen_ids)-1)/decode_s:.2f} tok/s text={text!r}", flush=True)
        if not finite:
            print("FAIL: non-finite logits (garbage output); see --log-rms bisect")
            break

    dtype_label = {
        "reference": "bf16 activations; resident block-FP8 weights (reference dequant forward)",
        "triton": "bf16 activations; resident block-FP8 weights (Triton fused dequant, M<=128)",
        "scaled-mm": "bf16 activations; resident block-FP8 weights (scaled-mm requested, reference fallback)",
    }[args.fp8_path]
    out = {
        "script": "scripts/run_qwen38_gpu.py",
        "snapshot": str(snapshot), "split": args.split,
        "devices": [args.device0, args.device1],
        "fp8_path": args.fp8_path, "sampling_sync": args.sampling_sync,
        "grad_mode": args.grad_mode,
        "dtype": dtype_label,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "max_new_tokens": args.max_new_tokens, "load_time_s": load_s,
        "load_report": {"ok": report["ok"], "loaded_count": report["loaded_count"],
                        "resident_targets": len(report["resident_targets"])},
        "peak_memory_bytes": {
            d: max(r["peak_memory_bytes"][d] for r in results) for d in (args.device0, args.device1)},
        "probes": results,
    }
    if args.log_rms:
        out["rms_log_first_prefill"] = rms_log[: config.num_hidden_layers]
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.dump_hidden:
        dp = Path(args.dump_hidden)
        dp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dump_payload, dp)
        print(f"wrote {dp} ({dp.stat().st_size / 1e6:.2f} MB hidden dump)")
    ok = len(results) == len(probes) and all(r["logits_finite"] for r in results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
