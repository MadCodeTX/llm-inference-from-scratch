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

from inference_from_scratch.fp8_linear import BlockFP8Linear
from inference_from_scratch.hybrid import (
    DeviceMappedHybridRunner, HybridConfig, HybridLM, pipeline_device_plan,
)
from inference_from_scratch.hybrid_checkpoint import build_manifest, load_into_model

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")


def _place(module: nn.Module, device: str) -> None:
    module.to_empty(device=device)  # meta -> device (load overwrites data)
    for p in module.parameters():
        p.data = p.data.to(dtype=torch.bfloat16)


def _sync(*devices: str) -> None:
    for d in devices:
        torch.cuda.synchronize(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--device0", default="cuda:0")
    ap.add_argument("--device1", default="cuda:1")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--probes", default="benchmarks/baseline/arm_a_quality_probes.json")
    ap.add_argument("--out", default="benchmarks/baseline/ours_pipeline_first_run.json")
    ap.add_argument("--log-rms", action="store_true")
    args = ap.parse_args()
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
    print(f"config: layers={config.num_hidden_layers} hidden={config.hidden_size} "
          f"fp8_modules={len(fp8_modules)}", flush=True)

    t_build = time.perf_counter()
    with torch.device("meta"):
        model = HybridLM(config)
        converted = 0
        for name, module in list(model.named_modules()):
            if name in fp8_modules and isinstance(module, nn.Linear) \
                    and not isinstance(module, BlockFP8Linear):
                repl = BlockFP8Linear(module.in_features, module.out_features,
                                      bias=False)
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

    runner = DeviceMappedHybridRunner(model, args.split, args.device0,
                                      args.device1, rms_hook=hook)
    for d in (args.device0, args.device1):
        torch.cuda.reset_peak_memory_stats(d)

    results = []
    for probe in probes:
        prompt_ids = tok.encode(probe["prompt"])
        assert prompt_ids == probe["prompt_token_ids"], (prompt_ids, probe)
        idx = torch.tensor([prompt_ids], dtype=torch.long)
        gen_ids: list[int] = []
        with torch.no_grad():
            _sync(args.device0, args.device1)
            t_pre = time.perf_counter()
            logits, state = runner.run(idx)
            finite = bool(torch.isfinite(logits).all())
            nxt = int(logits[0, -1].argmax().item())
            _sync(args.device0, args.device1)
            prefill_s = time.perf_counter() - t_pre
            gen_ids.append(nxt)
            t_dec = time.perf_counter()
            for _ in range(args.max_new_tokens - 1):
                step = torch.tensor([[nxt]], dtype=torch.long, device=args.device1)
                logits, state = runner.run(step, state=state)
                finite = finite and bool(torch.isfinite(logits).all())
                nxt = int(logits[0, -1].argmax().item())
                gen_ids.append(nxt)
            _sync(args.device0, args.device1)
            decode_s = time.perf_counter() - t_dec
        text = tok.decode(gen_ids)
        ref = probe["output_token_ids"]
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
            "prefill_s": prefill_s, "decode_s": decode_s,
            "decode_tok_per_s": (len(gen_ids) - 1) / decode_s if decode_s > 0 else None,
            "peak_memory_bytes": peak,
        }
        results.append(row)
        print(f"[{probe['id']}] finite={finite} match={matching}/{len(ref)} "
              f"first_div={first_div} prefill={prefill_s:.2f}s "
              f"decode={(len(gen_ids)-1)/decode_s:.2f} tok/s text={text!r}", flush=True)
        if not finite:
            print("FAIL: non-finite logits (garbage output); see --log-rms bisect")
            break

    out = {
        "script": "scripts/run_qwen38_gpu.py",
        "snapshot": str(snapshot), "split": args.split,
        "devices": [args.device0, args.device1],
        "dtype": "bf16 activations; resident block-FP8 weights (reference dequant forward)",
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
    ok = len(results) == len(probes) and all(r["logits_finite"] for r in results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
