#!/usr/bin/env python3
"""Capture target hidden trajectories for the MTP wiring sweep.

Greedily generates N tokens per prompt with the pipeline bridge
(scripts/run_qwen38_gpu.py's loader path: HybridLM on meta, resident
BlockFP8Linear conversion, layer-by-layer streamed load, split across
cuda:0/cuda:1) and, through DeviceMappedHybridRunner's hidden_hook,
saves per-position hidden states: hidden_pre_norm (after layer 63,
before the target's final norm) and hidden_post_norm (after it), both
[T, 5120] BF16 and aligned so row t is the hidden for token_ids[t].

Also exposes capture_trajectory() on any DeviceMappedHybridRunner so
the dump path is testable on a tiny CPU model (tests/test_mtp_wiring.py).

Run only in an approved GPU window (vLLM stopped):

    .venv-gpu/bin/python scripts/capture_target_hidden.py \
        --out /tmp/mtp_capture/trajectories.pt \
        --summary benchmarks/mtp/capture_summary.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference_from_scratch.fp8_linear import BlockFP8Linear  # noqa: E402
from inference_from_scratch.hybrid import (  # noqa: E402
    DeviceMappedHybridRunner, HybridConfig, HybridLM, pipeline_device_plan,
)
from inference_from_scratch.hybrid_checkpoint import build_manifest, load_into_model  # noqa: E402

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")

# Three neutral prompts written for this capture (the other three are
# the recorded quality probes in benchmarks/baseline/).
NEUTRAL_PROMPTS = [
    {"id": "neutral_code",
     "prompt": "def fibonacci(n):\n    # return the nth Fibonacci number\n"},
    {"id": "neutral_list",
     "prompt": "Shopping list:\n- milk\n- eggs\n-"},
    {"id": "neutral_prose",
     "prompt": "The old lighthouse stood at the edge of the harbour, and every evening"},
]


def capture_trajectory(runner, prompt_ids, max_new_tokens):
    """Greedy-generate via a pipeline runner, capturing final hiddens.

    Returns {"token_ids", "hidden_pre_norm", "hidden_post_norm"} with
    token_ids = prompt + generated (length L+N) and hiddens [L+N-1, D]
    BF16 on CPU: row t aligns with token_ids[t] (the last generated
    token has no hidden; its predictor is the last hidden row).
    """
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be >= 1")
    chunks_pre: list[torch.Tensor] = []
    chunks_post: list[torch.Tensor] = []

    def hook(pre, post):
        chunks_pre.append(pre.detach().to("cpu", torch.bfloat16).clone())
        chunks_post.append(post.detach().to("cpu", torch.bfloat16).clone())

    prev_hook = runner.hidden_hook
    runner.hidden_hook = hook
    try:
        dev1 = runner.device1
        idx = torch.tensor([list(prompt_ids)], dtype=torch.long)
        with torch.no_grad():
            logits, state = runner.run(idx)
            nxt = int(logits[0, -1].argmax().item())
            gen = [nxt]
            for _ in range(int(max_new_tokens) - 1):
                step = torch.tensor([[nxt]], dtype=torch.long, device=dev1)
                logits, state = runner.run(step, state=state)
                nxt = int(logits[0, -1].argmax().item())
                gen.append(nxt)
    finally:
        runner.hidden_hook = prev_hook
    hidden_pre = torch.cat(chunks_pre, dim=1)[0]
    hidden_post = torch.cat(chunks_post, dim=1)[0]
    return {"token_ids": [int(t) for t in list(prompt_ids)] + gen,
            "prompt_len": len(list(prompt_ids)),
            "hidden_pre_norm": hidden_pre, "hidden_post_norm": hidden_post}


def _place(module: nn.Module, device: str) -> None:
    module.to_empty(device=device)
    for p in module.parameters():
        p.data = p.data.to(dtype=torch.bfloat16)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--device0", default="cuda:0")
    ap.add_argument("--device1", default="cuda:1")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--probes", default="benchmarks/baseline/arm_a_quality_probes.json")
    ap.add_argument("--out", default="/tmp/mtp_capture/trajectories.pt")
    ap.add_argument("--summary", default="benchmarks/mtp/capture_summary.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("FAIL: need 2 CUDA devices")
        return 1
    from run_qwen38_gpu import fp8_kwargs_for_path
    snapshot = Path(args.snapshot)
    probes_doc = json.loads(Path(args.probes).read_text())
    prompts = [{"id": p["id"], "prompt": p["prompt"],
                "prompt_token_ids": p["prompt_token_ids"]}
               for p in probes_doc["probes"]] + [
        {"id": p["id"], "prompt": p["prompt"], "prompt_token_ids": None}
        for p in NEUTRAL_PROMPTS]

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
    with torch.device("meta"):
        model = HybridLM(config)
        converted = 0
        for name, module in list(model.named_modules()):
            if name in fp8_modules and isinstance(module, nn.Linear) \
                    and not isinstance(module, BlockFP8Linear):
                repl = BlockFP8Linear(module.in_features, module.out_features,
                                      bias=False, **fp8_kwargs_for_path("triton"))
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, repl)
                converted += 1
    print(f"meta build + fp8 convert ({converted})", flush=True)
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
    print(f"load: ok={report['ok']} loaded={report['loaded_count']} "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    if not report["ok"]:
        print(f"FAIL validation: {report}")
        return 1
    runner = DeviceMappedHybridRunner(model, args.split, args.device0, args.device1)

    payload: dict = {"snapshot": str(snapshot), "max_new_tokens": args.max_new_tokens,
                     "prompts": {}}
    summary_rows = []
    with torch.inference_mode():
        for p in prompts:
            prompt_ids = tok.encode(p["prompt"])
            if p["prompt_token_ids"] is not None:
                assert prompt_ids == p["prompt_token_ids"], p["id"]
            traj = capture_trajectory(runner, prompt_ids, args.max_new_tokens)
            payload["prompts"][p["id"]] = {
                "prompt": p["prompt"], "prompt_len": traj["prompt_len"],
                "token_ids": traj["token_ids"],
                "hidden_pre_norm": traj["hidden_pre_norm"],
                "hidden_post_norm": traj["hidden_post_norm"],
            }
            text = tok.decode(traj["token_ids"][traj["prompt_len"]:])
            row = {"id": p["id"], "prompt": p["prompt"],
                   "prompt_len": traj["prompt_len"],
                   "token_ids_len": len(traj["token_ids"]),
                   "hidden_shape": list(traj["hidden_pre_norm"].shape),
                   "generated_text": text}
            summary_rows.append(row)
            print(f"[{p['id']}] tokens={len(traj['token_ids'])} "
                  f"hidden={list(traj['hidden_pre_norm'].shape)} text={text!r}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.2f} MB; tensors stay out of git)", flush=True)
    summary = {"capture_file": str(out_path), "capture_bytes": out_path.stat().st_size,
               "snapshot": str(snapshot), "max_new_tokens": args.max_new_tokens,
               "prompts": summary_rows}
    sp = Path(args.summary)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {sp}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
