#!/usr/bin/env python3
"""Re-measure MTP acceptance THROUGH MTPDraftModule (not the sweep).

Loads the captured trajectories (scripts/capture_target_hidden.py) and
the real dequantized MTP block, then drafts teacher-forced one step at
a time through inference_from_scratch.mtp.MTPDraftModule with the
sweep's leading wiring, threading the draft state sequentially per
prompt (commit-on-accept on the teacher-forced trajectory). Reports
top-1 acceptance against the sweep winner in
benchmarks/mtp/wiring_sweep.json.

    .venv-gpu/bin/python scripts/mtp_module_acceptance.py \
        --capture /tmp/mtp_capture/trajectories.pt \
        --out benchmarks/mtp/module_acceptance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid_checkpoint import build_manifest, load_tensor  # noqa: E402
from inference_from_scratch.mtp import (  # noqa: E402
    LEADING_WIRING, MTPConfig, MTPDraftModule, MTPHead,
    teacher_forced_draft_indices)

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="/tmp/mtp_capture/trajectories.pt")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--sweep", default="benchmarks/mtp/wiring_sweep.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="benchmarks/mtp/module_acceptance.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("FAIL: need CUDA")
        return 1
    snapshot = Path(args.snapshot)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    manifest = build_manifest(snapshot)
    t0 = time.perf_counter()
    head = MTPHead(MTPConfig(),
                   fc_concat_order=LEADING_WIRING["fc_concat_order"],
                   norm_pairing=LEADING_WIRING["norm_pairing"],
                   lm_head_input=LEADING_WIRING["lm_head_input"])
    state = {}
    for key in head.state_dict():
        state[key] = load_tensor(snapshot, f"mtp.{key}", manifest=manifest)
    head.load_state_dict(state, strict=True)
    head = head.to(device=args.device, dtype=torch.bfloat16).eval()
    module = MTPDraftModule(head=head, wiring=LEADING_WIRING).eval()
    embed_w = load_tensor(snapshot, "model.language_model.embed_tokens.weight",
                          manifest=manifest).to(args.device)
    lm_head_w = load_tensor(snapshot, "lm_head.weight",
                            manifest=manifest).to(args.device)
    print(f"loaded module in {time.perf_counter() - t0:.1f}s", flush=True)

    per_prompt: dict[str, list] = {}
    tot_c = tot_n = 0
    with torch.no_grad():
        for pid, traj in capture["prompts"].items():
            token_ids = [int(t) for t in traj["token_ids"]]
            steps = teacher_forced_draft_indices(len(token_ids))
            hidden = traj["hidden_post_norm"].to(args.device)
            draft_state = module.initial_state()
            correct = 0
            for (t, t1, t2) in steps:
                draft_id, _logits, draft_state = module.draft_token(
                    hidden[t], token_ids[t1], draft_state, position=t,
                    embedding=lambda ids: torch.nn.functional.embedding(ids, embed_w),
                    lm_head=lambda h: torch.nn.functional.linear(h, lm_head_w))
                correct += int(int(draft_id) == token_ids[t2])
            per_prompt[pid] = [correct, len(steps)]
            tot_c += correct
            tot_n += len(steps)
            print(f"{pid}: {correct}/{len(steps)} = {correct / max(len(steps), 1):.4f}",
                  flush=True)
    acceptance = tot_c / tot_n if tot_n else 0.0
    sweep = json.loads(Path(args.sweep).read_text())
    sweep_best = max(c["mean_acceptance"] for c in sweep["candidates"])
    result = {
        "module": "MTPDraftModule",
        "wiring": LEADING_WIRING,
        "wiring_resolved": bool(sweep.get("resolved")),
        "acceptance": acceptance,
        "correct": tot_c, "total": tot_n,
        "per_prompt": {k: {"correct": v[0], "total": v[1],
                           "acceptance": v[0] / v[1] if v[1] else 0.0}
                       for k, v in per_prompt.items()},
        "sweep_winner_acceptance": sweep_best,
        "delta_vs_sweep_winner": acceptance - sweep_best,
        "capture": str(args.capture), "snapshot": str(snapshot),
        "device": args.device, "torch": torch.__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"module acceptance={acceptance:.4f} ({tot_c}/{tot_n}) "
          f"sweep winner={sweep_best:.4f} delta={acceptance - sweep_best:+.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
