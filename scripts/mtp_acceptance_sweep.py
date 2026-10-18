#!/usr/bin/env python3
"""MTP wiring acceptance sweep over captured target trajectories.

Builds ONE MTP block on cuda:0 from the real dequantized mtp.* tensors
(hybrid_checkpoint.load_tensor pairs each FP8 weight with its scale and
dequantizes to BF16) plus the shared embed_tokens / lm_head weights,
then runs every live candidate wiring from
inference_from_scratch.mtp.enumerate_wiring_candidates() (48) over the
trajectories saved by scripts/capture_target_hidden.py, teacher-forced:

    draft at step t: fuse hidden[t] with embed(token_ids[t+1]),
    predict token_ids[t+2]   (see mtp.teacher_forced_draft_indices)

Position rules map to absolute RoPE positions t / t+1 / t+2 for
hidden_pos / input_token_pos / predicted_pos. The MTP block runs as one
causal sequence pass per (prompt, candidate); the shared lm_head reads
mtp.norm's output or the raw block output per candidate.

A wiring is RESOLVED only if the best candidate dominates: mean top-1
acceptance at least 0.05 above the runner-up and at least 0.10 above
the trivial repeat-previous-token baseline. Otherwise the JSON reports
resolved=false and no wiring is picked.

    .venv-gpu/bin/python scripts/mtp_acceptance_sweep.py \
        --capture /tmp/mtp_capture/trajectories.pt \
        --out benchmarks/mtp/wiring_sweep.json
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
    MTPConfig, MTPHead, enumerate_wiring_candidates,
    teacher_forced_draft_indices,
)

DEFAULT_SNAPSHOT = (
    "/home/nick/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8"
    "/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")

POSITION_OFFSET = {"hidden_pos": 0, "input_token_pos": 1, "predicted_pos": 2}


def load_mtp_head(snapshot: Path, manifest, device: str) -> MTPHead:
    """One MTPHead on device, weights dequantized from real mtp.* tensors."""
    head = MTPHead(MTPConfig())
    state = {}
    for key in head.state_dict():
        ckpt = f"mtp.{key}"
        state[key] = load_tensor(snapshot, ckpt, manifest=manifest)
    head.load_state_dict(state, strict=True)
    head = head.to(device=device, dtype=torch.bfloat16).eval()
    return head


def candidate_id(cand: dict) -> str:
    return (f"fc={cand['fc_concat_order']}|norm={cand['norm_pairing']}"
            f"|hidden={cand['hidden_source']}|pos={cand['position_rule']}"
            f"|head={cand['lm_head_input']}")


def evaluate(capture: dict, head: MTPHead, embed_w: torch.Tensor,
             lm_head_w: torch.Tensor, device: str) -> dict:
    candidates = enumerate_wiring_candidates()
    # per candidate: per prompt (correct, total)
    counts: dict[str, dict[str, list]] = {
        candidate_id(c): {} for c in candidates}
    repeat_correct = repeat_total = 0
    prompt_ids_order = list(capture["prompts"])
    for pid in prompt_ids_order:
        traj = capture["prompts"][pid]
        token_ids = [int(t) for t in traj["token_ids"]]
        steps = teacher_forced_draft_indices(len(token_ids))
        hidden_pre = traj["hidden_pre_norm"].to(device)
        hidden_post = traj["hidden_post_norm"].to(device)
        # steps use hidden rows 0..len-3; hidden tensors hold len-1 rows.
        n = len(steps)
        targets = torch.tensor([token_ids[t2] for _, _, t2 in steps],
                               dtype=torch.long, device=device)
        repeat = torch.tensor([token_ids[t1] for _, t1, _ in steps],
                              dtype=torch.long, device=device)
        repeat_correct += int((repeat == targets).sum())
        repeat_total += n
        embed_seq_ids = torch.tensor([token_ids[t1] for _, t1, _ in steps],
                                     dtype=torch.long, device=device)
        with torch.no_grad():
            embeddings = torch.nn.functional.embedding(embed_seq_ids, embed_w)
            for cand in candidates:
                head.fc_concat_order = cand["fc_concat_order"]
                head.norm_pairing = cand["norm_pairing"]
                head.lm_head_input = cand["lm_head_input"]
                hidden = (hidden_pre[:n] if cand["hidden_source"] == "pre_final_norm"
                          else hidden_post[:n])
                offset = POSITION_OFFSET[cand["position_rule"]]
                pos = torch.arange(n, device=device) + offset
                _, logits, _ = head(
                    hidden.unsqueeze(0), embeddings.unsqueeze(0),
                    lm_head=lambda h: torch.nn.functional.linear(h, lm_head_w),
                    position_ids=pos)
                pred = logits[0].argmax(dim=-1)
                counts[candidate_id(cand)][pid] = [
                    int((pred == targets).sum()), n]
    rows = []
    for cand in candidates:
        cid = candidate_id(cand)
        per_prompt = {}
        tot_c = tot_n = 0
        for pid in prompt_ids_order:
            c, n = counts[cid][pid]
            per_prompt[pid] = c / n if n else 0.0
            tot_c += c; tot_n += n
        rows.append({"candidate": cand, "id": cid, "per_prompt": per_prompt,
                     "mean_acceptance": tot_c / tot_n if tot_n else 0.0,
                     "correct": tot_c, "total": tot_n})
    rows.sort(key=lambda r: r["mean_acceptance"], reverse=True)
    repeat_acc = repeat_correct / repeat_total if repeat_total else 0.0
    best = rows[0]["mean_acceptance"] if rows else 0.0
    second = rows[1]["mean_acceptance"] if len(rows) > 1 else 0.0
    resolved = bool(rows and (best - second) >= 0.05 and (best - repeat_acc) >= 0.10)
    axis_summary: dict = {}
    for axis in ("fc_concat_order", "norm_pairing", "hidden_source",
                 "position_rule", "lm_head_input"):
        axis_summary[axis] = {}
        for row in rows:
            axis_summary[axis].setdefault(row["candidate"][axis], []).append(
                row["mean_acceptance"])
        for opt, vals in axis_summary[axis].items():
            axis_summary[axis][opt] = {"min": min(vals), "max": max(vals),
                                       "mean": sum(vals) / len(vals)}
    return {"candidates": rows, "axis_summary": axis_summary,
            "repeat_previous_acceptance": repeat_acc,
            "random_acceptance": 1.0 / 248320,
            "resolved": resolved,
            "winner": rows[0]["candidate"] if resolved else None,
            "margin_over_second": best - second,
            "margin_over_repeat": best - repeat_acc}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="/tmp/mtp_capture/trajectories.pt")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="benchmarks/mtp/wiring_sweep.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        print("FAIL: need CUDA")
        return 1
    snapshot = Path(args.snapshot)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    manifest = build_manifest(snapshot)
    t0 = time.perf_counter()
    head = load_mtp_head(snapshot, manifest, args.device)
    embed_w = load_tensor(snapshot, "model.language_model.embed_tokens.weight",
                          manifest=manifest).to(args.device)
    lm_head_w = load_tensor(snapshot, "lm_head.weight",
                            manifest=manifest).to(args.device)
    print(f"loaded MTP head + embed/lm_head in {time.perf_counter() - t0:.1f}s",
          flush=True)
    result = evaluate(capture, head, embed_w, lm_head_w, args.device)
    result.update({"capture": str(args.capture), "snapshot": str(snapshot),
                   "device": args.device, "torch": torch.__version__})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["candidates"][:8]:
        print(f"{row['mean_acceptance']:.4f}  {row['id']}")
    print(f"repeat baseline={result['repeat_previous_acceptance']:.4f} "
          f"resolved={result['resolved']} winner={result['winner']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
