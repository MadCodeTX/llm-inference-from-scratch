#!/usr/bin/env python
"""Real two-process TP checkpoint load check vs dense converted model.

Run under torchrun with 2 processes, CPU (gloo) or GPU (NCCL):

  .venv/bin/torchrun --nproc_per_node=2 scripts/tp2_checkpoint_check.py --backend gloo --device cpu
  .venv-gpu/bin/torchrun --nproc_per_node=2 scripts/tp2_checkpoint_check.py

Rank 0 writes a tiny synthetic FP8 checkpoint (dense HybridLM weights,
FP8-target projections block-quantised) to a temp dir; every rank then
loads ONLY its shards via ``tp_checkpoint.load_rank_state`` (resident
FP8) and compares prefill + 6 decode steps against the dense converted
model (the same checkpoint loaded dequantised into ``HybridLM``).
Greedy tokens must match exactly; per-rank shard summaries are
gathered on rank 0 (counts/hashes of shapes only, never tensors).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridConfig, HybridLM  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import (  # noqa: E402
    expected_language_tensors, load_into_model)
from inference_from_scratch.quantization import quantize_block_fp8  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    gather_rank_summaries, load_rank_state, rank_shard_summary,
    validate_rank_coverage)
from inference_from_scratch.tp_distributed import DistributedTPContext  # noqa: E402


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, partial_rotary_factor=0.25,
        rope_theta=10000.0, mtp_num_hidden_layers=1)


def _ckpt_name(target: str) -> str:
    if target == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target == "norm.weight":
        return "model.language_model.norm.weight"
    if target == "lm_head.weight":
        return "lm_head.weight"
    if target.startswith("mtp."):
        return target
    return f"model.language_model.{target}"


def _write_fixture(root: Path, cfg: HybridConfig) -> None:
    torch.manual_seed(7)
    src = HybridLM(cfg).to(torch.bfloat16).eval()
    fp8_ckpts = {n for n, s in expected_language_tensors(cfg).items()
                 if s["kind"] == "fp8_weight"}
    tensors: dict[str, torch.Tensor] = {}
    for target, tensor in src.state_dict().items():
        ck = _ckpt_name(target)
        if ck in fp8_ckpts:
            q, scale = quantize_block_fp8(tensor.detach().to(torch.float32))
            tensors[ck] = q.contiguous()
            tensors[ck + "_scale_inv"] = scale.to(torch.bfloat16).contiguous()
        else:
            tensors[ck] = tensor.detach().to(torch.bfloat16).contiguous()
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text('{"text_config": {}}')


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, rank: int) -> None:
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
    print(f"[rank {rank}] {name}: max_abs_err={err:.3e} allclose={ok}", flush=True)
    if not ok:
        raise AssertionError(f"{name} mismatch on rank {rank}: max_abs_err={err}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=None, choices=["nccl", "gloo"])
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = p.parse_args()
    backend = args.backend or ("nccl" if args.device != "cpu" else "gloo")
    device_type = args.device or ("cuda" if backend == "nccl" else "cpu")

    ctx = DistributedTPContext.from_env(backend=backend, device_type=device_type)
    rank, dev = ctx.rank, ctx.device
    cfg = _tiny_cfg()
    fixture = Path(tempfile.gettempdir()) / "tp2_checkpoint_fixture"
    if rank == 0:
        _write_fixture(fixture, cfg)
    ctx.barrier()

    # Dense converted reference: same checkpoint, dequantised load.
    torch.manual_seed(99)
    dense = HybridLM(cfg).to(dev, torch.bfloat16)
    dense.eval()
    load_into_model(dense, fixture, strict=True)

    tp, report = load_rank_state(fixture, cfg, rank=rank, world=ctx.world_size,
                                 device=dev, fp8_mode="resident", ctx=ctx)
    print(f"[rank {rank}] loaded={report['loaded_count']} "
          f"resident_bytes={report['resident_bytes']}", flush=True)

    summary = rank_shard_summary(fixture, cfg, rank=rank, world=ctx.world_size)
    gathered = gather_rank_summaries(ctx, summary)
    coverage = validate_rank_coverage(fixture, cfg, world=ctx.world_size)
    print(f"[rank {rank}] coverage_ok={coverage['ok']}", flush=True)
    if not coverage["ok"]:
        raise AssertionError(f"rank coverage failed: {coverage['errors'][:3]}")
    if rank == 0:
        assert gathered is not None
        counts = [g["count"] for g in gathered]
        print(f"[rank 0] gathered counts={counts} hashes={[g['hash'][:12] for g in gathered]}", flush=True)
        if len(set(counts)) != 1:
            raise AssertionError(f"rank shard counts differ: {counts}")

    dense_bytes = int(sum(t.numel() * t.element_size() for t in dense.state_dict().values()))
    print(f"[rank {rank}] resident_bytes={report['resident_bytes']} dense_bytes={dense_bytes} "
          f"sharded_lt_dense={report['resident_bytes'] < dense_bytes}", flush=True)
    if not report["resident_bytes"] < dense_bytes:
        raise AssertionError("rank holds >= dense bytes")

    torch.manual_seed(77)
    idx = torch.randint(0, cfg.vocab_size, (2, 5), device=dev)
    with torch.no_grad():
        dense_logits, dense_state = dense.prefill(idx)
        tp_logits, tp_state = tp.prefill(idx)
        _check("prefill_logits", tp_logits, dense_logits, rank)
        if not torch.equal(dense_logits[:, -1].argmax(-1), tp_logits[:, -1].argmax(-1)):
            raise AssertionError("prefill greedy tokens differ")
        next_token = dense_logits[:, -1].argmax(dim=-1)
        for step_i in range(6):
            d_logits, dense_state = dense.step(next_token, dense_state)
            t_logits, tp_state = tp.step(next_token, tp_state)
            _check(f"decode_step{step_i}_logits", t_logits, d_logits, rank)
            if not torch.equal(d_logits[:, -1].argmax(-1), t_logits[:, -1].argmax(-1)):
                raise AssertionError(f"decode step {step_i} greedy tokens differ")
            next_token = d_logits[:, -1].argmax(dim=-1)
    print(f"[rank {rank}] greedy tokens equal over prefill + 6 decode steps", flush=True)

    ctx.barrier()
    if rank == 0:
        print("TP2_CHECKPOINT_CHECK PASS", flush=True)
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
