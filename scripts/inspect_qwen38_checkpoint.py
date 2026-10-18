#!/usr/bin/env python
"""Inspect the real Qwen3.8 checkpoint, headers only by default.

Usage:
    .venv/bin/python scripts/inspect_qwen38_checkpoint.py [--model-dir DIR] [--layer N]

Default mode reads safetensors headers only (no tensor data).
``--layer N`` additionally loads only small norm tensors of that
layer through the lazy path to prove single-tensor loading works,
printing their stats. It never loads a full real layer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference_from_scratch.hybrid import HybridConfig
from inference_from_scratch.hybrid_checkpoint import (
    build_manifest, find_default_snapshot, load_tensor, validate_manifest,
)


def _small_stats_names(layer_idx: int, manifest) -> list[str]:
    prefix = f"model.language_model.layers.{layer_idx}."
    names = []
    for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                   "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                   "linear_attn.A_log", "linear_attn.dt_bias",
                   "linear_attn.norm.weight"):
        name = prefix + suffix
        if any(e.checkpoint_name == name for e in manifest.entries):
            names.append(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="snapshot dir (default: auto-find in HF cache)")
    ap.add_argument("--layer", type=int, default=None,
                    help="also lazy-load small norm tensors of this layer")
    args = ap.parse_args()
    snap = args.model_dir or find_default_snapshot()
    if snap is None or not Path(snap).is_dir():
        print("error: Qwen3.8 snapshot not found; pass --model-dir", file=sys.stderr)
        return 2
    snap = Path(snap)
    config = HybridConfig.from_text_config(snap)
    manifest = build_manifest(snap)
    result = validate_manifest(manifest, config)
    print(f"snapshot: {snap}")
    print(f"shards: {len({e.shard for e in manifest.entries})}")
    print(f"manifest entries: {len(manifest.entries)}")
    print(f"kinds: {manifest.counts_by_kind()}")
    print(f"total_bytes_by_dtype: {result['total_bytes_by_dtype']}")
    print(f"counts: {result['counts']}")
    print(f"missing: {len(result['missing'])} {result['missing'][:5]}")
    print(f"unexpected: {len(result['unexpected'])} {result['unexpected'][:5]}")
    print(f"shape_mismatch: {len(result['shape_mismatch'])} {result['shape_mismatch'][:3]}")
    print(f"ok: {result['ok']}")
    if args.layer is not None:
        for name in _small_stats_names(args.layer, manifest):
            t = load_tensor(snap, name)
            tf = t.float()
            print(f"layer {args.layer} {name}: shape={list(t.shape)} dtype={t.dtype} "
                  f"numel={t.numel()} mean={tf.mean().item():.6g} "
                  f"std={tf.std().item():.6g} min={tf.min().item():.6g} max={tf.max().item():.6g}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
