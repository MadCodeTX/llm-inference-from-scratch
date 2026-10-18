"""Real single-layer parity vs transformers (CPU only, one layer at a time).

Builds the single HF ``Qwen3_5DecoderLayer`` for the REAL text_config at
``--layer N`` and our ``HybridBlock`` at the same index, loads dequantized
FP8->BF16 + BF16 tensors for ONLY that layer from the real snapshot into
BOTH (explicit name mapping, asserted), feeds identical seeded hidden
states ``[1, T, 5120]``, and reports per-submodule (input norm, token
mixer, post norm+MLP) and whole-block max-abs / relative diffs.

Usage:
    .venv/bin/python scripts/parity_real_layer.py --layer 0 --tokens 4 --seed 0
    .venv/bin/python scripts/parity_real_layer.py --layer 3 --tokens 4 --seed 0
    .venv/bin/python scripts/parity_real_layer.py --layer 3 --fp32   # dtype-noise control

Tolerance expectations: in bf16, block outputs will NOT agree to 1e-6.
bf16 has ~3 decimal digits; matmuls over 5120/17408-wide reductions and
the chunkwise (HF) vs sequential (ours, linear layers) re-association
give relative RMS differences of order 1e-3..1e-2. Judge by
``max_abs / ref_rms`` and ``diff_rms / ref_rms``; a math error (wrong
split, wrong norm gain, wrong RoPE width) gives O(1) relative error.
The ``--fp32`` rerun casts the SAME dequantized bf16 weights and inputs
to fp32 in both implementations: maths errors stay O(1) there, dtype
noise collapses to ~1e-6..1e-5, so it separates the two.

Only the requested layer's tensors are ever materialised (via
``hybrid_checkpoint.iter_layer_tensors``); memory is freed (del + gc)
before exit. No CUDA, no vLLM, checkpoint is read-only.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import HybridBlock, HybridConfig  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import (  # noqa: E402
    find_default_snapshot, iter_layer_tensors,
)


def _metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    r, g = ref.float(), got.float()
    diff = (r - g).abs()
    ref_rms = r.pow(2).mean().sqrt().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "diff_rms": (r - g).pow(2).mean().sqrt().item(),
        "ref_rms": ref_rms,
        "got_rms": g.pow(2).mean().sqrt().item(),
        "max_rel_to_rms": diff.max().item() / ref_rms if ref_rms else 0.0,
        "rel_rms": ((r - g).pow(2).mean().sqrt().item() / ref_rms) if ref_rms else 0.0,
    }


def _fmt(name: str, m: dict) -> str:
    return (f"{name}: max_abs={m['max_abs']:.6g} mean_abs={m['mean_abs']:.6g} "
            f"diff_rms={m['diff_rms']:.6g} ref_rms={m['ref_rms']:.6g} "
            f"got_rms={m['got_rms']:.6g} max_rel_to_rms={m['max_rel_to_rms']:.6g} "
            f"rel_rms={m['rel_rms']:.6g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--fp32", action="store_true",
                    help="cast both models+inputs to fp32 (dtype-noise control)")
    args = ap.parse_args()
    if args.tokens <= 0:
        ap.error("--tokens must be positive")

    snap = Path(args.snapshot) if args.snapshot else find_default_snapshot()
    if snap is None or not (snap / "config.json").is_file():
        print("real snapshot not found; cannot run real-layer parity")
        return 2
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    raw = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    text_cfg = raw["text_config"]

    # ---- configs ------------------------------------------------------
    our_cfg = HybridConfig.from_text_config(snap)
    if not (0 <= args.layer < our_cfg.num_hidden_layers):
        ap.error(f"--layer must be in [0, {our_cfg.num_hidden_layers}), got {args.layer}")
    layer_type = our_cfg.layer_types[args.layer]

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding)

    hf_cfg = Qwen3_5TextConfig(**text_cfg)
    hf_cfg._attn_implementation = "eager"

    # ---- build both single layers ------------------------------------
    hf_layer = Qwen3_5DecoderLayer(hf_cfg, args.layer).eval().to(dtype)
    our_block = HybridBlock(our_cfg, layer_idx=args.layer).eval().to(dtype)

    # ---- load ONLY this layer's tensors into both ---------------------
    # iter_layer_tensors yields checkpoint target names
    # "layers.{N}.<rest>"; <rest> is the state_dict key in BOTH modules
    # (HF Qwen3_5DecoderLayer and our HybridBlock share naming).
    loaded = {}
    for target_name, tensor in iter_layer_tensors(snap, args.layer):
        prefix = f"layers.{args.layer}."
        assert target_name.startswith(prefix), target_name
        loaded[target_name[len(prefix):]] = tensor.to(dtype)
    hf_keys = set(hf_layer.state_dict().keys())
    our_keys = set(our_block.state_dict().keys())
    print(f"layer={args.layer} type={layer_type} tokens={args.tokens} "
          f"seed={args.seed} dtype={dtype} tensors_loaded={len(loaded)}")
    print(f"hf_keys={len(hf_keys)} our_keys={len(our_keys)}")
    missing_hf = sorted(hf_keys - set(loaded))
    missing_ours = sorted(our_keys - set(loaded))
    extra = sorted(set(loaded) - hf_keys - our_keys)
    print(f"missing_hf={missing_hf} missing_ours={missing_ours} extra={extra}")
    assert not missing_hf, f"HF layer missing tensors: {missing_hf}"
    assert not missing_ours, f"our block missing tensors: {missing_ours}"
    assert not extra, f"loaded tensors belonging to neither module: {extra}"
    assert hf_keys == our_keys == set(loaded), "key sets must match exactly"
    hf_layer.load_state_dict(loaded, strict=True)
    our_block.load_state_dict(loaded, strict=True)
    del loaded
    gc.collect()

    # ---- identical seeded input ---------------------------------------
    gen = torch.Generator().manual_seed(args.seed)
    x32 = torch.randn(1, args.tokens, our_cfg.hidden_size, generator=gen) * 0.5
    x = x32.to(dtype)
    print(_fmt("input", _metrics(x, x)))

    rope = Qwen3_5TextRotaryEmbedding(hf_cfg)
    pos = torch.arange(args.tokens).unsqueeze(0)  # [1, T] for rope's (3,B,T) expand
    with torch.no_grad():
        cos, sin = rope(x, pos)  # [1, T, rotary_dim]
        if layer_type == "full_attention":
            T = args.tokens
            mask = torch.triu(
                torch.full((T, T), float("-inf"), dtype=dtype), diagonal=1
            ).view(1, 1, T, T)
        else:
            mask = None

        results = {}
        with torch.no_grad():
            # Isolated submodules on a COMMON input (bisect, not cascade).
            hf_n = hf_layer.input_layernorm(x)
            our_n = our_block.input_layernorm(x)
            results["input_layernorm"] = _metrics(hf_n, our_n)

            common = hf_n
            if layer_type == "full_attention":
                hf_mix, _ = hf_layer.self_attn(
                    hidden_states=common, position_embeddings=(cos, sin),
                    attention_mask=mask)
                our_mix = our_block.self_attn(common, cos[0], sin[0])
            else:
                hf_mix = hf_layer.linear_attn(
                    hidden_states=common, cache_params=None, attention_mask=None)
                our_mix = our_block.linear_attn(common)
            results["mixer"] = _metrics(hf_mix, our_mix)

            post_in = x + hf_mix  # common residual stream into post norm+MLP
            hf_pn = hf_layer.post_attention_layernorm(post_in)
            our_pn = our_block.post_attention_layernorm(post_in)
            results["post_layernorm"] = _metrics(hf_pn, our_pn)
            hf_mlp = hf_layer.mlp(hf_pn)
            our_mlp = our_block.mlp(hf_pn)  # common normed input isolates MLP
            results["mlp"] = _metrics(hf_mlp, our_mlp)

            # Whole block (each through its own full path).
            hf_out = hf_layer(x, position_embeddings=(cos, sin),
                              attention_mask=mask, position_ids=None,
                              past_key_values=None)
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]
            our_out = our_block(x)
            results["whole_block"] = _metrics(hf_out, our_out)

    for name in ("input_layernorm", "mixer", "post_layernorm", "mlp", "whole_block"):
        print(_fmt(name, results[name]))
    print(f"hf_out_norm={hf_out.float().norm().item():.6g} "
          f"our_out_norm={our_out.float().norm().item():.6g} "
          f"hf_out_rms={results['whole_block']['ref_rms']:.6g} "
          f"our_out_rms={results['whole_block']['got_rms']:.6g}")

    del hf_layer, our_block, hf_out, our_out, x, x32
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
