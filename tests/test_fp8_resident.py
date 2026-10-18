"""Resident-FP8 checkpoint loading into a converted hybrid (ch 22)."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inference_from_scratch.fp8_linear import BlockFP8Linear, replace_linears_with_fp8
from inference_from_scratch.hybrid import HybridConfig, HybridLM
from inference_from_scratch.hybrid_checkpoint import load_into_model, validate_state_against_model


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, partial_rotary_factor=0.25,
        rope_theta=10000.0, mtp_num_hidden_layers=1,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _fp8_targets(model):
    """True FP8 projections of the tiny stack (BF16 a/b + head excluded)."""
    out = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head" or name.endswith(("in_proj_a", "in_proj_b")):
            continue
        out.add(name)
    return out


def _ckpt_name(target: str) -> str:
    if target == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target == "norm.weight":
        return "model.language_model.norm.weight"
    if target == "lm_head.weight":
        return "lm_head.weight"
    assert target.startswith("layers.")
    return f"model.language_model.{target}"


def _write_checkpoint(root, src):
    tensors = {
        _ckpt_name(t): v.detach().contiguous()
        for t, v in src.state_dict().items()
    }
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = max(1, len(names) // 2)
    save_file({k: tensors[k] for k in names[:half]},
              str(root / "shard-a.safetensors"), metadata={"format": "pt"})
    save_file({k: tensors[k] for k in names[half:]},
              str(root / "shard-b.safetensors"), metadata={"format": "pt"})


def _model(seed):
    torch.manual_seed(seed)
    m = HybridLM(_tiny_config())
    m.to(torch.bfloat16)
    m.eval()
    return m


def test_resident_load_logits_equal_converted_original(tmp_path):
    src = _model(7)
    targets = _fp8_targets(src)
    assert len(targets) == 25  # 3x(3 lin + 3 mlp) + (4 attn + 3 mlp)
    replace_linears_with_fp8(src, targets)
    _write_checkpoint(tmp_path, src)

    dst = _model(99)
    replace_linears_with_fp8(dst, targets)
    assert isinstance(dst.layers[3].self_attn.q_proj, BlockFP8Linear)

    res = validate_state_against_model(dst, tmp_path)
    assert res["ok"] is True, res

    report = load_into_model(dst, tmp_path, strict=True, fp8_mode="resident")
    assert report["ok"] is True and report["fp8_mode"] == "resident"
    assert set(report["resident_targets"]) == targets
    # Every state entry landed: dense targets once, FP8 as weight+scale.
    assert report["loaded_count"] == len(list(dst.state_dict()))

    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    for name in targets:
        assert torch.equal(dst_sd[f"{name}.weight"], src_sd[f"{name}.weight"])
        assert torch.equal(
            dst_sd[f"{name}.weight_scale_inv"], src_sd[f"{name}.weight_scale_inv"]
        )
        assert dst_sd[f"{name}.weight"].dtype == torch.float8_e4m3fn

    idx = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        a, _ = src(idx)
        b, _ = dst(idx)
    assert torch.equal(a, b)


def test_resident_validate_rejects_bad_mode(tmp_path):
    dst = _model(1)
    replace_linears_with_fp8(dst, _fp8_targets(dst))
    with pytest.raises(ValueError, match="fp8_mode"):
        load_into_model(dst, tmp_path, fp8_mode="bogus")
