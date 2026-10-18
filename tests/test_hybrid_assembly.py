"""Full-model assembly: iter / meta-validate / streaming load (ch 22)."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inference_from_scratch.hybrid import HybridConfig, HybridLM, TinyHybridLM
from inference_from_scratch.hybrid_checkpoint import (
    build_manifest, iter_checkpoint_state, load_into_model,
    validate_state_against_model,
)
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8


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


def _ckpt_name(target: str) -> str:
    if target == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target == "norm.weight":
        return "model.language_model.norm.weight"
    if target == "lm_head.weight":
        return "lm_head.weight"
    if target.startswith("mtp."):
        return target
    assert target.startswith("layers.")
    return f"model.language_model.{target}"


def _build_fixture(root, src, *, quantize_target=None, drop_target=None,
                   add_mtp=False, mutate=None):
    """Multi-shard checkpoint from src's own state_dict via ckpt naming.

    All tensors stored BF16 except ``quantize_target`` (one 2-D weight
    stored FP8-block-quantized + its ``weight_scale_inv``).  Returns
    the dequantized reference for the quantized target (or None).
    """
    tensors: dict[str, torch.Tensor] = {}
    ref = None
    for target, tensor in src.state_dict().items():
        if target == drop_target:
            continue
        ckpt = _ckpt_name(target)
        if target == quantize_target:
            q, scale = quantize_block_fp8(tensor.detach().to(torch.float32))
            scale = scale.to(torch.bfloat16)
            tensors[ckpt] = q.contiguous()
            tensors[ckpt + "_scale_inv"] = scale.contiguous()
            ref = dequantize_block_fp8(q, scale.float(), out_dtype=torch.bfloat16)
        else:
            tensors[ckpt] = tensor.detach().to(torch.bfloat16).contiguous()
    if add_mtp:
        tensors["mtp.fc.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
        tensors["mtp.norm.weight"] = torch.randn(32, dtype=torch.bfloat16)
    if mutate:
        tensors = mutate(dict(tensors))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = max(1, len(names) // 2)
    save_file({k: tensors[k] for k in names[:half]},
              str(root / "shard-a.safetensors"), metadata={"format": "pt"})
    save_file({k: tensors[k] for k in names[half:]},
              str(root / "shard-b.safetensors"), metadata={"format": "pt"})
    return ref


def _src_model(seed=7):
    torch.manual_seed(seed)
    m = HybridLM(_tiny_config())
    m.to(torch.bfloat16)
    m.eval()
    return m


def _dst_model(seed=99):
    torch.manual_seed(seed)
    m = HybridLM(_tiny_config())
    m.to(torch.bfloat16)
    m.eval()
    return m


def test_hybridlm_alias_preserves_tiny_naming():
    cfg = _tiny_config()
    torch.manual_seed(0)
    a = HybridLM(cfg)
    torch.manual_seed(0)
    b = TinyHybridLM(cfg)
    assert list(a.state_dict()) == list(b.state_dict())
    assert a.state_dict()["embed_tokens.weight"].shape == (97, 32)
    assert "norm.weight" in a.state_dict()  # final norm
    assert "layers.3.self_attn.q_proj.weight" in a.state_dict()  # full
    assert "layers.0.linear_attn.in_proj_qkv.weight" in a.state_dict()  # linear


def test_meta_validation_ok_and_param_count(tmp_path):
    cfg = _tiny_config()
    src = _src_model()
    _build_fixture(tmp_path, src, add_mtp=True)
    with torch.device("meta"):
        meta = HybridLM(cfg)
    res = validate_state_against_model(meta, tmp_path)
    assert res["ok"] is True, res
    assert res["missing"] == [] and res["unexpected"] == []
    assert res["shape_mismatch"] == [] and res["dtype_mismatch"] == []
    expected = sum(p.numel() for p in src.parameters())
    assert res["parameter_count"] == expected
    # MTP explicitly included becomes unexpected (LM has no MTP params).
    res_mtp = validate_state_against_model(meta, tmp_path, include_mtp=True)
    assert sorted(res_mtp["unexpected"]) == ["mtp.fc.weight", "mtp.norm.weight"]


def test_iter_model_order_consumes_scale_and_mtp_excluded(tmp_path):
    cfg = _tiny_config()
    src = _src_model()
    qt = "layers.3.mlp.gate_proj.weight"
    ref = _build_fixture(tmp_path, src, quantize_target=qt, add_mtp=True)
    items = list(iter_checkpoint_state(tmp_path, cfg))
    names = [n for n, _ in items]
    assert names == list(src.state_dict())  # deterministic model order
    assert not any(n.endswith("weight_scale_inv") for n in names)
    assert not any(n.startswith("mtp.") for n in names)
    assert names[0] == "embed_tokens.weight"
    assert names[-1] == "lm_head.weight" and "norm.weight" in names
    got = dict(items)[qt]
    assert got.dtype == torch.bfloat16 and torch.equal(got, ref)
    names_mtp = [n for n, _ in iter_checkpoint_state(tmp_path, cfg, include_mtp=True)]
    assert names_mtp[:len(names)] == names
    assert names_mtp[len(names):] == ["mtp.fc.weight", "mtp.norm.weight"]


def test_streaming_load_reproduces_logits_exact_bf16(tmp_path):
    src = _src_model()
    _build_fixture(tmp_path, src)
    dst = _dst_model()
    report = load_into_model(dst, tmp_path, strict=True)
    assert report["ok"] is True and report["loaded_count"] == len(list(src.state_dict()))
    idx = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        a, _ = src(idx)
        b, _ = dst(idx)
    assert torch.equal(a, b)


def test_quantized_tensor_exact_and_logits_close(tmp_path):
    src = _src_model()
    qt = "layers.3.mlp.gate_proj.weight"
    ref = _build_fixture(tmp_path, src, quantize_target=qt)
    dst = _dst_model()
    load_into_model(dst, tmp_path, strict=True)
    loaded = dict(dst.state_dict())[qt]
    assert torch.equal(loaded, ref)  # dequantized path is exact
    idx = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        a, _ = src(idx)
        b, _ = dst(idx)
    # Tolerance only on the intentionally quantized path.
    assert torch.allclose(a.float(), b.float(), atol=0.5, rtol=0.1), (a - b).abs().max()


def test_missing_reported_and_strict_load_raises_before_mutating(tmp_path):
    src = _src_model()
    _build_fixture(tmp_path, src, drop_target="norm.weight")
    with torch.device("meta"):
        meta = HybridLM(_tiny_config())
    res = validate_state_against_model(meta, tmp_path)
    assert res["ok"] is False and res["missing"] == ["norm.weight"]

    dst = _dst_model()
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    with pytest.raises(RuntimeError, match="pre-validation"):
        load_into_model(dst, tmp_path, strict=True)
    # Pre-validation: no tensor (early or late) was mutated.
    for k, v in dst.state_dict().items():
        assert torch.equal(v, before[k]), k
    report = load_into_model(dst, tmp_path, strict=False)
    assert report["loaded_count"] == len(before) - 1
    assert torch.equal(dict(dst.state_dict())["embed_tokens.weight"],
                       dict(src.state_dict())["embed_tokens.weight"])


def test_shape_and_unexpected_reported(tmp_path):
    src = _src_model()

    def mutate(t):
        t["model.language_model.layers.0.input_layernorm.weight"] = \
            torch.zeros(7, dtype=torch.bfloat16)
        # A full-attn proj on a linear layer maps to a non-model target.
        t["model.language_model.layers.0.self_attn.q_proj.weight"] = \
            torch.zeros(128, 32, dtype=torch.bfloat16)
        return t

    _build_fixture(tmp_path, src, mutate=mutate)
    with torch.device("meta"):
        meta = HybridLM(_tiny_config())
    res = validate_state_against_model(meta, tmp_path)
    assert res["ok"] is False
    assert res["unexpected"] == ["layers.0.self_attn.q_proj.weight"]
    assert len(res["shape_mismatch"]) == 1
    m = res["shape_mismatch"][0]
    assert m["target_name"] == "layers.0.input_layernorm.weight"
    assert m["expected_shape"] == [32] and m["actual_shape"] == [7]
    with pytest.raises(RuntimeError, match="pre-validation"):
        load_into_model(_dst_model(), tmp_path, strict=True)
