"""Resident block-FP8 linear: quantise, forward, replace (ch 10/22)."""
import torch
import torch.nn.functional as F

from inference_from_scratch.fp8_linear import (
    BlockFP8Linear,
    replace_linears_with_fp8,
)
from inference_from_scratch.quantization import dequantize_block_fp8
from inference_from_scratch.linear_attention import GatedDeltaNet


def test_from_linear_roundtrip_close_within_fp8_error():
    torch.manual_seed(0)
    dense = torch.nn.Linear(256, 256, bias=True)
    with torch.no_grad():
        dense.weight.mul_(0.05)
    fp8 = BlockFP8Linear.from_linear(dense)
    assert fp8.weight.dtype == torch.float8_e4m3fn
    assert fp8.weight_scale_inv.dtype == torch.bfloat16
    assert fp8.weight.shape == (256, 256)
    assert fp8.weight_scale_inv.shape == (2, 2)
    x = torch.randn(4, 256)
    with torch.no_grad():
        got, ref = fp8(x), dense(x)
    # FP8 E4M3 has 3 mantissa bits: outputs track, not bit-equal.
    assert torch.allclose(got, ref, atol=0.05, rtol=0.1), (got - ref).abs().max()
    dense2 = fp8.to_dense()
    assert torch.allclose(
        dense2.weight, fp8.dequantized_weight(torch.float32)
    )


def test_forward_exact_vs_manual_dequant_matmul():
    torch.manual_seed(1)
    fp8 = BlockFP8Linear.from_linear(torch.nn.Linear(128, 128, bias=False))
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    w = dequantize_block_fp8(
        fp8.weight, fp8.weight_scale_inv, out_dtype=torch.bfloat16
    )
    with torch.no_grad():
        assert torch.equal(fp8(x), F.linear(x, w))
        # Fast-path flag falls back silently to the same reference.
        fp8_fast = BlockFP8Linear.from_linear(
            torch.nn.Linear(128, 128, bias=False), use_fast_path=True
        )
        fp8_fast.load_state_dict(fp8.state_dict())
        assert torch.equal(fp8_fast(x), F.linear(x, w))


def test_bias_and_non_divisible_shapes():
    torch.manual_seed(2)
    dense = torch.nn.Linear(300, 200, bias=True)
    fp8 = BlockFP8Linear.from_linear(dense, block=(128, 128))
    assert fp8.weight.shape == (200, 300)
    assert fp8.weight_scale_inv.shape == (2, 3)  # ceil(200/128), ceil(300/128)
    x = torch.randn(5, 300)
    with torch.no_grad():
        got = fp8(x)
    assert got.shape == (5, 200)
    assert fp8.bias is not None and torch.equal(fp8.bias, dense.bias)
    w = dequantize_block_fp8(
        fp8.weight, fp8.weight_scale_inv, out_dtype=torch.float32
    )
    with torch.no_grad():
        assert torch.equal(got, F.linear(x, w, fp8.bias))


def test_state_dict_keys_stable_and_loadable():
    torch.manual_seed(3)
    fp8 = BlockFP8Linear.from_linear(torch.nn.Linear(64, 32, bias=True), block=(32, 32))
    keys = set(fp8.state_dict())
    assert keys == {"weight", "weight_scale_inv", "bias"}
    fresh = BlockFP8Linear(64, 32, bias=True, block=(32, 32))
    fresh.load_state_dict(fp8.state_dict())
    x = torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(fp8(x), fresh(x))
    no_bias = BlockFP8Linear.from_linear(torch.nn.Linear(64, 32, bias=False))
    assert set(no_bias.state_dict()) == {"weight", "weight_scale_inv"}


def test_replace_linears_only_named_linears():
    torch.manual_seed(4)
    net = GatedDeltaNet(
        d_model=32, num_key_heads=2, key_dim=8,
        num_value_heads=4, value_dim=8, conv_kernel=4,
    )
    norm_weight_before = net.norm.weight.clone()
    replace_linears_with_fp8(net, {"in_proj_qkv", "in_proj_z", "norm", "missing"})
    assert isinstance(net.in_proj_qkv, BlockFP8Linear)
    assert isinstance(net.in_proj_z, BlockFP8Linear)
    # Unnamed Linears and the (non-Linear) norm stay untouched.
    assert type(net.out_proj) is torch.nn.Linear
    assert type(net.in_proj_a) is torch.nn.Linear
    assert torch.equal(net.norm.weight, norm_weight_before)
    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        out, _state = net.forward_prefill(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
