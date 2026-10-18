"""Tests for Ch 10: INT8 per-channel quantization + bytes accounting.

The error tests hold the roundtrip to the theory, not to a tuned
tolerance: rounding to a grid of step ``scale_i`` moves an element by
at most ``scale_i / 2``, so the per-row error must satisfy exactly
that bound (with tiny slack for the float divide/multiply itself).
Bytes tests are exact integer arithmetic, cross-checked against
hand-computed values. Tiny CPU tensors throughout.
"""

import pytest
import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.quantization import (
    BITS_PER_PARAM,
    bytes_for_dtype,
    bytes_for_params,
    dequantize_int8,
    kv_cache_bytes,
    quantization_error,
    quantize_int8_per_channel,
    worked_example,
)


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=32,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


# --- quantize / dequantize roundtrip ----------------------------------------


def test_roundtrip_error_small_and_bounded_by_half_scale():
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 2.0
    q, scales = quantize_int8_per_channel(w)

    assert q.dtype == torch.int8
    assert q.shape == w.shape
    assert scales.shape == (16,)
    assert int(q.abs().max()) <= 127

    w2 = dequantize_int8(q, scales)
    # Per-element bound: |w - w2| <= scale_i / 2 (row-wise), approx.
    err = (w - w2).abs()
    bound = (scales / 2).unsqueeze(1).expand_as(err)
    assert torch.all(err <= bound + 1e-5)

    metrics = quantization_error(w, w2)
    assert metrics["max_abs"] <= float(scales.max()) / 2 + 1e-5
    assert metrics["rmse"] >= 0.0
    assert metrics["rmse"] < 0.1  # small vs weights of scale ~2
    assert metrics["max_abs"] == pytest.approx(float(err.max()), abs=1e-6)


def test_quantize_is_symmetric_and_zero_is_exact():
    w = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    q, scales = quantize_int8_per_channel(w)
    assert q[0].tolist() == [64, -64, 127, -127]  # symmetric grid
    w2 = dequantize_int8(q, scales)
    # Grid points reproduce near-exactly; the max element is exact.
    assert w2[0, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert w2[0, 3].item() == pytest.approx(-1.0, abs=1e-6)


def test_zero_tensor_edge():
    w = torch.zeros(4, 8)
    q, scales = quantize_int8_per_channel(w)
    assert torch.equal(q, torch.zeros(4, 8, dtype=torch.int8))
    assert torch.all(scales > 0)  # pinned, never 0 -> no NaN path
    w2 = dequantize_int8(q, scales)
    assert torch.equal(w2, w)
    assert quantization_error(w, w2) == {
        "max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0,
    }


def test_zero_row_among_nonzero_rows():
    torch.manual_seed(1)
    w = torch.randn(3, 8)
    w[1] = 0.0
    q, scales = quantize_int8_per_channel(w)
    assert torch.equal(q[1], torch.zeros(8, dtype=torch.int8))
    w2 = dequantize_int8(q, scales)
    assert torch.equal(w2[1], w[1])  # zero row roundtrips exactly
    assert not torch.equal(w2[0], w[0])  # nonzero row: real grid error


def test_per_channel_scales_differ_per_row():
    w = torch.tensor([[1.0, 0.0], [100.0, 50.0]])
    q, scales = quantize_int8_per_channel(w)
    assert scales[0].item() == pytest.approx(1.0 / 127)
    assert scales[1].item() == pytest.approx(100.0 / 127)
    # Small row keeps fine resolution despite the large row's scale.
    assert q[0].tolist() == [127, 0]
    assert q[1, 0].item() == 127


def test_quantize_rejects_non_2d_and_error_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="2D"):
        quantize_int8_per_channel(torch.randn(4))
    with pytest.raises(ValueError, match="shape mismatch"):
        quantization_error(torch.zeros(2, 3), torch.zeros(3, 2))
    q, _ = quantize_int8_per_channel(torch.randn(4, 5))
    with pytest.raises(ValueError, match="scales"):
        dequantize_int8(q, torch.ones(3))


# --- bytes math --------------------------------------------------------------


def test_bytes_for_params_dtype_table():
    n = 1000
    assert bytes_for_params(n, 32) == 4000  # fp32
    assert bytes_for_params(n, 16) == 2000  # fp16 and bf16 (same bytes)
    assert bytes_for_params(n, 8) == 1000   # int8
    assert bytes_for_params(n, 4) == 500    # int4: packed pairs
    assert bytes_for_dtype(n, "bf16") == bytes_for_dtype(n, "fp16") == 2000
    assert BITS_PER_PARAM == {
        "fp32": 32, "fp16": 16, "bf16": 16, "int8": 8, "int4": 4,
    }
    with pytest.raises(ValueError):
        bytes_for_dtype(n, "fp64-ish")
    with pytest.raises(ValueError):
        bytes_for_params(-1, 8)


def test_bytes_for_params_int4_odd_count_rounds_up():
    assert bytes_for_params(1, 4) == 1  # half a byte still costs a byte
    assert bytes_for_params(3, 4) == 2
    assert bytes_for_params(2, 4) == 1
    assert bytes_for_params(0, 32) == 0


def test_kv_cache_bytes_formula_tiny_config():
    cfg = _tiny_config()  # L=2, H_kv=2, head_dim=32//4=8
    assert cfg.head_dim == 8
    # 2 * 2 layers * 1 batch * 2 kv-heads * 10 tokens * 8 * 2 bytes
    assert kv_cache_bytes(cfg, seq_len=10, batch=1, bytes_per_elem=2) == 1280
    assert kv_cache_bytes(cfg, seq_len=10, batch=2, bytes_per_elem=2) == 2560
    assert kv_cache_bytes(cfg, seq_len=0, batch=4) == 0
    # GQA lever: MHA (H_kv == H) costs exactly H/H_kv times more.
    mha = _tiny_config(n_kv_head=4)
    assert kv_cache_bytes(mha, 10, 1, 2) == 2 * kv_cache_bytes(cfg, 10, 1, 2)


def test_worked_example_matches_formulas_and_is_labeled_assumed():
    ex = worked_example()
    cfg = ModelConfig(
        vocab_size=128_000, d_model=5120, n_layer=64, n_head=40,
        n_kv_head=8, d_ff=13_824, max_seq_len=4096,
    )
    assert ex["weights_fp32_bytes"] == bytes_for_params(27_000_000_000, 32)
    assert ex["weights_fp16_bytes"] == ex["weights_bf16_bytes"] == 54_000_000_000
    assert ex["weights_int8_bytes"] == 27_000_000_000
    assert ex["weights_int4_bytes"] == 13_500_000_000
    assert ex["kv_fp16_b1_seq4096_bytes"] == kv_cache_bytes(cfg, 4096, 1, 2)
    assert ex["kv_fp16_b1_seq4096_bytes"] == 1_073_741_824  # 1 GiB
    assert ex["kv_fp16_per_token_bytes"] == 262_144  # 256 KiB
    # The example must announce itself as assumed, never as a real config.
    import inference_from_scratch.quantization as quant_mod

    assert "ASSUMED" in quant_mod.__doc__
    assert "NOT" in quant_mod.__doc__
