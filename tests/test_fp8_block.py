"""Tests for block FP8 (E4M3, 128x128) quantize/dequantize/matmul + header."""
import json
import struct

import pytest
import torch

from inference_from_scratch.quantization import (
    dequantize_block_fp8,
    fp8_block_matmul,
    quantize_block_fp8,
)
from inference_from_scratch.weights import read_safetensors_header


def test_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(256, 256, dtype=torch.float32) * 0.05
    q, s = quantize_block_fp8(w, block=(128, 128))
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert s.shape == (2, 2)
    deq = dequantize_block_fp8(q, s, out_dtype=torch.float32)
    err = (w - deq).abs()
    # FP8 E4M3 has ~3 mantissa bits: relative error bounded, small vs scale.
    assert torch.isfinite(deq).all()
    assert err.max() < 0.02
    assert err.mean() < 0.002
    # Per-block: dequant block absmax approx original block absmax (<= s*448).
    assert deq.abs().max() <= float(s.max()) * 448 + 1e-6


def test_roundtrip_zero_block():
    w = torch.zeros(128, 128)
    q, s = quantize_block_fp8(w)
    assert float(s[0, 0]) == 1.0
    deq = dequantize_block_fp8(q, s, out_dtype=torch.float32)
    assert torch.equal(deq, torch.zeros(128, 128))


def test_edge_cropping_non_divisible():
    torch.manual_seed(1)
    w = torch.randn(200, 300) * 0.05
    q, s = quantize_block_fp8(w, block=(128, 128))
    assert s.shape == (2, 3)  # ceil(200/128), ceil(300/128)
    deq = dequantize_block_fp8(q, s, out_dtype=torch.float32)
    assert deq.shape == (200, 300)
    assert (w - deq).abs().max() < 0.05

    # Small block, odd shape.
    w2 = torch.randn(10, 7) * 0.1
    q2, s2 = quantize_block_fp8(w2, block=(4, 4))
    assert s2.shape == (3, 2)
    assert dequantize_block_fp8(q2, s2, block=(4, 4), out_dtype=torch.float32).shape == (10, 7)


def test_dequantize_shape_validation():
    q = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequantize_block_fp8(q, torch.ones(2, 2))
    with pytest.raises(ValueError):
        dequantize_block_fp8(q.float(), torch.ones(1, 1))
    with pytest.raises(ValueError):
        quantize_block_fp8(torch.randn(3, 4, 5))


def test_matmul_vs_dequant_reference():
    torch.manual_seed(2)
    w = torch.randn(256, 128) * 0.05
    q, s = quantize_block_fp8(w, block=(128, 128))
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y = fp8_block_matmul(x, q, s)
    ref = x @ dequantize_block_fp8(q, s, out_dtype=torch.bfloat16).T
    assert y.shape == (4, 256)
    assert torch.equal(y, ref)


def _hand_built(dtype: str, shape: list[int], nbytes: int) -> bytes:
    header = json.dumps({"w": {"dtype": dtype, "shape": shape, "data_offsets": [0, nbytes]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * nbytes


def test_header_parses_f8_dtypes(tmp_path):
    for dtype, shape, n in [("F8_E4M3", [2, 3], 6), ("F8_E5M2", [4], 4)]:
        p = tmp_path / f"{dtype}.safetensors"
        p.write_bytes(_hand_built(dtype, shape, n))
        h = read_safetensors_header(p)
        assert h["w"] == {"dtype": dtype, "shape": shape, "data_offsets": [0, n]}


def test_header_rejects_bad_f8_span(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(_hand_built("F8_E4M3", [2, 3], 5))  # needs 6 bytes
    with pytest.raises(ValueError):
        read_safetensors_header(p)
