"""Tests for benchmarks/roofline_compute.py pure arithmetic.

No GPU and no checkpoint are touched: only the closed-form helpers
(ceiling, KV bytes, FLOPs) are exercised.
"""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "roofline_compute", _REPO / "benchmarks" / "roofline_compute.py"
)
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)


def test_ceiling_is_bandwidth_over_bytes():
    out = rc.ceiling(100e9, 10e9)
    assert out["seconds_per_token"] == 0.1
    assert out["tokens_per_s"] == 10.0


def test_kv_read_counts_only_full_layers():
    # 2 * 16 layers * 4 kv heads * 256 head_dim * S, FP8 = 1 byte.
    assert rc.kv_read_bytes(2048) == 2 * 16 * 4 * 256 * 2048
    assert rc.kv_read_bytes(1) == 32768


def test_decode_flops_two_params_plus_attention():
    base = rc.decode_flops_per_token(1000, 1)
    assert base == 2.0 * (1000 + 248320 * 5120) + 4.0 * 16 * 5120
    # Attention term grows linearly with context.
    assert rc.decode_flops_per_token(0, 2048) > rc.decode_flops_per_token(0, 1)
