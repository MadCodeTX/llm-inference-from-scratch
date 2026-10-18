"""Profiling rollup utilities (pure, CPU)."""
import pytest

from inference_from_scratch.profiling import categorize_op, rollup, roofline_ms


def test_categorize_known_ops():
    assert categorize_op("aten::mm") == "matmul_linear"
    assert categorize_op("aten::addmm") == "matmul_linear"
    assert categorize_op("aten::copy_") == "device_transfer"
    assert categorize_op("aten::_to_copy") == "device_transfer"
    assert categorize_op("aten::rsqrt") == "elementwise_norm"
    assert categorize_op("aten::item") == "host_sync_other"


def test_rollup_sums_and_shares():
    r = rollup([{"name": "aten::mm", "self_cuda_time_us": 300.0},
                {"name": "aten::copy_", "self_cuda_time_us": 100.0}])
    assert r["total_cuda_us"] == 400.0
    assert r["share"]["matmul_linear"] == pytest.approx(0.75)
    assert r["by_category_count"]["device_transfer"] == 1


def test_rollup_empty_is_zero_not_nan():
    r = rollup([])
    assert r["total_cuda_us"] == 0.0 and all(v == 0.0 for v in r["share"].values())


def test_roofline_ms():
    assert roofline_ms(920e9) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        roofline_ms(1.0, gbps=0)
