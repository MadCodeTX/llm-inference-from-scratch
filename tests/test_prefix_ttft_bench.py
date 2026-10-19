"""CPU tests for prefix_ttft_bench surface (no CUDA/distributed needed)."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prefix_ttft_bench.py"
_spec = importlib.util.spec_from_file_location("prefix_ttft_bench", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_parser_defaults_are_shipped_prefix_chunk512():
    a = _mod.build_parser().parse_args([])
    assert (a.chunk, a.snapshot_interval, a.fp8_config) == (512, 512, "gemv-bk512")
    assert (a.shared, a.unique, a.storage) == (2048, 256, "both")
    assert _mod.effective_chunk_size(a.chunk, a.snapshot_interval) == 512


def test_parser_v2_baseline_and_agent_shape_still_expressible():
    a = _mod.build_parser().parse_args(
        ["--chunk", "128", "--snapshot-interval", "256",
         "--shared", "8192", "--unique", "512", "--fp8-config", "gemv"])
    assert (a.chunk, a.snapshot_interval, a.fp8_config) == (128, 256, "gemv")
    assert (a.shared, a.unique) == (8192, 512)
    assert _mod.effective_chunk_size(a.chunk, a.snapshot_interval) == 128


def test_effective_chunk_capped_by_interval_and_bad_values_rejected():
    assert _mod.effective_chunk_size(512, 256) == 256
    assert _mod.effective_chunk_size(128, 512) == 128
    with pytest.raises(SystemExit):
        _mod.build_parser().parse_args(["--chunk", "0"])
    with pytest.raises(SystemExit):
        _mod.build_parser().parse_args(["--fp8-config", "bogus"])
