"""CPU tests for profile_qwen38_tp2 helpers (no CUDA/distributed needed)."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_qwen38_tp2.py"
_spec = importlib.util.spec_from_file_location("profile_qwen38_tp2", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_per_token_ms():
    assert _mod.per_token_ms({"fp8_linear": 160.0, "norms": 32.0}, 16) == {
        "fp8_linear": 10.0, "norms": 2.0}
    with pytest.raises(ValueError):
        _mod.per_token_ms({"fp8_linear": 1.0}, 0)


def test_host_gap_ms():
    assert _mod.host_gap_ms(44.0, 31.0) == 13.0
    assert _mod.host_gap_ms(20.0, 20.0) == 0.0


def test_parser_defaults():
    args = _mod.build_parser().parse_args([])
    assert args.tokens == 16
    assert args.warmup == 4
    assert args.fp8_path == "triton"
    assert args.norm == "manual"
    assert args.out.endswith("benchmarks/profile/tp2_profile.json")
    assert args.summary.endswith("benchmarks/profile/SUMMARY_tp2.md")
    args = _mod.build_parser().parse_args(["--norm", "fused"])
    assert args.norm == "fused"
