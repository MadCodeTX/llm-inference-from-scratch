"""CPU tests for run_qwen38_tp2 wiring (no CUDA/distributed needed)."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen38_tp2.py"
_spec = importlib.util.spec_from_file_location("run_qwen38_tp2", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_fp8_path_mapping():
    assert _mod.use_triton_for_path("triton") is True
    assert _mod.use_triton_for_path("reference") is False
    with pytest.raises(ValueError):
        _mod.use_triton_for_path("bogus")


def test_parser_defaults():
    args = _mod.build_parser().parse_args([])
    assert args.fp8_path == "triton"
    assert args.max_new_tokens == 32
    assert args.memory_guard_gib == 23.0
    assert args.debug_token_sync is False
    assert args.out.endswith("ours_tp2_first_run.json")
    args = _mod.build_parser().parse_args(["--fp8-path", "reference"])
    assert args.fp8_path == "reference"


def test_norm_mode_flag():
    import torch
    from inference_from_scratch.hybrid import QwenRMSNorm

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = QwenRMSNorm(8)
            self.sub = torch.nn.ModuleList([QwenRMSNorm(4)])

    args = _mod.build_parser().parse_args([])
    assert args.norm == "fused"
    args = _mod.build_parser().parse_args(["--norm", "manual"])
    assert args.norm == "manual"
    model = _M()
    assert _mod.apply_norm_mode(model, "fused") == 2
    assert model.norm.use_fused_rmsnorm is True
    assert model.sub[0].use_fused_rmsnorm is True
    assert _mod.apply_norm_mode(model, "manual") == 2
    assert model.norm.use_fused_rmsnorm is False
    with pytest.raises(ValueError):
        _mod.apply_norm_mode(model, "bogus")


def test_compare_ids():
    exact = _mod.compare_ids([1, 2, 3], [1, 2, 3])
    assert exact == {"exact_match": True, "matching_positions": 3,
                     "compared_tokens": 3, "first_divergence": None,
                     "generated_at_divergence": None,
                     "reference_at_divergence": None}
    div = _mod.compare_ids([1, 9, 3], [1, 2, 3])
    assert div["exact_match"] is False
    assert div["matching_positions"] == 2
    assert div["first_divergence"] == 1
    assert div["generated_at_divergence"] == 9
    assert div["reference_at_divergence"] == 2
