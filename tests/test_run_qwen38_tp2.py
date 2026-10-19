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


def test_microopt_flags():
    args = _mod.build_parser().parse_args([])
    # Defaults are measured winners only: vocab won paired A/B;
    # scale-cache is graph-scoped (round-2 confirmed on graph only).
    assert args.lm_head == "vocab" and args.scale_cache is None and args.probe is None
    assert _mod.resolve_scale_cache_mode(args.scale_cache, "graph") == "on"
    assert _mod.resolve_scale_cache_mode(args.scale_cache, "eager") == "off"
    assert _mod.resolve_scale_cache_mode("off", "graph") == "off"
    assert _mod.resolve_scale_cache_mode("on", "eager") == "on"
    args = _mod.build_parser().parse_args(
        ["--lm-head", "vocab", "--scale-cache", "on", "--probe", "capital"])
    assert args.lm_head == "vocab" and args.scale_cache == "on" and args.probe == "capital"
    import torch
    from inference_from_scratch.fp8_linear import BlockFP8Linear
    m = torch.nn.Sequential(BlockFP8Linear.from_linear(torch.nn.Linear(128, 128, bias=False)))
    assert _mod.apply_scale_cache_mode(m, "on") == 1
    assert m[0].cached_fp32_scale() is not None
    assert _mod.apply_scale_cache_mode(m, "off") == 1
    assert m[0].cached_fp32_scale() is None
    with pytest.raises(ValueError):
        _mod.apply_scale_cache_mode(m, "bogus")


def test_decode_mode_flag_and_plan():
    args = _mod.build_parser().parse_args([])
    assert args.decode_mode == "eager"  # default until graph measured
    args = _mod.build_parser().parse_args(["--decode-mode", "graph"])
    assert args.decode_mode == "graph"
    assert _mod.validate_decode_mode("graph") == "graph"
    assert _mod.validate_decode_mode("eager") == "eager"
    with pytest.raises(ValueError):
        _mod.validate_decode_mode("bogus")
    plan = _mod.graph_decode_plan(5, 32)
    assert plan["max_seq_len"] == 37 and plan["replays"] == 31
    assert plan["prompt_len"] + plan["replays"] <= plan["max_seq_len"]
    plan1 = _mod.graph_decode_plan(8, 1)
    assert plan1["replays"] == 0 and plan1["max_seq_len"] == 9
    with pytest.raises(ValueError):
        _mod.graph_decode_plan(0, 32)
    with pytest.raises(ValueError):
        _mod.graph_decode_plan(5, 0)


def test_fp8_config_flag():
    args = _mod.build_parser().parse_args([])
    assert args.fp8_config == "gemv-bk512"
    args = _mod.build_parser().parse_args(["--fp8-config", "old"])
    assert args.fp8_config == "old"
    args = _mod.build_parser().parse_args(["--fp8-config", "gemv"])
    assert args.fp8_config == "gemv"
    args = _mod.build_parser().parse_args(["--fp8-config", "gemv-bk512"])
    assert args.fp8_config == "gemv-bk512"
    try:
        assert _mod.validate_fp8_config("old") == "old"
        assert _mod.validate_fp8_config("gemv") == "gemv"
        assert _mod.validate_fp8_config("gemv-bk512") == "gemv-bk512"
        assert _mod.validate_fp8_config("new") == "new"
    finally:
        _mod.validate_fp8_config("gemv-bk512")
    with pytest.raises(ValueError):
        _mod.validate_fp8_config("bogus")
    with pytest.raises(ValueError):
        _mod.validate_fp8_config("prefill-s2")


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
