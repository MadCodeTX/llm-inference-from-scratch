"""CPU tests for run_qwen38_gpu fast-FP8 wiring (no CUDA needed)."""
import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen38_gpu.py"
_spec = importlib.util.spec_from_file_location("run_qwen38_gpu", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from inference_from_scratch.hybrid import (  # noqa: E402
    DeviceMappedHybridRunner, HybridConfig, TinyHybridLM,
)


def _tiny_config():
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def test_fp8_kwargs_mapping():
    assert _mod.fp8_kwargs_for_path("reference") == {
        "use_triton": False, "use_fast_path": False}
    assert _mod.fp8_kwargs_for_path("triton") == {
        "use_triton": True, "use_fast_path": False}
    assert _mod.fp8_kwargs_for_path("scaled-mm") == {
        "use_triton": False, "use_fast_path": True}
    with pytest.raises(ValueError):
        _mod.fp8_kwargs_for_path("bogus")


def test_parser_defaults_are_fastest_measured():
    args = _mod.build_parser().parse_args([])
    assert args.fp8_path == "triton"
    assert args.sampling_sync == "batched"
    assert args.grad_mode == "inference"
    args = _mod.build_parser().parse_args(
        ["--fp8-path", "reference", "--sampling-sync", "per-token",
         "--grad-mode", "no-grad"])
    assert args.fp8_path == "reference"
    assert args.sampling_sync == "per-token"
    assert args.grad_mode == "no-grad"
    with pytest.raises(SystemExit):
        _mod.build_parser().parse_args(["--fp8-path", "nope"])


def test_summarize_divergence_quantifies_logits():
    assert _mod.summarize_divergence([1, 2, 3], [1, 2, 3]) is None
    div = _mod.summarize_divergence(
        [1, 9, 3], [1, 2, 3],
        got_logits=[5.0, 4.0, 1.0], ref_logits=[5.0, 3.5, 1.0],
        top1_logits=[5.0, 4.0, 1.0], top2_logits=[4.0, 3.5, 0.5])
    assert div["step"] == 1
    assert div["generated_token"] == 9 and div["reference_token"] == 2
    assert div["logit_diff_got_minus_ref"] == pytest.approx(0.5)
    assert div["top1_top2_margin"] == pytest.approx(0.5)


def test_batched_matches_per_token_on_tiny_runner():
    torch.manual_seed(7)
    model = TinyHybridLM(_tiny_config()).eval()
    runner = DeviceMappedHybridRunner(model, split=2, device0="cpu", device1="cpu")
    prompt = [3, 11, 5, 42]
    expected = runner.generate_greedy(torch.tensor([prompt]), 6)
    with torch.no_grad():
        batched = _mod.greedy_generate_bridge(
            runner, prompt, 6, "cpu", sampling_sync="batched", ref_ids=expected)
        per_token = _mod.greedy_generate_bridge(
            runner, prompt, 6, "cpu", sampling_sync="per-token", ref_ids=expected)
    assert batched["gen_ids"] == expected
    assert per_token["gen_ids"] == expected
    assert batched["logits_finite"] and per_token["logits_finite"]
    assert batched["divergence_logits"] is None
    assert batched["prefill_s"] >= 0 and batched["decode_s"] >= 0
