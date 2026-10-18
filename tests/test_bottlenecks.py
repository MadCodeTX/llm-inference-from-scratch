"""Tests for Ch 13: bottlenecks (roofline / arithmetic intensity)."""

import pytest
import torch

from inference_from_scratch.bottlenecks import (
    HardwareSpec,
    arithmetic_intensity,
    classify,
    count_params,
    estimate_bytes_per_token,
    estimate_flops_per_token,
    make_example_spec,
    matmul_params,
    ridge_point,
    roofline_time_lower_bound,
    try_read_local_gpu,
    try_read_local_gpus,
)
from inference_from_scratch.config import ModelConfig
from inference_from_scratch.model import DecoderLM


def _config(**overrides):
    kwargs = dict(
        vocab_size=64, d_model=64, n_layer=2, n_head=4,
        n_kv_head=2, d_ff=128, max_seq_len=512,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def test_count_params_matches_model():
    for overrides in ({}, {"tie_embeddings": False}, {"mlp_type": "gelu"},
                      {"pos_type": "learned", "norm_type": "layernorm"}):
        cfg = _config(**overrides)
        torch.manual_seed(0)
        assert count_params(cfg) == DecoderLM(cfg).num_params()


def test_example_spec_ridge_is_clean_100():
    spec = make_example_spec()
    assert "not a real GPU" in spec.name
    assert ridge_point(spec) == pytest.approx(100.0)


def test_ridge_point_refuses_unknown_peaks():
    spec = HardwareSpec(name="x", peak_flops=None,
                        memory_bandwidth_bytes_s=1e12, memory_bytes=None)
    with pytest.raises(ValueError):
        ridge_point(spec)


def test_decode_intensity_far_below_prefill_at_batch1():
    cfg = _config()
    dec = classify(cfg, make_example_spec(), seq_len=128, phase="decode")
    pre = classify(cfg, make_example_spec(), seq_len=128, phase="prefill")
    assert dec["intensity"] < pre["intensity"] / 10
    assert dec["bound"] == "memory"


def test_intensity_grows_with_batch_decode():
    cfg = _config()
    b1 = arithmetic_intensity(
        estimate_flops_per_token(cfg, 64, "decode"),
        estimate_bytes_per_token(cfg, 64, "decode", batch=1))
    b8 = arithmetic_intensity(
        estimate_flops_per_token(cfg, 64, "decode"),
        estimate_bytes_per_token(cfg, 64, "decode", batch=8))
    assert b8 > b1


def test_classify_flips_across_synthetic_ridge():
    cfg = _config()
    low_ridge = HardwareSpec("low", peak_flops=1e9,
                             memory_bandwidth_bytes_s=1e12, memory_bytes=None)
    high_ridge = HardwareSpec("high", peak_flops=1e15,
                              memory_bandwidth_bytes_s=1e12, memory_bytes=None)
    assert classify(cfg, low_ridge, 128, "decode")["bound"] == "compute"
    assert classify(cfg, high_ridge, 128, "decode")["bound"] == "memory"


def test_roofline_lower_bound_is_max_term():
    spec = make_example_spec()
    # flops term wins: 1e12 flops / 1e14 = 0.01s vs 1e6 bytes / 1e12 tiny
    assert roofline_time_lower_bound(1e12, 1e6, spec) == pytest.approx(0.01)
    # bytes term wins
    assert roofline_time_lower_bound(1e6, 1e12, spec) == pytest.approx(1.0)
    # config form agrees with raw form
    cfg = _config()
    raw = roofline_time_lower_bound(
        estimate_flops_per_token(cfg, 32, "decode"),
        estimate_bytes_per_token(cfg, 32, "decode"), spec)
    via_cfg = roofline_time_lower_bound(cfg, spec, 32, "decode")
    assert via_cfg == pytest.approx(raw)


def test_prefill_bytes_amortize_weights():
    cfg = _config()
    b_short = estimate_bytes_per_token(cfg, 16, "prefill")
    b_long = estimate_bytes_per_token(cfg, 256, "prefill")
    assert b_long < b_short  # longer prompt amortizes the weight read


def test_invalid_phase_and_seq_raise():
    cfg = _config()
    with pytest.raises(ValueError):
        estimate_flops_per_token(cfg, 16, "train")
    with pytest.raises(ValueError):
        estimate_bytes_per_token(cfg, 0, "decode")
    with pytest.raises(ValueError):
        arithmetic_intensity(10, 0)


def test_matmul_params_positive_and_head_counted():
    cfg = _config()
    assert matmul_params(cfg) > 0
    assert matmul_params(cfg) >= cfg.vocab_size * cfg.d_model


def test_try_read_local_gpu_never_fabricates_peaks():
    spec = try_read_local_gpu()
    if spec is not None:  # GPU box: only name/VRAM may be filled
        assert spec.peak_flops is None
        assert spec.memory_bandwidth_bytes_s is None
        assert spec.memory_bytes and spec.memory_bytes > 0


def _fake_nvidia_smi(tmp_path, monkeypatch, body: str):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")


def test_try_read_local_gpus_enumerates_all(tmp_path, monkeypatch):
    _fake_nvidia_smi(
        tmp_path, monkeypatch,
        "printf 'NVIDIA GeForce RTX 4090, 24564\\nNVIDIA GeForce RTX 4090, 24564\\n'",
    )
    gpus = try_read_local_gpus()
    assert len(gpus) == 2
    assert all(g.name == "NVIDIA GeForce RTX 4090" for g in gpus)
    assert all(g.memory_bytes == 24564 * 1024 * 1024 for g in gpus)
    assert all(g.peak_flops is None for g in gpus)
    # Backward compat: singular helper returns the first GPU.
    assert try_read_local_gpu() == gpus[0]


def test_try_read_local_gpus_absent_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert try_read_local_gpus() == []
    assert try_read_local_gpu() is None


def test_try_read_local_gpus_skips_malformed_lines(tmp_path, monkeypatch):
    _fake_nvidia_smi(
        tmp_path, monkeypatch,
        "printf 'garbage-line\\nNVIDIA GeForce RTX 4090, 24564\\nBad, notanumber\\n'",
    )
    gpus = try_read_local_gpus()
    assert len(gpus) == 1
    assert gpus[0].memory_bytes == 24564 * 1024 * 1024
