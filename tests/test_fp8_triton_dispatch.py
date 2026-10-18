"""CPU dispatch/fallback tests for the Triton FP8 path (no CUDA needed)."""
import logging

import torch

import inference_from_scratch.fp8_linear as fp8_mod
from inference_from_scratch.fp8_linear import TRITON_MAX_M, BlockFP8Linear
from inference_from_scratch.fp8_triton import choose_config, triton_available


def _mod(use_triton=True):
    torch.manual_seed(0)
    m = BlockFP8Linear.from_linear(
        torch.nn.Linear(128, 128, bias=False), use_triton=use_triton
    )
    return m


def test_choose_config_heuristic():
    c = choose_config(1, 17408, 5120)
    assert c["BLOCK_K"] == 128 and c["BLOCK_M"] == 16
    assert 128 % c["BLOCK_N"] == 0
    assert choose_config(128, 5120, 5120)["BLOCK_M"] == 64
    # CPU env: triton module imports but reports unavailable w/o CUDA/triton.
    assert isinstance(triton_available(), bool)


def test_env_flag(monkeypatch):
    monkeypatch.setenv("INFERENCE_FP8_TRITON", "1")
    assert BlockFP8Linear(128, 128).use_triton is True
    monkeypatch.setenv("INFERENCE_FP8_TRITON", "0")
    assert BlockFP8Linear(128, 128).use_triton is False
    # Explicit flag wins over env.
    assert BlockFP8Linear(128, 128, use_triton=True).use_triton is True


def test_cpu_input_uses_reference_even_with_triton_flag():
    m = _mod(use_triton=True)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(m(x), m._try_triton(x) or m(x))


def test_fallback_on_launch_failure(monkeypatch, caplog):
    """GPU-shaped dispatch exercised via monkeypatched seams."""
    m = _mod(use_triton=True)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    import unittest.mock as mock

    fake_x = mock.MagicMock()
    fake_x.is_cuda = True
    fake_x.dim.return_value = 2
    fake_x.shape = (2, 128)
    fake_x.numel.return_value = 256
    fake_x.reshape.return_value = x
    monkeypatch.setattr(fp8_mod, "_triton_available", lambda: True)
    monkeypatch.setattr(
        fp8_mod, "_run_triton",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("launch boom")),
    )
    with caplog.at_level(logging.WARNING):
        assert m._try_triton(fake_x) is None
    assert "falling back" in caplog.text
    assert m._triton_fallback_reason is not None
    assert "launch boom" in m._triton_fallback_reason


def test_triton_used_when_available_and_in_range(monkeypatch):
    m = _mod(use_triton=True)
    sentinel = torch.randn(2, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(fp8_mod, "_triton_available", lambda: True)
    monkeypatch.setattr(fp8_mod, "_run_triton", lambda *a, **k: sentinel)
    import unittest.mock as mock

    fake_x = mock.MagicMock()
    fake_x.is_cuda = True
    fake_x.dim.return_value = 2
    fake_x.shape = (2, 128)
    fake_x.numel.return_value = 256
    fake_x.reshape.return_value = torch.randn(2, 128)
    out = m._try_triton(fake_x)
    assert torch.equal(out, sentinel)


def test_fp8_dispatch_thresholds():
    from inference_from_scratch.fp8_linear import fp8_dispatch
    assert fp8_dispatch(1) == "triton" and fp8_dispatch(128) == "triton"
    assert fp8_dispatch(512) == "triton" and fp8_dispatch(1024) == "triton"
    assert fp8_dispatch(1025) == "reference" and fp8_dispatch(0) == "reference"


def test_triton_skipped_above_threshold(monkeypatch):
    m = _mod(use_triton=True)
    monkeypatch.setattr(fp8_mod, "_triton_available", lambda: True)
    called = []
    monkeypatch.setattr(
        fp8_mod, "_run_triton", lambda *a, **k: called.append(1)
    )
    import unittest.mock as mock

    fake_x = mock.MagicMock()
    fake_x.is_cuda = True
    fake_x.dim.return_value = 2
    fake_x.shape = (TRITON_MAX_M + 1, 128)
    fake_x.numel.return_value = (TRITON_MAX_M + 1) * 128
    assert m._try_triton(fake_x) is None
    assert not called
