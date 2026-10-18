"""CPU dispatch/validation tests + skip-gated GPU correctness for the
fused gated-delta recurrence (src/inference_from_scratch/gated_delta_fused.py).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from inference_from_scratch.gated_delta_fused import (  # noqa: E402
    DELTA_MODES, fused_available, gated_delta_inplace,
    reference_gated_delta_inplace, validate_delta_mode,
    validate_recurrence_shapes)

_spec = importlib.util.spec_from_file_location(
    "run_qwen38_tp2_delta", ROOT / "scripts" / "run_qwen38_tp2.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def _inputs(bh=4, dk=8, dv=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(bh, dk, generator=gen)
    k = torch.randn(bh, dk, generator=gen)
    v = torch.randn(bh, dv, generator=gen)
    beta = torch.rand(bh, generator=gen)
    g = -torch.rand(bh, generator=gen)
    S = torch.randn(bh, dk, dv, generator=gen)
    return q, k, v, beta, g, S


def test_validate_delta_mode():
    assert DELTA_MODES == ("ref", "fused")
    assert validate_delta_mode("fused") == "fused"
    with pytest.raises(ValueError):
        validate_delta_mode("bogus")
    with pytest.raises(ValueError):
        gated_delta_inplace(*_inputs(), mode="bogus")


def test_validate_shapes_rejects_mismatch():
    q, k, v, beta, g, S = _inputs()
    assert validate_recurrence_shapes(q, k, v, beta, g, S) == (4, 8, 8)
    with pytest.raises(ValueError):
        validate_recurrence_shapes(q, k, v, beta, g, S[:, :, :4])
    with pytest.raises(ValueError):
        validate_recurrence_shapes(q, k, v, beta[:-1], g, S)
    with pytest.raises(ValueError):
        validate_recurrence_shapes(q, k, v, beta, g, S[0])


def test_cpu_fused_mode_falls_back_to_reference_exactly():
    # On CPU (or without CUDA) fused mode must fall back, bit-exact.
    q, k, v, beta, g, S0 = _inputs()
    S_ref, S_fused = S0.clone(), S0.clone()
    core_ref = reference_gated_delta_inplace(q, k, v, beta, g, S_ref)
    core_fused = gated_delta_inplace(q, k, v, beta, g, S_fused, mode="fused")
    assert torch.equal(core_ref, core_fused)
    assert torch.equal(S_ref, S_fused)


def test_reference_matches_manual_recurrence():
    q, k, v, beta, g, S0 = _inputs(bh=3, dk=8, dv=16)
    S = S0.clone()
    core = reference_gated_delta_inplace(q, k, v, beta, g, S)
    Sm = S0 * torch.exp(g)[:, None, None]
    kv = (Sm * k.unsqueeze(-1)).sum(dim=-2)
    Sm = Sm + k.unsqueeze(-1) * ((v - kv) * beta.unsqueeze(-1)).unsqueeze(-2)
    assert torch.allclose(S, Sm, atol=1e-6)
    assert torch.allclose(core, (Sm * q.unsqueeze(-1)).sum(dim=-2), atol=1e-6)


def test_runner_delta_flag_dispatch():
    assert _runner.DELTA_MODES == DELTA_MODES
    from inference_from_scratch.tp_distributed import DistributedGatedDeltaNet

    class _Ctx:
        rank = 0
        world_size = 1
        device = torch.device("cpu")

    from inference_from_scratch.linear_attention import GatedDeltaNet
    dense = GatedDeltaNet(16, 2, 8, 4, 8)
    layer = DistributedGatedDeltaNet(_Ctx(), dense)

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

    wrap = _Wrap(layer)
    assert _runner.apply_delta_mode(wrap, "ref") == 1
    assert layer.use_fused_delta is False
    assert _runner.apply_delta_mode(wrap, "fused") == 1
    assert layer.use_fused_delta is True
    with pytest.raises(ValueError):
        _runner.apply_delta_mode(wrap, "bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU correctness needs CUDA")
@pytest.mark.parametrize("bh,dk,dv", [(4, 8, 8), (24, 128, 128), (1, 128, 128)])
def test_gpu_fused_matches_reference(bh, dk, dv):
    if not fused_available():
        pytest.skip("Triton unavailable")
    q, k, v, beta, g, S0 = _inputs(bh=bh, dk=dk, dv=dv, seed=7)
    dev = "cuda"
    args = [t.to(dev) for t in (q, k, v, beta, g)]
    S_ref, S_fused = S0.to(dev).clone(), S0.to(dev).clone()
    core_ref = reference_gated_delta_inplace(*args, S_ref)
    core_fused = gated_delta_inplace(*args, S_fused, mode="fused")
    torch.cuda.synchronize()
    assert torch.allclose(S_fused, S_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(core_fused, core_ref, atol=1e-4, rtol=1e-4)
    # Second step on the mutated state (decode is sequential).
    core_ref2 = reference_gated_delta_inplace(*args, S_ref)
    core_fused2 = gated_delta_inplace(*args, S_fused, mode="fused")
    torch.cuda.synchronize()
    assert torch.allclose(S_fused, S_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(core_fused2, core_ref2, atol=1e-4, rtol=1e-4)
