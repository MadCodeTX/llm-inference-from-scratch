"""Tests for mlp.py: SwiGLU and GeluMLP shapes and manual formulas."""

import torch
import torch.nn.functional as F

from inference_from_scratch.mlp import GeluMLP, SwiGLU


def test_swiglu_shape():
    torch.manual_seed(0)
    mlp = SwiGLU(d_model=16, d_ff=32).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y = mlp(x)
    assert y.shape == (2, 5, 16)
    # Gate/up widen to d_ff, down returns to d_model.
    assert mlp.gate_proj.weight.shape == (32, 16)
    assert mlp.up_proj.weight.shape == (32, 16)
    assert mlp.down_proj.weight.shape == (16, 32)


def test_gelu_mlp_shape():
    torch.manual_seed(1)
    mlp = GeluMLP(d_model=16, d_ff=32).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y = mlp(x)
    assert y.shape == (2, 5, 16)
    assert mlp.up_proj.weight.shape == (32, 16)
    assert mlp.down_proj.weight.shape == (16, 32)


def test_swiglu_matches_manual_silu_gate_times_up():
    torch.manual_seed(2)
    mlp = SwiGLU(d_model=16, d_ff=24).eval()
    x = torch.randn(3, 4, 16)
    with torch.no_grad():
        y = mlp(x)
        manual = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(y, manual, atol=1e-6)


def test_gelu_mlp_matches_manual_gelu():
    torch.manual_seed(3)
    mlp = GeluMLP(d_model=16, d_ff=24).eval()
    x = torch.randn(3, 4, 16)
    with torch.no_grad():
        y = mlp(x)
        manual = mlp.down_proj(F.gelu(mlp.up_proj(x)))
    assert torch.allclose(y, manual, atol=1e-6)


def test_mlp_is_token_independent():
    # No cross-token mixing: changing one row leaves the others exact.
    torch.manual_seed(4)
    for mlp in (SwiGLU(16, 32).eval(), GeluMLP(16, 32).eval()):
        x = torch.randn(2, 5, 16)
        x_changed = x.clone()
        x_changed[:, 3] = torch.randn(2, 16) * 5
        with torch.no_grad():
            y, y_changed = mlp(x), mlp(x_changed)
        keep = [0, 1, 2, 4]
        assert torch.allclose(y[:, keep], y_changed[:, keep], atol=1e-6)
        assert not torch.allclose(y[:, 3], y_changed[:, 3], atol=1e-5)
