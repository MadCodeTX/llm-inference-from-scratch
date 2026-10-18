"""Fused (opt-in) QwenRMSNorm path vs the manual default maths."""
import torch

from inference_from_scratch.hybrid import QwenRMSNorm


def _norm(d=64, seed=0):
    torch.manual_seed(seed)
    n = QwenRMSNorm(d)
    with torch.no_grad():
        n.weight.copy_(torch.randn(d) * 0.2)
    return n


def test_fused_flag_defaults_off_and_matches_manual():
    n = _norm()
    assert n.use_fused_rmsnorm is False
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    ref = n(x)
    n.use_fused_rmsnorm = True
    out = n(x)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    # Same maths, fused reduction: agree within BF16 rounding.
    assert float((out.float() - ref.float()).abs().max()) < 0.05


def test_fused_zero_weight_is_identity_gain():
    n = QwenRMSNorm(32)  # weight zeros -> gain exactly 1 in both paths
    x = torch.randn(4, 32)
    ref = n(x)
    n.use_fused_rmsnorm = True
    assert torch.allclose(n(x), ref, atol=1e-6)


def test_fused_preserves_shape_for_head_norm_dims():
    n = _norm(d=256, seed=1)
    x = torch.randn(1, 2, 12, 256, dtype=torch.bfloat16)
    ref = n(x)
    n.use_fused_rmsnorm = True
    out = n(x)
    assert out.shape == x.shape
    assert float((out.float() - ref.float()).abs().max()) < 0.05
