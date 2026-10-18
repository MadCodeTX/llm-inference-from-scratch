"""Tests for layers.py: RMSNorm, LayerNorm, RoPE cache/apply."""

import torch
import torch.nn.functional as F

from inference_from_scratch.layers import (
    LayerNorm,
    RMSNorm,
    apply_rope,
    precompute_rope_cache,
)


def test_rmsnorm_output_rms_is_one_per_row():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    x = torch.randn(4, 7, 16) * 5.0  # arbitrary scale: norm must remove it
    y = norm(x)
    assert y.shape == x.shape
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_matches_manual_formula():
    torch.manual_seed(1)
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(8))  # non-trivial gain
    x = torch.randn(3, 8)
    y = norm(x)
    manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(y, manual, atol=1e-6)


def test_rmsnorm_matches_torch_reference_when_available():
    if not hasattr(F, "rms_norm"):
        return  # older torch: manual-formula test above already covers it
    torch.manual_seed(2)
    norm = RMSNorm(16, eps=1e-5)
    x = torch.randn(2, 5, 16)
    ref = F.rms_norm(x, (16,), weight=norm.weight, eps=1e-5)
    assert torch.allclose(norm(x), ref, atol=1e-6)


def test_layernorm_matches_nn_layernorm_no_affine():
    torch.manual_seed(3)
    d = 16
    ours = LayerNorm(d, eps=1e-5)
    ref = torch.nn.LayerNorm(d, eps=1e-5, elementwise_affine=False)
    with torch.no_grad():  # ours defaults to weight=1, bias=0 == no affine
        pass
    x = torch.randn(4, 6, d)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_layernorm_matches_nn_layernorm_with_affine():
    torch.manual_seed(4)
    d = 12
    ours = LayerNorm(d, eps=1e-5)
    ref = torch.nn.LayerNorm(d, eps=1e-5, elementwise_affine=True)
    with torch.no_grad():
        w, b = torch.randn(d), torch.randn(d)
        ours.weight.copy_(w)
        ours.bias.copy_(b)
        ref.weight.copy_(w)
        ref.bias.copy_(b)
    x = torch.randn(3, d)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)
    # LayerNorm rows (before affine) are zero-mean; RMSNorm's are not.
    assert torch.allclose(LayerNorm(d)(x).mean(-1), torch.zeros(3), atol=1e-5)


def test_rope_cache_shapes_and_pos0():
    cos, sin = precompute_rope_cache(32, 16, theta=10000.0)
    assert cos.shape == (32, 16) and sin.shape == (32, 16)
    assert torch.allclose(cos[0], torch.ones(16))
    assert torch.allclose(sin[0], torch.zeros(16))


def test_rope_position_zero_is_identity():
    cos, sin = precompute_rope_cache(16, 8)
    x = torch.randn(2, 3, 1, 8)
    y = apply_rope(x, cos, sin, start_pos=0)
    assert torch.allclose(y, x, atol=1e-6)


def test_rope_preserves_norm():
    torch.manual_seed(5)
    cos, sin = precompute_rope_cache(64, 16)
    x = torch.randn(2, 4, 10, 16)
    y = apply_rope(x, cos, sin, start_pos=3)
    assert torch.allclose(
        y.norm(dim=-1), x.norm(dim=-1), atol=1e-5
    )


def test_rope_relative_position_property():
    # dot(R_m q, R_n k) must depend only on m - n: equal differences,
    # equal scores, for every absolute offset -- RoPE's core trick.
    torch.manual_seed(6)
    D = 16
    cos, sin = precompute_rope_cache(64, D)
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def score(m, n):
        qm = apply_rope(q, cos, sin, start_pos=m)
        kn = apply_rope(k, cos, sin, start_pos=n)
        return (qm * kn).sum().item()

    base = score(5, 2)  # diff 3
    for m, n in [(3, 0), (10, 7), (20, 17)]:
        assert abs(score(m, n) - base) < 1e-5
    # A different difference must (generically) give a different score.
    assert abs(score(5, 1) - base) > 1e-4
