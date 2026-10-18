"""Cached FP32 FP8 scales: built once, invalidated honestly, no state_dict keys."""
import torch

import inference_from_scratch.fp8_linear as fp8_mod
from inference_from_scratch.fp8_linear import BlockFP8Linear, apply_scale_cache


def _mod(**kw):
    torch.manual_seed(0)
    return BlockFP8Linear.from_linear(
        torch.nn.Linear(128, 128, bias=False), **kw)


def test_cache_built_once_and_state_dict_keys_unchanged():
    m = _mod(use_scale_cache=True)
    # Eager build in from_linear.
    assert m._scale_cache_builds == 1
    assert set(m.state_dict()) == {"weight", "weight_scale_inv"}
    c1 = m.cached_fp32_scale()
    assert c1 is not None and c1.dtype == torch.float32
    assert torch.equal(c1, m.weight_scale_inv.float())
    c2 = m.cached_fp32_scale()
    assert c2.data_ptr() == c1.data_ptr()
    assert m._scale_cache_builds == 1  # built once
    # Reference output identical with cache on/off (cache only feeds Triton).
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    ref = _mod(use_scale_cache=False)
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert torch.equal(m(x), ref(x))


def test_cache_invalidation_on_inplace_and_replacement():
    m = _mod(use_scale_cache=True)
    c1 = m.cached_fp32_scale()
    with torch.no_grad():
        m.weight_scale_inv.mul_(2.0)
    c2 = m.cached_fp32_scale()
    assert m._scale_cache_builds == 2
    assert torch.equal(c2, m.weight_scale_inv.float())
    assert not torch.equal(c2, c1)
    # Tensor replacement (loader-style setattr) also rebuilds.
    with torch.no_grad():
        m.weight_scale_inv = torch.ones_like(m.weight_scale_inv)
    c3 = m.cached_fp32_scale()
    assert m._scale_cache_builds == 3
    assert torch.equal(c3, torch.ones_like(c3))
    # load_state_dict forces rebuild on next use.
    m2 = _mod(use_scale_cache=True)
    m2.load_state_dict(m.state_dict())
    before = m2._scale_cache_builds
    m2.cached_fp32_scale()
    assert m2._scale_cache_builds == before + 1


def test_cache_disabled_by_default_and_toggle():
    m = _mod()
    assert m.use_scale_cache is False
    assert m.cached_fp32_scale() is None
    assert "_scale_f32_cache" not in m.state_dict()
    m.set_use_scale_cache(True)
    assert m.cached_fp32_scale() is not None
    m.set_use_scale_cache(False)
    assert m.cached_fp32_scale() is None


def test_from_pre_sliced_eager_and_triton_receives_cached_fp32(monkeypatch):
    from inference_from_scratch.quantization import quantize_block_fp8
    torch.manual_seed(1)
    dense = torch.nn.Linear(128, 128, bias=False)
    q, scale = quantize_block_fp8(dense.weight.data.float())
    m = BlockFP8Linear.from_pre_sliced(
        q, scale.to(torch.bfloat16), use_triton=True, use_scale_cache=True)
    assert m._scale_cache_builds == 1
    seen = {}

    def fake_run(x, w, s, bias, block):
        seen["scale"] = s
        return torch.zeros(x.shape[0], w.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(fp8_mod, "_triton_available", lambda: True)
    monkeypatch.setattr(fp8_mod, "_run_triton", fake_run)
    import unittest.mock as mock
    fake_x = mock.MagicMock()
    fake_x.is_cuda = True
    fake_x.dim.return_value = 2
    fake_x.shape = (2, 128)
    fake_x.numel.return_value = 256
    fake_x.reshape.return_value = torch.randn(2, 128, dtype=torch.bfloat16)
    assert m._try_triton(fake_x) is not None
    assert seen["scale"].dtype == torch.float32
    assert seen["scale"].data_ptr() == m.cached_fp32_scale().data_ptr()
    assert m._scale_cache_builds == 1


def test_apply_scale_cache_walks_modules():
    net = torch.nn.Sequential(_mod(), _mod())
    assert apply_scale_cache(net, True) == 2
    assert all(m.cached_fp32_scale() is not None for m in net)
    assert apply_scale_cache(net, False) == 2
