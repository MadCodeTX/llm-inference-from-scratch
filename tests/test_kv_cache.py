"""Tests for kv_cache.py: KVCache (concat) and StaticKVCache."""

import pytest
import torch

from inference_from_scratch.kv_cache import KVCache, StaticKVCache


def _kv(T, base=0.0):
    # Distinct values per position so concatenation order is checkable.
    k = torch.full((1, 2, T, 4), base) + torch.arange(T).view(1, 1, T, 1).float()
    v = -k
    return k, v


def test_concat_cache_order_and_length():
    cache = KVCache(n_layer=2)
    assert cache.seq_len == 0
    k1, v1 = _kv(3, base=0.0)
    ka, va = cache.update(0, k1, v1)
    assert ka.shape == (1, 2, 3, 4) and cache.seq_len == 3
    k2, v2 = _kv(1, base=10.0)
    ka, va = cache.update(0, k2, v2)
    assert ka.shape == (1, 2, 4, 4) and cache.seq_len == 4
    # Old entries first, new token appended last (position == index).
    assert torch.equal(ka[:, :, :3], k1) and torch.equal(ka[:, :, 3:], k2)
    assert torch.equal(va[:, :, :3], v1) and torch.equal(va[:, :, 3:], v2)


def test_concat_cache_layers_independent_and_clear():
    cache = KVCache(n_layer=2)
    k, v = _kv(2)
    cache.update(1, k, v)  # only layer 1 written
    assert cache.keys[0] is None and cache.seq_len == 2
    cache.clear()
    assert cache.seq_len == 0 and all(t is None for t in cache.keys)


def test_static_cache_returns_only_filled_slice():
    cache = StaticKVCache(
        n_layer=2, batch_size=1, n_kv_head=2, max_seq_len=8, head_dim=4
    )
    assert cache.seq_len == 0
    k1, v1 = _kv(3)
    ka, va = cache.update(0, k1, v1, start_pos=0)
    assert ka.shape == (1, 2, 3, 4)  # not the full T_max=8 buffer
    assert torch.equal(ka, k1) and torch.equal(va, v1)
    k2, v2 = _kv(1, base=10.0)
    ka, _ = cache.update(0, k2, v2, start_pos=3)
    assert ka.shape == (1, 2, 4, 4) and cache.seq_len == 4
    assert torch.equal(ka[:, :, :3], k1) and torch.equal(ka[:, :, 3:], k2)


def test_static_cache_overflow_raises_and_clear():
    cache = StaticKVCache(
        n_layer=1, batch_size=1, n_kv_head=2, max_seq_len=4, head_dim=4
    )
    k, v = _kv(3)
    cache.update(0, k, v, start_pos=0)
    with pytest.raises(ValueError):
        cache.update(0, k, v, start_pos=2)  # 2+3 > 4
    cache.clear()
    assert cache.seq_len == 0
    ka, _ = cache.update(0, k, v, start_pos=0)  # reusable after clear
    assert ka.shape == (1, 2, 3, 4) and torch.equal(ka, k)
