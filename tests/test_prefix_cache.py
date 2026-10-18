"""Tests for Ch 14 prefix KV cache + prefill_with_prefix."""

import pytest
import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.kv_cache import KVCache
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.prefix_cache import PrefixKVCache, prefill_with_prefix


def _config(**overrides):
    kwargs = dict(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=64,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return DecoderLM(_config()).eval()


def _prefill(model, tokens):
    idx = torch.tensor([tokens], dtype=torch.long)
    cache = KVCache(n_layer=model.config.n_layer)
    with torch.no_grad():
        logits, cache = model(idx, kv_caches=cache, start_pos=0)
    return logits, cache


def test_lookup_longest_stored_prefix_and_stats():
    model = _tiny_model()
    cache = PrefixKVCache(max_entries=8)
    _, kv_short = _prefill(model, [1, 2, 3])
    _, kv_long = _prefill(model, [1, 2, 3, 4, 5])
    cache.put([1, 2, 3], kv_short)
    cache.put([1, 2, 3, 4, 5], kv_long)

    matched, cloned = cache.lookup([1, 2, 3, 4, 5, 6, 7])
    assert matched == 5
    assert cloned.seq_len == 5
    matched, _ = cache.lookup(torch.tensor([[1, 2, 3, 9]]))
    assert matched == 3
    matched, empty = cache.lookup([9, 9, 9])
    assert matched == 0 and empty.seq_len == 0

    assert cache.hits == 2 and cache.misses == 1
    assert cache.tokens_saved == 8
    assert cache.stats == {"hits": 2, "misses": 1, "tokens_saved": 8}
    assert cache.stats()["hits"] == 2


def test_lookup_returns_clone_isolated_from_store_and_put_source():
    model = _tiny_model()
    cache = PrefixKVCache(max_entries=4)
    _, kv = _prefill(model, [1, 2, 3])
    cache.put([1, 2, 3], kv)
    # Mutating the source cache after put must not corrupt the entry.
    kv.keys[0].fill_(999.0)  # type: ignore[union-attr]
    matched, cloned = cache.lookup([1, 2, 3, 4])
    assert matched == 3
    assert not torch.all(cloned.keys[0] == 999.0)
    # Mutating a returned clone must not corrupt the stored entry.
    cloned.keys[0].fill_(-999.0)  # type: ignore[union-attr]
    _, cloned2 = cache.lookup([1, 2, 3])
    assert not torch.all(cloned2.keys[0] == -999.0)


def test_store_only_at_put_boundaries():
    model = _tiny_model()
    cache = PrefixKVCache(max_entries=4)
    _, kv = _prefill(model, [1, 2, 3])
    cache.put([1, 2, 3], kv)
    # [1, 2] was never put, so a divergent continuation finds nothing.
    matched, _ = cache.lookup([1, 2, 9])
    assert matched == 0


def test_eviction_and_put_validation():
    model = _tiny_model()
    with pytest.raises(ValueError):
        PrefixKVCache(max_entries=0)
    cache = PrefixKVCache(max_entries=1)
    _, kv1 = _prefill(model, [1, 2])
    _, kv2 = _prefill(model, [3, 4])
    cache.put([1, 2], kv1)
    cache.put([3, 4], kv2)
    assert len(cache) == 1
    assert cache.lookup([1, 2, 5])[0] == 0
    assert cache.lookup([3, 4, 5])[0] == 2
    with pytest.raises(ValueError):
        cache.put([1, 2, 3, 4], kv1)  # cache shorter than key


def test_prefill_with_prefix_identical_logits_and_tokens():
    model = _tiny_model(seed=7)
    prefix = [5, 11, 7, 3, 20]
    prompts = [prefix + [9, 1, 2], prefix + [30, 31, 4, 4]]

    cache = PrefixKVCache(max_entries=8)
    _, prefix_kv = _prefill(model, prefix)
    cache.put(prefix, prefix_kv)

    for prompt in prompts:
        idx = torch.tensor([prompt], dtype=torch.long)
        logits_last, kv, n_reused = prefill_with_prefix(model, idx, cache)
        assert n_reused == len(prefix)
        assert kv.seq_len == len(prompt)
        full_logits, full_kv = _prefill(model, prompt)
        # Last-position logits match a full independent prefill.
        assert torch.allclose(logits_last, full_logits[:, -1], atol=1e-6)
        assert int(torch.argmax(logits_last, dim=-1)) == int(
            torch.argmax(full_logits[:, -1], dim=-1)
        )
        # The assembled KV contents match the full-prefill cache.
        for layer in range(model.config.n_layer):
            assert torch.allclose(kv.keys[layer], full_kv.keys[layer], atol=1e-6)
            assert torch.allclose(kv.values[layer], full_kv.values[layer], atol=1e-6)

    assert cache.hits == 2
    assert cache.tokens_saved == 2 * len(prefix)


def test_prefill_with_prefix_miss_and_full_hit():
    model = _tiny_model(seed=8)
    cache = PrefixKVCache(max_entries=4)
    idx = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    logits_last, kv, n_reused = prefill_with_prefix(model, idx, cache)
    assert n_reused == 0 and kv.seq_len == 4
    with torch.no_grad():
        full_logits, _ = model(idx)
    assert torch.allclose(logits_last, full_logits[:, -1], atol=1e-6)
    assert cache.misses == 1

    # Full-sequence hit: still returns correct last logits.
    cache.put([2, 4, 6, 8], kv)
    logits2, kv2, _ = prefill_with_prefix(model, idx, cache)
    assert kv2.seq_len == 4
    assert torch.allclose(logits2, full_logits[:, -1], atol=1e-6)
