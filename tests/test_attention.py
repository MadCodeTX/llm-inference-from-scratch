"""Tests for attention.py: manual SDPA, repeat_kv, CausalSelfAttention.

The cache-equivalence test is the heart of the repo: a token-by-token
cached decode must reproduce the full (no-cache) forward *exactly*
(eval mode, no dropout anywhere), or every later chapter's cache and
generation logic is built on sand.
"""

import torch
import torch.nn.functional as F

from inference_from_scratch.attention import (
    CausalSelfAttention,
    manual_scaled_dot_product_attention,
    repeat_kv,
)
from inference_from_scratch.config import ModelConfig
from inference_from_scratch.kv_cache import KVCache, StaticKVCache
from inference_from_scratch.layers import precompute_rope_cache


def _config(n_head=4, n_kv_head=2, d_model=32, max_seq_len=64):
    return ModelConfig(
        vocab_size=64,
        d_model=d_model,
        n_layer=1,
        n_head=n_head,
        n_kv_head=n_kv_head,
        d_ff=64,
        max_seq_len=max_seq_len,
    )


def _rope(config, T=None):
    return precompute_rope_cache(
        config.max_seq_len, config.head_dim, theta=config.rope_theta
    )


# --- manual_scaled_dot_product_attention ---------------------------------


def test_manual_sdpa_matches_f_sdpa_without_mask():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 8)
    ours = manual_scaled_dot_product_attention(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert ours.shape == (2, 4, 5, 8)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_manual_sdpa_matches_f_sdpa_with_causal_mask():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    ours = manual_scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_manual_sdpa_first_token_attends_only_to_itself():
    # With a causal mask, row 0 has exactly one visible key, so its
    # output must be v[:, :, 0] regardless of the later values.
    torch.manual_seed(2)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = manual_scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out[:, :, 0], v[:, :, 0], atol=1e-6)


# --- repeat_kv -------------------------------------------------------------


def test_repeat_kv_correctness_grouped_order():
    # Distinct constant per KV head makes the head order observable.
    x = torch.zeros(1, 2, 3, 4)
    x[:, 0] = 1.0
    x[:, 1] = 2.0
    y = repeat_kv(x, 3)
    assert y.shape == (1, 6, 3, 4)
    # Grouped: [kv0, kv0, kv0, kv1, kv1, kv1], copies identical to source.
    for head in range(3):
        assert torch.equal(y[:, head], x[:, 0])
    for head in range(3, 6):
        assert torch.equal(y[:, head], x[:, 1])


def test_repeat_kv_one_is_identity_and_general_shape():
    torch.manual_seed(3)
    x = torch.randn(2, 2, 5, 8)
    assert repeat_kv(x, 1) is x
    y = repeat_kv(x, 2)
    assert y.shape == (2, 4, 5, 8)
    assert torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, 1], x[:, 0])
    assert torch.equal(y[:, 2], x[:, 1]) and torch.equal(y[:, 3], x[:, 1])


# --- CausalSelfAttention ----------------------------------------------------


def test_causal_attention_output_shape_mha_and_gqa():
    torch.manual_seed(4)
    for n_kv_head in (4, 2, 1):  # MHA, GQA, MQA
        config = _config(n_head=4, n_kv_head=n_kv_head)
        attn = CausalSelfAttention(config).eval()
        cos, sin = _rope(config)
        x = torch.randn(2, 5, config.d_model)
        with torch.no_grad():
            y = attn(x, cos, sin)
        assert y.shape == (2, 5, config.d_model)


def test_gqa_projection_shapes():
    config = _config(n_head=4, n_kv_head=2, d_model=32)  # head_dim=8
    attn = CausalSelfAttention(config)
    # q spans all heads; k/v only the KV heads; o merges back to d_model.
    assert attn.q_proj.weight.shape == (32, 32)
    assert attn.k_proj.weight.shape == (16, 32)
    assert attn.v_proj.weight.shape == (16, 32)
    assert attn.o_proj.weight.shape == (32, 32)
    assert attn.n_rep == 2


def test_causality_future_token_does_not_change_past_outputs():
    torch.manual_seed(5)
    config = _config()
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    x = torch.randn(2, 6, config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, config.d_model) * 10  # rewrite future
    with torch.no_grad():
        y = attn(x, cos, sin)
        y_changed = attn(x_changed, cos, sin)
    # Every position before the changed one is bit-for-bit unaffected...
    assert torch.allclose(y[:, :-1], y_changed[:, :-1], atol=1e-6)
    # ...and the changed position itself does move (test is not vacuous).
    assert not torch.allclose(y[:, -1], y_changed[:, -1], atol=1e-5)


def test_cached_decode_step_equals_full_forward_last_position():
    torch.manual_seed(6)
    config = _config(n_head=4, n_kv_head=2)
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    B, T = 2, 7
    x = torch.randn(B, T, config.d_model)

    with torch.no_grad():
        y_full = attn(x, cos, sin)  # [B, T, D] no cache

        cache = KVCache(n_layer=1)
        # Prefill all but the last token, then one cached decode step.
        attn(x[:, :-1], cos, sin, kv_cache=cache, layer_idx=0, start_pos=0)
        assert cache.seq_len == T - 1
        y_step = attn(
            x[:, -1:], cos, sin, kv_cache=cache, layer_idx=0, start_pos=T - 1
        )
    assert cache.seq_len == T
    assert y_step.shape == (B, 1, config.d_model)
    assert torch.allclose(y_step[:, 0], y_full[:, -1], atol=1e-6)


def test_cached_token_by_token_decode_equals_full_forward():
    torch.manual_seed(7)
    config = _config(n_head=4, n_kv_head=1)  # MQA: hardest repeat case
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    B, T = 2, 8
    x = torch.randn(B, T, config.d_model)

    with torch.no_grad():
        y_full = attn(x, cos, sin)
        cache = KVCache(n_layer=1)
        steps = [
            attn(
                x[:, t : t + 1], cos, sin,
                kv_cache=cache, layer_idx=0, start_pos=t,
            )
            for t in range(T)
        ]
        y_cached = torch.cat(steps, dim=1)  # [B, T, D]
    assert torch.allclose(y_cached, y_full, atol=1e-6)


def test_static_cache_decode_equals_full_forward():
    torch.manual_seed(8)
    config = _config(n_head=4, n_kv_head=2)
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    B, T = 1, 6
    x = torch.randn(B, T, config.d_model)

    with torch.no_grad():
        y_full = attn(x, cos, sin)
        cache = StaticKVCache(
            n_layer=1, batch_size=B, n_kv_head=config.n_kv_head,
            max_seq_len=config.max_seq_len, head_dim=config.head_dim,
        )
        attn(x[:, :-1], cos, sin, kv_cache=cache, layer_idx=0, start_pos=0)
        y_step = attn(
            x[:, -1:], cos, sin, kv_cache=cache, layer_idx=0, start_pos=T - 1
        )
    assert torch.allclose(y_step[:, 0], y_full[:, -1], atol=1e-6)


def test_biased_attention_shapes_forward_and_effect():
    torch.manual_seed(10)
    config = _config(n_head=4, n_kv_head=2)
    object.__setattr__(config, "attention_bias", True)
    # Rebuild with the field set properly (dataclass is not frozen).
    from inference_from_scratch.config import ModelConfig as _MC
    config = _MC(
        vocab_size=64, d_model=32, n_layer=1, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=64, attention_bias=True,
    )
    attn = CausalSelfAttention(config).eval()
    assert attn.q_proj.bias is not None and attn.q_proj.bias.shape == (32,)
    assert attn.k_proj.bias is not None and attn.k_proj.bias.shape == (16,)
    assert attn.v_proj.bias is not None and attn.v_proj.bias.shape == (16,)
    assert attn.o_proj.bias is None
    cos, sin = _rope(config)
    x = torch.randn(2, 5, config.d_model)
    with torch.no_grad():
        y = attn(x, cos, sin)
    assert y.shape == (2, 5, config.d_model)
    # Zero biases must reproduce the bias-free result; nonzero must move it.
    with torch.no_grad():
        for p in (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias):
            p.zero_()
        y_zero = attn(x, cos, sin)
        for p in (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias):
            p.fill_(0.5)
        y_biased = attn(x, cos, sin)
    assert not torch.allclose(y_zero, y_biased, atol=1e-6)


def test_biased_cached_decode_equals_full_forward():
    torch.manual_seed(11)
    from inference_from_scratch.config import ModelConfig as _MC
    config = _MC(
        vocab_size=64, d_model=32, n_layer=1, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=64, attention_bias=True,
    )
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    B, T = 2, 7
    x = torch.randn(B, T, config.d_model)
    with torch.no_grad():
        y_full = attn(x, cos, sin)
        cache = KVCache(n_layer=1)
        steps = [
            attn(
                x[:, t : t + 1], cos, sin,
                kv_cache=cache, layer_idx=0, start_pos=t,
            )
            for t in range(T)
        ]
        y_cached = torch.cat(steps, dim=1)
    assert torch.allclose(y_cached, y_full, atol=1e-6)


def test_chunked_prefill_with_cache_equals_full_forward():
    # T>1 at start_pos>0: second chunk must see chunk 1 + itself causally.
    torch.manual_seed(9)
    config = _config()
    attn = CausalSelfAttention(config).eval()
    cos, sin = _rope(config)
    x = torch.randn(1, 6, config.d_model)
    with torch.no_grad():
        y_full = attn(x, cos, sin)
        cache = KVCache(n_layer=1)
        y1 = attn(x[:, :4], cos, sin, kv_cache=cache, layer_idx=0, start_pos=0)
        y2 = attn(x[:, 4:], cos, sin, kv_cache=cache, layer_idx=0, start_pos=4)
        y_cached = torch.cat([y1, y2], dim=1)
    assert torch.allclose(y_cached, y_full, atol=1e-6)
