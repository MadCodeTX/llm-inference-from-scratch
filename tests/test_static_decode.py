"""Tests for static (CUDA-graph-friendly) decode: StaticHybridState,
GatedDeltaNet.step_inplace, TinyHybridLM.decode_token_static.

Static token-by-token logits must equal the legacy immutable step path
on tiny dims, storage pointers must not move across steps or reset,
reset+rerun must be identical, and overflow must raise a clear error.
CPU only.
"""

import pytest
import torch

from inference_from_scratch.hybrid import (
    HybridConfig,
    StaticHybridState,
    TinyHybridLM,
)
from inference_from_scratch.kv_cache import StaticKVCache
from inference_from_scratch.linear_attention import GatedDeltaNet

D, VOCAB = 32, 97


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _tiny_lm(seed=3):
    torch.manual_seed(seed)
    m = TinyHybridLM(_tiny_config())
    m.eval()
    return m


# -- GatedDeltaNet.step_inplace ---------------------------------------------


def test_step_inplace_equals_legacy_step_and_ptrs_stable():
    torch.manual_seed(0)
    g = GatedDeltaNet(d_model=32, num_key_heads=2, key_dim=8,
                      num_value_heads=4, value_dim=8, conv_kernel=4).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        state = g.initial_state(2)
        legacy_outs = []
        for t in range(6):
            o, state = g.step(x[:, t], state)
            legacy_outs.append(o)
        S, conv = g.initial_state(2)
        ptr_s, ptr_c = S.data_ptr(), conv.data_ptr()
        inplace_outs = [g.step_inplace(x[:, t], S, conv) for t in range(6)]
    assert torch.equal(torch.stack(legacy_outs), torch.stack(inplace_outs))
    assert torch.equal(state[0], S) and torch.equal(state[1], conv)
    assert S.data_ptr() == ptr_s and conv.data_ptr() == ptr_c
    # Rank preservation: [B, 1, D] in -> [B, 1, D] out.
    S2, conv2 = g.initial_state(2)
    with torch.no_grad():
        o3 = g.step_inplace(x[:, :1, :], S2, conv2)
    assert o3.shape == (2, 1, 32)
    with pytest.raises(ValueError):
        g.step_inplace(x[:, 0], torch.zeros(2, 4, 8, 7), conv)


def test_step_inplace_legacy_step_still_does_not_mutate():
    torch.manual_seed(1)
    g = GatedDeltaNet(d_model=32, num_key_heads=2, key_dim=8,
                      num_value_heads=4, value_dim=8, conv_kernel=4).eval()
    x = torch.randn(1, 32)
    with torch.no_grad():
        S0, conv0 = g.initial_state(1)
        snap = g.snapshot_state((S0, conv0))
        g.step(x, (S0, conv0))
    assert torch.equal(S0, snap[0]) and torch.equal(conv0, snap[1])


# -- StaticHybridState --------------------------------------------------------


def test_static_state_shapes_and_position_tensor():
    cfg = _tiny_config()
    st = StaticHybridState(cfg, batch_size=2, max_seq_len=16)
    assert isinstance(st.kv_cache, StaticKVCache)
    # Full cache: [n_full=1, B, H_kv, T_max, head_dim]
    assert st.kv_cache.k_cache.shape == (1, 2, 2, 16, 16)
    assert st.linear_S[0].shape == (2, 4, 8, 8)  # [B, Hv, Dk, Dv]
    assert st.linear_conv_buffers[0].shape == (2, 64, 3)  # [B, C, K-1]
    assert st.linear_S[3] is None and st.linear_conv_buffers[3] is None
    assert st.position.dim() == 0 and st.position.dtype == torch.long
    assert st.seq_len == 0 and len(st) == 4
    S, conv = st.linear_state(0)
    assert S is st.linear_S[0] and conv is st.linear_conv_buffers[0]
    with pytest.raises(ValueError):
        st.linear_state(3)


def test_static_state_defaults_follow_model_dtype():
    # The factory must not silently hand back an fp32/CPU state for a
    # model placed elsewhere: dtype (CPU-checkable) and, by the same
    # code path, device follow the model's parameters unless overridden.
    m = _tiny_lm(seed=8).to(dtype=torch.float64)
    st = m.static_state(batch_size=2, max_seq_len=8)
    assert st.kv_cache.k_cache.dtype == torch.float64
    assert st.linear_S[0].dtype == torch.float64
    assert st.device == next(m.parameters()).device
    st32 = m.static_state(batch_size=2, max_seq_len=8, dtype=torch.float32)
    assert st32.kv_cache.k_cache.dtype == torch.float32  # explicit wins


def test_static_snapshot_is_explicit_clone_and_copy_from_restores():
    m = _tiny_lm(seed=4)
    st = m.static_state(batch_size=2, max_seq_len=16)
    idx = torch.randint(0, VOCAB, (2, 3))
    with torch.no_grad():
        for t in range(3):
            m.decode_token_static(idx[:, t], st)
        snap = st.snapshot()
        assert snap.seq_len == 3
        # Snapshot tensors are clones (different storage), values equal.
        assert snap.linear_S[0].data_ptr() != st.linear_S[0].data_ptr()
        assert torch.equal(snap.linear_S[0], st.linear_S[0])
        logits_snap = m.decode_token_static(idx[:, 0], snap)
        # Continuing the snapshot does not disturb the original.
        assert st.seq_len == 3
        # copy_from restores a diverged state in place, ptrs stable.
        ptrs = st.data_ptrs()
        for t in range(2):
            m.decode_token_static(idx[:, t], st)
        assert st.seq_len == 5
        st.copy_from(snap)
        assert st.seq_len == snap.seq_len == 4 and st.data_ptrs() == ptrs
        logits_again = m.decode_token_static(idx[:, 1], st)
        logits_snap2 = m.decode_token_static(idx[:, 1], snap)
    assert torch.equal(logits_snap2, logits_again)
    assert logits_snap.shape == (2, VOCAB)


# -- TinyHybridLM.decode_token_static -----------------------------------------


def test_static_logits_equal_legacy_step_path():
    m = _tiny_lm()
    idx = torch.randint(0, VOCAB, (2, 7))
    with torch.no_grad():
        legacy_state = m.initial_state(2)
        legacy = []
        for t in range(7):
            logits, legacy_state = m.step(idx[:, t], legacy_state)
            legacy.append(logits[:, 0, :])
        legacy = torch.stack(legacy, dim=1)

        st = m.static_state(batch_size=2, max_seq_len=16)
        static = torch.stack(
            [m.decode_token_static(idx[:, t], st) for t in range(7)], dim=1)
    assert static.shape == (2, 7, VOCAB)
    assert st.seq_len == 7
    assert torch.allclose(static, legacy, atol=1e-6), (static - legacy).abs().max()
    # Token-by-token static also matches the single prefill forward.
    with torch.no_grad():
        full, _ = m(idx)
    assert torch.allclose(static, full, atol=1e-6), (static - full).abs().max()


def test_static_ptrs_unchanged_across_steps_and_reset_rerun_identical():
    m = _tiny_lm(seed=5)
    idx = torch.randint(0, VOCAB, (2, 6))
    st = m.static_state(batch_size=2, max_seq_len=16)
    ptrs = st.data_ptrs()
    with torch.no_grad():
        first = [m.decode_token_static(idx[:, t], st) for t in range(6)]
        assert st.data_ptrs() == ptrs and st.seq_len == 6
        st.reset()
        assert st.data_ptrs() == ptrs and st.seq_len == 0
        assert all(s.abs().max().item() == 0.0
                   for s in st.linear_S if s is not None)
        assert st.kv_cache.k_cache.abs().max().item() == 0.0
        assert st.kv_cache.seq_len == 0
        second = [m.decode_token_static(idx[:, t], st) for t in range(6)]
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_static_overflow_raises_clear_error_before_mutation():
    m = _tiny_lm(seed=6)
    st = m.static_state(batch_size=1, max_seq_len=4)
    idx = torch.randint(0, VOCAB, (1, 5))
    with torch.no_grad():
        for t in range(4):
            m.decode_token_static(idx[:, t], st)
        assert st.seq_len == 4
        with pytest.raises(ValueError, match="overflow.*max_seq_len=4"):
            m.decode_token_static(idx[:, 4], st)
    # Failed step mutated nothing: still exactly full, reset still works.
    assert st.seq_len == 4
    st.reset()
    assert st.seq_len == 0
    with torch.no_grad():
        m.decode_token_static(idx[:, 0], st)
    assert st.seq_len == 1


def test_static_explicit_position_forms():
    m = _tiny_lm(seed=7)
    idx = torch.randint(0, VOCAB, (2, 3))
    st = m.static_state(batch_size=2, max_seq_len=8)
    with torch.no_grad():
        # Python-int position and 0-dim tensor position both accepted.
        m.decode_token_static(idx[:, 0], st, position=0)
        m.decode_token_static(idx[:, 1], st, position=st.position)
        with pytest.raises(ValueError, match="does not match"):
            m.decode_token_static(idx[:, 2], st, position=99)
        with pytest.raises(ValueError):
            m.decode_token_static(idx[:, 2], st, position=torch.arange(2))
