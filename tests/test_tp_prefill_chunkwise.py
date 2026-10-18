"""Chunkwise TP prefill: DistributedHybridLM.prefill_chunkwise_static.

Single-rank tests use the world=1 local context (identity collectives)
and pin chunkwise prefill (chunk_size=4) against ``prefill_static``
(token loop), the distributed ``prefill`` and the dense model: last
logits fp32-close, state contents (KV filled region, S, conv) close,
greedy continuation identical, and non-empty-state continuation equal
to the one-shot prefill. The subprocess test runs the real 2-process
gloo proof in ``scripts/tp2_chunkwise_check.py``.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, _LocalContext)

ATOL = RTOL = 1e-5


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def _tp_and_dense():
    torch.manual_seed(7)
    dense = TinyHybridLM(_tiny_cfg())
    dense.eval()
    tp = DistributedHybridLM.from_dense(
        dense, _LocalContext(0, 1, torch.device("cpu")), config=_tiny_cfg())
    tp.eval()
    return tp, dense


def _assert_state_close(a, b, filled: int):
    assert a.seq_len == b.seq_len == filled
    for i in range(len(a.layer_types)):
        if a.kv_k[i] is not None:
            # Only the filled region is meaningful; tail stays zero in both.
            assert torch.allclose(a.kv_k[i][:, :, :filled], b.kv_k[i][:, :, :filled],
                                  atol=ATOL, rtol=RTOL)
            assert torch.allclose(a.kv_v[i][:, :, :filled], b.kv_v[i][:, :, :filled],
                                  atol=ATOL, rtol=RTOL)
            assert torch.equal(a.kv_k[i][:, :, filled:], b.kv_k[i][:, :, filled:])
        if a.linear_S[i] is not None:
            assert torch.allclose(a.linear_S[i], b.linear_S[i], atol=ATOL, rtol=RTOL)
            assert torch.allclose(a.linear_conv_buffers[i], b.linear_conv_buffers[i],
                                  atol=ATOL, rtol=RTOL)


def test_chunkwise_matches_static_dense_and_state():
    tp, dense = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        d_logits, _ = dense.prefill(idx)
        static_state = tp.static_state(batch_size=2, max_seq_len=32)
        s_logits = tp.prefill_static(idx, static_state)
        ptrs0 = static_state.data_ptrs()

        state = tp.static_state(batch_size=2, max_seq_len=32)
        c_ptrs0 = state.data_ptrs()
        c_logits, c_hidden = tp.prefill_chunkwise_static(
            idx, state, chunk_size=4, return_hidden=True)
        # Last-position logits only: [B, V], no [B, T, V] tensor.
        assert c_logits.shape == (2, 97)
        assert torch.allclose(c_logits, s_logits[:, -1], atol=ATOL, rtol=RTOL)
        assert torch.allclose(c_logits, d_logits[:, -1], atol=ATOL, rtol=RTOL)
        _, _, d_hidden = dense.prefill(idx, return_hidden=True)
        assert c_hidden.shape == (2, 11, 32)
        assert torch.allclose(c_hidden, d_hidden, atol=ATOL, rtol=RTOL)
        _assert_state_close(state, static_state, filled=11)
        assert state.data_ptrs() == c_ptrs0
        assert static_state.data_ptrs() == ptrs0


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 128])
def test_chunkwise_chunk_sizes(chunk_size):
    tp, _ = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        ref_state = tp.static_state(batch_size=2, max_seq_len=32)
        ref = tp.prefill_static(idx, ref_state)[:, -1]
        state = tp.static_state(batch_size=2, max_seq_len=32)
        got = tp.prefill_chunkwise_static(idx, state, chunk_size=chunk_size)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL)
        _assert_state_close(state, ref_state, filled=11)


def test_chunkwise_greedy_continuation_identical():
    tp, dense = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 5))
    with torch.no_grad():
        d_logits, d_state = dense.prefill(idx)
        s_state = tp.static_state(batch_size=2, max_seq_len=32)
        tp.prefill_static(idx, s_state)
        c_state = tp.static_state(batch_size=2, max_seq_len=32)
        c_last = tp.prefill_chunkwise_static(idx, c_state, chunk_size=4)
        assert torch.equal(c_last.argmax(dim=-1), d_logits[:, -1].argmax(dim=-1))
        nxt = d_logits[:, -1].argmax(dim=-1)
        for _ in range(6):
            d_logits, d_state = dense.step(nxt, d_state)
            s_out = tp.decode_token_static(nxt, s_state)
            c_out = tp.decode_token_static(nxt, c_state)
            assert torch.allclose(c_out, s_out, atol=ATOL, rtol=RTOL)
            assert torch.equal(c_out.argmax(dim=-1), d_logits[:, 0].argmax(dim=-1))
            nxt = d_logits[:, 0].argmax(dim=-1)


def test_chunkwise_nonempty_state_continuation_equals_oneshot():
    tp, _ = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        one_state = tp.static_state(batch_size=2, max_seq_len=32)
        one = tp.prefill_chunkwise_static(idx, one_state, chunk_size=4)

        split_state = tp.static_state(batch_size=2, max_seq_len=32)
        tp.prefill_static(idx[:, :5], split_state)
        assert split_state.seq_len == 5
        two = tp.prefill_chunkwise_static(idx[:, 5:], split_state, chunk_size=4)
        assert torch.allclose(two, one, atol=ATOL, rtol=RTOL)
        _assert_state_close(split_state, one_state, filled=11)

        # Chunkwise prefix + chunkwise continuation (uneven split) too.
        split2 = tp.static_state(batch_size=2, max_seq_len=32)
        tp.prefill_chunkwise_static(idx[:, :7], split2, chunk_size=4)
        three = tp.prefill_chunkwise_static(idx[:, 7:], split2, chunk_size=4)
        assert torch.allclose(three, one, atol=ATOL, rtol=RTOL)
        _assert_state_close(split2, one_state, filled=11)


def test_chunkwise_position_ids_and_errors_no_mutation():
    tp, _ = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        state = tp.static_state(batch_size=2, max_seq_len=16)
        ref_state = tp.static_state(batch_size=2, max_seq_len=16)
        ref = tp.prefill_chunkwise_static(idx, ref_state, chunk_size=4)
        pos = torch.arange(6).unsqueeze(0).expand(2, 6).contiguous()
        got = tp.prefill_chunkwise_static(idx, state, chunk_size=4, position_ids=pos)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL)

        small = tp.static_state(batch_size=2, max_seq_len=4)
        tp.prefill_chunkwise_static(idx[:, :3], small, chunk_size=4)
        kv_before, s_before = small.kv_k[3].clone(), small.linear_S[0].clone()
        with pytest.raises(ValueError, match="overflow"):
            tp.prefill_chunkwise_static(idx, small, chunk_size=4)
        assert small.seq_len == 3
        assert torch.equal(small.kv_k[3], kv_before)
        assert torch.equal(small.linear_S[0], s_before)
        for bad in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match="chunk_size"):
                tp.prefill_chunkwise_static(idx, tp.static_state(batch_size=2, max_seq_len=16),
                                            chunk_size=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="position_ids"):
            tp.prefill_chunkwise_static(idx, tp.static_state(batch_size=2, max_seq_len=16),
                                        chunk_size=4, position_ids=torch.arange(5))
        empty = tp.prefill_chunkwise_static(
            idx[:, :0], tp.static_state(batch_size=2, max_seq_len=16), chunk_size=4)
        assert empty.shape == (2, 0, 97)


@pytest.mark.skipif(not (ROOT / "scripts/tp2_chunkwise_check.py").exists(),
                    reason="check script missing")
def test_tp2_chunkwise_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_chunkwise_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_CHUNKWISE_CHECK PASS" in proc.stdout
    assert "chunkwise_vs_static:" in proc.stdout
    assert "nonempty_continuation:" in proc.stdout
