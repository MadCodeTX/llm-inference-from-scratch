"""Truly batched TP static decode: per-row positions + selective reset.

Single-rank tests use the world=1 local context and pin the batched
path against independent B=1 runs; the subprocess test runs the real
2-process gloo proof in ``scripts/tp2_batched_check.py`` (B=3
different prompts, staggered reset of row 1 mid-decode).
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, TPStaticState, _LocalContext)

ATOL = RTOL = 1e-5
PROMPTS = torch.tensor([[3, 11, 42, 7, 90], [55, 1, 2, 3, 4], [94, 20, 20, 8, 13]])


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


def _tp():
    torch.manual_seed(7)
    dense = TinyHybridLM(_tiny_cfg())
    dense.eval()
    tp = DistributedHybridLM.from_dense(
        dense, _LocalContext(0, 1, torch.device("cpu")), config=_tiny_cfg())
    tp.eval()
    return tp


def test_position_is_per_row_and_seq_len_contract():
    cfg = _tiny_cfg()
    state = TPStaticState(cfg, batch_size=3, max_seq_len=16, device="cpu",
                          ctx=_LocalContext(0, 1, torch.device("cpu")))
    assert state.position.shape == (3,) and state.position.dtype == torch.int64
    assert state.positions == [0, 0, 0] and state.seq_len == 0
    state.position.copy_(torch.tensor([2, 5, 2]))
    assert state.positions == [2, 5, 2]
    with pytest.raises(ValueError, match="diverge"):
        _ = state.seq_len
    state.seq_len = 4
    assert state.positions == [4, 4, 4] and state.scalar_position == 4
    state.seq_len = [1, 2, 3]
    assert state.positions == [1, 2, 3]
    # Legacy scalar position input broadcasts in the attention helper.
    from inference_from_scratch.tp_distributed import _row_positions
    assert _row_positions(torch.tensor(7), 3).tolist() == [7, 7, 7]
    assert _row_positions(torch.tensor([7]), 3).tolist() == [7, 7, 7]


def test_selective_reset_zeroes_only_selected_rows():
    tp = _tp()
    with torch.no_grad():
        state = tp.static_state(batch_size=3, max_seq_len=32)
        tp.prefill_chunkwise_static(PROMPTS, state, chunk_size=2)
        ptrs0 = state.data_ptrs()
        kv_before = state.kv_k[3].clone()
        s_before = state.linear_S[0].clone()
        conv_before = state.linear_conv_buffers[0].clone()
        state.reset(rows=[1])
        assert state.data_ptrs() == ptrs0
        assert state.positions == [5, 0, 5]
        for buf, before in ((state.kv_k[3], kv_before), (state.linear_S[0], s_before),
                            (state.linear_conv_buffers[0], conv_before)):
            assert torch.equal(buf[[0, 2]], before[[0, 2]])
            assert int(buf[1].abs().max()) == 0
        # Tensor / mask / int spellings.
        state.reset(rows=torch.tensor([0]))
        assert state.positions == [0, 0, 5]
        state.reset(rows=2)
        assert state.positions == [0, 0, 0]
        with pytest.raises(ValueError):
            state.reset(rows=[3])
        with pytest.raises(ValueError):
            state.reset(rows=[True])
        assert state.positions == [0, 0, 0]


def test_generate_static_batched_matches_single_runs():
    tp = _tp()
    with torch.no_grad():
        state = tp.static_state(batch_size=3, max_seq_len=32)
        batched = tp.generate_static_batched(PROMPTS, 6, state, chunk_size=2)
        assert batched.shape == (3, 6)
        assert state.positions == [5 + 5] * 3  # prompt + fed-back tokens
        for r in range(3):
            s1 = tp.static_state(batch_size=1, max_seq_len=32)
            single = tp.generate_static_batched(PROMPTS[r:r + 1], 6, s1, chunk_size=2)
            assert torch.equal(batched[r:r + 1], single), (batched[r], single[0])
        # B=1 backward compat: same helper on a [T] prompt.
        s1 = tp.static_state(batch_size=1, max_seq_len=32)
        one = tp.generate_static_batched(PROMPTS[0], 4, s1)
        assert one.shape == (1, 4) and s1.seq_len == 5 + 3
        with pytest.raises(ValueError):
            tp.generate_static_batched(PROMPTS, -1, tp.static_state(batch_size=3, max_seq_len=32))
        with pytest.raises(ValueError):
            tp.generate_static_batched(PROMPTS[:2], 2, tp.static_state(batch_size=3, max_seq_len=32))


def test_batched_prefill_logits_match_singles_at_divergent_starts():
    tp = _tp()
    with torch.no_grad():
        # Rows starting at different positions (row 1 selectively reset
        # after a longer prefill) must still match fresh B=1 prefills.
        state = tp.static_state(batch_size=3, max_seq_len=32)
        tp.prefill_chunkwise_static(PROMPTS, state, chunk_size=2)
        state.reset(rows=[1])
        got = tp.prefill_chunkwise_static(PROMPTS, state, chunk_size=2)
        # Row 1 restarted from 0 == fresh single prefill of its prompt.
        s1 = tp.static_state(batch_size=1, max_seq_len=32)
        ref1 = tp.prefill_chunkwise_static(PROMPTS[1:2], s1, chunk_size=2)
        assert torch.allclose(got[1:2], ref1, atol=ATOL, rtol=RTOL)
        assert state.positions == [10, 5, 10]


def test_staggered_reset_rows_continue_and_row1_restarts():
    tp = _tp()
    with torch.no_grad():
        total_ref = 3 + 5 + 3
        ref_state = tp.static_state(batch_size=3, max_seq_len=64)
        ref0 = tp.prefill_chunkwise_static(PROMPTS, ref_state, chunk_size=2)
        ref_logits, nxt = [], ref0.argmax(dim=-1)
        for _ in range(total_ref):
            lg = tp.decode_token_static(nxt, ref_state)
            ref_logits.append(lg)
            nxt = lg.argmax(dim=-1)

        st = tp.static_state(batch_size=3, max_seq_len=64)
        lg0 = tp.prefill_chunkwise_static(PROMPTS, st, chunk_size=2)
        assert torch.allclose(lg0, ref0, atol=ATOL, rtol=RTOL)
        nxt = lg0.argmax(dim=-1)
        for i in range(3):
            lg = tp.decode_token_static(nxt, st)
            assert torch.allclose(lg[[0, 2]], ref_logits[i][[0, 2]], atol=ATOL, rtol=RTOL)
            nxt = lg.argmax(dim=-1)
        st.reset(rows=[1])
        assert st.positions == [8, 0, 8]

        fresh = tp.static_state(batch_size=1, max_seq_len=64)
        fresh_prefill = tp.prefill_static(PROMPTS[1:2], fresh)
        step, last_row1 = 3, None
        for t in range(5):
            inp = torch.stack([nxt[0], PROMPTS[1, t], nxt[2]])
            lg = tp.decode_token_static(inp, st)
            assert torch.allclose(lg[[0, 2]], ref_logits[step][[0, 2]], atol=ATOL, rtol=RTOL)
            last_row1, nxt, step = lg[1:2], lg.argmax(dim=-1), step + 1
        assert torch.allclose(last_row1, fresh_prefill[:, -1], atol=ATOL, rtol=RTOL)
        fresh_nxt = fresh_prefill[:, -1].argmax(dim=-1)
        for i in range(3):
            lg = tp.decode_token_static(nxt, st)
            fresh_lg = tp.decode_token_static(fresh_nxt, fresh)
            assert torch.allclose(lg[[0, 2]], ref_logits[step][[0, 2]], atol=ATOL, rtol=RTOL)
            assert torch.allclose(lg[1:2], fresh_lg, atol=ATOL, rtol=RTOL)
            nxt, fresh_nxt, step = lg.argmax(dim=-1), fresh_lg.argmax(dim=-1), step + 1


def test_per_row_overflow_before_mutation():
    tp = _tp()
    with torch.no_grad():
        state = tp.static_state(batch_size=3, max_seq_len=8)
        tp.prefill_chunkwise_static(PROMPTS, state, chunk_size=2)  # positions [5]*3
        state.reset(rows=[1])
        # Divergence: row 1 at 0 (reset), rows 0/2 at 5.
        assert state.positions == [5, 0, 5]
        kv_before, s_before = state.kv_k[3].clone(), state.linear_S[0].clone()
        pos_before = state.positions
        with pytest.raises(ValueError, match="overflow"):
            tp.generate_static_batched(PROMPTS, 6, state)  # 5+5+5 > 8 for every row count check
        assert state.positions == pos_before
        assert torch.equal(state.kv_k[3], kv_before) and torch.equal(state.linear_S[0], s_before)
        # Only row 1 would overflow: fill its position to the limit.
        state.position[1] = 8
        kv_before, s_before = state.kv_k[3].clone(), state.linear_S[0].clone()
        with pytest.raises(ValueError, match="overflow.*rows \\[1\\]"):
            tp.decode_token_static(torch.tensor([1, 2, 3]), state)
        assert state.positions == [5, 8, 5]
        assert torch.equal(state.kv_k[3], kv_before) and torch.equal(state.linear_S[0], s_before)


@pytest.mark.skipif(not (ROOT / "scripts/tp2_batched_check.py").exists(),
                    reason="check script missing")
def test_tp2_batched_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_batched_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_BATCHED_CHECK PASS" in proc.stdout
    assert "row2_batched_eq_single=True" in proc.stdout
    assert "staggered_reset_proof=True" in proc.stdout
