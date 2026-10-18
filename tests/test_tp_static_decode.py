"""Static TP decode: TPStaticState + DistributedHybridLM.decode_token_static.

Single-rank tests use the world=1 local context (identity collectives)
and pin static token-by-token logits against the existing distributed
``step`` path and the dense model; the subprocess test runs the real
2-process gloo proof in ``scripts/tp2_static_check.py``.
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


def test_static_state_shapes_and_validation():
    cfg = _tiny_cfg()
    state = TPStaticState(cfg, batch_size=2, max_seq_len=16,
                          device="cpu", dtype=torch.float32,
                          ctx=_LocalContext(1, 2, torch.device("cpu")))
    # Rank-local slices at world=2: 1 KV head, 2 value heads, half conv.
    assert tuple(state.kv_k[3].shape) == (2, 1, 16, 16)
    assert tuple(state.kv_v[3].shape) == (2, 1, 16, 16)
    assert tuple(state.linear_S[0].shape) == (2, 2, 8, 8)
    assert tuple(state.linear_conv_buffers[0].shape) == (2, 32, 3)
    assert state.kv_k[0] is None and state.linear_S[3] is None
    assert state.position.shape == (2,) and state.position.dtype == torch.long
    assert state.seq_len == 0
    with pytest.raises(ValueError):
        TPStaticState(cfg, batch_size=0, max_seq_len=4)
    with pytest.raises(ValueError):
        TPStaticState(cfg, batch_size=1, max_seq_len=0)


def test_static_decode_matches_legacy_step_and_dense():
    tp, dense = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 5))
    with torch.no_grad():
        d_logits, d_state = dense.prefill(idx)
        t_logits, t_state = tp.prefill(idx)
        state = tp.static_state(batch_size=2, max_seq_len=16)
        s_logits = tp.prefill_static(idx, state)
        assert torch.allclose(s_logits, t_logits, atol=1e-5, rtol=1e-5)
        assert torch.allclose(s_logits, d_logits, atol=1e-5, rtol=1e-5)
        assert state.seq_len == 5
        out = torch.empty(2, 97)
        nxt = d_logits[:, -1].argmax(dim=-1)
        for _ in range(6):
            d_logits, d_state = dense.step(nxt, d_state)
            t_logits, t_state = tp.step(nxt, t_state)
            got = tp.decode_token_static(nxt, state, out=out)
            assert got is out
            assert torch.allclose(out, t_logits[:, 0], atol=1e-5, rtol=1e-5)
            assert torch.allclose(out, d_logits[:, 0], atol=1e-5, rtol=1e-5)
            assert torch.equal(out.argmax(dim=-1), d_logits[:, 0].argmax(dim=-1))
            nxt = d_logits[:, 0].argmax(dim=-1)
        # [B, 1] token spelling is accepted too.
        got = tp.decode_token_static(nxt.unsqueeze(1), state)
        assert got.shape == (2, 97)


def test_static_ptrs_stable_reset_rerun_and_overflow_no_mutation():
    tp, dense = _tp_and_dense()
    torch.manual_seed(77)
    idx = torch.randint(0, 97, (2, 5))
    with torch.no_grad():
        d_logits, d_state = dense.prefill(idx)
        state = tp.static_state(batch_size=2, max_seq_len=16)
        ptrs0 = state.data_ptrs()
        first = [tp.prefill_static(idx, state)]
        nxt = d_logits[:, -1].argmax(dim=-1)
        for _ in range(4):
            d_logits, d_state = dense.step(nxt, d_state)
            first.append(tp.decode_token_static(nxt, state).clone())
            nxt = d_logits[:, 0].argmax(dim=-1)
        assert state.data_ptrs() == ptrs0
        state.reset()
        assert state.data_ptrs() == ptrs0 and state.seq_len == 0
        d_logits, d_state = dense.prefill(idx)
        second = [tp.prefill_static(idx, state)]
        nxt = d_logits[:, -1].argmax(dim=-1)
        for _ in range(4):
            d_logits, d_state = dense.step(nxt, d_state)
            second.append(tp.decode_token_static(nxt, state).clone())
            nxt = d_logits[:, 0].argmax(dim=-1)
        for a, b in zip(first, second):
            assert torch.equal(a, b)

        small = tp.static_state(batch_size=2, max_seq_len=3)
        tp.prefill_static(idx[:, :3], small)
        kv_before = small.kv_k[3].clone()
        s_before = small.linear_S[0].clone()
        with pytest.raises(ValueError, match="overflow"):
            tp.decode_token_static(nxt, small)
        assert small.positions == [3, 3]
        assert torch.equal(small.kv_k[3], kv_before)
        assert torch.equal(small.linear_S[0], s_before)
        with pytest.raises(ValueError, match="overflow"):
            tp.prefill_static(idx, tp.static_state(batch_size=2, max_seq_len=4))


@pytest.mark.skipif(not (ROOT / "scripts/tp2_static_check.py").exists(),
                    reason="check script missing")
def test_tp2_static_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_static_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_STATIC_CHECK PASS" in proc.stdout
    assert "prefill_static_vs_dense:" in proc.stdout
    assert "ptrs_stable_across_steps=True" in proc.stdout
    assert "overflow_no_mutation=True" in proc.stdout
