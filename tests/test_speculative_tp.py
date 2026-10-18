"""Greedy MTP speculative decoding on the TP static path (gamma=1).

Pins the Kiely sec. 5.2 invariant: speculative ids EXACTLY equal
``DistributedHybridLM`` greedy ids, for a real (random) TPMTP draft,
a rigged always-wrong draft (acceptance 0, every cycle rolls back)
and a rigged target-shifted draft (every draft accepted). The reject
path additionally pins state equality with plain generation.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.mtp import (  # noqa: E402
    MTPConfig, MTPHead, TPMTPDraftModule)
from inference_from_scratch.speculation import speculative_generate_tp  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, _LocalContext)

VOCAB = 97


def _cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
        mtp_num_hidden_layers=1)


def _tp_and_mtp():
    torch.manual_seed(7)
    dense = TinyHybridLM(_cfg()).eval()
    ctx = _LocalContext(0, 1, torch.device("cpu"))
    tp = DistributedHybridLM.from_dense(dense, ctx, config=_cfg()).eval()
    torch.manual_seed(11)
    head = MTPHead(MTPConfig.from_hybrid_config(_cfg())).eval()
    mtp = TPMTPDraftModule.from_dense(
        ctx, head, embedding=tp.embed_tokens, lm_head=tp.lm_head).eval()
    return tp, dense, mtp


def _plain_greedy(tp, idx, max_new):
    with torch.no_grad():
        logits, state = tp.prefill(idx)
        gen = [int(logits[0, -1].argmax())]
        for _ in range(max_new - 1):
            logits, state = tp.step(torch.tensor([gen[-1]]), state)
            gen.append(int(logits[0, 0].argmax()))
    return gen


class _AlwaysWrongDraft:
    """Rigged draft: proposes (target's next token + 1) % vocab.

    Built from the precomputed reference sequence; with every cycle
    rejecting, cycle k's target token is ``ref[k + 1]`` by induction.
    """

    wiring: dict = {}

    def __init__(self, ref):
        self.ref = list(ref)
        self.calls = 0

    def initial_state(self):
        return None

    def draft_token(self, hidden_t, token_id_t1, state=None, position=0, **kw):
        wrong = (self.ref[self.calls + 1] + 1) % VOCAB
        self.calls += 1
        return torch.tensor(wrong), None, state


class _TargetShiftedDraft:
    """Rigged draft: proposes exactly the target's next token.

    Tracks its place in the reference sequence; every proposal is
    accepted, so each call advances the last-accepted pointer by 2
    (draft + bonus), except possibly the final truncated cycle.
    """

    wiring: dict = {}

    def __init__(self, ref):
        self.ref = list(ref)
        self.ptr = 0  # index of last_accepted in ref

    def initial_state(self):
        return None

    def draft_token(self, hidden_t, token_id_t1, state=None, position=0, **kw):
        draft = self.ref[self.ptr + 1]
        self.ptr += 2
        return torch.tensor(draft), None, state


def test_chunk_static_matches_two_single_steps():
    tp, _dense, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7]])
    with torch.no_grad():
        s1 = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s1)
        a = tp.decode_token_static(torch.tensor([4]), s1)
        b = tp.decode_token_static(torch.tensor([9]), s1)
        s2 = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s2)
        logits, hidden = tp.decode_chunk_static(
            torch.tensor([[4, 9]]), s2, return_hidden=True)
        assert int(s2.position.item()) == int(s1.position.item()) == 6
        # First chunk position does not see the second token (causal).
        assert torch.allclose(logits[0, 0], a, atol=1e-5, rtol=1e-5)
        assert torch.allclose(logits[0, 1], b, atol=1e-5, rtol=1e-5)
        assert hidden.shape == (1, 2, 32)
        with pytest.raises(ValueError, match="overflow"):
            small = tp.static_state(batch_size=1, max_seq_len=5)
            tp.prefill_static(idx, small)
            tp.decode_chunk_static(torch.tensor([[4, 9]]), small)
        assert int(small.position.item()) == 4


def test_static_state_snapshot_copy_from_restores():
    tp, _dense, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5]])
    with torch.no_grad():
        state = tp.static_state(batch_size=1, max_seq_len=16)
        tp.prefill_static(idx, state)
        snap = state.snapshot()
        s_before = state.linear_S[0].clone()
        first = tp.decode_token_static(torch.tensor([4]), state)
        assert not torch.equal(state.linear_S[0], s_before)
        state.copy_from(snap)
        assert torch.equal(state.linear_S[0], s_before)
        assert int(state.position.item()) == 3
        second = tp.decode_token_static(torch.tensor([4]), state)
        assert torch.equal(first, second)


def test_speculative_tp_exact_real_mtp_draft():
    tp, _dense, mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    ref = _plain_greedy(tp, idx, 12)
    with torch.no_grad():
        out, stats = speculative_generate_tp(tp, mtp, idx, 12)
    assert out[0, 5:].tolist() == ref
    assert stats["proposed"] == stats["mtp_forwards"] == stats["cycles"]
    assert stats["accepted"] + stats["rollbacks"] == stats["cycles"]


def test_speculative_tp_always_wrong_draft_acceptance_zero():
    tp, _dense, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    ref = _plain_greedy(tp, idx, 10)
    stub = _AlwaysWrongDraft(ref)
    with torch.no_grad():
        out, stats = speculative_generate_tp(tp, stub, idx, 10)
    assert out[0, 5:].tolist() == ref
    assert stats["accepted"] == 0 and stats["proposed"] > 0
    assert stats["rollbacks"] == stats["cycles"] == stats["proposed"]


def test_speculative_tp_target_shifted_draft_acceptance_high():
    tp, _dense, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    ref = _plain_greedy(tp, idx, 12)
    stub = _TargetShiftedDraft(ref)
    with torch.no_grad():
        out, stats = speculative_generate_tp(tp, stub, idx, 12)
    assert out[0, 5:].tolist() == ref
    assert stats["accepted"] == stats["proposed"] and stats["rollbacks"] == 0
    # Each accepted cycle finalises 2 tokens (draft + bonus).
    assert stats["target_forwards"] == stats["cycles"] < 12


def test_speculative_tp_reject_state_equals_plain_generation():
    tp, _dense, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    ref = _plain_greedy(tp, idx, 8)
    captured = {}

    def factory(batch_size, max_seq_len):
        captured["state"] = tp.static_state(
            batch_size=batch_size, max_seq_len=max_seq_len)
        return captured["state"]

    stub = _AlwaysWrongDraft(ref)
    with torch.no_grad():
        out, _stats = speculative_generate_tp(
            tp, stub, idx, 8, state_factory=factory)
        assert out[0, 5:].tolist() == ref
        spec_state = captured["state"]
        plain = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, plain)
        for tok in ref[:-1]:
            tp.decode_token_static(torch.tensor([tok]), plain)
        assert int(spec_state.position.item()) == int(plain.position.item())
        spec_logits = tp.decode_token_static(torch.tensor([ref[-1]]), spec_state)
        plain_logits = tp.decode_token_static(torch.tensor([ref[-1]]), plain)
    assert torch.allclose(spec_logits, plain_logits, atol=1e-5, rtol=1e-5)


def test_speculative_tp2_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_speculative_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_SPECULATIVE_CHECK PASS" in proc.stdout
