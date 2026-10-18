"""Graph-form pieces of MTP speculation: static draft, chunk buffers,
profile phases and the mtp1-graph runner flag (CPU; CUDA capture itself
is proven by scripts/tp2_spec_graph_check.py under NCCL).
"""
import importlib.util
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

_SCRIPT = ROOT / "scripts" / "run_qwen38_tp2.py"
_spec = importlib.util.spec_from_file_location("run_qwen38_tp2_g", _SCRIPT)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def _cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
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
    return tp, mtp


def test_chunk_static_out_buffers_match_fresh_returns():
    tp, _mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7]])
    with torch.no_grad():
        s1 = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s1)
        logits, hidden = tp.decode_chunk_static(
            torch.tensor([[4, 9]]), s1, return_hidden=True)
        s2 = tp.static_state(batch_size=1, max_seq_len=32)
        tp.prefill_static(idx, s2)
        out = torch.empty(1, 2, 97)
        hid = torch.empty(1, 2, 32)
        got_logits, got_hidden = tp.decode_chunk_static(
            torch.tensor([[4, 9]]), s2, return_hidden=True,
            out=out, hidden_out=hid)
    assert got_logits is out and got_hidden is hid
    assert torch.equal(out, logits) and torch.equal(hid, hidden)
    with pytest.raises(ValueError):
        tp.decode_chunk_static(torch.tensor([[4, 9]]), s2,
                               out=torch.empty(1, 2, 5))


def test_draft_token_static_matches_functional_draft():
    tp, mtp = _tp_and_mtp()
    torch.manual_seed(3)
    hiddens = [torch.randn(32) for _ in range(3)]
    tokens = [5, 9, 2]
    positions = [10, 12, 14]  # absolute RoPE positions, slots stay sequential
    with torch.no_grad():
        state = None
        want = []
        for h, tok, pos in zip(hiddens, tokens, positions):
            draft_id, logits, state = mtp.draft_token(
                h, tok, state, position=pos,
                embedding=tp.embed_tokens, lm_head=tp.lm_head)
            want.append((int(draft_id), logits.clone()))
        static_state = mtp.initial_static_state(batch_size=1, max_seq_len=16)
        for (h, tok, pos), (want_id, want_logits) in zip(
                zip(hiddens, tokens, positions), want):
            draft_id, logits = mtp.draft_token_static(
                h, tok, static_state, rope_position=pos,
                embedding=tp.embed_tokens, lm_head=tp.lm_head)
            assert int(draft_id) == want_id
            assert torch.allclose(logits, want_logits, atol=1e-5, rtol=1e-5)
            static_state.position.add_(1)  # commit, as the loop does on accept
    assert int(static_state.position.item()) == 3


def test_profile_phases_filled_on_eager_run():
    tp, mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    profile: dict = {}
    with torch.no_grad():
        _out, stats = speculative_generate_tp(
            tp, mtp, idx, 8, profile=profile)
    assert stats["cycles"] > 0
    for key in ("draft_ms", "verify_ms", "snapshot_ms", "rollback_copy_ms",
                "refeed_ms", "host_overhead_ms", "wall_ms"):
        assert key in profile and profile[key] >= 0.0
    assert profile["wall_ms"] > 0.0


def test_graph_flags_require_cuda_and_static_draft():
    tp, mtp = _tp_and_mtp()
    idx = torch.tensor([[3, 11, 5, 7, 9]])
    with pytest.raises(ValueError, match="CUDA"):
        speculative_generate_tp(tp, mtp, idx, 4, graph_verify=True)


def test_runner_mtp1_graph_validation():
    assert _runner.validate_speculative_mode("mtp1-graph", "eager") == "mtp1-graph"
    assert "mtp1-graph" in _runner.SPECULATIVE_MODES
    with pytest.raises(ValueError):
        _runner.validate_speculative_mode("mtp1-graph", "graph")
    args = _runner.build_parser().parse_args(["--speculative", "mtp1-graph"])
    assert args.speculative == "mtp1-graph"
