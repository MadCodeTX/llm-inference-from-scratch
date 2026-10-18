"""CPU tests: graph2 packing/validation + preallocated snapshot semantics."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from inference_from_scratch.hybrid import HybridConfig  # noqa: E402
from inference_from_scratch.speculation import (  # noqa: E402
    decode_acceptance_triple, pack_acceptance_triple, speculative_generate_tp)
from inference_from_scratch.tp_distributed import TPStaticState  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_qwen38_tp2_g2", ROOT / "scripts" / "run_qwen38_tp2.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def test_pack_triple_accept_and_reject():
    assert pack_acceptance_triple(5, 5, 9) == (5, 5, 9)
    d = decode_acceptance_triple((5, 5, 9))
    assert d == {"draft_token": 5, "correction_token": 5,
                 "bonus_token": 9, "accepted": True}
    d2 = decode_acceptance_triple([7, 3, 3])
    assert d2["accepted"] is False and d2["correction_token"] == 3


def test_pack_triple_tensor_and_validation():
    d = decode_acceptance_triple(torch.tensor([4, 4, 8]))
    assert d["accepted"] is True
    with pytest.raises(ValueError):
        decode_acceptance_triple([1, 2])
    with pytest.raises(ValueError):
        pack_acceptance_triple(-1, 2, 3)
    with pytest.raises(ValueError):
        pack_acceptance_triple(True, 2, 3)
    with pytest.raises(ValueError):
        pack_acceptance_triple(1.5, 2, 3)


def _cfg():
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


def test_snapshot_into_preallocated_and_restore_linear_only():
    cfg = _cfg()
    s = TPStaticState(cfg, batch_size=1, max_seq_len=16,
                      device=torch.device("cpu"))
    snap = s.alloc_snapshot()
    ptrs = [t.data_ptr() for t in snap.linear_S if t is not None]
    # Mutate live, snapshot into preallocated buffers (same ptrs).
    for t in s.linear_S:
        if t is not None:
            t.fill_(2.0)
    for t in s.linear_conv_buffers:
        if t is not None:
            t.fill_(3.0)
    s.position.fill_(5)
    s.snapshot_into(snap)
    assert [t.data_ptr() for t in snap.linear_S if t is not None] == ptrs
    assert int(snap.position.item()) == 5
    # Mutate live again incl. KV; restore linear only: KV stays mutated.
    for t in s.linear_S:
        if t is not None:
            t.fill_(9.0)
    kv = next(t for t in s.kv_k if t is not None)
    kv.fill_(7.0)
    s.position.fill_(9)
    s.restore_linear_from(snap)
    assert all(float(t.flatten()[0]) == 2.0 for t in s.linear_S if t is not None)
    assert float(kv.flatten()[0]) == 7.0  # KV deliberately not restored
    assert int(s.position.item()) == 5


def test_snapshot_into_rejects_mismatched_buffers():
    cfg = _cfg()
    s = TPStaticState(cfg, batch_size=1, max_seq_len=16,
                      device=torch.device("cpu"))
    other = TPStaticState(cfg, batch_size=1, max_seq_len=8,
                          device=torch.device("cpu"))
    with pytest.raises(ValueError):
        s.snapshot_into(other)


def test_graph2_requires_cuda_on_cpu():
    # Validation happens after prefill in the real path, so only test the
    # runner-level mode validation here plus exclusivity via CPU stub is
    # covered by test_graph_flags in test_speculative_graph.py.
    assert _runner.validate_speculative_mode("mtp1-graph2", "eager") == "mtp1-graph2"
    assert "mtp1-graph2" in _runner.SPECULATIVE_MODES
    with pytest.raises(ValueError):
        _runner.validate_speculative_mode("mtp1-graph2", "graph")
    args = _runner.build_parser().parse_args(["--speculative", "mtp1-graph2"])
    assert args.speculative == "mtp1-graph2"
