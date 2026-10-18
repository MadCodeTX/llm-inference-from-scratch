"""Tests for mtp.py: native MTP head foundation (Kiely sec. 5.2).

The tensor spec is pinned against the real checkpoint's 22 ``mtp.*``
tensors (headers only, skipped when the snapshot is absent) and
against formula-derived shapes at tiny dims. No HF MTP class exists
in transformers 5.20.1 to parity against -- that absence is itself
asserted, with its search evidence, instead of faked.
"""

from pathlib import Path

import pytest
import torch

from inference_from_scratch.hybrid import HybridConfig
from inference_from_scratch.mtp import INFERRED_JOINS, MTPConfig, MTPHead, mtp_tensor_spec
from inference_from_scratch.speculation import acceptance_stats, accepted_prefix_len

D, VOCAB = 32, 97


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0, mtp_num_hidden_layers=1,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _tiny_head(seed=0):
    torch.manual_seed(seed)
    return MTPHead(MTPConfig.from_hybrid_config(_tiny_config())).eval()


def _snapshot_dir():
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots"
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "mtp.safetensors").is_file():
                return snap
    return None


def test_mtp_config_from_hybrid_and_text_config():
    cfg = MTPConfig.from_hybrid_config(_tiny_config())
    assert cfg.hidden_size == D and cfg.mtp_num_hidden_layers == 1
    assert cfg.num_attention_heads == 4 and cfg.num_key_value_heads == 2
    assert cfg.head_dim == 16 and cfg.intermediate_size == 64
    assert cfg.mtp_use_dedicated_embeddings is False
    assert cfg.to_hybrid_config().layer_types == ["full_attention"]
    text = {
        "hidden_size": D, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64,
        "vocab_size": VOCAB, "max_position_embeddings": 128,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "linear_num_key_heads": 2, "linear_key_head_dim": 8,
        "linear_num_value_heads": 4, "linear_value_head_dim": 8,
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
    }
    from_text = MTPConfig.from_text_config(text)
    assert from_text.hidden_size == D and from_text.head_dim == 16
    assert from_text.mtp_use_dedicated_embeddings is False
    assert MTPHead.layer_type == "full_attention"
    assert len(INFERRED_JOINS) >= 1


def test_tiny_module_keys_match_checkpoint_spec_minus_scales():
    head = _tiny_head()
    spec = mtp_tensor_spec(head.config)
    assert len(spec) == 22
    module_keys = {f"mtp.{k}" for k in head.state_dict()}
    spec_module_keys = {k for k, v in spec.items() if v["kind"] != "scale_inv"}
    assert module_keys == spec_module_keys
    sd = head.state_dict()
    assert sd["fc.weight"].shape == (D, 2 * D)
    assert sd["layers.0.self_attn.q_proj.weight"].shape == (2 * 4 * 16, D)
    assert sd["layers.0.self_attn.k_proj.weight"].shape == (2 * 16, D)
    assert sd["layers.0.self_attn.o_proj.weight"].shape == (D, 4 * 16)
    assert sd["layers.0.mlp.gate_proj.weight"].shape == (64, D)
    assert sd["pre_fc_norm_hidden.weight"].shape == (D,)
    assert sd["norm.weight"].shape == (D,)
    assert not any("linear_attn" in k for k in sd)


@pytest.mark.skipif(_snapshot_dir() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_real_mtp_tensor_spec_matches_shard_headers():
    from inference_from_scratch.weights import read_safetensors_header
    snap = _snapshot_dir()
    header = read_safetensors_header(snap / "mtp.safetensors")
    actual = {k: v for k, v in header.items() if k != "__metadata__"}
    assert len(actual) == 22
    spec = mtp_tensor_spec(MTPConfig())
    assert set(spec) == set(actual)
    for name, entry in actual.items():
        assert spec[name]["shape"] == list(entry["shape"]), name
        assert spec[name]["dtype"] == entry["dtype"], name


def test_mtp_forward_shapes_ids_and_embedding_paths_agree():
    head = _tiny_head()
    emb = torch.nn.Embedding(VOCAB, D)
    lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    hidden = torch.randn(2, D)
    ids = torch.tensor([3, 4])
    with torch.no_grad():
        dh, logits, state = head(hidden, ids, embedding=emb, lm_head=lm_head, start_pos=5)
        dh2, logits2, _ = head(hidden, emb(ids), lm_head=lm_head, start_pos=5)
        dh3, logits3, _ = head(torch.randn(2, 3, D), torch.randn(2, 3, D), lm_head=lm_head)
    assert dh.shape == (2, D) and logits.shape == (2, VOCAB)
    assert torch.equal(logits, logits2) and torch.equal(dh, dh2)
    assert dh3.shape == (2, 3, D) and logits3.shape == (2, 3, VOCAB)
    assert state[0][0].shape == (2, 2, 1, 16)
    with torch.no_grad():
        dh_only, no_logits, _ = head(hidden, ids, embedding=emb)
    assert no_logits is None and dh_only.shape == (2, D)
    with pytest.raises(ValueError):
        head(hidden, ids)


def test_mtp_draft_step_deterministic_and_replayable():
    head = _tiny_head(seed=7)
    emb = torch.nn.Embedding(VOCAB, D)
    lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    hidden = torch.randn(2, D)
    ids = torch.tensor([10, 20])
    with torch.no_grad():
        tok_a, dh_a, _ = head.draft_token(hidden, ids, embedding=emb, lm_head=lm_head, start_pos=3)
        tok_b, dh_b, _ = head.draft_token(hidden, ids, embedding=emb, lm_head=lm_head, start_pos=3)
        s0 = head.initial_state()
        _, logits_a, _ = head(hidden, ids, embedding=emb, lm_head=lm_head, start_pos=3, state=s0)
        _, logits_b, _ = head(hidden, ids, embedding=emb, lm_head=lm_head, start_pos=3, state=s0)
    assert torch.equal(tok_a, tok_b) and torch.equal(dh_a, dh_b)
    assert torch.equal(logits_a, logits_b) and s0 == [None]
    assert torch.equal(tok_a, torch.argmax(logits_a, dim=-1))


def test_acceptance_stats_math_and_edges():
    s = acceptance_stats([[1, 2, 3], [4, 5]], [[1, 2], [4]])
    assert s == {"accept_rate": 1.0, "mean_accepted_len": 1.5, "expected_tokens_per_target_forward": 2.5}
    s = acceptance_stats([[9], [8]], [[1], [2]])
    assert s["accept_rate"] == 0.0 and s["mean_accepted_len"] == 0.0
    assert s["expected_tokens_per_target_forward"] == 1.0
    assert accepted_prefix_len([1, 2, 3], [1, 9, 3]) == 1
    s = acceptance_stats([[1, 2, 9], [5, 5]], [[1, 2, 3], [5, 6]])
    assert s["accept_rate"] == pytest.approx(3 / 5)
    assert s["mean_accepted_len"] == pytest.approx(1.5)
    assert acceptance_stats([], []) == {"accept_rate": 0.0, "mean_accepted_len": 0.0, "expected_tokens_per_target_forward": 0.0}
    assert acceptance_stats([[]], [[]])["expected_tokens_per_target_forward"] == 1.0
    assert acceptance_stats([1, 2], [1, 2])["accept_rate"] == 1.0
    assert acceptance_stats(torch.tensor([[1, 2]]), torch.tensor([[1, 9]]))["mean_accepted_len"] == 1.0
    with pytest.raises(ValueError):
        acceptance_stats([[1], [2]], [[1]])


def test_hf_mtp_parity_absent_in_transformers_5_20_1():
    """No runnable HF MTP class exists; assert the documented absence.

    Search evidence (transformers 5.20.1): ``mtp`` occurs in the
    qwen3_5 modeling/config/modular files only as
    ``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]`` -- HF
    ignores the checkpoint's mtp tensors on load. If a future
    transformers ships a real MTP class, this test fails loudly so a
    genuine parity test replaces it, rather than passing vacuously.
    """
    import importlib
    modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    candidates = [n for n in dir(modeling) if "mtp" in n.lower()]
    assert candidates == [], (
        f"transformers now exposes {candidates}; write a real HF-parity "
        "test for the MTP head instead of asserting absence")
    assert any("mtp" in pat for pat in modeling.Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected)
