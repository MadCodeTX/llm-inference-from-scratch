"""Tests for tensor_parallel.py: TP2 partitioning maths on CPU.

Tiny hybrid dims (same family as test_hybrid.py): D=32, 3 linear + 1
full layer, Q/KV 4/2 heads of 16, gated-delta K/V 2x8 / 4x8 (repeat
factor 2), intermediate 64. The real-config test reads only the local
checkpoint's config/headers via the report's manifest cross-check.
"""

import pytest
import torch
import torch.nn.functional as F

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM
from inference_from_scratch.tensor_parallel import (
    merge_state_dicts,
    parameter_bytes_by_rank,
    partitioned_linear,
    real_config_report,
    shard_spec_for,
    shard_state_dict,
)

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


def _tiny_state_dict(seed=0):
    torch.manual_seed(seed)
    return TinyHybridLM(_tiny_config()).state_dict()


# -- shard specs ------------------------------------------------------------


def test_spec_kinds_cover_every_tensor_family():
    cfg = _tiny_config()
    cases = {
        "layers.3.self_attn.q_proj.weight": ((128, D), "head_column"),
        "layers.3.self_attn.k_proj.weight": ((32, D), "head_column"),
        "layers.3.self_attn.v_proj.weight": ((32, D), "head_column"),
        "layers.3.self_attn.o_proj.weight": ((D, 64), "row"),
        "layers.3.self_attn.q_norm.weight": ((16,), "replicated"),
        "layers.0.linear_attn.in_proj_qkv.weight": ((64, D), "segmented_qkv"),
        "layers.0.linear_attn.conv1d.weight": ((64, 1, 4), "segmented_qkv"),
        "layers.0.linear_attn.in_proj_z.weight": ((32, D), "head_column"),
        "layers.0.linear_attn.out_proj.weight": ((D, 32), "row"),
        "layers.0.linear_attn.in_proj_a.weight": ((4, D), "head_column"),
        "layers.0.linear_attn.in_proj_b.weight": ((4, D), "head_column"),
        "layers.0.linear_attn.A_log": ((4,), "head_column"),
        "layers.0.linear_attn.dt_bias": ((4,), "head_column"),
        # [Dv] gain shared by all value heads: nothing per-head to split.
        "layers.0.linear_attn.norm.weight": ((8,), "replicated"),
        "layers.0.mlp.gate_proj.weight": ((64, D), "column"),
        "layers.0.mlp.up_proj.weight": ((64, D), "column"),
        "layers.0.mlp.down_proj.weight": ((D, 64), "row"),
        "layers.0.input_layernorm.weight": ((D,), "replicated"),
        "embed_tokens.weight": ((VOCAB, D), "replicated"),
        "lm_head.weight": ((VOCAB, D), "replicated"),
        "norm.weight": ((D,), "replicated"),
    }
    for name, (shape, kind) in cases.items():
        spec = shard_spec_for(name, shape, cfg)
        assert spec.kind == kind, (name, spec)
    # Embedding/head record the production vocab option without using it.
    for name in ("embed_tokens.weight", "lm_head.weight"):
        spec = shard_spec_for(name, (VOCAB, D), cfg)
        assert spec.kind == "replicated" and spec.future_kind == "vocab"


def test_spec_mtp_follows_layer_rules_and_fc_is_replicated():
    cfg = HybridConfig()  # real dims via from-default correction in _dims
    q = shard_spec_for("mtp.layers.0.self_attn.q_proj.weight", (12288, 5120), cfg)
    assert q.kind == "head_column" and q.unit_size == 512  # 2 * 256 pair
    fc = shard_spec_for("mtp.fc.weight", (5120, 10240), cfg)
    assert fc.kind == "replicated" and "fuses" in fc.reason


def test_spec_real_kv_heads_split_two_per_rank_at_world2():
    cfg = _tiny_config()
    spec = shard_spec_for("layers.3.self_attn.k_proj.weight", (32, D), cfg)
    assert spec.indices_for_rank(0, 2) == list(range(0, 16))
    assert spec.indices_for_rank(1, 2) == list(range(16, 32))


# -- the q/gate pairing (the split that silently corrupts if guessed) -------


def test_q_gate_pair_travels_with_its_head():
    cfg = _tiny_config()
    name = "layers.3.self_attn.q_proj.weight"
    n_head, head_dim = cfg.num_attention_heads, cfg.head_dim
    # Identity-like rows: row i reads input coord 0 with gain i+1, so
    # output feature i literally reports which dense row produced it.
    w = torch.zeros(2 * n_head * head_dim, D)
    for i in range(w.shape[0]):
        w[i, 0] = float(i + 1)
    sd = {name: w}
    shards = [shard_state_dict(sd, r, 2, cfg)[name] for r in range(2)]
    spec = shard_spec_for(name, tuple(w.shape), cfg)
    for rank in range(2):
        idx = spec.indices_for_rank(rank, 2)
        heads = list(range(rank * 2, rank * 2 + 2))  # 4 heads / world 2
        # Whole [q|gate] pairs, in head order, nothing interleaved away.
        expected = [row for h in heads
                    for row in range(h * 2 * head_dim, (h + 1) * 2 * head_dim)]
        assert idx == expected
        assert torch.equal(shards[rank], w[expected])
        # Each head's gate block (second half of its pair) is present
        # contiguously right after its query block in the same shard.
        for local_h, h in enumerate(heads):
            pair = shards[rank][local_h * 2 * head_dim:(local_h + 1) * 2 * head_dim]
            q_rows = pair[:head_dim, 0]
            gate_rows = pair[head_dim:, 0]
            assert q_rows[0].item() == h * 2 * head_dim + 1
            assert gate_rows[0].item() == h * 2 * head_dim + head_dim + 1
    # And the partitioned projection still reproduces the dense one.
    x = torch.zeros(1, 1, D)
    x[..., 0] = 1.0
    y = partitioned_linear(x, shards, spec)
    assert torch.equal(y, F.linear(x, w))
    assert y[0, 0, head_dim].item() == head_dim + 1  # head 0's gate feature


# -- gated-delta segmented qkv ----------------------------------------------


def test_delta_qkv_segment_indices_at_tiny_dims():
    cfg = _tiny_config()
    # Layout [q 0:16 | k 16:32 | v 32:64]; K=2 heads x8, V=4 heads x8.
    # world=2 -> each rank owns 2 value heads -> 1 key head (repeat 2).
    spec = shard_spec_for("layers.0.linear_attn.in_proj_qkv.weight", (64, D), cfg)
    assert spec.kind == "segmented_qkv"
    assert spec.indices_for_rank(0, 2) == (
        list(range(0, 8)) + list(range(16, 24)) + list(range(32, 48)))
    assert spec.indices_for_rank(1, 2) == (
        list(range(8, 16)) + list(range(24, 32)) + list(range(48, 64)))
    assert spec.local_segments(0, 2) == [
        ("q", 0, 8, 0), ("k", 8, 8, 16), ("v", 16, 16, 32)]
    assert spec.local_segments(1, 2) == [
        ("q", 0, 8, 8), ("k", 8, 8, 24), ("v", 16, 16, 48)]
    # conv1d channels use the identical concatenated order.
    conv = shard_spec_for("layers.0.linear_attn.conv1d.weight", (64, 1, 4), cfg)
    assert conv.indices_for_rank(1, 2) == spec.indices_for_rank(1, 2)


def test_delta_qkv_shard_keeps_concatenated_order():
    cfg = _tiny_config()
    w = torch.arange(64 * D, dtype=torch.float32).reshape(64, D)
    sd = {"layers.0.linear_attn.in_proj_qkv.weight": w}
    shard1 = shard_state_dict(sd, 1, 2, cfg)["layers.0.linear_attn.in_proj_qkv.weight"]
    # Rank 1 storage order is [its q rows | its k rows | its v rows].
    assert torch.equal(shard1[:8], w[8:16])     # q of key head 1
    assert torch.equal(shard1[8:16], w[24:32])  # k of key head 1
    assert torch.equal(shard1[16:], w[48:64])   # v of value heads 2-3


# -- shard / merge roundtrip ---------------------------------------------------


def test_shard_merge_roundtrip_exact_tiny_hybrid():
    cfg = _tiny_config()
    sd = _tiny_state_dict()
    original = {k: v.clone() for k, v in sd.items()}
    shards = [shard_state_dict(sd, r, 2, cfg) for r in range(2)]
    # Sharding does not mutate the dense dict and actually splits rows.
    for k in sd:
        assert torch.equal(sd[k], original[k])
    assert shards[0]["layers.3.self_attn.q_proj.weight"].shape == (64, D)
    assert shards[0]["layers.0.linear_attn.in_proj_qkv.weight"].shape == (32, D)
    assert shards[0]["layers.0.mlp.down_proj.weight"].shape == (D, 32)
    merged = merge_state_dicts(shards, cfg)
    assert set(merged) == set(sd)
    for k in sd:
        assert torch.equal(merged[k], sd[k]), k
    # Rank-order mapping form merges identically; world=1 is identity.
    assert all(torch.equal(merge_state_dicts({0: shards[0], 1: shards[1]}, cfg)[k], sd[k])
               for k in sd)
    solo = shard_state_dict(sd, 0, 1, cfg)
    assert all(torch.equal(merge_state_dicts([solo], cfg)[k], sd[k]) for k in sd)


def test_merge_rejects_disagreeing_replicated_copies():
    cfg = _tiny_config()
    sd = _tiny_state_dict()
    shards = [shard_state_dict(sd, r, 2, cfg) for r in range(2)]
    shards[1]["norm.weight"] = shards[1]["norm.weight"] + 1
    with pytest.raises(ValueError, match="replicated"):
        merge_state_dicts(shards, cfg)


def test_world_must_divide_head_groups():
    cfg = _tiny_config()
    sd = _tiny_state_dict()
    with pytest.raises(ValueError):
        shard_state_dict(sd, 0, 4, cfg)  # 1 value head/rank < repeat factor 2


# -- partitioned linear ---------------------------------------------------------


@pytest.mark.parametrize("layer,proj", [
    ("layers.3", "self_attn.q_proj"), ("layers.3", "self_attn.k_proj"),
    ("layers.3", "self_attn.o_proj"), ("layers.0", "linear_attn.in_proj_qkv"),
    ("layers.0", "linear_attn.in_proj_z"), ("layers.0", "linear_attn.out_proj"),
    ("layers.0", "mlp.gate_proj"), ("layers.0", "mlp.down_proj"),
])
def test_partitioned_linear_matches_dense(layer, proj):
    cfg = _tiny_config()
    sd = _tiny_state_dict(seed=3)
    name = f"{layer}.{proj}.weight"
    w = sd[name]
    spec = shard_spec_for(name, tuple(w.shape), cfg)
    shards = [shard_state_dict(sd, r, 2, cfg)[name] for r in range(2)]
    torch.manual_seed(5)
    x = torch.randn(2, 3, w.shape[1])
    y = partitioned_linear(x, shards, spec)
    dense = F.linear(x, w)
    if spec.dim == 1:  # row-parallel sum re-associates: fp tolerance only
        assert torch.allclose(y, dense, atol=1e-6)
    else:              # column family reassembles slices exactly
        assert torch.equal(y, dense)


# -- byte accounting -------------------------------------------------------------


def test_parameter_bytes_by_rank_tiny_formula():
    cfg = _tiny_config()
    per_rank = parameter_bytes_by_rank(cfg, 2)
    assert set(per_rank) == {0, 1}
    assert per_rank[0] == per_rank[1]  # symmetric split at world 2
    assert per_rank[0]["total_bytes"] > 0
    # Custom dtype rates are honoured (fp8 counted at 2 bytes here).
    wide = parameter_bytes_by_rank(cfg, 2, {"fp8_weight": 2})
    assert wide[0]["fp8_weight_bytes"] == 2 * wide[0]["fp8_weight"]


def test_real_config_report_under_24gib_with_sane_breakdown():
    report = real_config_report(world=2)
    total = report["per_rank_total_gib"]
    assert report["fits_24gib_per_rank"] is True
    assert 12.0 < total < 20.0  # range, not fabricated precision
    rank0 = report["per_rank_bytes"][0]
    fp8_gib = rank0["fp8_weight_bytes"] / 1024 ** 3
    bf16_gib = rank0["bf16_replicated_bytes"] / 1024 ** 3
    scale_gib = rank0["scale_inv_bytes"] / 1024 ** 3
    assert 10.0 < fp8_gib < 13.5   # ~half of the 24.7 GB dense FP8 payload
    assert 4.0 < bf16_gib < 6.0    # dominated by replicated embed + lm_head
    assert 0.0 < scale_gib < 0.1   # [128,128] block scales are a rounding term
    assert report["config"]["layers"] == 64
    assert report["config"]["full_layers"] == 16
    assert report["config"]["linear_layers"] == 48
    # Cross-check against the real manifest when present (headers only):
    # dense FP8 bytes must equal 2x the per-rank FP8 bytes (world=2).
    manifest = report["manifest_cross_check"]
    if manifest is not None:
        dense_fp8 = manifest["total_bytes_by_dtype"]["F8_E4M3"]
        assert dense_fp8 == pytest.approx(2 * rank0["fp8_weight_bytes"], rel=0.01)
