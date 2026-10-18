"""Tests for hybrid.py: gated full attention + hybrid stack (Qwen3.8).

The real-config test reads the actual Qwen3.8-27B-FP8 snapshot
``config.json`` from the local HF cache (skipped when absent); every
other test uses tiny dims. Names/checkpoint shapes are pinned against
the real layer-3 (full) and layer-0 (linear) shard headers.
"""

import json
from pathlib import Path

import pytest
import torch

from inference_from_scratch.hybrid import (
    GatedFullAttention,
    HybridBlock,
    HybridConfig,
    HybridState,
    TinyHybridLM,
    apply_partial_rope,
    generate_hybrid,
    list_hybrid_tensor_map,
    precompute_mrope_cos_sin,
    precompute_partial_rope_cache,
)
from inference_from_scratch.kv_cache import KVCache

# Tiny hybrid: 3 linear + 1 full, GQA 4q/2kv, head_dim 16 (rotary 4).
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


def _snapshot_dir():
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots")
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


# -- config -----------------------------------------------------------------


def test_config_from_text_config_synthetic_and_aliases():
    cfg = HybridConfig.from_text_config({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 128,
        "vocab_size": 1000, "max_position_embeddings": 512,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16,
        "rope_parameters": {"rope_theta": 500000.0, "partial_rotary_factor": 0.5,
                            "mrope_section": [5, 5, 6], "mrope_interleaved": True},
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
    })
    assert cfg.hidden_size == cfg.d_model == 64
    assert cfg.num_hidden_layers == cfg.n_layer == 2
    assert cfg.num_attention_heads == cfg.n_head == 4
    assert cfg.num_key_value_heads == cfg.n_kv_head == 2
    assert cfg.intermediate_size == cfg.d_ff == 128
    assert cfg.max_position_embeddings == cfg.max_seq_len == 512
    assert cfg.head_dim == 32 and cfg.rotary_dim == 16
    assert cfg.rope_theta == 500000.0 and cfg.mrope_section == [5, 5, 6]
    assert cfg.rms_norm_eps == cfg.rms_eps == 1e-5
    assert cfg.tie_word_embeddings is cfg.tie_embeddings is True
    assert cfg.num_linear_layers == cfg.num_full_layers == 1
    # Repo-style construction spells the same config.
    cfg2 = HybridConfig(vocab_size=1000, d_model=64, n_layer=2, n_head=4,
                        n_kv_head=2, head_dim=32, d_ff=128, max_seq_len=512,
                        layer_types=["linear", "full"])
    assert cfg2.layer_types == ["linear_attention", "full_attention"]
    assert cfg2.d_model == 64 and cfg2.n_layer == 2


@pytest.mark.skipif(_snapshot_dir() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_config_parses_real_snapshot_file():
    snap = _snapshot_dir()
    raw = json.loads((snap / "config.json").read_text())
    cfg = HybridConfig.from_text_config(raw)          # full file
    cfg_text = HybridConfig.from_text_config(raw["text_config"])  # text only
    for c in (cfg, cfg_text):
        assert c.hidden_size == 5120 and c.d_model == 5120
        assert c.num_hidden_layers == 64 and len(c.layer_types) == 64
        assert c.num_linear_layers == 48 and c.num_full_layers == 16
        assert c.layer_types[3] == "full_attention"
        assert c.layer_types[0] == "linear_attention"
        assert c.num_attention_heads == 24 and c.num_key_value_heads == 4
        assert c.head_dim == 256 and c.rotary_dim == 64
        assert c.intermediate_size == 17408 and c.vocab_size == 248320
        assert c.max_position_embeddings == 262144
        assert c.partial_rotary_factor == 0.25
        assert c.rope_theta == 10000000
        assert c.mrope_section == [11, 11, 10] and c.mrope_interleaved is True
        assert c.linear_num_key_heads == 16 and c.linear_key_head_dim == 128
        assert c.linear_num_value_heads == 48 and c.linear_value_head_dim == 128
        assert c.linear_conv_kernel_dim == 4
        assert c.rms_norm_eps == 1e-6 and c.attention_bias is False
        assert c.tie_word_embeddings is False


# -- gated full attention -----------------------------------------------------


def test_gated_full_attention_shapes_mirror_layer3():
    cfg = _tiny_config()
    attn = GatedFullAttention(cfg)
    # Real layer 3: q [12288, 5120] = 2*24*256; k/v [1024, 5120] = 4*256;
    # o [5120, 6144]; norms [256]. Tiny analogues:
    assert attn.q_proj.weight.shape == (2 * 4 * 16, D)
    assert attn.k_proj.weight.shape == (2 * 16, D)
    assert attn.v_proj.weight.shape == (2 * 16, D)
    assert attn.o_proj.weight.shape == (D, 4 * 16)
    assert attn.q_norm.weight.shape == (16,)
    assert attn.k_norm.weight.shape == (16,)
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    assert attn.rotary_dim == 4
    with torch.no_grad():
        out = attn(torch.randn(2, 5, D))
    assert out.shape == (2, 5, D)


def test_manual_vs_sdpa_paths_allclose():
    cfg = _tiny_config()
    torch.manual_seed(0)
    manual = GatedFullAttention(cfg, attn_implementation="manual").eval()
    sdpa = GatedFullAttention(cfg, attn_implementation="sdpa").eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 6, D)
    with torch.no_grad():
        a, b = manual(x), sdpa(x)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_partial_rope_rotates_only_prefix_dims():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 5, 16)
    cos, sin, rotary_dim = precompute_partial_rope_cache(
        16, head_dim=16, partial_rotary_factor=0.25, theta=10000.0)
    assert rotary_dim == 4
    y = apply_partial_rope(x, cos, sin, start_pos=2, rotary_dim=rotary_dim)
    # Tail dims are bit-identical; prefix actually rotates (past pos 0).
    assert torch.equal(y[..., rotary_dim:], x[..., rotary_dim:])
    assert not torch.allclose(y[..., :rotary_dim], x[..., :rotary_dim])
    # Rotation is orthogonal per plane: norms preserved.
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


def test_hf_parity_gated_full_attention():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention, Qwen3_5TextRotaryEmbedding)

    torch.manual_seed(2)
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=D, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-6, attention_bias=False,
        partial_rotary_factor=0.25)
    hf = Qwen3_5Attention(hf_cfg, layer_idx=3).eval()
    ours = GatedFullAttention(_tiny_config()).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)  # same names/shapes
    x = torch.randn(2, 5, D)
    rope = Qwen3_5TextRotaryEmbedding(hf_cfg)
    pos = torch.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = rope(x, pos)
    mask = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1).view(1, 1, 5, 5)
    with torch.no_grad():
        ref, _ = hf(x, position_embeddings=(cos, sin), attention_mask=mask)
        got = ours(x, cos[0], sin[0])
    assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()


# -- blocks and the tiny LM ----------------------------------------------------


def test_block_shapes_both_types():
    cfg = _tiny_config()
    x = torch.randn(2, 5, D)
    linear = HybridBlock(cfg, layer_idx=0)
    full = HybridBlock(cfg, layer_idx=3)
    assert linear.layer_type == "linear_attention"
    assert linear.linear_attn is not None and linear.self_attn is None
    assert full.layer_type == "full_attention"
    assert full.self_attn is not None and full.linear_attn is None
    assert full.self_attn.q_proj.weight.shape == (2 * 4 * 16, D)
    with torch.no_grad():
        assert linear(x).shape == (2, 5, D)
        assert full(x).shape == (2, 5, D)
        out, state = full.forward_with_state(x, None)
        assert out.shape == (2, 5, D)
        assert state[0].shape == (2, 2, 5, 16)  # [B, H_kv, T, head_dim]


def _tiny_lm(seed=3):
    torch.manual_seed(seed)
    m = TinyHybridLM(_tiny_config())
    m.eval()
    return m


def test_tiny_hybrid_prefill_vs_tokenstep_logits_allclose():
    m = _tiny_lm()
    idx = torch.randint(0, VOCAB, (2, 7))
    with torch.no_grad():
        full, state_full = m(idx)
        pre, state = m.prefill(idx[:, :4])
        pieces = [pre]
        for t in range(4, 7):
            logits, state = m.decode_step(idx[:, t], state)
            pieces.append(logits)
        joined = torch.cat(pieces, dim=1)
        # Also: one-by-one from scratch equals the single prefill.
        state1 = m.initial_state(2)
        singles = []
        for t in range(7):
            logits, state1 = m.step(idx[:, t], state1)
            singles.append(logits)
        joined1 = torch.cat(singles, dim=1)
    assert state_full.seq_len == state.seq_len == 7
    assert torch.allclose(full, joined, atol=1e-6), (full - joined).abs().max()
    assert torch.allclose(full, joined1, atol=1e-6), (full - joined1).abs().max()


def test_snapshot_restore_continuation_identical():
    m = _tiny_lm(seed=4)
    idx = torch.randint(0, VOCAB, (2, 5))
    nxt = torch.randint(0, VOCAB, (2,))
    with torch.no_grad():
        _, state = m.prefill(idx)
        snap = m.snapshot_state(state)
        out_a, state_a = m.decode_step(nxt, state)
        restored = m.restore_state(snap)
        out_b, state_b = m.decode_step(nxt, restored)
        # The same snapshot restores again, identically.
        out_c, _ = m.decode_step(nxt, m.restore_state(snap))
    assert torch.equal(out_a, out_b) and torch.equal(out_a, out_c)
    # Mixed state: full layer holds KV, linear layers hold (S, conv).
    assert state_a.layer_states[3][0].shape[2] == 6  # KV grew by one
    assert state_a.layer_states[0][0].shape == (2, 4, 8, 8)  # S [B, Hv, Dk, Dv]
    for la, lb in zip(state_a.layer_states, state_b.layer_states):
        assert torch.equal(la[0], lb[0]) and torch.equal(la[1], lb[1])
    # Snapshot is independent of later mutation of a restored copy.
    with torch.no_grad():
        restored.layer_states[0][0].add_(10.0)
    assert not torch.equal(restored.layer_states[0][0], snap.layer_states[0][0])


def test_lm_head_untied_by_default():
    m = _tiny_lm(seed=5)
    assert m.lm_head.weight is not m.embed_tokens.weight
    assert m.lm_head.weight.shape == (VOCAB, D)
    tied = TinyHybridLM(_tiny_config(tie_word_embeddings=True))
    assert tied.lm_head.weight is tied.embed_tokens.weight


# -- checkpoint tensor map -----------------------------------------------------


def test_tensor_map_from_checkpoint_style_names():
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ]
    mp = list_hybrid_tensor_map(names)
    assert mp["model.language_model.embed_tokens.weight"] == "embed_tokens.weight"
    assert mp["model.language_model.norm.weight"] == "norm.weight"
    assert mp["lm_head.weight"] == "lm_head.weight"
    assert mp["model.language_model.layers.3.self_attn.q_proj.weight"] == \
        "layers.3.self_attn.q_proj.weight"
    assert mp["model.language_model.layers.3.self_attn.q_proj.weight_scale_inv"] == \
        "layers.3.self_attn.q_proj.weight_scale_inv"
    assert mp["model.language_model.layers.3.self_attn.q_norm.weight"] == \
        "layers.3.self_attn.q_norm.weight"
    assert mp["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"] == \
        "layers.0.linear_attn.in_proj_qkv.weight"
    assert mp["model.language_model.layers.0.linear_attn.conv1d.weight"] == \
        "layers.0.linear_attn.conv1d.weight"
    assert mp["model.language_model.layers.0.linear_attn.A_log"] == \
        "layers.0.linear_attn.A_log"
    assert mp["model.language_model.layers.0.mlp.gate_proj.weight"] == \
        "layers.0.mlp.gate_proj.weight"
    assert mp["mtp.fc.weight"] == "mtp.fc.weight"
    assert "model.visual.blocks.0.attn.qkv.weight" not in mp  # vision deferred


# -- explicit positions (mRoPE) and the output gate ---------------------------


def test_position_ids_text_repeat_equals_default():
    torch.manual_seed(10)
    attn = GatedFullAttention(_tiny_config()).eval()
    x = torch.randn(2, 6, D)
    pos = torch.arange(6)
    with torch.no_grad():
        ref = attn(x)
        by_t = attn(x, position_ids=pos)
        by_3t = attn(x, position_ids=pos.unsqueeze(0).expand(3, -1))
    # Text semantics: one axis repeated to T/H/W; the interleaved
    # recomposition is then the identity, bit-for-bit.
    assert torch.equal(ref, by_t)
    assert torch.equal(ref, by_3t)
    # The mrope tables themselves equal the plain partial-RoPE tables.
    cos_m, sin_m = precompute_mrope_cos_sin(
        pos, attn.rotary_dim, theta=10000.0, mrope_section=[11, 11, 10])
    cos_p, sin_p, _ = precompute_partial_rope_cache(
        6, head_dim=16, partial_rotary_factor=0.25, theta=10000.0)
    assert torch.equal(cos_m, cos_p) and torch.equal(sin_m, sin_p)


def test_position_ids_differing_axes_touch_only_their_slots():
    # Rotary width 8 -> 4 pairs; section [2, 1, 1]: H owns pair slot 1
    # (slice(1, 3, 3)), W owns slot 2 (slice(2, 3, 3)), T the rest.
    base = torch.arange(5)
    pos3 = torch.stack([base, base + 100, base + 200])
    cos3, sin3 = precompute_mrope_cos_sin(
        pos3, 8, theta=10000.0, mrope_section=[2, 1, 1])
    cos1, sin1 = precompute_mrope_cos_sin(
        base, 8, theta=10000.0, mrope_section=[2, 1, 1])
    for tables in ((cos3, cos1), (sin3, sin1)):
        changed = [j for j in range(4)
                   if not torch.equal(tables[0][:, j], tables[1][:, j])]
        assert changed == [1, 2]
    # Rotate-half layout duplicates each pair into the second half.
    assert torch.equal(cos3[:, 1], cos3[:, 5])
    assert torch.equal(cos3[:, 0], cos3[:, 4])
    # At the attention level the differing axes really do move the
    # output (the slots above are not decorative).
    torch.manual_seed(11)
    cfg = _tiny_config(head_dim=32, partial_rotary_factor=0.25,
                       mrope_section=[2, 1, 1])
    attn = GatedFullAttention(cfg).eval()
    x = torch.randn(1, 5, D)
    with torch.no_grad():
        ref = attn(x, position_ids=base)
        moved = attn(x, position_ids=pos3)
    assert not torch.allclose(ref, moved)


def test_position_ids_lm_level_and_validation():
    m = _tiny_lm(seed=12)
    idx = torch.randint(0, VOCAB, (2, 7))
    pos = torch.arange(7)
    with torch.no_grad():
        ref, _ = m(idx)
        by_t, _ = m(idx, position_ids=pos)
        by_3t, _ = m(idx, position_ids=pos.unsqueeze(0).expand(3, -1))
    assert torch.equal(ref, by_t) and torch.equal(ref, by_3t)
    with pytest.raises(ValueError):
        m(idx, position_ids=torch.arange(3))  # one position per token
    with pytest.raises(ValueError):
        GatedFullAttention(_tiny_config())(torch.randn(1, 4, D),
                                           position_ids=torch.arange(9))
    with pytest.raises(ValueError):
        precompute_mrope_cos_sin(torch.zeros(2, 2, 2, 2), 4)


def test_output_gate_swish_label_means_sigmoid():
    # HF applies attn_output * sigmoid(gate) (modeling line 804); the
    # config's output_gate_type="swish" is a label for that op, and
    # silu(g) = g * sigmoid(g) must never be selected by it.
    cfg = _tiny_config(output_gate_type="swish")
    assert GatedFullAttention(cfg).gate_activation == "sigmoid"
    assert GatedFullAttention(_tiny_config(),
                              gate_activation="swish").gate_activation == "sigmoid"
    assert GatedFullAttention(_tiny_config(),
                              gate_activation="silu").gate_activation == "silu"
    torch.manual_seed(13)
    sig = GatedFullAttention(_tiny_config()).eval()
    swish = GatedFullAttention(cfg).eval()
    swish.load_state_dict(sig.state_dict())
    silu = GatedFullAttention(_tiny_config(), gate_activation="silu").eval()
    silu.load_state_dict(sig.state_dict())
    x = torch.randn(2, 5, D)
    with torch.no_grad():
        assert torch.equal(sig(x), swish(x))
        assert not torch.allclose(sig(x), silu(x))


# -- typed HybridState ---------------------------------------------------------


def test_hybrid_state_typed_entries_and_clone():
    m = _tiny_lm(seed=14)
    fresh = m.initial_state(2)
    assert [tag for tag, _ in fresh.entries] == [
        "linear", "linear", "linear", "kv"]
    assert isinstance(fresh.kv_cache(3), KVCache)
    assert fresh.linear_state(0) is None  # lazy zero state
    assert fresh.seq_len == 0 and len(fresh) == 4

    idx = torch.randint(0, VOCAB, (2, 4))
    with torch.no_grad():
        _, state = m.prefill(idx)
    assert state.seq_len == 4
    tag, payload = state.entry(0)
    assert tag == "linear" and payload[0].shape == (2, 4, 8, 8)
    assert state.kv_cache(3).keys[0].shape == (2, 2, 4, 16)
    # Legacy view still reads as (k, v) / (S, conv) tensor pairs.
    assert state.layer_states[3][0].shape[2] == 4
    assert state.layer_states[0][0].shape == (2, 4, 8, 8)

    snap = state.clone()
    with torch.no_grad():
        _, continued = m.decode_step(torch.randint(0, VOCAB, (2,)), state)
    assert continued.seq_len == 5
    # The clone is untouched by continuing the original.
    assert snap.seq_len == 4
    assert torch.equal(snap.kv_cache(3).keys[0], state.kv_cache(3).keys[0])
    assert snap.kv_cache(3).keys[0].shape[2] == 4
    # And it restores into an independently continuable state.
    restored = snap.restore()
    assert restored.seq_len == 4 and restored.entry(3)[0] == "kv"
    # Legacy construction (no layer_types) still feeds the model.
    legacy = HybridState(layer_states=[None] * 4, seq_len=0)
    with torch.no_grad():
        logits, out_state = m(idx, legacy)
    assert out_state.seq_len == 4


# -- stateful generation --------------------------------------------------------


def test_generate_hybrid_greedy_prefill_methods_agree():
    m = _tiny_lm(seed=15)
    prompt = torch.randint(0, VOCAB, (2, 5))
    with torch.no_grad():
        seq = generate_hybrid(m, prompt, 6, temperature=0,
                              prefill_method="sequential")
        chunk = generate_hybrid(m, prompt, 6, temperature=0,
                                prefill_method="chunkwise", chunk_size=4)
    assert torch.equal(seq, chunk)
    assert seq.shape == (2, 11)
    assert torch.equal(seq[:, :5], prompt)  # prompt preserved verbatim
    with pytest.raises(ValueError):
        generate_hybrid(m, prompt, 2, temperature=0, prefill_method="bogus")
    assert generate_hybrid(m, prompt, 0, temperature=0).shape == (2, 5)


def test_generate_hybrid_sampling_reproducible_and_eos():
    m = _tiny_lm(seed=16)
    prompt = torch.randint(0, VOCAB, (1, 4))
    g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    with torch.no_grad():
        a = generate_hybrid(m, prompt, 5, temperature=0.8, top_k=10, generator=g1)
        b = generate_hybrid(m, prompt, 5, temperature=0.8, top_k=10, generator=g2)
        greedy = generate_hybrid(m, prompt, 6, temperature=0)
        stopped = generate_hybrid(m, prompt, 6, temperature=0,
                                  eos_id=int(greedy[0, 4]))
        stopped_alias = generate_hybrid(m, prompt, 6, temperature=0,
                                        eos=int(greedy[0, 4]))
    assert torch.equal(a, b)  # same seed, same ids
    assert stopped.shape == (1, 5)  # EOS (included) ends generation
    assert torch.equal(stopped, greedy[:, :5])
    assert torch.equal(stopped_alias, stopped)


def test_branching_snapshot_suffixes_equal_fresh_generations():
    m = _tiny_lm(seed=17)
    prompt = torch.randint(0, VOCAB, (1, 5))

    def greedy_suffix(state, forced_token, n_new):
        """Feed ``forced_token``, then argmax-chain ``n_new`` ids."""
        ids, logits = [], None
        logits, s = m.decode_step(torch.tensor([[forced_token]]), state)
        nxt = int(logits[:, -1].argmax())
        ids.append(nxt)
        for _ in range(n_new - 1):
            logits, s = m.decode_step(torch.tensor([[nxt]]), s)
            nxt = int(logits[:, -1].argmax())
            ids.append(nxt)
        return ids, logits

    def fresh_suffix(forced_token, n_new):
        """The same suffix built token-by-token from an empty state."""
        s = m.initial_state(1)
        for t in range(prompt.shape[1]):
            _, s = m.decode_step(prompt[:, t], s)
        return greedy_suffix(s, forced_token, n_new)

    with torch.no_grad():
        _, prompt_state = m.prefill(prompt)
        snap = m.snapshot_state(prompt_state)
        branch_a_ids, branch_a_logits = greedy_suffix(
            m.restore_state(snap), forced_token=3, n_new=4)
        branch_b_ids, branch_b_logits = greedy_suffix(
            m.restore_state(snap), forced_token=90, n_new=4)
        fresh_a_ids, fresh_a_logits = fresh_suffix(forced_token=3, n_new=4)
        fresh_b_ids, fresh_b_logits = fresh_suffix(forced_token=90, n_new=4)
        # A single fresh full forward over each branch's whole token
        # stream lands on the same final logits (fp tolerance: the
        # batched prefill re-associates the arithmetic).
        full_a, _ = m(torch.cat(
            [prompt, torch.tensor([[3] + branch_a_ids[:-1]])], dim=1))
        full_b, _ = m(torch.cat(
            [prompt, torch.tensor([[90] + branch_b_ids[:-1]])], dim=1))
    # Each branch's greedy suffix equals the fresh generation's.
    assert branch_a_ids == fresh_a_ids and branch_b_ids == fresh_b_ids
    assert torch.allclose(branch_a_logits, fresh_a_logits, atol=1e-6)
    assert torch.allclose(branch_b_logits, fresh_b_logits, atol=1e-6)
    assert torch.allclose(branch_a_logits, full_a[:, -1:], atol=1e-6)
    assert torch.allclose(branch_b_logits, full_b[:, -1:], atol=1e-6)
    # The two branches really diverged, and neither disturbed the other
    # or the shared snapshot: replaying branch A still matches.
    assert branch_a_ids != branch_b_ids
    assert snap.seq_len == 5 and prompt_state.seq_len == 5
    with torch.no_grad():
        replay_ids, _ = greedy_suffix(m.restore_state(snap), 3, 4)
    assert replay_ids == branch_a_ids


@pytest.mark.skipif(_snapshot_dir() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_tensor_map_from_real_shard_headers():
    snap = _snapshot_dir()
    mp = list_hybrid_tensor_map(snap)  # index/header names only
    # 1250 language + 22 MTP + lm_head; vision (333) excluded.
    assert len(mp) == 1273
    assert mp["model.language_model.layers.3.self_attn.k_proj.weight"] == \
        "layers.3.self_attn.k_proj.weight"
    assert mp["model.language_model.layers.63.self_attn.o_proj.weight"] == \
        "layers.63.self_attn.o_proj.weight"
    assert mp["mtp.layers.0.self_attn.q_proj.weight"] == \
        "mtp.layers.0.self_attn.q_proj.weight"
    assert not any(k.startswith("model.visual") for k in mp)
    # Every mapped language-model target exists in a (tiny-shaped) model
    # built from the real layer pattern's first layers -- names, not dims.
    cfg = _tiny_config()
    ours = TinyHybridLM(cfg)
    sd = ours.state_dict()
    for layer in range(4):
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                       "mlp.gate_proj.weight"):
            assert f"layers.{layer}.{suffix}" in sd
    assert "layers.3.self_attn.q_proj.weight" in sd
    assert "layers.0.linear_attn.in_proj_qkv.weight" in sd
    assert "embed_tokens.weight" in sd and "norm.weight" in sd
