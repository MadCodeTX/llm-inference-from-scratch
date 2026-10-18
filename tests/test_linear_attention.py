"""Tests for linear_attention.py: gated delta rule (Qwen3.5 linear attn).

Heart of the file: prefill (explicit sequential scan) and token-by-token
decode (`step`) must agree exactly in fp32, a snapshot/restore must
continue identically, and prefill must be causal. A parity test against
HF `Qwen3_5GatedDeltaNet` (transformers 5.20.1) runs when importable.
"""

import torch

from inference_from_scratch.linear_attention import GatedDeltaNet

# Tiny dims: Hv (4) != Hk (2) exercises the repeat_interleave rule (n_rep=2).
D_MODEL, N_K, K_DIM, N_V, V_DIM = 32, 2, 8, 4, 8


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    kwargs = dict(
        d_model=D_MODEL, num_key_heads=N_K, key_dim=K_DIM,
        num_value_heads=N_V, value_dim=V_DIM, conv_kernel=4,
    )
    kwargs.update(overrides)
    m = GatedDeltaNet(**kwargs)
    m.eval()
    return m


def test_parameter_shapes_mirror_checkpoint():
    m = _model()
    # conv_dim = 2*2*8 + 4*8 = 64; value_total = 32.
    assert m.in_proj_qkv.weight.shape == (64, 32)
    assert m.in_proj_z.weight.shape == (32, 32)
    assert m.in_proj_a.weight.shape == (4, 32)
    assert m.in_proj_b.weight.shape == (4, 32)
    assert m.conv1d.weight.shape == (64, 1, 4)
    assert m.A_log.shape == (4,)
    assert m.dt_bias.shape == (4,)
    assert m.norm.weight.shape == (8,)
    assert m.out_proj.weight.shape == (32, 32)
    for mod in (m.in_proj_qkv, m.in_proj_z, m.in_proj_a, m.in_proj_b, m.out_proj):
        assert mod.bias is None


def test_prefill_vs_step_allclose_and_state_shapes():
    torch.manual_seed(1)
    m = _model(seed=1)
    x = torch.randn(2, 6, D_MODEL)
    with torch.no_grad():
        out_pre, state = m.forward_prefill(x)
        S, conv_buffer = state
        assert S.shape == (2, N_V, K_DIM, V_DIM)
        assert conv_buffer.shape == (2, 64, 3)  # K-1 raw vectors
        assert out_pre.shape == (2, 6, D_MODEL)

        state_step = m.initial_state(2, device=x.device)
        outs = []
        for t in range(6):
            o, state_step = m.step(x[:, t, :], state_step)
            outs.append(o)
        out_step = torch.stack(outs, dim=1)
    assert torch.allclose(out_pre, out_step, atol=1e-6), (
        (out_pre - out_step).abs().max()
    )
    assert torch.allclose(state[0], state_step[0], atol=1e-6)
    assert torch.allclose(state[1], state_step[1], atol=1e-6)


def test_prefill_then_step_continues_from_final_state():
    torch.manual_seed(2)
    m = _model(seed=2)
    x = torch.randn(1, 7, D_MODEL)
    with torch.no_grad():
        full, _ = m.forward_prefill(x)
        pre, state = m.forward_prefill(x[:, :4])
        outs = [pre]
        st = state
        for t in range(4, 7):
            o, st = m.step(x[:, t, :], st)
            outs.append(o.unsqueeze(1))
        joined = torch.cat(outs, dim=1)
    assert torch.allclose(full, joined, atol=1e-6)


def test_snapshot_restore_continues_identically():
    torch.manual_seed(3)
    m = _model(seed=3)
    x = torch.randn(2, 5, D_MODEL)
    extra = torch.randn(2, D_MODEL)
    with torch.no_grad():
        _, state = m.forward_prefill(x)
        snap = m.snapshot_state(state)
        out_a, state_a = m.step(extra, state)
        # Snapshot is a deep copy: corrupting it must not touch `state`,
        # and restoring twice must give identical continuations.
        snap_mut = m.snapshot_state(state)
        snap_mut[0].add_(100.0)
        assert not torch.allclose(snap_mut[0], state[0])
        restored = m.restore_state(snap)
        out_b, state_b = m.step(extra, restored)
        restored2 = m.restore_state(snap)
        out_c, _ = m.step(extra, restored2)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_c)
    assert torch.equal(state_a[0], state_b[0])
    assert torch.equal(state_a[1], state_b[1])
    # Restored state is independent of the snapshot it came from.
    with torch.no_grad():
        restored[0].add_(1.0)
    assert not torch.equal(restored[0], snap[0])


def test_causality_future_token_change_leaves_past_unchanged():
    torch.manual_seed(4)
    m = _model(seed=4)
    x = torch.randn(1, 6, D_MODEL)
    x2 = x.clone()
    x2[:, -1, :] += 5.0  # change only the last token
    with torch.no_grad():
        out1, _ = m.forward_prefill(x)
        out2, _ = m.forward_prefill(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)


def test_shapes_equal_heads_and_batched_step_ranks():
    torch.manual_seed(5)
    m = _model(seed=5, num_key_heads=4, num_value_heads=4)  # n_rep = 1
    x = torch.randn(3, 4, D_MODEL)
    with torch.no_grad():
        out, state = m.forward_prefill(x)
        o2, _ = m.step(x[:, 0, :])          # [B, D] in -> [B, D] out
        o3, _ = m.step(x[:, 0:1, :], None)  # [B, 1, D] preserved
    assert out.shape == (3, 4, D_MODEL)
    assert o2.shape == (3, D_MODEL)
    assert o3.shape == (3, 1, D_MODEL)
    assert state[0].shape == (3, 4, K_DIM, V_DIM)


def test_repeat_factor_must_be_integer():
    try:
        GatedDeltaNet(d_model=16, num_key_heads=3, key_dim=4,
                      num_value_heads=4, value_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for Hv % Hk != 0")


def test_gradient_free_determinism():
    torch.manual_seed(6)
    m = _model(seed=6)
    x = torch.randn(2, 5, D_MODEL)
    with torch.no_grad():
        a, sa = m.forward_prefill(x)
        b, sb = m.forward_prefill(x)
    assert torch.equal(a, b)
    assert torch.equal(sa[0], sb[0]) and torch.equal(sa[1], sb[1])
    assert not a.requires_grad


def test_hf_parity_with_copied_weights():
    transformers = __import__("pytest").importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    torch.manual_seed(7)
    cfg = Qwen3_5TextConfig(
        hidden_size=D_MODEL, linear_num_key_heads=N_K, linear_key_head_dim=K_DIM,
        linear_num_value_heads=N_V, linear_value_head_dim=V_DIM,
        linear_conv_kernel_dim=4, num_hidden_layers=1,
        layer_types=["linear_attention"], hidden_act="silu", rms_norm_eps=1e-6,
    )
    hf = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
    ours = _model(seed=99)
    # Same parameter names/shapes: strict load must succeed outright.
    ours.load_state_dict(hf.state_dict(), strict=True)
    x = torch.randn(2, 5, D_MODEL)
    with torch.no_grad():
        ref = hf(x)
        got, _ = ours.forward_prefill(x)
    # Chunkwise (HF prefill) vs explicit recurrent scan: mathematically
    # identical, fp32 reduction order differs. Observed max-abs diff on
    # this machine is 1.19e-07; 1e-5 is the honest tolerance.
    diff = (ref - got).abs().max().item()
    assert torch.allclose(ref, got, atol=1e-5), f"max abs diff {diff}"
