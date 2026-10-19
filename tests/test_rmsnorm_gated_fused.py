"""Fused RMSNormGated (--gated-norm fused, shipped default) vs manual.

Shipped (decode round 5): replace the per-layer pow/mean/rsqrt/cast
chain in ``RMSNormGated.forward`` (48 linear layers/token in the TP2
graph profile) with one ``F.rms_norm`` kernel, preserving the
intermediate dtype round-trip. Module flag off by default; the TP2
runner defaults to fused, ``--gated-norm manual`` is the A/B baseline.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from inference_from_scratch.linear_attention import RMSNormGated

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_qwen38_tp2_gated", ROOT / "scripts" / "run_qwen38_tp2.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def _norm(d=128, seed=0):
    torch.manual_seed(seed)
    n = RMSNormGated(d)
    with torch.no_grad():
        n.weight.copy_(1.0 + torch.randn(d) * 0.2)
    return n


def test_fused_flag_defaults_off_and_counter_proves_execution():
    n = _norm()
    assert n.use_fused_rmsnorm is False
    assert n.fused_rmsnorm_calls == 0
    core = torch.randn(24, 128)
    gate = torch.randn(24, 128)
    ref = n(core, gate)
    assert n.fused_rmsnorm_calls == 0  # manual path never counts
    n.use_fused_rmsnorm = True
    out = n(core, gate)
    assert n.fused_rmsnorm_calls == 1  # fused branch actually ran
    n(core, gate)
    assert n.fused_rmsnorm_calls == 2
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_fused_matches_manual_bf16_decode_dtypes():
    n = _norm(seed=1)
    core = torch.randn(24, 128, dtype=torch.bfloat16)
    gate = torch.randn(24, 128, dtype=torch.bfloat16)
    ref = n(core, gate)
    n.use_fused_rmsnorm = True
    out = n(core, gate)
    assert out.dtype == torch.bfloat16
    # Same maths, fused reduction: agree within BF16 rounding.
    assert float((out.float() - ref.float()).abs().max()) < 0.05


def test_fused_matches_manual_fp32_core_bf16_gate():
    # TP static decode: core is fp32 (fused-delta output), gate bf16.
    n = _norm(seed=2)
    core = torch.randn(24, 128)
    gate = torch.randn(24, 128, dtype=torch.bfloat16)
    ref = n(core, gate)
    n.use_fused_rmsnorm = True
    out = n(core, gate)
    assert out.dtype == ref.dtype
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gated_delta_step_equivalence_cpu():
    from inference_from_scratch.linear_attention import GatedDeltaNet
    torch.manual_seed(3)
    net = GatedDeltaNet(32, 2, 8, 4, 8, conv_kernel=4)
    x = torch.randn(2, 32)
    state = net.initial_state(2)
    out_ref, s_ref = net.step(x, state)
    net.norm.use_fused_rmsnorm = True
    out, s = net.step(x, net.initial_state(2))
    assert net.norm.fused_rmsnorm_calls == 1
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(s[0], s_ref[0], atol=1e-6, rtol=1e-6)
    assert torch.equal(s[1], s_ref[1])


def test_runner_gated_norm_flag_defaults_fused():
    args = _runner.build_parser().parse_args([])
    assert args.gated_norm == "fused"  # shipped default (round 5)
    args = _runner.build_parser().parse_args(["--gated-norm", "manual"])
    assert args.gated_norm == "manual"  # A/B baseline still selectable
    assert _runner.GATED_NORM_MODES == ("manual", "fused")

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = RMSNormGated(8)
            self.sub = torch.nn.ModuleList([RMSNormGated(8)])

    m = _Wrap()
    assert _runner.apply_gated_norm_mode(m, "fused") == 2
    assert m.a.use_fused_rmsnorm is True and m.sub[0].use_fused_rmsnorm is True
    assert _runner.apply_gated_norm_mode(m, "manual") == 2
    assert m.a.use_fused_rmsnorm is False
    with pytest.raises(ValueError):
        _runner.apply_gated_norm_mode(m, "bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU graph test needs CUDA")
def test_gpu_fused_kernel_and_graph_replay():
    from torch.profiler import ProfilerActivity, profile
    n = _norm(seed=4).to("cuda")
    core = torch.randn(24, 128, device="cuda")
    gate = torch.randn(24, 128, dtype=torch.bfloat16, device="cuda")
    ref = n(core, gate)
    n.use_fused_rmsnorm = True
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = n(core, gate)
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    # Proof the fused path executed on GPU, not a silent fallback.
    assert any("layer_norm_kernel" in name for name in names), names
    assert n.fused_rmsnorm_calls == 1
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    # CUDA-graph capture/replay over static buffers (decode pattern).
    static_core = core.clone()
    static_gate = gate.clone()
    static_out = torch.empty_like(ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out.copy_(n(static_core, static_gate))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_out, ref, atol=1e-5, rtol=1e-5)
    static_core.copy_(torch.randn(24, 128, device="cuda"))
    expected = n(static_core, static_gate)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(static_out, expected, atol=1e-5, rtol=1e-5)
    del graph
