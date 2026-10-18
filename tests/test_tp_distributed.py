"""Distributed TP (tp_distributed): shard construction + real gloo check.

Shard-construction tests use a stub context (rank/world/device only, no
process group) so they run in the plain CPU suite. The subprocess test
spawns the real 2-process gloo check; it measured ~1.5 s wall locally,
well under the 20 s budget, so it is kept durable rather than script-only.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.linear_attention import GatedDeltaNet  # noqa: E402
from inference_from_scratch.mlp import SwiGLU  # noqa: E402
from inference_from_scratch.tensor_parallel import shard_state_dict  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedGatedDeltaNet, DistributedSwiGLU, TPColumnLinear, TPRowLinear,
    shard_weight_for_rank)


class _StubCtx:
    def __init__(self, rank, world=2):
        self.rank, self.world_size = rank, world
        self.device = torch.device("cpu")


def test_column_row_shard_construction_matches_dense_slices():
    torch.manual_seed(0)
    col_dense = nn.Linear(16, 24, bias=True)
    shards = [TPColumnLinear.from_dense(col_dense, "mlp.gate_proj.weight", _StubCtx(r)) for r in range(2)]
    assert [tuple(s.weight.shape) for s in shards] == [(12, 16), (12, 16)]
    assert torch.equal(torch.cat([s.weight.data for s in shards]), col_dense.weight.data)
    assert torch.equal(torch.cat([s.bias.data for s in shards]), col_dense.bias.data)
    row_dense = nn.Linear(24, 16, bias=True)
    rshards = [TPRowLinear.from_dense(row_dense, "mlp.down_proj.weight", _StubCtx(r)) for r in range(2)]
    assert [tuple(s.weight.shape) for s in rshards] == [(16, 12), (16, 12)]
    assert torch.equal(torch.cat([s.weight.data for s in rshards], dim=1), row_dense.weight.data)
    # Each module holds ONLY its shard (half the dense rows/cols).
    assert shards[0].weight.numel() == col_dense.weight.numel() // 2
    assert rshards[0].weight.numel() == row_dense.weight.numel() // 2


def test_shard_weight_for_rank_matches_simulated_specs():
    torch.manual_seed(1)
    w = torch.randn(64, 32)
    for r in range(2):
        expected = shard_state_dict({"mlp.gate_proj.weight": w}, r, 2)["mlp.gate_proj.weight"]
        assert torch.equal(shard_weight_for_rank("mlp.gate_proj.weight", w, r, 2), expected)


def test_distributed_mlp_and_delta_hold_only_rank_shards():
    torch.manual_seed(2)
    mlp = DistributedSwiGLU.from_dense(SwiGLU(32, 64), _StubCtx(0))
    assert tuple(mlp.gate_proj.weight.shape) == (32, 32)
    assert tuple(mlp.down_proj.weight.shape) == (32, 32)
    net = GatedDeltaNet(d_model=32, num_key_heads=2, key_dim=8,
                        num_value_heads=4, value_dim=8, conv_kernel=4)
    delta = DistributedGatedDeltaNet(_StubCtx(1), net)
    assert delta.shard_shapes["linear_attn.in_proj_qkv.weight"] == (32, 32)
    assert delta.shard_shapes["linear_attn.out_proj.weight"] == (32, 16)
    assert delta.shard_shapes["linear_attn.A_log"] == (2,)
    with pytest.raises(ValueError):
        TPColumnLinear.from_dense(nn.Linear(8, 7), "mlp.gate_proj.weight", _StubCtx(0))


@pytest.mark.skipif(not (ROOT / "scripts/tp2_tiny_check.py").exists(), reason="check script missing")
def test_tp2_tiny_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_tiny_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "TP2_TINY_CHECK PASS" in proc.stdout
    assert "mlp:" in proc.stdout and "attention:" in proc.stdout and "delta_step:" in proc.stdout
