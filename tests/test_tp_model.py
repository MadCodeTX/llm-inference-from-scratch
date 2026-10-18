"""DistributedHybridLM: shard construction (stub ctx) + real gloo proof.

Construction tests use a stub context (rank/world/device only, no
process group) so they run in the plain CPU suite. The subprocess test
spawns ``scripts/tp2_model_check.py`` under 2-process gloo; it measured
~2 s wall locally, well under the 20 s budget, so it stays durable.
"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tensor_parallel import shard_state_dict  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, TPState)


class _StubCtx:
    def __init__(self, rank, world=2):
        self.rank, self.world_size = rank, world
        self.device = torch.device("cpu")


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


def _dense() -> TinyHybridLM:
    torch.manual_seed(7)
    model = TinyHybridLM(_tiny_cfg())
    model.eval()
    return model


def test_from_dense_holds_only_shards_plus_replicated():
    dense = _dense()
    for rank in range(2):
        tp = DistributedHybridLM.from_dense(dense, _StubCtx(rank))
        # Replicated: embedding, lm_head, norms keep dense shapes/values.
        assert tp.embed_tokens.weight.shape == dense.embed_tokens.weight.shape
        assert torch.equal(tp.embed_tokens.weight.data, dense.embed_tokens.weight.data)
        assert torch.equal(tp.lm_head.weight.data, dense.lm_head.weight.data)
        assert torch.equal(tp.norm.weight.data, dense.norm.weight.data)
        assert torch.equal(tp.layers[0].input_layernorm.weight.data,
                           dense.layers[0].input_layernorm.weight.data)
        # Sharded: MLP gate/up rows halved, down cols halved.
        assert tuple(tp.layers[0].mlp.gate_proj.weight.shape) == (32, 32)
        assert tuple(tp.layers[0].mlp.down_proj.weight.shape) == (32, 32)
        # Full layer: q by head pairs (4 heads -> 2/rank), o cols halved.
        assert tuple(tp.layers[3].self_attn.q_weight.shape) == (64, 32)
        assert tuple(tp.layers[3].self_attn.o_weight.shape) == (32, 32)
        # Linear layer: segmented qkv + out shards per tensor_parallel specs.
        assert tuple(tp.layers[0].linear_attn.in_proj_qkv_w.shape) == (32, 32)
        assert tuple(tp.layers[0].linear_attn.out_proj_w.shape) == (32, 16)
        # Each rank holds strictly fewer bytes than the dense model.
        assert tp.local_parameter_bytes() < tp.dense_parameter_bytes(dense)


def test_from_dense_shards_match_shard_state_dict_specs():
    dense = _dense()
    cfg = _tiny_cfg()
    sd = {
        "mlp.gate_proj.weight": dense.layers[0].mlp.gate_proj.weight.data,
        "self_attn.q_proj.weight": dense.layers[3].self_attn.q_proj.weight.data,
        "linear_attn.in_proj_qkv.weight": dense.layers[0].linear_attn.in_proj_qkv.weight.data,
    }
    for rank in range(2):
        tp = DistributedHybridLM.from_dense(dense, _StubCtx(rank))
        expected = shard_state_dict(sd, rank, 2, cfg)
        assert torch.equal(tp.layers[0].mlp.gate_proj.weight.data, expected["mlp.gate_proj.weight"])
        assert torch.equal(tp.layers[3].self_attn.q_weight.data, expected["self_attn.q_proj.weight"])
        assert torch.equal(tp.layers[0].linear_attn.in_proj_qkv_w.data,
                           expected["linear_attn.in_proj_qkv.weight"])


def test_tp_state_empty_tags_and_clone():
    state = TPState.empty(["linear_attention", "full_attention"])
    assert state.seq_len == 0
    assert state.entries == [("linear", None), ("kv", None)]
    filled = TPState(
        entries=[("linear", (torch.zeros(1, 2, 8, 8), torch.zeros(1, 32, 3))),
                 ("kv", (torch.zeros(1, 1, 5, 16), torch.zeros(1, 1, 5, 16)))],
        seq_len=5, layer_types=["linear_attention", "full_attention"])
    cloned = filled.clone()
    assert cloned.seq_len == 5
    assert torch.equal(cloned.entries[0][1][0], filled.entries[0][1][0])
    assert cloned.entries[0][1][0] is not filled.entries[0][1][0]


def test_tp2_model_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_model_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_MODEL_CHECK PASS" in proc.stdout
    assert "prefill_logits:" in proc.stdout
    assert "decode_step5_logits:" in proc.stdout
    assert "sharded_lt_dense=True" in proc.stdout
