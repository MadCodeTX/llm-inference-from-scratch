"""Executable simulated TP runtime tests (CPU, numbers executed)."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from inference_from_scratch.hybrid import GatedFullAttention, HybridConfig
from inference_from_scratch.kv_cache import KVCache
from inference_from_scratch.linear_attention import GatedDeltaNet
from inference_from_scratch.mlp import SwiGLU
from inference_from_scratch.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, SimulatedProcessGroup,
    TPFullAttention, TPGatedDeltaNet, TPMLP,
    merge_linear_state, shard_linear_state, state_bytes, state_bytes_table,
)

def _tiny_cfg():
    return HybridConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention","linear_attention","linear_attention","full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)

def test_pg_collectives_and_errors():
    pg = SimulatedProcessGroup(2)
    a, b = torch.ones(2,3), torch.full((2,3), 2.0)
    assert torch.equal(pg.all_reduce_sum([a,b]), torch.full((2,3),3.0))
    assert pg.all_reduce_calls == 1
    assert torch.equal(pg.gather_concat([a,b], dim=-1), torch.cat([a,b],dim=-1))
    with pytest.raises(ValueError): pg.all_reduce_sum([a])
    with pytest.raises(ValueError): SimulatedProcessGroup(0)

def test_column_and_row_parallel_equal_dense_and_are_slices():
    torch.manual_seed(0)
    dense = nn.Linear(16, 24, bias=True)
    x = torch.randn(2,3,16)
    col = ColumnParallelLinear.from_dense(dense, world=2)
    # shards are exact dense slices, not re-initialised
    assert torch.equal(torch.cat([w.data for w in col.shard_weights]), dense.weight.data)
    assert torch.allclose(col(x), F.linear(x, dense.weight, dense.bias), atol=1e-6)
    dense2 = nn.Linear(24, 16, bias=True)
    row = RowParallelLinear.from_dense(dense2, world=2)
    assert torch.equal(torch.cat([w.data for w in row.shard_weights], dim=1), dense2.weight.data)
    x2 = torch.randn(2,3,24)
    assert torch.allclose(row(x2), F.linear(x2, dense2.weight, dense2.bias), atol=1e-6)
    assert row.process_group.all_reduce_calls == 1
    with pytest.raises(ValueError): ColumnParallelLinear.from_dense(nn.Linear(8,7), world=2)

def test_tp_mlp_equals_swiglu_one_allreduce():
    torch.manual_seed(1)
    dense = SwiGLU(32, 64)
    pg = SimulatedProcessGroup(2)
    tp = TPMLP.from_dense(dense, world=2, process_group=pg)
    x = torch.randn(2,5,32)
    before = pg.all_reduce_calls
    out = tp(x)
    assert pg.all_reduce_calls - before == 1
    assert torch.allclose(out, dense(x), atol=1e-6)
    assert torch.equal(torch.cat([w.data for w in tp.gate_proj.shard_weights]), dense.gate_proj.weight.data)
    assert torch.equal(torch.cat([w.data for w in tp.down_proj.shard_weights], dim=1), dense.down_proj.weight.data)

def test_tp_full_attention_equals_dense():
    torch.manual_seed(2)
    cfg = _tiny_cfg()
    dense = GatedFullAttention(cfg)
    dense.eval()
    pg = SimulatedProcessGroup(2)
    tp = TPFullAttention.from_dense(dense, world=2, process_group=pg)
    assert tp.q_heads_per_rank == [[0,1],[2,3]]
    assert tp.kv_heads_per_rank == [[0],[1]]
    x = torch.randn(2,4,32)
    with torch.no_grad():
        expected = dense(x)
        before = pg.all_reduce_calls
        out = tp(x)
    assert pg.all_reduce_calls - before == 1
    assert torch.allclose(out, expected, atol=1e-6), (out-expected).abs().max()

def test_tp_full_attention_kv_cache_equals_dense():
    torch.manual_seed(3)
    cfg = _tiny_cfg()
    dense = GatedFullAttention(cfg); dense.eval()
    tp = TPFullAttention.from_dense(dense, world=2)
    x = torch.randn(1,3,32)
    c1, c2 = KVCache(1), KVCache(1)
    with torch.no_grad():
        exp = dense(x, kv_cache=c1, layer_idx=0)
        out = tp(x, kv_cache=c2, layer_idx=0)
        # one decode token through both caches
        x1 = torch.randn(1,1,32)
        exp1 = dense(x1, kv_cache=c1, layer_idx=0, start_pos=3)
        out1 = tp(x1, kv_cache=c2, layer_idx=0, start_pos=3)
    assert torch.allclose(out, exp, atol=1e-6)
    assert torch.allclose(out1, exp1, atol=1e-6)

def test_linear_state_shard_roundtrip_and_tp_step_equals_dense():
    torch.manual_seed(4)
    net = GatedDeltaNet(d_model=32, num_key_heads=2, key_dim=8, num_value_heads=4, value_dim=8, conv_kernel=4)
    net.eval()
    x0 = torch.randn(2,32)
    with torch.no_grad():
        _, state = net.step(x0, None)
        shards = [shard_linear_state(state, r, 2) for r in range(2)]
        assert shards[0][0].shape == (2,2,8,8)
        # conv per rank: kh_per=1 -> q8+k8+v16=32 channels
        assert shards[0][1].shape == (2,32,3)
        merged = merge_linear_state(shards)
        assert torch.equal(merged[0], state[0]) and torch.equal(merged[1], state[1])
        # per-rank step on next token
        tp = TPGatedDeltaNet.from_dense(net, world=2)
        x1 = torch.randn(2,32)
        exp_out, exp_state = net.step(x1, state)
        tp_out, tp_rank_states = tp.step(x1, shards)
        tp_merged = tp.merge_state(tp_rank_states)
    assert torch.allclose(tp_out, exp_out, atol=1e-6), (tp_out-exp_out).abs().max()
    assert torch.allclose(tp_merged[0], exp_state[0], atol=1e-6)
    assert torch.allclose(tp_merged[1], exp_state[1], atol=1e-6)
    assert tp.process_group.all_reduce_calls == 1
    # also dense-pair input form and [B,1,D] shape
    with torch.no_grad():
        out3, _ = tp.step(x1.unsqueeze(1), state)
    assert out3.shape == (2,1,32)

def test_state_bytes_real_config_formulas():
    r4096 = state_bytes(seq_len=4096, batch=1, kv_dtype_bytes=2, world=2)
    # real: 16 full, Hkv/world=2, head 256: 16*2*1*4096*2*256*2
    assert r4096["full_kv_bytes_per_rank"] == 16*2*4096*2*256*2
    assert r4096["full_layers"] == 16 and r4096["linear_layers"] == 48
    # linear: rec 48*1*24*128*128*4 ; conv 48*1*5120*3*2
    assert r4096["linear_recurrent_bytes_per_rank"] == 48*24*128*128*4
    assert r4096["linear_conv_bytes_per_rank"] == 48*5120*3*2
    assert r4096["conv_dim_per_rank"] == 5120
    table = state_bytes_table()
    assert [r["seq_len"] for r in table] == [4096,32768,262144]
    assert table[0]["linear_bytes_per_rank"] == table[1]["linear_bytes_per_rank"] == table[2]["linear_bytes_per_rank"]
    assert table[2]["full_kv_bytes_per_rank"] == 64 * table[0]["full_kv_bytes_per_rank"]
    # world=1 doubles per-rank KV and linear vs world=2
    r_w1 = state_bytes(seq_len=4096, world=1)
    assert r_w1["full_kv_bytes_per_rank"] == 2*r4096["full_kv_bytes_per_rank"]
    assert r_w1["linear_bytes_per_rank"] == 2*r4096["linear_bytes_per_rank"]
    # batch scales linearly
    assert state_bytes(seq_len=4096, batch=2, world=2)["total_bytes_per_rank"] == 2*r4096["total_bytes_per_rank"]
