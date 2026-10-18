"""Per-rank checkpoint loading: FP8 pair slicing, rank loader, coverage.

The fixture is a tiny dense HybridLM checkpoint whose FP8-target
projections are stored block-FP8 (assembly-fixture style). The
subprocess test spawns ``scripts/tp2_checkpoint_check.py`` under
2-process gloo and compares generation against the dense converted
model; it measured a few seconds locally, so it stays durable.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.fp8_linear import BlockFP8Linear, shard_fp8_pair  # noqa: E402
from inference_from_scratch.hybrid import HybridConfig, HybridLM  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import (  # noqa: E402
    expected_language_tensors, load_into_model)
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8  # noqa: E402
from inference_from_scratch.tensor_parallel import shard_state_dict  # noqa: E402
from inference_from_scratch.tp_checkpoint import (  # noqa: E402
    load_rank_state, rank_shard_summary, resident_memory_report,
    validate_rank_coverage)
from inference_from_scratch.tp_distributed import TPBlockFP8Column, TPBlockFP8Row  # noqa: E402


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, partial_rotary_factor=0.25,
        rope_theta=10000.0, mtp_num_hidden_layers=1)


def _ckpt_name(target: str) -> str:
    if target == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target == "norm.weight":
        return "model.language_model.norm.weight"
    if target == "lm_head.weight":
        return "lm_head.weight"
    if target.startswith("mtp."):
        return target
    return f"model.language_model.{target}"


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    cfg = _tiny_cfg()
    root = tmp_path_factory.mktemp("tiny_fp8_ckpt")
    torch.manual_seed(7)
    src = HybridLM(cfg).to(torch.bfloat16).eval()
    fp8_ckpts = {n for n, s in expected_language_tensors(cfg).items()
                 if s["kind"] == "fp8_weight"}
    tensors: dict[str, torch.Tensor] = {}
    for target, tensor in src.state_dict().items():
        ck = _ckpt_name(target)
        if ck in fp8_ckpts:
            q, scale = quantize_block_fp8(tensor.detach().to(torch.float32))
            tensors[ck] = q.contiguous()
            tensors[ck + "_scale_inv"] = scale.to(torch.bfloat16).contiguous()
        else:
            tensors[ck] = tensor.detach().to(torch.bfloat16).contiguous()
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text('{"text_config": {}}')
    torch.manual_seed(99)
    dense = HybridLM(cfg).to(torch.bfloat16).eval()
    load_into_model(dense, root, strict=True)
    return root, cfg, dense


def test_shard_fp8_pair_dequant_equals_dense_shard(fixture):
    _, cfg, dense = fixture
    sd = dict(dense.state_dict())
    names = [
        "layers.0.mlp.gate_proj.weight",       # column
        "layers.0.mlp.down_proj.weight",       # row
        "layers.3.self_attn.q_proj.weight",    # head_column
        "layers.0.linear_attn.in_proj_qkv.weight",  # segmented
        "layers.0.linear_attn.out_proj.weight",     # row (value heads)
    ]
    for name in names:
        dense_w = sd[name]
        q, scale = quantize_block_fp8(dense_w.to(torch.float32))
        scale = scale.to(torch.bfloat16)
        deq = dequantize_block_fp8(q, scale, out_dtype=torch.bfloat16)
        for rank in range(2):
            w_shard, s_shard = shard_fp8_pair(name, q, scale, rank, 2, cfg)
            got = dequantize_block_fp8(w_shard, s_shard, out_dtype=torch.bfloat16)
            expected = shard_state_dict({name: deq}, rank, 2, cfg)[name]
            assert torch.equal(got, expected), (name, rank)
            mod = BlockFP8Linear.from_pre_sliced(w_shard, s_shard)
            assert tuple(mod.weight.shape) == tuple(w_shard.shape)


class _StubCtx:
    def __init__(self, rank=0, world=1):
        self.rank, self.world_size = rank, world
        self.device = torch.device("cpu")

    def all_reduce_sum(self, tensor):
        return tensor

    def all_gather_concat(self, tensor, dim=0):
        return tensor


def test_tp_block_fp8_column_row_semantics():
    torch.manual_seed(0)
    dense = torch.nn.Linear(256, 512, bias=False)
    q, scale = quantize_block_fp8(dense.weight.data.float())
    scale = scale.to(torch.bfloat16)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    # Column: rank shards stay sharded; gathered == dense FP8 output.
    cols = []
    for rank in range(2):
        w_s, s_s = shard_fp8_pair("mlp.gate_proj.weight", q, scale, rank, 2, None)
        cols.append(TPBlockFP8Column(BlockFP8Linear.from_pre_sliced(w_s, s_s), _StubCtx(rank, 2)))
    local = cols[0](x)
    assert local.shape == (4, 256)
    gathered = torch.cat([c(x) for c in cols], dim=-1)
    ref = BlockFP8Linear.from_pre_sliced(q, scale)(x)
    assert torch.equal(gathered, ref)
    # Row: local partials summed (the one all-reduce, emulated here
    # with an explicit sum) == dense FP8 output.
    row_dense = torch.nn.Linear(512, 256, bias=False)
    rq, rscale = quantize_block_fp8(row_dense.weight.data.float())
    rscale = rscale.to(torch.bfloat16)
    xr = torch.randn(4, 512, dtype=torch.bfloat16)
    rows = []
    for rank in range(2):
        w_s, s_s = shard_fp8_pair("mlp.down_proj.weight", rq, rscale, rank, 2, None)
        rows.append(TPBlockFP8Row(BlockFP8Linear.from_pre_sliced(w_s, s_s),
                                  _StubCtx(rank, 2), col_start=rank * 256))
    summed = rows[0].local(xr[:, :256]) + rows[1].local(xr[:, 256:])
    ref_row = BlockFP8Linear.from_pre_sliced(rq, rscale)(xr)
    # Row partials re-associate BF16 addition (docs/23): fp tolerance.
    assert torch.allclose(summed.float(), ref_row.float(), atol=1e-2, rtol=1e-2)


def test_load_rank_state_holds_only_shards(fixture):
    root, cfg, dense = fixture
    dense_sd = dict(dense.state_dict())
    for rank in range(2):
        model, report = load_rank_state(root, cfg, rank=rank, world=2, fp8_mode="resident")
        assert report["missing"] == [] and report["not_loaded"] == []
        # Replicated: embedding/lm_head whole and equal on every rank.
        assert torch.equal(model.embed_tokens.weight.data, dense_sd["embed_tokens.weight"])
        assert torch.equal(model.lm_head.weight.data, dense_sd["lm_head.weight"])
        # Sharded FP8: dequantised shard == dense converted shard exactly.
        local = model.layers[0].mlp.gate_proj.local
        assert tuple(local.weight.shape) == (32, 32)
        got = dequantize_block_fp8(local.weight, local.weight_scale_inv, out_dtype=torch.bfloat16)
        exp = shard_state_dict(
            {"layers.0.mlp.gate_proj.weight": dense_sd["layers.0.mlp.gate_proj.weight"]},
            rank, 2, cfg)["layers.0.mlp.gate_proj.weight"]
        assert torch.equal(got, exp)
        assert report["resident_bytes"] < int(
            sum(t.numel() * t.element_size() for t in dense_sd.values()))


def test_coverage_and_memory_report(fixture):
    root, cfg, _ = fixture
    cov = validate_rank_coverage(root, cfg, world=2)
    assert cov["ok"] is True, cov["errors"][:3]
    assert cov["per_rank"][0]["count"] == cov["per_rank"][1]["count"] > 0
    s0 = rank_shard_summary(root, cfg, rank=0, world=2)
    assert s0["count"] == cov["per_rank"][0]["count"] and len(s0["hash"]) == 64
    mem = resident_memory_report(root, cfg, world=2)
    row = mem["per_rank_bytes"][0]
    assert row["total"] == (row["fp8_weight"] + row["scale_inv"]
                            + row["bf16_sharded"] + row["bf16_replicated"])
    assert mem["estimate_total_gib"] > 0 and mem["explanations"]


def _real_snapshot():
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots")
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


@pytest.mark.skipif(_real_snapshot() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_real_snapshot_memory_report_and_coverage():
    snap = _real_snapshot()
    mem = resident_memory_report(snap, world=2)
    # Measured on the real manifest: 16.362 GiB vs 16.364 GiB estimate.
    assert 16.0 < mem["per_rank_total_gib"] < 17.0
    assert 16.0 < mem["estimate_total_gib"] < 17.0
    assert abs(mem["delta_gib"]) < 0.1
    assert mem["fits_24gib_per_rank"] is True
    cov = validate_rank_coverage(snap, world=2)
    assert cov["ok"] is True and cov["missing"] == []
    assert cov["per_rank"][0]["count"] == 1273


def test_tp2_checkpoint_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_checkpoint_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_CHECKPOINT_CHECK PASS" in proc.stdout
    assert "greedy tokens equal" in proc.stdout
