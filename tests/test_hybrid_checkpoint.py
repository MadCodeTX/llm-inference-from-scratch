"""Tests for hybrid_checkpoint: synthetic multi-shard fixtures + real guard."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inference_from_scratch.hybrid import HybridConfig
from inference_from_scratch.hybrid_checkpoint import (
    build_manifest, discover_shards, expected_language_tensors,
    iter_layer_tensors, load_tensor, validate_manifest,
)
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, partial_rotary_factor=0.25,
        rope_theta=10000.0, mtp_num_hidden_layers=1,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _bf16(shape):
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)


def _fp8_pair(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, dtype=torch.float32)
    q, scale = quantize_block_fp8(w.to(torch.bfloat16))
    return q, scale.to(torch.bfloat16)


def _write_fixture(root, config=None, mutate=None):
    """Two shards: shard-a = embed+layer0, shard-b = rest (by expected map)."""
    cfg = config or _tiny_config()
    expected = expected_language_tensors(cfg)
    tensors: dict[str, torch.Tensor] = {}
    ref: dict[str, torch.Tensor] = {}
    for idx, (name, spec) in enumerate(sorted(expected.items())):
        shape = spec["shape"]
        if spec["kind"] == "fp8_weight":
            continue  # created with its scale below
        if name.endswith("weight_scale_inv"):
            continue
        tensors[name] = _bf16(shape)
    # one explicit FP8 pair (layer-0 qkv) replaces the placeholder path
    fp8_names = [n for n, s in expected.items() if s["kind"] == "fp8_weight"]
    for i, name in enumerate(fp8_names):
        q, scale = _fp8_pair(expected[name]["shape"], seed=100 + i)
        tensors[name] = q
        tensors[name + "_scale_inv"] = scale
        ref[name] = dequantize_block_fp8(q, scale.float(), out_dtype=torch.bfloat16)
    if mutate:
        tensors = mutate(tensors)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = max(1, len(names) // 2)
    save_file({k: tensors[k].contiguous() for k in names[:half]},
              str(root / "shard-a.safetensors"), metadata={"format": "pt"})
    save_file({k: tensors[k].contiguous() for k in names[half:]},
              str(root / "shard-b.safetensors"), metadata={"format": "pt"})
    # minimal config.json so HybridConfig.from_text_config(dir) also works
    (root / "config.json").write_text('{"text_config": {}}')
    return tensors, ref


def test_manifest_discovery_multi_shard(tmp_path):
    _write_fixture(tmp_path)
    shards = discover_shards(tmp_path)
    assert [p.name for p in shards] == ["shard-a.safetensors", "shard-b.safetensors"]
    man = build_manifest(tmp_path)
    assert len(man.entries) == len(expected_language_tensors(_tiny_config()))
    assert man.counts_by_kind()["fp8_weight"] > 0
    assert man.counts_by_kind()["scale_inv"] > 0
    assert man.counts_by_kind()["bf16"] > 0
    e = man.by_name()["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    assert e.kind == "fp8_weight" and e.dtype == "F8_E4M3" and e.bytes > 0
    assert e.target_name == "layers.0.linear_attn.in_proj_qkv.weight"
    assert e.shard in ("shard-a.safetensors", "shard-b.safetensors")


def test_validation_ok_and_catches_missing_and_shape(tmp_path):
    cfg = _tiny_config()
    _write_fixture(tmp_path, cfg)
    assert validate_manifest(build_manifest(tmp_path), cfg)["ok"] is True

    def drop(t):
        t = dict(t)
        t.pop("model.language_model.layers.3.self_attn.q_norm.weight", None)
        t.pop("model.language_model.norm.weight", None)
        return t

    bad = tmp_path / "bad"
    _write_fixture(bad, cfg, mutate=drop)
    res = validate_manifest(build_manifest(bad), cfg)
    assert res["ok"] is False and len(res["missing"]) == 2

    def warp(t):
        t = dict(t)
        k = "model.language_model.layers.0.input_layernorm.weight"
        t[k] = torch.zeros(7, dtype=torch.bfloat16)
        t["model.language_model.layers.0.bogus.weight"] = torch.zeros(3, dtype=torch.bfloat16)
        return t

    bad2 = tmp_path / "bad2"
    _write_fixture(bad2, cfg, mutate=warp)
    res2 = validate_manifest(build_manifest(bad2), cfg)
    assert res2["ok"] is False
    assert any(m["checkpoint_name"].endswith("input_layernorm.weight")
               for m in res2["shape_mismatch"])
    assert any("bogus" in u for u in res2["unexpected"])


def test_lazy_fp8_dequant_equals_reference(tmp_path):
    _, ref = _write_fixture(tmp_path)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    got = load_tensor(tmp_path, name)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref[name])
    raw = load_tensor(tmp_path, name, dequantize=False)
    assert raw.dtype == torch.float8_e4m3fn
    with pytest.raises(ValueError, match="scale"):
        load_tensor(tmp_path, name + "_scale_inv")
    with pytest.raises((KeyError, FileNotFoundError), match="unknown|no tensor|does.not.exist"):
        load_tensor(tmp_path, "model.language_model.layers.0.does.not.exist.weight")


def test_iteration_scope(tmp_path):
    _write_fixture(tmp_path)
    items = list(iter_layer_tensors(tmp_path, 0))
    names = [n for n, _ in items]
    assert names and all(n.startswith("layers.0.") for n in names)
    assert not any(n.startswith("layers.1.") for n in names)
    assert not any(n.endswith("weight_scale_inv") for n in names)
    other = list(iter_layer_tensors(tmp_path, 3))
    assert other and all(n.startswith("layers.3.") for n, _ in other)


def _real_snapshot():
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots")
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


@pytest.mark.skipif(_real_snapshot() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_real_snapshot_counts_only():
    snap = _real_snapshot()
    cfg = HybridConfig.from_text_config(snap)
    man = build_manifest(snap)
    assert len(man.entries) == 1606
    res = validate_manifest(man, cfg)
    assert res["counts"]["expected"] == 1273  # 1250 language + MTP/lm_head scope
    assert res["ok"] is True, res
