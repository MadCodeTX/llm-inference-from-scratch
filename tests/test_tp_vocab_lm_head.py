"""Vocab-parallel lm_head: sharding, padding, greedy fast path, loader."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state  # noqa: E402
from inference_from_scratch.tp_distributed import (  # noqa: E402
    DistributedHybridLM, vocab_shard_layout)


class _StubCtx:
    def __init__(self, rank, world=2):
        self.rank, self.world_size = rank, world
        self.device = torch.device("cpu")


def _cfg(vocab=96, **kw):
    base = dict(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)
    base.update(kw)
    return HybridConfig(**base)


def test_vocab_layout_padding():
    assert vocab_shard_layout(248320, 2) == (248320, 124160)
    assert vocab_shard_layout(97, 2) == (98, 49)
    assert vocab_shard_layout(96, 2) == (96, 48)


def test_from_dense_vocab_shards_only_lm_head():
    torch.manual_seed(0)
    cfg = _cfg(96)
    dense = TinyHybridLM(cfg).eval()
    for rank in range(2):
        tp = DistributedHybridLM.from_dense(dense, _StubCtx(rank), cfg, lm_head_mode="vocab")
        assert tuple(tp.lm_head.weight.shape) == (48, 32)
        s = tp.vocab_start
        assert torch.equal(tp.lm_head.weight.data, dense.lm_head.weight.data[s:s + 48])
        # Embeddings stay replicated.
        assert torch.equal(tp.embed_tokens.weight.data, dense.embed_tokens.weight.data)
    # Replicated default unchanged.
    tp = DistributedHybridLM.from_dense(dense, _StubCtx(0), cfg)
    assert tuple(tp.lm_head.weight.shape) == (96, 32)


def test_uneven_vocab_padded_shard_and_tied_error():
    cfg = _cfg(97)
    torch.manual_seed(0)
    dense = TinyHybridLM(cfg).eval()
    tp1 = DistributedHybridLM.from_dense(dense, _StubCtx(1), cfg, lm_head_mode="vocab")
    assert tuple(tp1.lm_head.weight.shape) == (49, 32)
    assert tp1.vocab_end == 97 and tp1.vocab_padded == 98
    assert torch.equal(tp1.lm_head.weight.data[:48], dense.lm_head.weight.data[49:97])
    assert torch.equal(tp1.lm_head.weight.data[48], torch.zeros(32))
    with pytest.raises(ValueError, match="untied"):
        DistributedHybridLM.from_dense(
            dense, _StubCtx(0), _cfg(96, tie_word_embeddings=True), lm_head_mode="vocab")
    with pytest.raises(ValueError):
        DistributedHybridLM.from_dense(dense, _StubCtx(0), cfg, lm_head_mode="bogus")


def test_greedy_next_token_world1_matches_dense_argmax():
    torch.manual_seed(0)
    cfg = _cfg(97)
    dense = TinyHybridLM(cfg).eval()
    tp = DistributedHybridLM.from_dense(dense, _StubCtx(0, world=1), cfg, lm_head_mode="vocab")
    hidden = torch.randn(3, 32)
    tok, val = tp.greedy_next_token(hidden)
    ref_logits = torch.nn.functional.linear(hidden, dense.lm_head.weight.data)
    assert torch.equal(tok, ref_logits.argmax(-1))
    assert torch.allclose(val, ref_logits.max(-1).values)


def test_loader_vocab_slices_lm_head(tmp_path):
    from safetensors.torch import save_file
    from inference_from_scratch.hybrid import HybridLM
    from inference_from_scratch.hybrid_checkpoint import expected_language_tensors, load_into_model
    from inference_from_scratch.quantization import quantize_block_fp8

    def ck_name(target):
        if target == "embed_tokens.weight":
            return "model.language_model.embed_tokens.weight"
        if target == "norm.weight":
            return "model.language_model.norm.weight"
        if target == "lm_head.weight":
            return "lm_head.weight"
        if target.startswith("mtp."):
            return target
        return f"model.language_model.{target}"

    cfg = _cfg(97, num_hidden_layers=4,
               layer_types=["linear_attention", "linear_attention",
                            "linear_attention", "full_attention"],
               mtp_num_hidden_layers=1, linear_conv_kernel_dim=4,
               linear_key_head_dim=8, linear_value_head_dim=8)
    # _cfg already sets compatible dims; rebuild via HybridConfig directly.
    torch.manual_seed(7)
    src = HybridLM(cfg).to(torch.bfloat16).eval()
    fp8_ckpts = {n for n, s in expected_language_tensors(cfg).items() if s["kind"] == "fp8_weight"}
    tensors = {}
    for target, tensor in src.state_dict().items():
        ck = ck_name(target)
        if ck in fp8_ckpts:
            q, scale = quantize_block_fp8(tensor.detach().to(torch.float32))
            tensors[ck] = q.contiguous()
            tensors[ck + "_scale_inv"] = scale.to(torch.bfloat16).contiguous()
        else:
            tensors[ck] = tensor.detach().to(torch.bfloat16).contiguous()
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    (tmp_path / "config.json").write_text('{"text_config": {}}')
    torch.manual_seed(99)
    dense = HybridLM(cfg).to(torch.bfloat16).eval()
    load_into_model(dense, tmp_path, strict=True)
    for rank in range(2):
        model, report = load_rank_state(
            tmp_path, cfg, rank=rank, world=2, fp8_mode="resident",
            lm_head_mode="vocab", use_scale_cache=True)
        assert report["lm_head_mode"] == "vocab" and report["scale_cache_modules"] > 0
        assert tuple(model.lm_head.weight.shape) == (49, 32)
        s = model.vocab_start
        valid = model.vocab_end - s
        assert torch.equal(model.lm_head.weight.data[:valid],
                           dense.lm_head.weight.data[s:model.vocab_end])
        if valid < 49:
            assert torch.equal(model.lm_head.weight.data[valid:], torch.zeros(49 - valid, 32, dtype=model.lm_head.weight.dtype))
        # Embeddings replicated, lm_head sharded: resident < replicated load.
        assert torch.equal(model.embed_tokens.weight.data, dense.embed_tokens.weight.data)


def test_tp2_model_check_vocab_gloo():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_model_check.py"),
         "--backend", "gloo", "--device", "cpu", "--lm-head", "vocab"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_MODEL_CHECK PASS" in proc.stdout
    assert "lm_head_mode=vocab" in proc.stdout
    assert "prefill_greedy" in proc.stdout


def test_tp2_checkpoint_check_vocab_scale_cache_gloo():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp2_checkpoint_check.py"),
         "--backend", "gloo", "--device", "cpu", "--lm-head", "vocab",
         "--scale-cache"],
        capture_output=True, text=True, timeout=240, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP2_CHECKPOINT_CHECK PASS" in proc.stdout
    assert "lm_head_mode=vocab" in proc.stdout
    assert "greedy tokens equal" in proc.stdout
