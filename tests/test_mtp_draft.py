"""MTPDraftModule / TPMTPDraftModule: draft step, hidden capture, loader."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, HybridLM, TinyHybridLM  # noqa: E402
from inference_from_scratch.hybrid_checkpoint import expected_language_tensors  # noqa: E402
from inference_from_scratch.mtp import (  # noqa: E402
    LEADING_WIRING, MTPConfig, MTPDraftModule, MTPHead, TPMTPDraftModule,
    mtp_state_length, resolve_wiring)
from inference_from_scratch.quantization import dequantize_block_fp8, quantize_block_fp8  # noqa: E402
from inference_from_scratch.tensor_parallel import shard_state_dict  # noqa: E402
from inference_from_scratch.tp_checkpoint import load_rank_state, rank_shard_summary  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedHybridLM  # noqa: E402

D, VOCAB = 32, 97


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
        mtp_num_hidden_layers=1)
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _module(seed=0):
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    mod = MTPDraftModule(MTPConfig.from_hybrid_config(_cfg()),
                         embedding=emb, lm_head=lm_head).eval()
    return mod, emb, lm_head


def test_leading_wiring_is_sweep_leader_and_validated():
    assert LEADING_WIRING["fc_concat_order"] == "embedding_first"
    assert resolve_wiring() == LEADING_WIRING
    assert resolve_wiring({"position_rule": "predicted_pos"})["position_rule"] == "predicted_pos"
    with pytest.raises(ValueError):
        resolve_wiring({"fc_concat_order": "bogus"})
    with pytest.raises(ValueError):
        resolve_wiring({"embedding_source": "dedicated_mtp_embedding"})


def test_draft_token_shapes_topk_and_determinism():
    mod, _emb, _lm = _module(seed=1)
    hidden = torch.randn(2, D)
    with torch.no_grad():
        tok_a, logits_a, state_a = mod.draft_token(hidden, torch.tensor([3, 4]), position=5)
        tok_b, logits_b, _ = mod.draft_token(hidden, torch.tensor([3, 4]), position=5)
        tok1, (vals, idx), _ = mod.draft_token(hidden[0], 3, position=5, top_k=3)
    assert tok_a.shape == (2,) and logits_a.shape == (2, VOCAB)
    assert torch.equal(tok_a, tok_b) and torch.equal(logits_a, logits_b)
    assert torch.equal(tok_a, logits_a.argmax(dim=-1))
    assert vals.shape == (3,) and int(idx[0]) == int(tok1) == int(tok_a[0])
    assert mtp_state_length(state_a) == 1
    with pytest.raises(ValueError):
        MTPDraftModule(MTPConfig.from_hybrid_config(_cfg())).draft_token(hidden, 3)


def test_draft_state_advances_only_on_commit():
    mod, _emb, _lm = _module(seed=2)
    hidden = torch.randn(2, D)
    with torch.no_grad():
        s0 = mod.initial_state()
        assert mtp_state_length(s0) == 0
        tok_a, logits_a, s1 = mod.draft_token(hidden, torch.tensor([3, 4]), s0, position=0)
        # Input state untouched: replaying from it is bit-identical.
        assert mtp_state_length(s0) == 0
        tok_b, logits_b, _ = mod.draft_token(hidden, torch.tensor([3, 4]), s0, position=0)
        assert torch.equal(tok_a, tok_b) and torch.equal(logits_a, logits_b)
        # Reject = keep s0 (above). Accept = commit s1, length grows.
        assert mtp_state_length(s1) == 1
        snap = MTPDraftModule.snapshot_state(s1)
        _t, _l, s2 = mod.draft_token(hidden, torch.tensor([5, 6]), s1, position=1)
        assert mtp_state_length(s2) == 2
        restored = MTPDraftModule.restore_state(snap)
        assert mtp_state_length(restored) == 1
        tok_c, logits_c, _ = mod.draft_token(hidden, torch.tensor([5, 6]), restored, position=1)
        tok_d, logits_d, _ = mod.draft_token(hidden, torch.tensor([5, 6]), s1, position=1)
        assert torch.equal(tok_c, tok_d) and torch.equal(logits_c, logits_d)


def test_tiny_hidden_capture_matches_manual_final_norm_input():
    torch.manual_seed(3)
    model = TinyHybridLM(_cfg()).eval()
    idx = torch.tensor([[3, 11, 5, 7]])
    with torch.no_grad():
        plain_logits, plain_state = model(idx)
        assert isinstance(plain_logits, torch.Tensor)
        logits, state, hidden_pre, hidden_post = model.forward_with_hidden(idx)
        logits2, state2, hidden2 = model.prefill(idx, return_hidden=True)
        # Manual recomputation of the final-norm input: embed + blocks.
        x = model.embed_tokens(idx)
        manual_state = model.initial_state()
        for i, block in enumerate(model.layers):
            tag, payload = manual_state.entry(i)
            payload = payload if tag == "kv" else (
                None if payload is None else payload)
            if block.layer_type == "full_attention":
                x, kv = block.forward_with_state(x, payload, start_pos=0, layer_idx=i)
                from inference_from_scratch.kv_cache import KVCache
                cache = KVCache(1)
                cache.keys[0], cache.values[0] = kv
                manual_state.entries[i] = ("kv", cache)
            else:
                x, lin = block.forward_with_state(x, payload, start_pos=0, layer_idx=i)
                manual_state.entries[i] = ("linear", lin)
    assert torch.equal(logits, plain_logits)
    assert torch.equal(hidden_pre, x)  # equals manual final-norm input
    assert torch.equal(hidden_post, model.norm(hidden_pre))
    assert torch.equal(model.lm_head(hidden_post), logits)
    assert torch.equal(logits2, logits) and torch.equal(hidden2, hidden_pre)
    with torch.no_grad():
        step_logits, step_state, step_hidden = model.decode_step(
            torch.tensor([9]), state, return_hidden=True)
        ref_logits, _ref_state = model.decode_step(torch.tensor([9]), state)
    assert torch.equal(step_logits, ref_logits) and step_hidden.shape == (1, 1, D)


class _World1Ctx:
    rank, world_size = 0, 1
    device = torch.device("cpu")
    backend = "none"

    def all_reduce_sum(self, tensor):
        return tensor

    def all_gather(self, tensor):
        return [tensor]

    def all_gather_concat(self, tensor, dim=0):
        return tensor


def test_distributed_hidden_capture_matches_dense():
    torch.manual_seed(4)
    dense = TinyHybridLM(_cfg()).eval()
    tp = DistributedHybridLM.from_dense(dense, _World1Ctx())
    idx = torch.tensor([[3, 11, 5]])
    with torch.no_grad():
        d_logits, _ds, d_pre, d_post = dense.forward_with_hidden(idx)
        t_logits, _ts, t_pre, t_post = tp.forward_with_hidden(idx)
        plain = tp.prefill(idx)
        assert len(plain) == 2  # default off: unchanged 2-tuple
        s_logits, s_state, s_hidden = tp.step(torch.tensor([7]), _ts, return_hidden=True)
    assert torch.allclose(t_logits.float(), d_logits.float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(t_pre.float(), d_pre.float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(t_post.float(), dense.norm(d_pre).float(), atol=1e-5, rtol=1e-5)
    assert s_hidden.shape == (1, 1, D)


def test_tp_draft_module_world1_matches_dense():
    mod, emb, lm_head = _module(seed=5)
    tp = TPMTPDraftModule.from_dense(_World1Ctx(), mod.head, embedding=emb,
                                     lm_head=lm_head).eval()
    # fc replicated whole; block q shard is the full width at world 1.
    assert tuple(tp.fc.weight.shape) == (D, 2 * D)
    assert torch.equal(tp.fc.weight.data, mod.head.fc.weight.data)
    hidden = torch.randn(2, D)
    with torch.no_grad():
        d_id, d_logits, _ = mod.draft_token(hidden, torch.tensor([3, 4]), position=2)
        t_id, t_logits, t_state = tp.draft_token(hidden, torch.tensor([3, 4]), position=2)
    assert torch.allclose(t_logits.float(), d_logits.float(), atol=1e-5, rtol=1e-5)
    assert torch.equal(t_id, d_id) and mtp_state_length(t_state) == 1


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
def mtp_fixture(tmp_path_factory):
    cfg = _cfg()
    root = tmp_path_factory.mktemp("tiny_mtp_ckpt")
    torch.manual_seed(7)
    src = HybridLM(cfg).to(torch.bfloat16).eval()
    torch.manual_seed(8)
    head = MTPHead(MTPConfig.from_hybrid_config(cfg)).to(torch.bfloat16).eval()
    fp8_ckpts = {n for n, s in expected_language_tensors(cfg).items()
                 if s["kind"] == "fp8_weight"}
    tensors: dict[str, torch.Tensor] = {}
    items = list(src.state_dict().items()) + [
        (f"mtp.{k}", v) for k, v in head.state_dict().items()]
    for target, tensor in items:
        ck = _ckpt_name(target)
        if ck in fp8_ckpts:
            q, scale = quantize_block_fp8(tensor.detach().to(torch.float32))
            tensors[ck] = q.contiguous()
            tensors[ck + "_scale_inv"] = scale.to(torch.bfloat16).contiguous()
        else:
            tensors[ck] = tensor.detach().to(torch.bfloat16).contiguous()
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text('{"text_config": {}}')
    return root, cfg, head


def test_tp_checkpoint_include_mtp_loads_draft_module(mtp_fixture):
    root, cfg, head = mtp_fixture
    for rank in range(2):
        model, report = load_rank_state(root, cfg, rank=rank, world=2,
                                        fp8_mode="resident", include_mtp=True)
        assert report["include_mtp"] is True
        assert report["mtp_loaded_count"] == 22  # 15 weights + 7 scales
        assert model.mtp_draft is not None
        # fc replicated: whole dense weight on every rank.
        got_fc = model.mtp_draft.fc.weight.data
        assert tuple(got_fc.shape) == (D, 2 * D)
        assert torch.equal(got_fc, head.fc.weight.data)
        # Block q shard dequantises to the dense shard exactly.
        local = model.mtp_draft.layers[0].self_attn.q_fp8
        deq = dequantize_block_fp8(local.weight, local.weight_scale_inv,
                                   out_dtype=torch.bfloat16)
        q_full, s_full = quantize_block_fp8(
            head.layers[0].self_attn.q_proj.weight.data.to(torch.float32))
        deq_full = dequantize_block_fp8(q_full, s_full.to(torch.bfloat16),
                                        out_dtype=torch.bfloat16)
        exp = shard_state_dict(
            {"mtp.layers.0.self_attn.q_proj.weight": deq_full},
            rank, 2, cfg)["mtp.layers.0.self_attn.q_proj.weight"]
        assert torch.equal(deq, exp)
    model2, report2 = load_rank_state(root, cfg, rank=0, world=2,
                                      fp8_mode="dequantize", include_mtp=True)
    assert report2["mtp_loaded_count"] == 15  # dense mode: no scale yields
    assert model2.mtp_draft is not None
    plain, _rep = load_rank_state(root, cfg, rank=0, world=2, fp8_mode="resident")
    assert plain.mtp_draft is None  # default off


def _real_snapshot():
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots")
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "mtp.safetensors").is_file():
                return snap
    return None


@pytest.mark.skipif(_real_snapshot() is None, reason="Qwen3.8 snapshot not in HF cache")
def test_real_manifest_mtp_coverage_count():
    summary = rank_shard_summary(_real_snapshot(), rank=0, world=2)
    mtp_shapes = {k: v for k, v in summary["shapes"].items() if k.startswith("mtp.")}
    assert len(mtp_shapes) == 22  # real mtp.* coverage, headers only
    assert mtp_shapes["mtp.fc.weight"] == [5120, 10240]  # replicated


def test_mtp_tp_check_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/mtp_tp_check.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "MTP_TP_CHECK PASS" in proc.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NCCL check needs CUDA")
def test_mtp_tp_check_nccl_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/mtp_tp_check.py"),
         "--backend", "nccl", "--device", "cuda"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "MTP_TP_CHECK PASS" in proc.stdout
