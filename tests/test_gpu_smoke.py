"""CUDA smoke tests for the tiny hybrid stack (skip-gated).

Every test here skips when CUDA is unavailable, so the CPU suite stays
green without a GPU -- and a skip is never proof of anything. Run with
the GPU venv to actually exercise them:
``.venv-gpu/bin/python -m pytest tests/test_gpu_smoke.py -q``.
Dims/config mirror ``tests/test_hybrid.py``.
"""

import pytest
import torch

from inference_from_scratch.hybrid import (
    HybridConfig, TinyHybridLM, generate_hybrid,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available")

D, VOCAB = 32, 97
DEV = "cuda:0"


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


def _pair(seed=3):
    torch.manual_seed(seed)
    cpu = TinyHybridLM(_tiny_config()).eval()
    torch.manual_seed(seed)
    gpu = TinyHybridLM(_tiny_config()).eval().to(DEV)
    return cpu, gpu


def test_gpu_fp32_prefill_decode_matches_cpu():
    cpu, gpu = _pair()
    torch.manual_seed(1234)
    idx_cpu = torch.randint(0, VOCAB, (2, 7))
    idx = idx_cpu.to(DEV)
    with torch.no_grad():
        ref, _ = cpu(idx_cpu)
        got, _ = gpu(idx)
        _, state = gpu.prefill(idx[:, :4])
        steps = []
        for t in range(4, 7):
            logits, state = gpu.decode_step(idx[:, t], state)
            steps.append(logits)
        joined = torch.cat(steps, dim=1)
    assert (got.cpu() - ref).abs().max() < 1e-5
    assert torch.equal(got.argmax(-1).cpu(), ref.argmax(-1))
    assert (joined.cpu() - ref[:, 4:]).abs().max() < 1e-5


def test_gpu_static_decode_matches_legacy_and_defaults_to_model_device():
    _, gpu = _pair()
    idx = torch.randint(0, VOCAB, (2, 7), device=DEV)
    with torch.no_grad():
        state = gpu.initial_state(2)
        legacy = []
        for t in range(7):
            logits, state = gpu.step(idx[:, t], state)
            legacy.append(logits[:, 0, :])
        legacy = torch.stack(legacy, dim=1)
        st = gpu.static_state(batch_size=2, max_seq_len=16)  # no device given
        assert st.kv_cache.k_cache.device.type == "cuda"
        static = torch.stack(
            [gpu.decode_token_static(idx[:, t], st) for t in range(7)], dim=1)
    assert st.seq_len == 7
    assert (static - legacy).abs().max() < 1e-5


def test_gpu_chunkwise_matches_sequential():
    _, gpu = _pair()
    idx = torch.randint(0, VOCAB, (2, 7), device=DEV)
    with torch.no_grad():
        seq, _ = gpu(idx)
        chu, _ = gpu.prefill(idx, prefill_method="chunkwise", chunk_size=4)
    assert (chu - seq).abs().max() < 1e-5


def test_gpu_sampling_with_cpu_generator_is_reproducible():
    _, gpu = _pair()
    prompt = torch.randint(0, VOCAB, (2, 4), device=DEV)
    with torch.no_grad():
        a = generate_hybrid(gpu, prompt.clone(), 3, temperature=0.8,
                            top_k=10,
                            generator=torch.Generator().manual_seed(42))
        b = generate_hybrid(gpu, prompt.clone(), 3, temperature=0.8,
                            top_k=10,
                            generator=torch.Generator().manual_seed(42))
    assert a.device.type == "cuda"
    assert torch.equal(a, b)
