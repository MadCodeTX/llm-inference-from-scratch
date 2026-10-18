"""Tests for Ch 07/08: DecoderLM (model.py) and generate() (generate.py).

Cache-vs-no-cache greedy identity is the load-bearing test: with
temperature=0 there is no randomness anywhere, so any token
divergence between the two paths is a cache / mask / position bug.
Everything runs on tiny CPU dims per the repo contract
(d_model<=64, n_layer<=2) and finishes in well under a second.
"""

import torch
import torch.nn.functional as F
from torch import nn

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.generate import generate
from inference_from_scratch.kv_cache import KVCache
from inference_from_scratch.model import DecoderLM, TransformerBlock


def _config(**overrides):
    kwargs = dict(
        vocab_size=48,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_ff=64,
        max_seq_len=64,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return DecoderLM(_config(**overrides)).eval()


# --- DecoderLM forward / structure ------------------------------------------


def test_forward_logits_shape_rope_and_learned():
    for pos_type in ("rope", "learned"):
        model = _tiny_model(pos_type=pos_type)
        idx = torch.randint(0, 48, (2, 7))
        with torch.no_grad():
            logits, caches = model(idx)
        assert logits.shape == (2, 7, 48)
        assert caches is None  # no cache in, no cache out


def test_forward_with_cache_returns_same_object_and_fills_it():
    model = _tiny_model()
    idx = torch.randint(0, 48, (1, 5))
    cache = KVCache(n_layer=2)
    with torch.no_grad():
        logits, out_cache = model(idx, kv_caches=cache, start_pos=0)
    assert logits.shape == (1, 5, 48)
    assert out_cache is cache
    assert cache.seq_len == 5


def test_learned_pos_variant_forward_and_decode_positions():
    # Learned-pos must add rows start_pos.. (not always 0..): a cached
    # token-by-token decode has to reproduce the full forward, which
    # only holds if decode steps pass their absolute position.
    torch.manual_seed(1)
    model = _tiny_model(pos_type="learned")
    assert model.pos_embedding is not None
    assert model.pos_embedding.weight.shape == (64, 32)
    idx = torch.randint(0, 48, (1, 6))
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(n_layer=2)
        steps = []
        for t in range(6):
            logits, cache = model(
                idx[:, t : t + 1], kv_caches=cache, start_pos=t,
            )
            steps.append(logits)
        stepped = torch.cat(steps, dim=1)
    assert torch.allclose(stepped, full, atol=1e-6)


def test_tied_embeddings_share_weight():
    model = _tiny_model(tie_embeddings=True)
    assert model.lm_head.weight is model.tok_embedding.weight
    # One shared [V, D] matrix, counted once.
    assert model.num_params() == sum(
        p.numel() for p in model.parameters()
    )


def test_untied_embeddings_have_separate_weights():
    model = _tiny_model(tie_embeddings=False)
    assert model.lm_head.weight is not model.tok_embedding.weight
    assert model.lm_head.weight.shape == (48, 32)
    assert model.tok_embedding.weight.shape == (48, 32)


def test_num_params_matches_state_dict_and_positive():
    model = _tiny_model()
    n = model.num_params()
    assert n > 0
    # parameters() deduplicates the tied head; state_dict lists the
    # shared [V, D] matrix under both names, so num_params (unique
    # tensors, buffers excluded) must be strictly smaller here.
    sd_total = sum(t.numel() for t in model.state_dict().values()
                   if t.dtype in (torch.float32, torch.float64))
    assert n == sum(p.numel() for p in model.parameters())
    assert n < sd_total  # tied lm_head.weight double-listed in state_dict


def test_transformer_block_pre_norm_residual_shape():
    torch.manual_seed(2)
    block = TransformerBlock(_config(n_layer=1)).eval()
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (2, 5, 32)
    # Pre-norm residual: zeroing both sublayers would return x itself;
    # with real weights the output must at least differ (not vacuous).
    assert not torch.allclose(y, x)


def test_norm_and_mlp_variants_forward():
    # layernorm + gelu (GPT-2 flavour) must run through the same code.
    model = _tiny_model(norm_type="layernorm", mlp_type="gelu",
                        pos_type="learned", tie_embeddings=False)
    idx = torch.randint(0, 48, (2, 4))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (2, 4, 48)
    assert torch.isfinite(logits).all()


# --- generate(): cache vs no-cache, length, EOS ------------------------------


def test_greedy_generate_cache_vs_no_cache_identical():
    model = _tiny_model(seed=3)
    prompt = torch.randint(0, 48, (2, 6))
    cached = generate(model, prompt.clone(), max_new_tokens=12,
                      temperature=0, use_cache=True)
    plain = generate(model, prompt.clone(), max_new_tokens=12,
                     temperature=0, use_cache=False)
    assert cached.shape == plain.shape == (2, 18)
    assert torch.equal(cached, plain)  # IDENTICAL token ids
    assert torch.equal(cached[:, :6], prompt)  # prompt preserved


def test_greedy_generate_identical_gqa_and_mqa_configs():
    for n_kv_head in (4, 1):
        model = _tiny_model(seed=4, n_kv_head=n_kv_head)
        prompt = torch.randint(0, 48, (1, 5))
        cached = generate(model, prompt, max_new_tokens=8,
                          temperature=0, use_cache=True)
        plain = generate(model, prompt, max_new_tokens=8,
                         temperature=0, use_cache=False)
        assert torch.equal(cached, plain)


def test_generate_respects_max_new_tokens():
    model = _tiny_model(seed=5)
    prompt = torch.randint(0, 48, (1, 4))
    for n in (0, 1, 5):
        out = generate(model, prompt, max_new_tokens=n, temperature=0)
        assert out.shape == (1, 4 + n)
    assert torch.equal(out[:, :4], prompt)


class _StubModel(nn.Module):
    """Fake LM: ignores input content, emits a scripted token cycle.

    Lets the EOS / length tests pin generate()'s loop behaviour
    without depending on a real model's (untrained) distribution.
    Emits token 7 twice, then EOS (3), then 7 forever -- so a correct
    generate() with eos_id=3 must stop after exactly 3 new tokens.
    """

    def __init__(self, vocab_size=16, eos_id=3):
        super().__init__()
        self.config = _config(vocab_size=vocab_size)
        self.eos_id = eos_id
        self._calls = 0
        # One real parameter so eval()/train() and .parameters() work.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, kv_caches=None, start_pos=0):
        self._calls += 1
        B, T = idx.shape
        logits = torch.full((B, T, self.config.vocab_size), -10.0)
        # Script by total generated count: calls 1,2 -> 7; call 3 -> EOS.
        token = 7 if self._calls <= 2 else self.eos_id
        if self._calls > 3:
            token = 7  # would continue forever without EOS handling
        logits[:, -1, token] = 10.0
        return logits, kv_caches


def test_generate_eos_stop_with_stub_model():
    stub = _StubModel()
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out = generate(stub, prompt, max_new_tokens=10, temperature=0,
                   eos_id=3, use_cache=True)
    # 3 prompt + exactly 3 generated (7, 7, EOS), not 10.
    assert out.shape == (1, 6)
    assert out[0, 3:].tolist() == [7, 7, 3]

    stub2 = _StubModel()
    out_nc = generate(stub2, prompt, max_new_tokens=10, temperature=0,
                      eos_id=3, use_cache=False)
    assert out_nc.shape == (1, 6)
    assert out_nc[0, 3:].tolist() == [7, 7, 3]


def test_generate_without_eos_uses_full_budget_with_stub():
    stub = _StubModel()
    prompt = torch.zeros(1, 2, dtype=torch.long)
    out = generate(stub, prompt, max_new_tokens=5, temperature=0,
                   eos_id=None, use_cache=True)
    assert out.shape == (1, 7)  # no early stop without eos_id


def test_generate_sampling_seeded_reproducible_and_in_vocab():
    model = _tiny_model(seed=6)
    prompt = torch.randint(0, 48, (1, 4))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = generate(model, prompt, max_new_tokens=8, temperature=0.9,
                 top_k=10, generator=g1)
    b = generate(model, prompt, max_new_tokens=8, temperature=0.9,
                 top_k=10, generator=g2)
    assert torch.equal(a, b)
    assert a.min() >= 0 and a.max() < 48
