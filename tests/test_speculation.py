"""Tests for Ch 14 speculative decoding: exact target-greedy parity."""

import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.generate import generate
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.speculation import speculative_generate


def _config(**overrides):
    kwargs = dict(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=64,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return DecoderLM(_config()).eval()


def test_draft_is_target_all_accepted_fewer_target_forwards():
    model = _tiny_model(seed=3)
    prompt = torch.tensor([[4, 9, 2, 17]], dtype=torch.long)
    tokens, stats = speculative_generate(
        model, model, prompt.clone(), max_new_tokens=12, gamma=4
    )
    reference = generate(model, prompt.clone(), max_new_tokens=12, temperature=0)
    assert torch.equal(tokens, reference)
    assert tokens.shape == (1, 16)
    # Every drafted token agrees with itself; bonus tokens are extra.
    assert stats["accepted"] > 0
    assert stats["accepted"] == stats["drafted"]
    assert stats["target_forwards"] < 12
    assert stats["target_forwards"] >= 1


def test_different_draft_still_exact_target_greedy():
    target = _tiny_model(seed=10)
    draft = _tiny_model(seed=99)  # different tiny model, same shapes/vocab
    prompt = torch.tensor([[1, 5, 8, 3, 21]], dtype=torch.long)
    tokens, stats = speculative_generate(
        target, draft, prompt.clone(), max_new_tokens=10, gamma=3
    )
    reference = generate(target, prompt.clone(), max_new_tokens=10, temperature=0)
    assert torch.equal(tokens, reference)
    assert stats["drafted"] > 0
    assert stats["target_forwards"] >= 1
    # Accepted proposals can never exceed what was drafted.
    assert 0 <= stats["accepted"] <= stats["drafted"]


def test_speculative_zero_budget_and_validation():
    model = _tiny_model(seed=1)
    prompt = torch.tensor([[2, 3]], dtype=torch.long)
    tokens, stats = speculative_generate(model, model, prompt, max_new_tokens=0)
    assert torch.equal(tokens, prompt)
    assert stats == {"accepted": 0, "drafted": 0, "target_forwards": 0}
    try:
        speculative_generate(model, model, prompt, max_new_tokens=1, gamma=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("gamma=0 should raise ValueError")
