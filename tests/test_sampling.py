import pytest
import torch
from inference_from_scratch.sampling import (
    greedy, apply_temperature, top_k_filter, top_p_filter, sample_next,
)


def _logits():
    return torch.tensor([[0.0, 1.0, 5.0, 2.0], [3.0, 0.0, 1.0, 0.5]])


def test_greedy_argmax():
    out = greedy(_logits())
    assert out.shape == (2, 1)
    assert out.dtype == torch.long
    assert out[0, 0].item() == 2
    assert out[1, 0].item() == 0


def test_temperature_zero_is_greedy():
    logits = _logits()
    assert torch.equal(sample_next(logits, temperature=0), greedy(logits))


def test_topk1_is_greedy():
    logits = _logits()
    for _ in range(5):
        assert torch.equal(sample_next(logits, temperature=1.0, top_k=1), greedy(logits))


def test_top_p_keeps_smallest_set():
    # probs approx: token2 dominates
    logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    filtered = top_p_filter(logits, p=0.9)
    # only the dominant token survives (its prob >0.99 >=0.9 alone)
    kept = torch.isfinite(filtered[0]).sum().item()
    assert kept == 1
    assert torch.isfinite(filtered[0, 2])
    # larger vocab: check cumulative property on sorted probs
    logits2 = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    f2 = top_p_filter(logits2, p=0.8)
    kept_idx = torch.where(torch.isfinite(f2[0]))[0]
    probs = torch.softmax(logits2[0], dim=-1)
    kept_mass = probs[kept_idx].sum().item()
    assert kept_mass >= 0.8
    # removing any kept token (the smallest kept) would drop below p => smallest
    smallest_kept_prob = probs[kept_idx].min().item()
    assert kept_mass - smallest_kept_prob < 0.8 + 1e-6


def test_top_k_filter():
    logits = _logits()
    f = top_k_filter(logits, k=2)
    assert torch.isfinite(f[0]).sum().item() == 2
    assert torch.isfinite(f[0, 2]) and torch.isfinite(f[0, 3])


def test_seeded_reproducible():
    logits = torch.randn(3, 50)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_next(logits, temperature=1.0, generator=g1)
    b = sample_next(logits, temperature=1.0, generator=g2)
    assert a.shape == (3, 1)
    assert torch.equal(a, b)


def test_invalid_temperature_raises():
    logits = _logits()
    with pytest.raises(ValueError):
        sample_next(logits, temperature=-1.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -0.5)
    with pytest.raises(ValueError):
        top_p_filter(logits, p=0.0)
    with pytest.raises(ValueError):
        top_p_filter(logits, p=1.5)
    with pytest.raises(ValueError):
        top_k_filter(logits, k=0)


def test_temperature_sharpens():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = torch.softmax(apply_temperature(logits, 0.5), dim=-1)
    hot = torch.softmax(apply_temperature(logits, 2.0), dim=-1)
    assert cold[0, 2] > hot[0, 2]  # lower T -> more peaked
