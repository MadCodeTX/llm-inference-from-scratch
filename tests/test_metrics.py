"""Tests for Ch 13 / sec. 1.4: metrics (percentiles, measure_generate)."""

import pytest
import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.metrics import measure_generate, percentile, summarize
from inference_from_scratch.model import DecoderLM


def test_percentile_known_values():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == pytest.approx(1.0)
    assert percentile(xs, 100) == pytest.approx(4.0)
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile([0.0, 10.0], 25) == pytest.approx(2.5)
    assert percentile([5.0], 99) == pytest.approx(5.0)
    # unsorted input, odd count -> exact middle at p50
    assert percentile([3.0, 1.0, 2.0], 50) == pytest.approx(2.0)


def test_percentile_validated():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_keys_and_values():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["p50"] == pytest.approx(2.5)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(4.0)
    assert s["p99"] >= s["p90"] >= s["p50"]
    with pytest.raises(ValueError):
        summarize([])


def _tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=64, max_seq_len=64)
    return DecoderLM(cfg).eval()


def test_measure_generate_tiny_model():
    model = _tiny_model()
    idx = torch.randint(0, 48, (1, 4))
    r = measure_generate(model, idx, 6, temperature=0)
    assert r["ttft_s"] > 0
    assert r["tps"] > 0
    assert r["tokens"] == 6
    assert r["decode_s"] >= 0
    assert len(r["token_ids"]) == 4 + 6


def test_measure_generate_zero_budget():
    model = _tiny_model()
    idx = torch.randint(0, 48, (1, 3))
    r = measure_generate(model, idx, 0, temperature=0)
    assert r["tokens"] == 0 and r["ttft_s"] == 0.0 and r["tps"] == 0.0
