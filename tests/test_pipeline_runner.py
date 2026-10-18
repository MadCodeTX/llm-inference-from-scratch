"""Pipeline-by-layers device plan + boundary-transfer runner (tiny CPU)."""
import pytest
import torch

from inference_from_scratch.hybrid import (
    DeviceMappedHybridRunner, HybridConfig, TinyHybridLM,
    generate_hybrid, pipeline_device_plan,
)


def _tiny_config():
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def test_pipeline_device_plan_assignment():
    plan = pipeline_device_plan(64, 32)
    assert plan["embed"] == "cuda:0" and plan["norm"] == "cuda:1"
    assert plan["lm_head"] == "cuda:1" and plan["split"] == 32
    assert plan["layers"][:32] == ["cuda:0"] * 32
    assert plan["layers"][32:] == ["cuda:1"] * 32
    edge = pipeline_device_plan(4, 0, "cpu", "cpu")
    assert edge["layers"] == ["cpu"] * 4
    with pytest.raises(ValueError):
        pipeline_device_plan(4, 5)
    with pytest.raises(ValueError):
        pipeline_device_plan(0, 0)


def test_runner_matches_single_device_greedy():
    torch.manual_seed(7)
    model = TinyHybridLM(_tiny_config()).eval()
    runner = DeviceMappedHybridRunner(model, split=2, device0="cpu", device1="cpu")
    idx = torch.tensor([[3, 11, 5, 42]])
    with torch.no_grad():
        ref_logits, _ = model(idx)
        got_logits, _ = runner.run(idx)
    assert torch.equal(ref_logits, got_logits)
    expected = generate_hybrid(model, idx, 6, temperature=0.0)[0, 4:].tolist()
    assert runner.generate_greedy(idx, 6) == expected
    seen = []
    hooked = DeviceMappedHybridRunner(
        model, split=1, device0="cpu", device1="cpu",
        rms_hook=lambda i, h: seen.append((i, float(h.float().pow(2).mean().sqrt()))))
    hooked.run(idx)
    assert [i for i, _ in seen] == [0, 1, 2, 3]
    assert all(v > 0 for _, v in seen)
