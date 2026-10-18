"""Wiring-space, teacher-index and hidden-dump tests for the MTP sweep."""
import importlib.util
from pathlib import Path

import pytest
import torch

from inference_from_scratch.hybrid import (
    DeviceMappedHybridRunner, HybridConfig, TinyHybridLM,
)
from inference_from_scratch.mtp import (
    PRUNED_WIRING, WIRING_AXES, MTPConfig, MTPHead,
    enumerate_wiring_candidates, teacher_forced_draft_indices,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wiring_space_enumeration_and_pruning():
    cands = enumerate_wiring_candidates()
    assert len(cands) == 48  # 2*2*2*3*2 live; dedicated embedding pruned
    assert all(c["embedding_source"] == "shared_embed_tokens" for c in cands)
    assert {c["fc_concat_order"] for c in cands} == {"hidden_first", "embedding_first"}
    assert {c["position_rule"] for c in cands} == {
        "hidden_pos", "input_token_pos", "predicted_pos"}
    assert any(p["option"] == "dedicated_mtp_embedding" for p in PRUNED_WIRING)
    assert WIRING_AXES["embedding_source"] == ("shared_embed_tokens",)


def test_teacher_forced_draft_indices_toy_sequence():
    seq = [10, 11, 12, 13, 14]
    steps = teacher_forced_draft_indices(len(seq))
    # (hidden t, embed token, predicted target): predicts 12, 13, 14.
    assert steps == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert [(seq[e], seq[t]) for _, e, t in steps] == [(11, 12), (12, 13), (13, 14)]
    assert teacher_forced_draft_indices(2) == []
    assert teacher_forced_draft_indices(3) == [(0, 1, 2)]
    with pytest.raises(ValueError):
        teacher_forced_draft_indices(-1)


def test_mtp_head_wiring_options_change_output():
    torch.manual_seed(0)
    cfg = MTPConfig(hidden_size=32, vocab_size=97, mtp_num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    intermediate_size=64)
    hidden = torch.randn(2, 32)
    emb = torch.randn(2, 32)
    lm = torch.nn.Linear(32, 97, bias=False)
    torch.manual_seed(5)
    base = MTPHead(cfg).eval()
    outs = {}
    for opts in (dict(), dict(fc_concat_order="embedding_first"),
                 dict(norm_pairing="swapped"), dict(lm_head_input="raw_block")):
        head = MTPHead(cfg, **opts).eval()
        head.load_state_dict(base.state_dict())  # identical weights
        with torch.no_grad():
            dh, logits, _ = head(hidden, emb, lm_head=lm)
        outs[str(opts)] = dh
    assert all(v.shape == (2, 32) for v in outs.values())
    vals = list(outs.values())
    assert any(not torch.equal(vals[0], v) for v in vals[1:])
    with pytest.raises(ValueError):
        MTPHead(cfg, fc_concat_order="bogus")


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


def test_hidden_hook_default_off_and_capture_trajectory_tiny():
    torch.manual_seed(3)
    model = TinyHybridLM(_tiny_config()).eval()
    plain = DeviceMappedHybridRunner(model, split=2, device0="cpu", device1="cpu")
    prompt = [3, 11, 5]
    with torch.no_grad():
        ref_logits, _ = plain.run(torch.tensor([prompt]))
    seen = []
    hooked = DeviceMappedHybridRunner(
        model, split=2, device0="cpu", device1="cpu",
        hidden_hook=lambda pre, post: seen.append((pre.clone(), post.clone())))
    with torch.no_grad():
        got_logits, _ = hooked.run(torch.tensor([prompt]))
    assert torch.equal(ref_logits, got_logits)  # hook changes nothing
    pre, post = seen[0]
    assert pre.shape == (1, 3, 32) and torch.equal(post, model.norm(pre))

    cap = _load_script("capture_target_hidden")
    traj = cap.capture_trajectory(hooked, prompt, 4)
    assert hooked.hidden_hook is not None  # restored after capture (was set)
    assert len(traj["token_ids"]) == 3 + 4
    assert traj["hidden_pre_norm"].shape == (3 + 4 - 1, 32)
    assert traj["hidden_pre_norm"].dtype == torch.bfloat16
    # Alignment: last hidden (post-norm -> lm_head) predicts the last token.
    with torch.no_grad():
        last_logits = model.lm_head(traj["hidden_post_norm"][-1].float())
    assert int(last_logits.argmax()) == traj["token_ids"][-1]
    assert traj["token_ids"][:3] == prompt


def test_run_qwen38_gpu_dump_hidden_parser_default_off():
    mod = _load_script("run_qwen38_gpu")
    assert mod.build_parser().parse_args([]).dump_hidden is None
    args = mod.build_parser().parse_args(["--dump-hidden", "/tmp/x.pt"])
    assert args.dump_hidden == "/tmp/x.pt"
