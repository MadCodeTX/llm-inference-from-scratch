"""TPEngineCore: continuous batching on ONE batched TPStaticState.

Tiny proofs (world=1 here; the 2-process gloo proof is
``scripts/tp_engine_smoke.py``, run by the subprocess test below):
5 staggered requests on 2 slots == independent B=1 greedy runs,
slot reuse after EOS, and per-step token budget respected.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM  # noqa: E402
from inference_from_scratch.tp_distributed import DistributedHybridLM, _LocalContext  # noqa: E402
from inference_from_scratch.tp_engine import TPEngineCore  # noqa: E402

PROMPTS = [[3, 11, 42, 7, 90], [55, 1, 2, 3, 4], [94, 20, 20, 8, 13],
           [10, 10, 10, 10, 10], [1, 96, 5, 5, 77]]


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0)


def _tp():
    torch.manual_seed(7)
    dense = TinyHybridLM(_tiny_cfg())
    dense.eval()
    tp = DistributedHybridLM.from_dense(
        dense, _LocalContext(0, 1, torch.device("cpu")), config=_tiny_cfg())
    tp.eval()
    return tp


def _b1(tp, prompt, max_new):
    s1 = tp.static_state(batch_size=1, max_seq_len=64)
    return tp.generate_static_batched(
        torch.tensor([prompt]), max_new, s1, chunk_size=2)[0].tolist()


def test_staggered_five_on_two_slots_exact_and_budget():
    tp = _tp()
    refs = [_b1(tp, p, 6) for p in PROMPTS]
    eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    ids = [eng.submit(PROMPTS[0], 6), eng.submit(PROMPTS[1], 6)]
    eng.step(); eng.step()  # staggered arrivals below
    ids.append(eng.submit(PROMPTS[2], 6))
    eng.step()
    ids.append(eng.submit(PROMPTS[3], 6)); ids.append(eng.submit(PROMPTS[4], 6))
    eng.run()
    for rid, ref in zip(ids, refs):
        out = eng.poll(rid)
        assert out["status"] == "finished" and out["token_ids"] == ref, (out, ref)
        assert out["ttft_steps"] is not None and out["ttft_steps"] >= 1
        assert out["finish_step"] is not None and out["finish_step"] >= out["ttft_steps"] - 1
    assert all(r.total_tokens <= 8 for r in eng.history)
    assert sum(r.prefill_tokens for r in eng.history) == 5 * 5
    # Determinism: replay same interleaving on a fresh engine.
    eng2 = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    j = [eng2.submit(PROMPTS[0], 6), eng2.submit(PROMPTS[1], 6)]
    eng2.step(); eng2.step(); j.append(eng2.submit(PROMPTS[2], 6)); eng2.step()
    j.append(eng2.submit(PROMPTS[3], 6)); j.append(eng2.submit(PROMPTS[4], 6))
    eng2.run()
    assert [eng2.poll(r)["token_ids"] for r in j] == refs


def test_chunked_prefill_small_budget_exact():
    tp = _tp()
    ref = _b1(tp, PROMPTS[0], 5)
    eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64, token_budget=2)
    rid = eng.submit(PROMPTS[0], 5)
    assert eng.poll(rid)["status"] == "waiting"
    eng.run()
    assert eng.poll(rid)["token_ids"] == ref
    assert all(r.total_tokens <= 2 for r in eng.history)
    assert any(r.prefill_tokens == 2 for r in eng.history)  # prompt really chunked


def test_slot_reuse_after_eos():
    tp = _tp()
    first = _b1(tp, PROMPTS[0], 6)[0]  # token that EOS-terminates request 0
    refs = [_b1(tp, p, 4) for p in PROMPTS[1:3]]
    eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64,
                       token_budget=16, eos_id=first)
    r0 = eng.submit(PROMPTS[0], 6)
    eng.run()
    out0 = eng.poll(r0)
    assert out0["token_ids"] == [first]  # stopped at EOS, slot freed
    r1 = eng.submit(PROMPTS[1], 4); r2 = eng.submit(PROMPTS[2], 4)
    eng.run()  # single slot must be reused serially, exactly
    assert eng.poll(r1)["token_ids"] == refs[0]
    assert eng.poll(r2)["token_ids"] == refs[1]


def test_validation_and_greedy_only():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    with pytest.raises(ValueError):
        eng.submit(PROMPTS[0], 4, sampling="temperature")
    with pytest.raises(ValueError):
        eng.submit([], 4)
    with pytest.raises(ValueError):
        eng.submit(PROMPTS[0], -1)
    with pytest.raises(ValueError):
        eng.submit([0] * 70, 4)
    with pytest.raises(KeyError):
        eng.poll(999)
    with pytest.raises(ValueError):
        TPEngineCore(tp, tp.ctx, max_slots=0, max_seq_len=64, token_budget=8)
    rid = eng.submit(PROMPTS[0], 0)
    eng.run()
    assert eng.poll(rid) == {"status": "finished", "token_ids": [],
                             "ttft_steps": None, "finish_step": 0}


def test_sampled_temperature_zero_equals_greedy():
    tp = _tp()
    ref = _b1(tp, PROMPTS[0], 6)
    eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64, token_budget=16)
    rid = eng.submit(PROMPTS[0], 6, temperature=0.0, top_p=0.5, top_k=3, seed=99)
    eng.run()
    assert eng.poll(rid)["token_ids"] == ref


def test_sampled_seeded_reproducible_and_rank_agreement_seam():
    # Two engines with identical weights = two TP ranks holding
    # identical full logits; same seed must decide identically.
    outs = []
    for _ in range(2):
        tp = _tp()
        eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=16)
        rid = eng.submit(PROMPTS[0], 8, temperature=1.0, top_p=0.9, seed=1234)
        eng.run()
        outs.append(eng.poll(rid)["token_ids"])
    assert len(outs[0]) == 8 and outs[0] == outs[1]


def test_sampled_topk1_and_tiny_topp_equal_greedy():
    tp = _tp()
    ref = _b1(tp, PROMPTS[0], 6)
    for kw in ({"temperature": 1.3, "top_k": 1},
               {"temperature": 1.3, "top_p": 0.01}):
        eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64, token_budget=16)
        rid = eng.submit(PROMPTS[0], 6, seed=7, **kw)
        eng.run()
        assert eng.poll(rid)["token_ids"] == ref, kw


def test_sampled_validation_and_graph_never_sampled():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64, token_budget=16)
    for kw in ({"temperature": -0.1}, {"temperature": float("nan")},
               {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0},
               {"top_k": True}, {"seed": -1}, {"seed": True}):
        with pytest.raises(ValueError):
            eng.submit(PROMPTS[0], 4, **kw)
    rid = eng.submit(PROMPTS[0], 4, temperature=0.7, top_p=0.8, seed=1234)
    eng.step()  # prefill consumes the first sampled token
    decoding = [r for r in eng._slots if r is not None and r.status == "decoding"]
    if decoding:  # graph must refuse sampled rows even if forced on
        eng.decode_mode = "graph"
        assert eng._graph_eligible(decoding) is False
    eng.run()
    assert len(eng.poll(rid)["token_ids"]) >= 1


def test_ignore_eos_per_request():
    tp = _tp()
    first = _b1(tp, PROMPTS[0], 6)[0]
    eng = TPEngineCore(tp, tp.ctx, max_slots=1, max_seq_len=64,
                       token_budget=16, eos_id=first)
    rid = eng.submit(PROMPTS[0], 3, ignore_eos=True)
    eng.run()
    assert len(eng.poll(rid)["token_ids"]) == 3  # EOS token does not finish it


def test_decode_mode_default_and_validation():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    assert eng.decode_mode == "eager"  # CPU context resolves the default to eager
    assert eng.graph_captures == 0 and eng.graph_replays == 0
    with pytest.raises(ValueError):
        TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8,
                     decode_mode="turbo")
    # Graph fails closed off CUDA instead of silently falling back.
    with pytest.raises(ValueError):
        TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8,
                     decode_mode="graph")
    eng.close()  # no graph captured: close is a no-op


def test_graph_eligibility_requires_all_occupied_slots_decoding():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    # Eager engine is never graph-eligible, whatever the slot state.
    r0 = eng.submit(PROMPTS[0], 4)
    eng.step()  # prefills (budget 8 >= prompt 5 + decode 0) -> decoding
    assert eng._graph_eligible([]) is False
    decoding = [r for r in eng._slots if r is not None and r.status == "decoding"]
    assert decoding and eng._graph_eligible(decoding) is False  # mode is eager
    assert eng.poll(r0)["status"] in ("decoding", "finished")


@pytest.mark.skipif(not (ROOT / "scripts/tp_engine_smoke.py").exists(),
                    reason="smoke script missing")
def test_tp_engine_smoke_gloo_two_process():
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", str(ROOT / "scripts/tp_engine_smoke.py"),
         "--backend", "gloo", "--device", "cpu"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TP_ENGINE_SMOKE PASS" in proc.stdout
    assert "staggered_exact=True" in proc.stdout
    assert "budget_respected=True" in proc.stdout
