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


def test_eager_protected_rows_classification():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=8)
    rids = [eng.submit(PROMPTS[i], 4) for i in range(3)]
    reqs = [eng._requests[r] for r in rids]
    for i, req in enumerate(reqs):
        req.slot = i
        eng._slots[i] = req
    reqs[0].status = "decoding"                       # planned below
    reqs[1].status, reqs[1].prefill_done = "prefilling", 0   # stateless
    reqs[2].status, reqs[2].prefill_done = "prefilling", 2   # live state
    # slot 3 stays free (stateless)
    snap_rows, reset_rows = eng._eager_protected_rows([reqs[0]])
    assert snap_rows == [2]
    assert reset_rows == [1, 3]


def test_mixed_steps_exact_with_reduced_snapshots():
    # Budget forces mixed prefill+decode steps; reduced snapshot/reset
    # handling must stay token-identical to independent B=1 runs, and
    # no snapshot taken during the run may include a free row clone of
    # the old full width (snapshotted rows are at most the live rows).
    tp = _tp()
    refs = [_b1(tp, p, 5) for p in PROMPTS[:3]]
    eng = TPEngineCore(tp, tp.ctx, max_slots=3, max_seq_len=64, token_budget=3)
    seen = []
    orig = eng._snapshot_rows

    def spy(rows):
        seen.append(list(rows))
        return orig(rows)

    eng._snapshot_rows = spy
    ids = [eng.submit(p, 5) for p in PROMPTS[:3]]
    eng.run()
    assert [eng.poll(r)["token_ids"] for r in ids] == refs
    assert seen and all(len(rows) < 3 for rows in seen)


def test_graph_capture_plan_no_second_full_state_and_guard():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=8)
    rid = eng.submit(PROMPTS[0], 4)
    eng.step()  # admits + prefills -> one occupied row
    plan = eng._graph_capture_plan()
    assert plan["mode"] == "real-state" and plan["scratch_bytes"] == 0
    assert plan["snapshot_rows"] == [eng._requests[rid].slot]
    assert plan["transient_bytes"] < eng._state_bytes()  # not a 2nd state
    scratch_eng = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64,
                               token_budget=8, graph_capture_mode="scratch")
    assert scratch_eng._graph_capture_plan()["scratch_bytes"] == \
        scratch_eng._state_bytes()
    # Guard fails closed (capture blocked -> eager) before any OOM.
    orig = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda dev=None: (0, 1 << 30)  # noqa: E731
    try:
        assert eng._graph_capture_allowed() is False
    finally:
        torch.cuda.mem_get_info = orig
    assert eng.graph_capture_blocked and eng.graph_capture_block_reason
    assert plan["graph_pool_estimate_bytes"] > 0
    # Fail closed when the memory oracle itself fails.
    eng2 = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    orig_info = torch.cuda.mem_get_info

    def _boom(dev=None):
        raise RuntimeError("no oracle")

    torch.cuda.mem_get_info = _boom  # noqa: E731
    try:
        assert eng2._graph_capture_allowed() is False
    finally:
        torch.cuda.mem_get_info = orig_info
    assert eng2.graph_capture_blocked
    # Capture-time RuntimeError (incl. CUDA OOM) -> eager, not a crash.
    eng3 = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8)
    eng3._graph_capture_allowed = lambda plan=None: True  # noqa: E731

    def _oom(inp):
        raise RuntimeError("CUDA out of memory")

    eng3._capture_graph = _oom  # noqa: E731
    inp = torch.zeros(eng3.max_slots, dtype=torch.long)
    assert eng3._decode_graphed(inp) is None and eng3.graph_capture_blocked
    with pytest.raises(ValueError):
        TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=8,
                     graph_capture_mode="bogus")


def test_real_state_warmup_restores_exactly_and_allocates_no_state():
    tp = _tp()
    ref = _b1(tp, PROMPTS[0], 6)
    eng = TPEngineCore(tp, tp.ctx, max_slots=2, max_seq_len=64, token_budget=16)
    rid = eng.submit(PROMPTS[0], 6)
    eng.step()  # prefill -> decoding, one generated token
    before_pos = eng.state.position.clone()
    before = {}
    for name in ("kv_k", "kv_v", "linear_S", "linear_conv_buffers"):
        before[name] = [t.clone() if t is not None else None
                        for t in getattr(eng.state, name)]
    calls = []
    orig_static = tp.static_state
    tp.static_state = lambda **kw: (calls.append(kw), orig_static(**kw))[1]
    try:
        inp = torch.zeros(eng.max_slots, dtype=torch.long)
        inp[int(eng._requests[rid].slot)] = eng._requests[rid].generated[-1]
        eng._warmup_on_real_state(inp)
    finally:
        tp.static_state = orig_static
    assert calls == []  # no scratch / second full state allocated
    assert torch.equal(eng.state.position, before_pos)
    for name, saved in before.items():
        for cur, old in zip(getattr(eng.state, name), saved):
            if old is not None:
                assert torch.equal(cur, old), name
    eng.run()  # generation after warmup+restore is still exact
    assert eng.poll(rid)["token_ids"] == ref


def _fake_req(slot, temperature=0.0):
    from inference_from_scratch.tp_engine import EngineRequest
    req = EngineRequest(request_id=slot, prompt_ids=[1], max_new=2,
                        arrival_step=0, submit_seq=slot,
                        temperature=temperature)
    req.slot, req.status = slot, "decoding"
    return req


def test_bucket_choice_prefix_scattered_sampled_and_defaults():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=8, max_seq_len=64, token_budget=16,
                       graph_bucketed=True)
    assert eng.graph_buckets == (1, 2, 4, 8)
    assert eng._bucket_choice([_fake_req(0)]) == 1
    assert eng._bucket_choice([_fake_req(0), _fake_req(1)]) == 2
    assert eng._bucket_choice([_fake_req(i) for i in range(3)]) == 4
    assert eng._bucket_choice([_fake_req(i) for i in range(5)]) == 8
    # Scattered active rows are NOT a prefix view: no bucket.
    assert eng._bucket_choice([_fake_req(1)]) is None
    assert eng._bucket_choice([_fake_req(0), _fake_req(2)]) is None
    assert eng._bucket_choice([_fake_req(2), _fake_req(3)]) is None
    # Sampled rows never take a graph bucket (argmax-only reduction).
    assert eng._bucket_choice([_fake_req(0, temperature=1.0)]) is None
    assert eng._bucket_choice([]) is None
    # Opt-in: default engine never chooses a bucket.
    plain = TPEngineCore(tp, tp.ctx, max_slots=8, max_seq_len=64, token_budget=16)
    assert plain.graph_bucketed is False
    assert plain._bucket_choice([_fake_req(0)]) is None
    for bad in ((2, 1), (0,), (True,), ()):
        with pytest.raises(ValueError):
            TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64,
                         token_budget=8, graph_buckets=bad)


def test_prefix_state_shares_storage_reset_and_decode():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=16)
    view = eng._prefix_state(2)
    assert view.batch_size == 2 and view.batch == 2
    for name in ("kv_k", "kv_v", "linear_S", "linear_conv_buffers"):
        for parent, child in zip(getattr(eng.state, name), getattr(view, name)):
            if parent is not None:
                assert child.data_ptr() == parent.data_ptr()
                assert child.shape[0] == 2 and child.shape[1:] == parent.shape[1:]
    assert view.position.data_ptr() == eng.state.position.data_ptr()
    eng.state.position.copy_(torch.tensor([5, 6, 7, 8]))
    view.reset(rows=[1])
    assert eng.state.position.tolist() == [5, 0, 7, 8]  # only view row 1
    # Exactness: decode over the prefix view == full-width decode rows.
    idx = torch.tensor([PROMPTS[0], PROMPTS[1], PROMPTS[0], PROMPTS[1]])
    eng.state.reset()
    tp.prefill_chunkwise_static(idx, eng.state, chunk_size=2)
    got = tp.decode_token_static(torch.tensor([3, 4]), eng._prefix_state(2))
    eng2 = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=16)
    eng2.state.reset()
    tp.prefill_chunkwise_static(idx, eng2.state, chunk_size=2)
    ref = tp.decode_token_static(
        torch.tensor([3, 4, 0, 0]), eng2.state)[:2]
    # Token-exact (the engine contract); logits may differ in low bits
    # because GEMM reduction order depends on the batch width M.
    assert got.argmax(dim=-1).tolist() == ref.argmax(dim=-1).tolist()
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    assert eng.state.position[:2].tolist() == eng2.state.position[:2].tolist()


def test_bucket_capture_guard_fallback_and_pool_accounting():
    tp = _tp()
    eng = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=8,
                       graph_bucketed=True)
    plan1 = eng._graph_capture_plan(1)
    assert plan1["width"] == 1 and plan1["graph_out_bytes"] < \
        eng._graph_capture_plan()["graph_out_bytes"]
    assert 0 < plan1["graph_pool_estimate_bytes"] < \
        eng._graph_capture_plan()["graph_pool_estimate_bytes"]
    eng._bucket_graphs[1] = {}  # already-captured pool must be charged
    assert eng._captured_pool_estimate() == eng._pool_estimate(1)
    assert eng._graph_capture_plan(2)["captured_pool_estimate_bytes"] == \
        eng._pool_estimate(1)
    eng._bucket_graphs = {}
    inp = torch.zeros(eng.max_slots, dtype=torch.long)
    orig = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda dev=None: (0, 1 << 30)  # noqa: E731
    try:
        assert eng._decode_bucketed_graph(inp, 1) is None  # guard -> fallback
    finally:
        torch.cuda.mem_get_info = orig
    assert 1 in eng._bucket_blocked
    # Capture failure also falls back (None), never raises into step().
    eng2 = TPEngineCore(tp, tp.ctx, max_slots=4, max_seq_len=64, token_budget=8,
                        graph_bucketed=True)
    eng2._graph_capture_allowed = lambda plan=None: True  # noqa: E731
    def _boom(bucket, inp_):
        raise RuntimeError("CUDA out of memory")
    eng2._capture_bucket_graph = _boom  # noqa: E731
    assert eng2._decode_bucketed_graph(inp, 2) is None
    assert 2 in eng2._bucket_blocked and eng2._bucket_graphs == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bucketed_graph_gpu_exact_occupancies_1_2_4():
    from inference_from_scratch.tp_distributed import _LocalContext as LC
    torch.manual_seed(7)
    dense = TinyHybridLM(_tiny_cfg()).eval()
    tp = DistributedHybridLM.from_dense(
        dense, LC(0, 1, torch.device("cuda")), config=_tiny_cfg())
    tp.eval()
    refs = []
    for p in PROMPTS[:4]:
        s1 = tp.static_state(batch_size=1, max_seq_len=64)
        refs.append(tp.generate_static_batched(
            torch.tensor([p], device="cuda"), 6, s1, chunk_size=2)[0].tolist())
    for occ, bucket in ((1, 1), (2, 2), (4, 4)):
        # max_slots=8 so occupancy 4 is a real bucket, not the full graph.
        eng = TPEngineCore(tp, tp.ctx, max_slots=8, max_seq_len=64,
                           token_budget=16, decode_mode="graph",
                           graph_bucketed=True)
        ids = [eng.submit(PROMPTS[i], 6) for i in range(occ)]
        eng.run()
        assert [eng.poll(r)["token_ids"] for r in ids] == refs[:occ]
        assert eng.graph_replays_by_bucket.get(bucket, 0) > 0, \
            eng.graph_replays_by_bucket
        assert eng.bucket_graph_captures == 1
        eng.close()
    # Scattered: middle request finishes first, planned slots become
    # {0, 2}; the bucket path must stand down (full graph), tokens exact.
    eng = TPEngineCore(tp, tp.ctx, max_slots=8, max_seq_len=64,
                       token_budget=16, decode_mode="graph",
                       graph_bucketed=True)
    ids = [eng.submit(PROMPTS[0], 6), eng.submit(PROMPTS[1], 2),
           eng.submit(PROMPTS[2], 6)]
    eng.run()
    assert eng.poll(ids[0])["token_ids"] == refs[0]
    assert eng.poll(ids[1])["token_ids"] == refs[1][:2]
    assert eng.poll(ids[2])["token_ids"] == refs[2]
    assert eng.graph_replays > sum(eng.graph_replays_by_bucket.values())
    eng.close()


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
