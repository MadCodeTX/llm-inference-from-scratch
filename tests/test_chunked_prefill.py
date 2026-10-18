"""Tests for ChunkedPrefillScheduler (serving.py, future TP server).

Hand-computed scenarios: every step's decode ids, prefill chunks,
TTFT and completion steps are derived by pencil arithmetic in the
comments before being asserted.
"""

import pytest

from inference_from_scratch.serving import (
    ChunkedPrefillRequest,
    ChunkedPrefillScheduler,
)


def _run(sched, max_steps=1000):
    records = []
    while sched.has_work and len(records) < max_steps:
        records.append(sched.step())
    assert not sched.has_work
    return records


def test_hand_computed_scenario_exact():
    # Budget 10. A: prompt25/new3 @0, B: prompt5/new2 @0,
    # C: prompt10 cached8 (effective2)/new1 @2.
    # step0: prefill A10          (A 10/25)
    # step1: prefill A10          (A 20/25)
    # step2: prefill A5+B5        -> A,B first tokens (gen1 each)
    # step3: decode A,B (2) + prefill C2 -> B gen2 done, C first=done
    # step4: decode A (1)         -> A gen3 done
    # Cost = 2.0/decode-token + 0.5/prefill-token:
    # steps: 5,5,5, (4+1)=5, 2  -> cumulative 22.0
    sched = ChunkedPrefillScheduler(
        token_budget=10,
        step_cost_model=lambda w: w.decode_tokens * 2.0 + w.prefill_tokens * 0.5,
    )
    a = sched.submit("A", prompt_tokens=25, max_new=3, arrival_step=0)
    b = sched.submit("B", prompt_tokens=5, max_new=2, arrival_step=0)
    c = sched.submit(
        "C", prompt_tokens=10, max_new=1, cached_prefix_tokens=8, arrival_step=2
    )
    records = _run(sched)

    assert [r.step for r in records] == [0, 1, 2, 3, 4]
    assert [(r.decode_request_ids, [(x.request_id, x.tokens) for x in r.prefill_chunks]) for r in records] == [
        ((), [("A", 10)]),
        ((), [("A", 10)]),
        ((), [("A", 5), ("B", 5)]),
        (("A", "B"), [("C", 2)]),
        (("A",), []),
    ]
    assert [r.first_token_request_ids for r in records] == [(), (), ("A", "B"), ("C",), ()]
    assert [r.finished_request_ids for r in records] == [(), (), (), ("B", "C"), ("A",)]
    assert (a.first_token_step, a.ttft_steps, a.completion_step, a.completion_steps) == (2, 3, 4, 5)
    assert (b.first_token_step, b.ttft_steps, b.completion_step) == (2, 3, 3)
    assert (c.first_token_step, c.ttft_steps, c.completion_step) == (3, 2, 3)
    assert (a.generated_tokens, b.generated_tokens, c.generated_tokens) == (3, 2, 1)
    assert c.prefill_tokens_done == 2  # only uncached suffix computed
    assert [x.request_id for x in sched.finished] == ["B", "C", "A"]
    assert [r.cost_s for r in records] == [5.0, 5.0, 5.0, 5.0, 2.0]
    assert sched.cumulative_cost_s == 22.0
    assert all(r.total_tokens <= 10 for r in records)
    assert all(x.preemptions == 0 for x in (a, b, c))


def test_decode_never_starved_by_long_prefill():
    # Budget 10. D prompt5/new5 @0 finishes prefill step0 (first token),
    # then must decode every step while L (prompt100 @1) chunks 9/step.
    sched = ChunkedPrefillScheduler(token_budget=10)
    d = sched.submit("D", prompt_tokens=5, max_new=5, arrival_step=0)
    long_req = sched.submit("L", prompt_tokens=100, max_new=1, arrival_step=1)
    records = _run(sched)
    # D: first token step0, decode steps1-4 -> done step4.
    assert d.completion_step == 4
    for rec in records[1:5]:
        assert rec.decode_request_ids == ("D",)
        assert [(c.request_id, c.tokens) for c in rec.prefill_chunks] == [("L", 9)]
        assert rec.total_tokens == 10
    # After D done, L gets full 10/step: 100 = 9*4 + 10*6 + 4
    assert long_req.prefill_tokens_done == 100
    assert long_req.first_token_step == records[-1].step


def test_budget_never_exceeded_and_totals_accounted():
    sched = ChunkedPrefillScheduler(token_budget=7)
    reqs = [
        sched.submit(f"r{i}", prompt_tokens=13 + i * 7, max_new=3, arrival_step=i % 3)
        for i in range(6)
    ]
    records = _run(sched)
    assert all(r.total_tokens <= 7 for r in records)
    assert all(r.decode_tokens + r.prefill_tokens == r.total_tokens for r in records)
    for req in reqs:
        chunked = sum(
            c.tokens for r in records for c in r.prefill_chunks
            if c.request_id == req.request_id
        )
        assert chunked == req.effective_prefill_tokens
        assert req.generated_tokens == req.max_new_tokens
        assert req.preemptions == 0


def test_cached_prefix_reduces_prefill_steps_exactly():
    def ttft(prompt, cached):
        sched = ChunkedPrefillScheduler(token_budget=10)
        r = sched.submit("x", prompt_tokens=prompt, max_new=1,
                         cached_prefix_tokens=cached)
        records = _run(sched)
        return r, records

    cold, cold_recs = ttft(100, 0)
    warm, warm_recs = ttft(100, 90)
    full, full_recs = ttft(100, 100)
    assert (cold.first_token_step, cold.ttft_steps) == (9, 10)   # 10 chunks of 10
    assert (warm.first_token_step, warm.ttft_steps) == (0, 1)    # 1 chunk of 10
    assert cold.ttft_steps - warm.ttft_steps == 9
    assert sum(c.tokens for r in warm_recs for c in r.prefill_chunks) == 10
    # Fully cached: zero prefill tokens, first token at first eligible step.
    assert (full.first_token_step, full.ttft_steps) == (0, 1)
    assert all(not r.prefill_chunks for r in full_recs)


def test_fifo_prefill_order():
    # Budget 10, both @0: first-submitted fills first; second gets no
    # chunk until first's prefill completes in the same step leftover.
    sched = ChunkedPrefillScheduler(token_budget=10)
    first = sched.submit("first", prompt_tokens=15, max_new=1)
    second = sched.submit("second", prompt_tokens=15, max_new=1)
    records = _run(sched)
    assert [(c.request_id, c.tokens) for c in records[0].prefill_chunks] == [("first", 10)]
    assert [(c.request_id, c.tokens) for c in records[1].prefill_chunks] == [
        ("first", 5), ("second", 5)]
    assert [(c.request_id, c.tokens) for c in records[2].prefill_chunks] == [("second", 10)]
    assert first.first_token_step == 1 and second.first_token_step == 2


def test_arrival_step_and_future_requests_wait():
    sched = ChunkedPrefillScheduler(token_budget=10)
    r = sched.submit("late", prompt_tokens=5, max_new=1, arrival_step=3)
    records = _run(sched)
    assert [rec.total_tokens for rec in records[:3]] == [0, 0, 0]
    assert (r.first_token_step, r.ttft_steps, r.completion_step) == (3, 1, 3)


def test_zero_max_new_and_validation_and_cost_model():
    sched = ChunkedPrefillScheduler(token_budget=4)
    z = sched.submit("z", prompt_tokens=4, max_new=0)
    _run(sched)
    assert z.finished and z.first_token_step is None and z.generated_tokens == 0
    assert z.completion_step == 0

    with pytest.raises(ValueError):
        ChunkedPrefillRequest("bad", prompt_tokens=5, max_new_tokens=1,
                              cached_prefix_tokens=6)
    with pytest.raises(ValueError):
        ChunkedPrefillRequest("bad", prompt_tokens=-1, max_new_tokens=1)
    with pytest.raises(ValueError):
        ChunkedPrefillScheduler(token_budget=0)
    sched2 = ChunkedPrefillScheduler(token_budget=4)
    sched2.submit("dup", prompt_tokens=1, max_new=1)
    with pytest.raises(ValueError, match="duplicate"):
        sched2.submit("dup", prompt_tokens=1, max_new=1)
    # Negative cost rejected.
    sched3 = ChunkedPrefillScheduler(token_budget=4, step_cost_model=lambda w: -1.0)
    sched3.submit("n", prompt_tokens=1, max_new=1)
    with pytest.raises(ValueError, match="negative"):
        sched3.step()


def test_max_prefill_chunk_cap_spreads_leftover_fifo():
    # Cap 4/request/step, budget 10: A gets 4, B gets 4, leftover 2 unused
    # (only two prefilling) -- decode still budget-accounted next steps.
    sched = ChunkedPrefillScheduler(token_budget=10, max_prefill_chunk_tokens=4)
    sched.submit("A", prompt_tokens=8, max_new=1)
    sched.submit("B", prompt_tokens=8, max_new=1)
    rec0 = sched.step()
    assert [(c.request_id, c.tokens) for c in rec0.prefill_chunks] == [("A", 4), ("B", 4)]
    assert rec0.total_tokens == 8
