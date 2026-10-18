"""Tests for Ch 11: continuous batching + paged block allocation.

Scheduler tests drive the batcher with deterministic counting
samplers (no model): the property under test is *scheduling* --
requests finishing at different steps, new requests admitted
mid-flight, slots refilled -- so each request's finish step is
asserted exactly. Allocator tests assert exact block counts and the
paging property (non-contiguous reuse after fragmentation). The
latency helper is smoke-run on a tiny real model: timings must be
present and sane, token ids must match generate() greedy exactly --
never that timing takes any particular value.
"""

import pytest
import torch

from inference_from_scratch.config import ModelConfig
from inference_from_scratch.generate import generate
from inference_from_scratch.model import DecoderLM
from inference_from_scratch.serving import (
    BlockAllocator,
    ContinuousBatcher,
    Request,
    compute_throughput,
    compute_tpot,
    measure_prefill_decode,
)


def _counting_sampler(request, step):
    """One token per step; token value encodes the request's own step."""
    return len(request.generated)  # 0, 1, 2, ... per request


# --- Request ------------------------------------------------------------------


def test_request_finished_property():
    r = Request(prompt_ids=[1, 2], max_new_tokens=2)
    assert not r.finished
    assert r.generated == [] and r.total_len == 2
    r.generated.append(5)
    assert not r.finished
    r.generated.append(6)
    assert r.finished
    assert r.total_len == 4

    eos = Request(prompt_ids=[1], max_new_tokens=10, eos_id=99)
    eos.generated.extend([3, 99])
    assert eos.finished  # EOS finishes early, EOS id included

    zero = Request(prompt_ids=[1], max_new_tokens=0)
    assert zero.finished


# --- ContinuousBatcher ---------------------------------------------------------


def test_batcher_finishes_requests_at_different_steps():
    batcher = ContinuousBatcher()
    short = batcher.add(Request([1], max_new_tokens=1))
    long_ = batcher.add(Request([2], max_new_tokens=3))

    batcher.step(_counting_sampler)  # step 1: short finishes
    assert batcher.finished == [short]
    assert batcher.running == [long_]
    assert short.generated == [0]

    batcher.step(_counting_sampler)  # step 2
    assert batcher.finished == [short]
    batcher.step(_counting_sampler)  # step 3: long finishes
    assert batcher.finished == [short, long_]
    assert long_.generated == [0, 1, 2]
    assert not batcher.has_work


def test_batcher_admits_new_requests_mid_flight():
    batcher = ContinuousBatcher()
    first = batcher.add(Request([1], max_new_tokens=3))
    batcher.step(_counting_sampler)
    assert len(first.generated) == 1

    # Added while `first` is mid-flight: admitted at the NEXT step,
    # without waiting for `first` to finish (continuous batching).
    second = batcher.add(Request([2], max_new_tokens=1))
    batcher.step(_counting_sampler)
    assert batcher.finished == [second]
    assert len(first.generated) == 2

    batcher.step(_counting_sampler)
    assert batcher.finished == [second, first]


def test_batcher_respects_max_batch_size():
    batcher = ContinuousBatcher(max_batch_size=1)
    a = batcher.add(Request([1], max_new_tokens=1))
    b = batcher.add(Request([2], max_new_tokens=1))
    batcher.step(_counting_sampler)
    assert batcher.finished == [a]
    assert batcher.waiting == [b]  # slot freed only after a's step
    batcher.step(_counting_sampler)
    assert batcher.finished == [a, b]


def test_batcher_default_rule_and_run_and_eos():
    batcher = ContinuousBatcher()
    r = batcher.add(Request([10, 11], max_new_tokens=3))
    out = batcher.run()  # default rule: last token + 1 (mod 256)
    assert out == [r]
    assert r.generated == [12, 13, 14]

    eos_batcher = ContinuousBatcher(eos_id=7)
    r2 = eos_batcher.add(Request([1], max_new_tokens=100))
    eos_batcher.run(lambda req, step: 7 if step == 2 else 1)
    assert r2.generated == [1, 7]
    assert r2.finished


def test_batcher_single_arg_sampler_accepted():
    batcher = ContinuousBatcher()
    r = batcher.add(Request([5], max_new_tokens=2))
    batcher.run(lambda req: 42)  # 1-arg sampler (no step argument)
    assert r.generated == [42, 42]


# --- BlockAllocator -------------------------------------------------------------


def test_allocator_refuses_over_allocation_atomically():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.free_blocks == 4
    assert alloc.allocate(n_tokens=65) is None  # needs 5 blocks > 4
    assert alloc.free_blocks == 4  # refused: pool untouched
    assert alloc.allocated_blocks == 0
    ids = alloc.allocate(n_tokens=64)
    assert ids is not None and len(ids) == 4
    assert alloc.allocate(n_tokens=1) is None  # pool now empty
    assert alloc.free_blocks == 0


def test_allocator_free_returns_blocks_and_double_free_raises():
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    ids = alloc.allocate(n_tokens=8)  # 2 blocks
    assert alloc.free_blocks == 1
    alloc.free(ids)
    assert alloc.free_blocks == 3
    assert alloc.allocated_blocks == 0
    with pytest.raises(ValueError, match="double free|not currently"):
        alloc.free(ids)
    with pytest.raises(ValueError):
        alloc.free([999])


def test_allocator_block_count_math_and_zero_tokens():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    assert alloc.blocks_for_tokens(0) == 0
    assert alloc.blocks_for_tokens(16) == 1
    assert alloc.blocks_for_tokens(17) == 2
    assert alloc.wasted_slots(17) == 15  # internal fragmentation
    assert alloc.allocate(0) == []
    assert alloc.free_blocks == 8


def test_allocator_fragmentation_still_allocates_noncontiguous():
    # Classic paging demo: allocate 4 blocks, free the middle two so
    # free blocks are scattered; a new 2-block request still succeeds
    # because paged blocks need not be contiguous.
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    a = alloc.allocate(4)   # 1 block
    b = alloc.allocate(4)   # 1 block
    c = alloc.allocate(4)   # 1 block
    d = alloc.allocate(4)   # 1 block
    assert alloc.free_blocks == 0
    alloc.free(b)
    alloc.free(d)  # free blocks are now non-adjacent (b and d slots)
    assert alloc.free_blocks == 2

    e = alloc.allocate(8)  # needs 2 blocks: succeeds despite the holes
    assert e is not None and len(e) == 2
    assert set(e) == set(b) | set(d)  # exactly the scattered holes reused
    assert alloc.free_blocks == 0
    assert all(alloc.is_allocated(x) for x in a + c + e)


# --- TTFT / TPOT / throughput ---------------------------------------------------


def test_tpot_throughput_definitions():
    assert compute_tpot(2.0, 4) == 0.5
    assert compute_tpot(1.0, 0) == 0.0
    assert compute_throughput(100, 2.0) == 50.0
    assert compute_throughput(10, 0.0) == 0.0


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=64, max_seq_len=64,
    )
    return DecoderLM(cfg).eval()


def test_measure_prefill_decode_matches_generate_greedy():
    model = _tiny_model()
    prompt = [1, 2, 3]
    stats = measure_prefill_decode(model, prompt, max_new_tokens=5)
    assert stats["n_new_tokens"] == 5
    assert stats["ttft_s"] >= 0.0
    assert stats["tpot_s"] >= 0.0
    assert stats["throughput_tok_s"] > 0.0
    assert stats["total_s"] == pytest.approx(
        stats["ttft_s"] + stats["decode_s"]
    )
    want = generate(
        model, torch.tensor([prompt]), max_new_tokens=5, temperature=0,
    )
    assert stats["token_ids"] == want.squeeze(0).tolist()  # same tokens
