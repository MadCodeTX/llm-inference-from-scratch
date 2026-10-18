"""Durable CPU tests for the speculative host-overhead instrumentation."""
import pytest

from inference_from_scratch.speculation import (
    SEGMENT_NAMES, SegmentAccumulator, summarize_segments)


def test_segment_accumulator_per_cycle_and_bookkeeping_residual():
    acc = SegmentAccumulator()
    acc.add("draft_replay_launch_ms", 2.0)
    acc.add("draft_token_readback_ms", 6.0)
    acc.add("snapshot_ms", 1.0)
    acc.add_loop_wall(12.0)
    acc.bump("item")
    acc.bump("tolist")
    acc.bump("copy_", 3)
    out: dict = {}
    acc.finish_into(out, cycles=2)
    assert out["segment_per_cycle_ms"]["draft_token_readback_ms"] == 3.0
    # residual bookkeeping = loop wall - explicit segments
    assert out["segment_per_cycle_ms"]["loop_bookkeeping_ms"] == 1.5
    assert out["sync_per_cycle"]["copy_"] == 1.5
    assert set(out["segment_totals_ms"]) == set(SEGMENT_NAMES)


def test_segment_accumulator_rejects_unknown_names():
    acc = SegmentAccumulator()
    with pytest.raises(ValueError):
        acc.add("nope_ms", 1.0)
    with pytest.raises(ValueError):
        acc.bump("nope")


def test_summarize_segments_ranks_descending():
    rows = summarize_segments({"a_ms": 1.0, "b_ms": 5.0, "c_ms": 3.0})
    assert [r["segment"] for r in rows] == ["b_ms", "c_ms", "a_ms"]
    assert rows[0]["ms_per_cycle"] == 5.0
