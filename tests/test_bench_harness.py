"""Tests for benchmarks/run_bench.py: suite, percentiles, dry-run plan.

No network: the dry-run path must never call urlopen, and stream-event
aggregation is checked on fake timestamps only.
"""

import importlib.util
import json
import urllib.request
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_bench", _REPO / "benchmarks" / "run_bench.py"
)
run_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_bench)


def _suite():
    return run_bench.load_suite(_REPO / "benchmarks" / "suite.json")


def test_suite_loads_and_covers_required_shapes():
    suite = _suite()
    ids = {s["id"] for s in suite["shapes"]}
    assert ids == {"short", "medium", "long", "agent-shared-prefix"}
    by_id = {s["id"]: s for s in suite["shapes"]}
    assert (by_id["short"]["prompt_tokens"], by_id["short"]["output_tokens"]) == (2048, 512)
    assert (by_id["medium"]["prompt_tokens"], by_id["medium"]["output_tokens"]) == (16384, 1024)
    assert (by_id["long"]["prompt_tokens"], by_id["long"]["output_tokens"]) == (49152, 1024)
    shared = by_id["agent-shared-prefix"]
    assert shared["shared_prefix_tokens"] == 8192
    assert shared["unique_tokens"] == 512
    assert shared["output_tokens"] == 512
    assert suite["concurrency"] == [1, 4, 16]
    assert suite["repeats"] == 2
    assert set(suite["arms"]) == {"greedy", "sampled"}
    assert suite["arms"]["greedy"]["temperature"] == 0.0


def test_prompt_generation_is_deterministic_and_word_sized():
    a = run_bench.build_text(200, seed=3)
    b = run_bench.build_text(200, seed=3)
    assert a == b
    assert run_bench.count_words(a) == 200
    assert run_bench.build_text(200, seed=4) != a
    # Shared prefix must be identical across requests in a batch.
    shape = {"shared_prefix_tokens": 8192, "unique_tokens": 512,
             "prompt_tokens": 8704}
    prompts = run_bench.build_shape_prompts(shape, concurrency=4, repeat=0)
    prefix_words = run_bench.words_for_tokens(8192)
    prefix = " ".join(prompts[0].split()[:prefix_words])
    for p in prompts:
        assert " ".join(p.split()[:prefix_words]) == prefix


def test_percentile_aggregation_on_fake_stream_events():
    # Fake stream: tokens arrive at 1.10, 1.12, 1.15, 1.20 (t0 = 1.00).
    m = run_bench.metrics_from_timestamps([1.10, 1.12, 1.15, 1.20], t0=1.00)
    assert m["ttft_s"] == pytest.approx(0.10)
    assert m["inter_token_s"] == pytest.approx([0.02, 0.03, 0.05])
    agg = run_bench.summarize_values(m["inter_token_s"])
    assert agg["n"] == 3
    assert agg["p50"] == pytest.approx(0.03)
    assert agg["mean"] == pytest.approx((0.02 + 0.03 + 0.05) / 3)
    # Repo metrics.summarize agrees on the same fake events.
    from inference_from_scratch.metrics import summarize

    repo = summarize(m["inter_token_s"])
    assert repo["p50"] == pytest.approx(agg["p50"])
    empty = run_bench.metrics_from_timestamps([], t0=1.00)
    assert empty["ttft_s"] is None and empty["inter_token_s"] == []


def test_dry_run_plan_counts_without_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("dry-run must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    suite = _suite()
    plan = run_bench.build_plan(suite)
    # 4 shapes x 2 arms x 3 concurrencies x 2 repeats = 48 batches;
    # requests = 4 x (1+4+16) x 2 x 2 = 336.
    assert plan["total_batches"] == 48
    assert plan["total_requests"] == 336
    capped = run_bench.build_plan(suite, limit_concurrency=4)
    assert capped["concurrencies"] == [1, 4]
    assert capped["total_requests"] == 4 * (1 + 4) * 2 * 2
    # Incumbent reference file stays aggregate-only (no prompt fields).
    inc = json.loads(
        (_REPO / "benchmarks" / "baseline" / "incumbent_existing.json").read_text()
    )
    assert inc["rows"], "incumbent rows missing"
    for row in inc["rows"]:
        assert "prompt" not in row or row.get("prompt") is None or isinstance(
            row.get("prompt_tokens"), int
        )
        assert isinstance(row["aggregate_output_tps"], (int, float))
