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


class _FakeTokenizer:
    """Word-level fake: each whitespace word is one token id."""

    def __init__(self):
        self._vocab = {}
        self._rev = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab) + 1
                self._rev[self._vocab[w]] = w
            ids.append(self._vocab[w])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self._rev[int(i)] for i in ids)


def test_exact_token_lengths_with_fake_tokenizer():
    tok = _FakeTokenizer()
    assert len(run_bench.build_token_ids(37, tok, seed=0)) == 37
    assert run_bench.build_token_ids(37, tok, seed=0) == run_bench.build_token_ids(37, tok, seed=0)
    assert len(run_bench.build_token_ids(0, tok)) == 0
    shape = {"id": "x", "prompt_tokens": 64, "output_tokens": 8,
             "shared_prefix_tokens": 0, "unique_tokens": 64}
    id_lists = run_bench.build_shape_prompt_ids(shape, 3, 0, tok)
    assert [len(x) for x in id_lists] == [64, 64, 64]
    prompts = run_bench.build_shape_prompts(shape, 3, 0, tokenizer=tok)
    assert [len(tok.encode(p)) for p in prompts] == [64, 64, 64]
    # Dry-run plan records exact actuals per request, not approximate.
    suite = {"shapes": [shape], "concurrency": [2], "repeats": 1,
             "arms": {"greedy": {}}}
    plan = run_bench.build_plan(suite, tokenizer=tok)
    assert plan["tokenizer_available"] is True
    assert plan["target_tokens_approximate"] is False
    assert plan["batches"][0]["prompt_tokens_actual"] == [64, 64]
    assert plan["batches"][0]["target_tokens_approximate"] is False
    # Without a tokenizer the fallback labels targets approximate.
    plan2 = run_bench.build_plan(suite)
    assert plan2["target_tokens_approximate"] is True
    assert plan2["batches"][0]["prompt_tokens_actual"] == [None, None]


def test_shared_prefix_identity_at_token_level():
    tok = _FakeTokenizer()
    shape = {"id": "agent-shared-prefix", "prompt_tokens": 8704,
             "output_tokens": 512, "shared_prefix_tokens": 8192,
             "unique_tokens": 512}
    small = dict(shape, prompt_tokens=40, shared_prefix_tokens=24,
                 unique_tokens=16)
    id_lists = run_bench.build_shape_prompt_ids(small, 4, 0, tok)
    assert [len(x) for x in id_lists] == [40, 40, 40, 40]
    prefix = id_lists[0][:24]
    for ids in id_lists:
        assert ids[:24] == prefix
    # Unique suffixes differ across the batch.
    assert len({tuple(x[24:]) for x in id_lists}) == 4


def test_arms_configs_parse():
    arms = _REPO / "benchmarks" / "arms"
    a = json.loads((arms / "arm-a-incumbent.json").read_text())
    b = json.loads((arms / "arm-b-mtp.json").read_text())
    assert a["version"] == 1 and a["arm"] == "A"
    assert a["model"] == "Qwen/Qwen3.8-27B-FP8"
    assert a["speculative_config"] is None
    assert b["version"] == 1 and b["arm"] == "B"
    assert b["status"] in ("UNMEASURED", "FAILED_TO_START", "MEASURED", "MEASURED-WITH-DEVIATION")
    assert b["num_speculative_tokens_sweep"] == [1, 2, 3]
    assert [e["speculative_config"]["num_speculative_tokens"] for e in b["entries"]] == [1, 2, 3]
    assert all(e["speculative_config"]["method"] == "mtp" for e in b["entries"])
    assert all(
        e["status"] in ("UNMEASURED", "FAILED_TO_START", "NOT_ATTEMPTED", "MEASURED", "MEASURED-WITH-DEVIATION")
        for e in b["entries"]
    )
    if b["status"] == "FAILED_TO_START":
        assert b["entries"][0]["status"] == "FAILED_TO_START"
        assert "KV cache" in b["entries"][0]["failure"]
    if b["status"] == "MEASURED-WITH-DEVIATION":
        assert b["entries"][0]["status"] == "MEASURED-WITH-DEVIATION"
        assert "256608" in b["entries"][0]["deviation"]
    assert (arms / "arm-a-incumbent.md").exists()
    assert (arms / "arm-b-mtp.md").exists()


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


def test_arms_filter_selects_subset():
    suite = _suite()
    assert run_bench.selected_arms(suite) == ["greedy", "sampled"]
    assert run_bench.selected_arms(suite, ["greedy"]) == ["greedy"]
    assert run_bench.selected_arms(suite, ["greedy,sampled"]) == [
        "greedy", "sampled"]
    with pytest.raises(ValueError):
        run_bench.selected_arms(suite, ["nope"])
    plan = run_bench.build_plan(suite, limit_concurrency=4, arms=["greedy"])
    assert plan["arms"] == ["greedy"]
    assert plan["total_batches"] == 4 * 2 * 2  # shapes x conc x repeats
    assert {b["arm"] for b in plan["batches"]} == {"greedy"}


def test_quality_probe_payload_and_parsing():
    spec = importlib.util.spec_from_file_location(
        "quality_probe", _REPO / "benchmarks" / "quality_probe.py")
    qp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qp)
    assert len(qp.PROBES) == 3
    payload = qp.build_payload("m", "hi")
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 32
    assert payload["repetition_penalty"] == 1.0
    assert payload["ignore_eos"] is False
    tok = _FakeTokenizer()
    body = {"choices": [{"text": " Paris", "finish_reason": "stop",
                         "logprobs": {"token_ids": [11, 12]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    parsed = qp.parse_completion(body, tok)
    assert parsed["output_token_ids"] == [11, 12]
    assert parsed["output_token_ids_source"] == "server_logprobs"
    assert parsed["finish_reason"] == "stop"
    body2 = {"choices": [{"text": " Paris", "finish_reason": "length"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    parsed2 = qp.parse_completion(body2, tok)
    assert parsed2["output_token_ids_source"] == "retokenized_local"


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
