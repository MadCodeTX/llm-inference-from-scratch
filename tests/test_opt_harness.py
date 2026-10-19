"""Tests for the recursive optimisation harness (GPU-free, deterministic)."""

import json
import sys
from pathlib import Path

import pytest

from inference_from_scratch.opt_harness import (
    Candidate,
    CommandExecutor,
    Experience,
    FakeExecutor,
    FileProposer,
    Ledger,
    Library,
    MetricSpec,
    OptimizationTask,
    ProbeResult,
    ProposerError,
    ScriptedProposer,
    assign_ids,
    expected_rollouts,
    expected_sessions,
    pareto_frontier,
    run_parallel,
    run_sequential,
    select_winner,
)
from inference_from_scratch.opt_harness.demo import demo_task, load_metrics_file, run_demo
from inference_from_scratch.opt_harness.ledger import BudgetError
from inference_from_scratch.opt_harness.proposer import ProposalContext


def _metrics():
    return (MetricSpec("throughput", "higher", 1), MetricSpec("latency_ms", "lower", 2))


# --- spec JSON ---------------------------------------------------------

def test_spec_json_round_trips():
    task = demo_task()
    assert OptimizationTask.from_dict(json.loads(json.dumps(task.to_dict()))) == task
    cand = Candidate("c1", settings={"gain": 3}, component_edits={"attention": "x"},
                     provenance="test", hypothesis="faster")
    assert Candidate.from_dict(json.loads(json.dumps(cand.to_dict()))) == cand
    res = ProbeResult(label="agg1", candidate_name="c1", metrics={"throughput": 42.0},
                      exact=True, delivered=True, artifacts={"a": "b"}, errors=[])
    back = ProbeResult.from_dict(json.loads(json.dumps(res.to_dict())))
    assert back == res
    with pytest.raises(ValueError):
        MetricSpec("x", "sideways")


# --- ledger ------------------------------------------------------------

def test_ledger_identity_and_duplicates():
    assert (expected_rollouts(1), expected_sessions(1)) == (4, 3)
    assert (expected_rollouts(2), expected_sessions(2)) == (8, 6)
    led = Ledger(mode="agg", task_id="t", k=1)
    for lbl in ("base1", "inv1", "dom1", "agg1"):
        led.rollout(lbl)
    for lbl in ("inv1", "dom1", "agg1"):
        led.session(lbl)
    led.assert_identity()
    payload = json.loads(led.to_json())
    assert payload["R"] == 4 and payload["expected_R"] == 4
    assert payload["P"] == 3 and payload["expected_P"] == 3

    dup = Ledger(mode="agg", task_id="t", k=1)
    for lbl in ("base1", "base1", "dom1", "agg1"):
        dup.rollout(lbl)
    for lbl in ("inv1", "dom1", "agg1"):
        dup.session(lbl)
    with pytest.raises(BudgetError, match="DUPLICATE"):
        dup.assert_identity()

    short = Ledger(mode="agg", task_id="t", k=2)
    short.rollout("base1")
    with pytest.raises(BudgetError):
        short.assert_identity()
    with pytest.raises(ValueError):
        expected_rollouts(0)


# --- selection ---------------------------------------------------------

def _res(label, thr, lat, *, exact=True, delivered=True, errors=None):
    return ProbeResult(label=label, candidate_name=label, metrics={"throughput": thr, "latency_ms": lat},
                       exact=exact, delivered=delivered, errors=errors or [])


def test_selection_exactness_gate_and_pareto():
    fast_inexact = _res("fast", 1000.0, 1.0, exact=False)
    good = _res("good", 100.0, 50.0)
    slow = _res("slow", 50.0, 90.0)
    undelivered = _res("ghost", 500.0, 5.0, delivered=False)
    sel = select_winner([fast_inexact, good, slow, undelivered], _metrics(), task_id="t", k=1, mode="agg")
    assert sel.ok and sel.chosen_label == "good"
    assert set(sel.dropped) == {"fast", "ghost"}
    assert "not exact" in sel.drop_reasons["fast"]
    frontier = pareto_frontier([good, slow], _metrics())
    assert [r.label for r in frontier] == ["good"]
    # priority: throughput first, so higher throughput on the frontier wins
    a = _res("a", 120.0, 80.0)
    b = _res("b", 100.0, 10.0)
    assert {r.label for r in pareto_frontier([a, b], _metrics())} == {"a", "b"}
    assert select_winner([b, a], _metrics()).chosen_label == "a"
    # anonymised ids are stable and cover every label
    m1 = assign_ids(["b", "a"], task_id="t", k=1, mode="agg")
    assert m1 == assign_ids(["a", "b"], task_id="t", k=1, mode="agg")
    assert sorted(m1.values()) == ["a", "b"] and set(m1) == {"A", "B"}
    # nobody eligible -> not ok, never picks the inexact one
    none_sel = select_winner([fast_inexact], _metrics())
    assert not none_sel.ok and none_sel.chosen_label == ""


# --- library -----------------------------------------------------------

def test_library_adoption_index_and_forget(tmp_path):
    lib = Library(tmp_path / "lib", "decode")
    cand = Candidate("winner", settings={"gain": 9})
    incomplete = Experience(cue="c", lesson="l", evidence="e")  # no confidence
    assert lib.add("task-1", cand, experience=incomplete) is None
    assert lib.entries() == []
    full = Experience(cue="cue text", lesson="lesson text", evidence="ev", confidence="high")
    entry = lib.add("task-1", cand, experience=full, label="agg1")
    assert entry is not None and lib.entries() == ["task-1"]
    text = lib.read("task-1")
    assert text and "cue text" in text and "confidence" in text
    assert lib.read_candidate("task-1") == cand
    index = (lib.dir / "INDEX.md").read_text()
    assert "task-1" in index and "cue text" in index
    lib.add("task-2", cand, experience=full, label="agg1")
    assert lib.entries() == ["task-1", "task-2"]
    lib.forget(["task-1", "unknown-task"])
    assert lib.entries() == ["task-2"]
    index = lib.rebuild_index().read_text()
    assert "task-2" in index and "task-1" not in index
    lib.forget(["task-2"])
    assert lib.entries() == [] and "empty" in lib.index.read_text()


# --- file proposer -----------------------------------------------------

def _ctx(label="inv1", kind="invariant"):
    return ProposalContext(task=demo_task(), kind=kind, label=label, index=1)


def test_file_proposer_loads_and_fails_visibly(tmp_path):
    (tmp_path / "inv1.json").write_text(json.dumps(
        {"name": "from-file", "settings": {"gain": 7}, "hypothesis": "file candidate"}))
    prop = FileProposer(tmp_path)
    cand = prop.propose(_ctx())
    assert cand.name == "from-file" and cand.settings["gain"] == 7
    with pytest.raises(ProposerError, match="missing artefact"):
        prop.propose(_ctx(label="dom9", kind="domain"))
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ProposerError, match="invalid JSON"):
        prop.propose(_ctx(label="bad"))
    (tmp_path / "agg1.json").write_text(json.dumps({"settings": {}}))  # no name
    with pytest.raises(ProposerError, match="invalid candidate"):
        prop.propose(_ctx(label="agg1", kind="composition"))
    with pytest.raises(ProposerError, match="missing artefact"):
        prop.reflect(demo_task(), "agg1", cand, None)
    (tmp_path / "reflection.json").write_text(json.dumps(
        {"cue": "c", "lesson": "l", "evidence": "e", "confidence": "high"}))
    exp = prop.reflect(demo_task(), "agg1", cand, None)
    assert exp is not None and exp.is_complete


# --- command executor ----------------------------------------------------

_WRITER = (
    "import json,sys;"
    "json.dump({'throughput': 99.0, 'latency_ms': 12.0, 'exact': True}, open('metrics.json','w'));"
    "print('hello trajectory')"
)


def test_command_executor_success_and_failure(tmp_path):
    task = demo_task()
    ok_exec = CommandExecutor([sys.executable, "-c", _WRITER])
    res = ok_exec.execute(task, Candidate("c"), "run1", tmp_path / "run1")
    assert res.metrics == {"throughput": 99.0, "latency_ms": 12.0}
    assert res.exact and res.delivered and not res.errors
    assert "hello trajectory" in res.trajectory

    fail_exec = CommandExecutor([sys.executable, "-c", "import sys; sys.exit(3)"])
    bad = fail_exec.execute(task, Candidate("c"), "run2", tmp_path / "run2")
    assert bad.errors and "exited 3" in bad.errors[0] and not bad.delivered

    missing = CommandExecutor([sys.executable, "-c", "print('no metrics')"]).execute(
        task, Candidate("c"), "run3", tmp_path / "run3")
    assert missing.errors and "metrics file not found" in missing.errors[0]


# --- recursion -----------------------------------------------------------

@pytest.mark.parametrize("mode_fn,mode,final_prefix", [
    (run_parallel, "parallel", "agg"), (run_sequential, "sequential", "c")])
@pytest.mark.parametrize("k", [1, 2])
def test_recursion_both_modes_k1_k2(tmp_path, mode_fn, mode, final_prefix, k):
    lib = Library(tmp_path / "lib", "decode")
    result = mode_fn(demo_task(), k, proposer=ScriptedProposer(),
                     executor=FakeExecutor(), work_dir=tmp_path / "work", library=lib)
    assert result.ok, result.errors
    ledger = result.ledger.to_dict()
    assert ledger["R"] == expected_rollouts(k) == 4 * k
    assert ledger["P"] == expected_sessions(k) == 3 * k
    assert result.finals == [f"{final_prefix}{i}" for i in range(1, k + 1)]
    assert result.winner_label == f"{final_prefix}{k}"
    winner = result.results[result.winner_label]
    assert winner.exact and winner.delivered
    # reflection adopted into the domain library
    assert lib.entries() == ["demo-decode"]
    assert result.library_entry


def test_recursion_inexact_fast_final_never_wins(tmp_path):
    overrides = {
        "agg1": Candidate("agg1", settings={"gain": 100.0, "exact": False}),
        "agg2": Candidate("agg2", settings={"gain": 1.0}),
    }
    result = run_parallel(demo_task(), 2, proposer=ScriptedProposer(overrides),
                          executor=FakeExecutor(), work_dir=tmp_path / "w")
    assert result.selection.chosen_label == "agg2"
    assert "agg1" in result.selection.dropped
    assert result.ledger.R == 8 and result.ledger.P == 6


def test_recursion_failed_probe_does_not_crash_round(tmp_path):
    overrides = {"agg1": Candidate("agg1", settings={"gain": 1.0, "fail": True})}
    result = run_parallel(demo_task(), 1, proposer=ScriptedProposer(overrides),
                          executor=FakeExecutor(), work_dir=tmp_path / "w")
    assert result.results["agg1"].errors  # recorded
    assert not result.selection.ok  # sole final gated out, visibly


# --- demo / metrics import ----------------------------------------------

def test_demo_and_metrics_import(tmp_path):
    result = run_demo(tmp_path / "demo", k=1, mode="parallel")
    assert result.ok and result.winner_label == "agg1"
    artefact = tmp_path / "metrics.json"
    artefact.write_text(json.dumps({
        "slow-exact": {"throughput": 10.0, "latency_ms": 90.0, "exact": True, "delivered": True},
        "fast-inexact": {"throughput": 999.0, "latency_ms": 1.0, "exact": False, "delivered": True},
    }))
    results = load_metrics_file(artefact)
    sel = select_winner(results, _metrics(), task_id="imported")
    assert sel.ok and sel.chosen_label == "slow-exact"


def test_decode_round_artifact_import_selects_exact_fastest():
    artifact = Path(__file__).resolve().parents[1] / "benchmarks/harness/decode_round1.json"
    results = load_metrics_file(artifact)
    by_label = {r.label: r for r in results}
    assert set(by_label) == {f"C{i}_{suffix}" for i, suffix in enumerate(
        ["base", "scale_cache_on", "lm_head_replicated", "norm_manual",
         "fp8_config_old", "delta_ref"])}
    # ABBA medians, not first probes, for the repeated candidates.
    assert by_label["C0_base"].metrics["latency_ms"] == pytest.approx(23.7264, abs=1e-3)
    assert by_label["C1_scale_cache_on"].metrics["latency_ms"] == pytest.approx(22.8657, abs=1e-3)
    assert all(r.exact and r.delivered for r in results)
    sel = select_winner(results, _metrics(), task_id="decode-tpot")
    assert sel.ok and sel.chosen_label == "C1_scale_cache_on"
