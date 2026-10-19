"""Deterministic, GPU-free fake task for the demo / CLI path.

Adapted in concept from Harness² (Google, Apache-2.0) ``demo.py``: a small
deterministic adapter plus scripted proposer exercising the real loop.
Original code for inference metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .executor import FakeExecutor
from .library import Library
from .recursion import ModeResult, run_mode
from .spec import Candidate, MetricSpec, OptimizationTask


def demo_task() -> OptimizationTask:
    return OptimizationTask(
        task_id="demo-decode",
        domain="decode",
        description="Maximise decode throughput, then minimise latency, exactly.",
        metrics=(
            MetricSpec("throughput", "higher", priority=1),
            MetricSpec("latency_ms", "lower", priority=2),
        ),
        base_candidate=Candidate(
            "base", settings={"gain": 0.0}, hypothesis="unmodified baseline"
        ),
    )


def run_demo(
    work_dir: str | Path, *, k: int = 1, mode: str = "parallel", use_library: bool = True
) -> ModeResult:
    from .proposer import ScriptedProposer

    work = Path(work_dir)
    library = Library(work / "library", "decode") if use_library else None
    return run_mode(
        mode,
        demo_task(),
        k,
        proposer=ScriptedProposer(),
        executor=FakeExecutor(),
        work_dir=work / "run",
        library=library,
    )


def _decode_round_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a ``benchmarks/harness/*_round*.json`` artifact to probes.

    Uses the ABBA repeat medians when present, otherwise the first probe.
    Exactness requires both recorded references exact and finite logits.
    """
    from .spec import ProbeResult

    repeats = payload.get("repeats") or {}
    out: list[dict[str, Any]] = []
    for cand in payload.get("candidates") or []:
        cid = str(cand.get("candidate_id", ""))
        runs = cand.get("probe_runs") or {}
        first = next(iter(runs.values()), {})
        rep = repeats.get(cid) or {}
        tpot = rep.get("tpot_ms_median", first.get("tpot_ms"))
        thru = rep.get("decode_tok_per_s_median", first.get("decode_tok_per_s"))
        if tpot is None or thru is None:
            continue
        exact = bool(
            (first.get("vs_bridge") or {}).get("exact_match")
            and (first.get("vs_arm_a_recorded_vllm_probe") or {}).get("exact_match")
            and first.get("logits_finite")
        )
        if "all_exact_vs_bridge_and_arm_a" in rep:
            exact = exact and bool(rep["all_exact_vs_bridge_and_arm_a"])
        out.append(ProbeResult(
            label=cid,
            candidate_name=cid,
            metrics={"throughput": float(thru), "latency_ms": float(tpot)},
            exact=exact,
            delivered=cand.get("return_code") == 0 and not cand.get("failure_text"),
        ).to_dict())
    return out


def import_metrics_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalise an imported metrics JSON artefact to probe-result dicts.

    Accepts a list of probe-result objects, or an object mapping
    label -> {metrics, exact, delivered, candidate_name}.
    """
    from .spec import ProbeResult

    out: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            out.append(ProbeResult.from_dict(item).to_dict())
        return out
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        return _decode_round_results(payload)
    if isinstance(payload, dict):
        for label, value in payload.items():
            if not isinstance(value, dict):
                raise ValueError(f"metrics entry {label!r} must be an object")
            merged = dict(value)
            merged.setdefault("label", label)
            if "metrics" not in merged:
                merged = {
                    "label": label,
                    "candidate_name": str(value.get("candidate_name", label)),
                    "metrics": {
                        kk: vv for kk, vv in value.items()
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool)
                    },
                    "exact": value.get("exact", True),
                    "delivered": value.get("delivered", True),
                }
            out.append(ProbeResult.from_dict(merged).to_dict())
        return out
    raise ValueError("metrics JSON must be a list or an object keyed by label")


def load_metrics_file(path: str | Path) -> list:
    from .spec import ProbeResult

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ProbeResult.from_dict(d) for d in import_metrics_payload(payload)]


__all__ = ["demo_task", "import_metrics_payload", "load_metrics_file", "run_demo"]
