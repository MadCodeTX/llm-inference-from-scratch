"""Executors: run one candidate once and return a :class:`ProbeResult`.

Adapted in concept from Harness² (Google, Apache-2.0): a dead rollout is
recorded, never raised, so one failed probe cannot crash a whole round.
Original code for this repository.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .spec import Candidate, OptimizationTask, ProbeResult


@runtime_checkable
class Executor(Protocol):
    def execute(
        self, task: OptimizationTask, candidate: Candidate, label: str, run_dir: Path
    ) -> ProbeResult: ...


class FakeExecutor:
    """Deterministic in-memory executor for tests and the demo.

    Metrics derive from ``candidate.settings['gain']``:
    ``throughput = 10 + 5*gain`` and ``latency_ms = 100 - 2*gain`` (floored).
    ``exact``/``delivered`` default to True and can be forced per candidate
    via settings, which lets tests inject a fast-but-inexact candidate.
    An optional ``metric_fn(task, candidate) -> (metrics, exact, delivered)``
    overrides the default formula.
    """

    def __init__(
        self,
        metric_fn: Callable[
            [OptimizationTask, Candidate], tuple[dict[str, float], bool, bool]
        ]
        | None = None,
    ):
        self.metric_fn = metric_fn
        self.calls: list[str] = []

    def execute(
        self, task: OptimizationTask, candidate: Candidate, label: str, run_dir: Path
    ) -> ProbeResult:
        self.calls.append(label)
        if self.metric_fn is not None:
            metrics, exact, delivered = self.metric_fn(task, candidate)
            return ProbeResult(
                label=label,
                candidate_name=candidate.name,
                metrics=dict(metrics),
                exact=bool(exact),
                delivered=bool(delivered),
                run_dir=str(run_dir),
                trajectory=f"[fake] {label} gain={candidate.settings.get('gain')}",
            )
        gain = candidate.settings.get("gain", 0.0)
        try:
            gain_f = float(gain)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            gain_f = 0.0
        exact = bool(candidate.settings.get("exact", True))
        delivered = bool(candidate.settings.get("delivered", True))
        metrics = {
            "throughput": 10.0 + 5.0 * gain_f,
            "latency_ms": max(1.0, 100.0 - 2.0 * gain_f),
        }
        # Respect the task's declared metrics when they differ.
        if task.metrics:
            metrics = {m.name: metrics.get(m.name, 10.0 + 5.0 * gain_f) for m in task.metrics}
            # Keep latency semantics for a declared lower-is-better metric.
            for m in task.metrics:
                if m.direction == "lower":
                    metrics[m.name] = max(1.0, 100.0 - 2.0 * gain_f)
        errors: list[str] = []
        if candidate.settings.get("fail"):
            errors.append("fake executor: candidate requested failure")
            delivered = False
        return ProbeResult(
            label=label,
            candidate_name=candidate.name,
            metrics=metrics,
            exact=exact,
            delivered=delivered,
            errors=errors,
            artifacts={"settings": json.dumps(candidate.settings, sort_keys=True)},
            run_dir=str(run_dir),
            trajectory=f"[fake] executed {label} ({candidate.name})",
        )


class CommandExecutor:
    """Run an argv template in a run directory and read a metrics JSON file.

    Each argv element may contain ``{run_dir}``, ``{candidate}``,
    ``{candidate_name}``, ``{task_id}``, ``{label}`` and ``{settings}``
    (a JSON file written into the run dir). Stdout/stderr are captured as
    the trajectory. Any failure (non-zero exit, missing/invalid metrics
    file) is recorded on the returned result instead of raising.
    """

    def __init__(
        self,
        argv_template: Sequence[str],
        *,
        metrics_file: str = "metrics.json",
        timeout: float = 60.0,
        exact_key: str = "exact",
        delivered_key: str = "delivered",
    ):
        if not argv_template:
            raise ValueError("CommandExecutor needs a non-empty argv template")
        self.argv_template = [str(a) for a in argv_template]
        self.metrics_file = metrics_file
        self.timeout = float(timeout)
        self.exact_key = exact_key
        self.delivered_key = delivered_key

    def execute(
        self, task: OptimizationTask, candidate: Candidate, label: str, run_dir: Path
    ) -> ProbeResult:
        run_dir = Path(run_dir)
        result = ProbeResult(label=label, candidate_name=candidate.name, run_dir=str(run_dir))
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            settings_path = run_dir / "candidate.json"
            settings_path.write_text(
                json.dumps(candidate.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            fmt = {
                "run_dir": str(run_dir),
                "candidate": str(settings_path),
                "candidate_name": candidate.name,
                "task_id": task.task_id,
                "label": label,
                "settings": str(settings_path),
            }
            argv = []
            for part in self.argv_template:
                rendered = part
                for key, value in fmt.items():
                    rendered = rendered.replace("{" + key + "}", value)
                argv.append(rendered)
            proc = subprocess.run(
                argv,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            result.trajectory = (
                f"$ {' '.join(argv)}\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}"
            )
            result.artifacts["argv"] = json.dumps(argv)
            if proc.returncode != 0:
                result.errors.append(f"command exited {proc.returncode}")
                result.delivered = False
                result.exact = False
                return result
            metrics_path = run_dir / self.metrics_file
            if not metrics_path.is_file():
                result.errors.append(f"metrics file not found: {self.metrics_file}")
                result.delivered = False
                result.exact = False
                return result
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                result.errors.append(f"invalid metrics JSON: {exc}")
                result.delivered = False
                result.exact = False
                return result
            if not isinstance(payload, dict):
                result.errors.append("metrics JSON must be an object")
                result.delivered = False
                result.exact = False
                return result
            result.exact = bool(payload.pop(self.exact_key, True))
            result.delivered = bool(payload.pop(self.delivered_key, True))
            metrics: dict[str, float] = {}
            for key, value in payload.items():
                if isinstance(value, bool):
                    continue
                try:
                    metrics[str(key)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
            result.metrics = metrics
            result.artifacts["metrics_file"] = str(metrics_path)
            return result
        except subprocess.TimeoutExpired:
            result.errors.append(f"command timed out after {self.timeout}s")
            result.delivered = False
            result.exact = False
            return result
        except Exception as exc:  # recorded, never raised: one probe must not kill a round
            result.errors.append(f"{type(exc).__name__}: {exc}")
            result.delivered = False
            result.exact = False
            return result


__all__ = ["CommandExecutor", "Executor", "FakeExecutor"]
