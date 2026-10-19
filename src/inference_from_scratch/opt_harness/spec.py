"""Typed specifications for inference optimization tasks.

Adapted in concept from Harness² (Google, Apache-2.0) ``core/spec.py``:
the idea of a small typed seam describing a task before any search runs.
This adaptation is original code for LLM-inference metrics and candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DIRECTIONS = ("higher", "lower")


@dataclass(frozen=True)
class MetricSpec:
    """One declared metric, its optimisation direction and priority.

    Lower ``priority`` numbers are considered first during selection.
    """

    name: str
    direction: str = "higher"
    priority: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MetricSpec.name must not be empty")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"MetricSpec.direction must be one of {_DIRECTIONS}, got {self.direction!r}"
            )
        if int(self.priority) < 1:
            raise ValueError("MetricSpec.priority must be >= 1")

    def better_key(self, value: float) -> float:
        """Return a key where smaller is always better."""
        return -float(value) if self.direction == "higher" else float(value)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "priority": self.priority}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricSpec":
        return cls(
            name=str(data["name"]),
            direction=str(data.get("direction", "higher")),
            priority=int(data.get("priority", 1)),
        )


@dataclass(frozen=True)
class Candidate:
    """One proposed configuration / component change set."""

    name: str
    settings: dict[str, Any] = field(default_factory=dict)
    component_edits: dict[str, str] = field(default_factory=dict)
    provenance: str = ""
    hypothesis: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Candidate.name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "settings": dict(self.settings),
            "component_edits": dict(self.component_edits),
            "provenance": self.provenance,
            "hypothesis": self.hypothesis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        if not isinstance(data, dict):
            raise ValueError(f"Candidate payload must be an object, got {type(data).__name__}")
        edits = data.get("component_edits") or {}
        if isinstance(edits, list):
            # Accept [{"component": ..., "change": ...}, ...] artefacts.
            converted: dict[str, str] = {}
            for item in edits:
                if isinstance(item, dict) and "component" in item:
                    converted[str(item["component"])] = str(item.get("change", ""))
                else:
                    raise ValueError(f"invalid component_edits entry: {item!r}")
            edits = converted
        return cls(
            name=str(data["name"]),
            settings=dict(data.get("settings") or {}),
            component_edits={str(k): str(v) for k, v in dict(edits).items()},
            provenance=str(data.get("provenance", "")),
            hypothesis=str(data.get("hypothesis", "")),
        )


@dataclass(frozen=True)
class OptimizationTask:
    """A single optimisation problem in one domain."""

    task_id: str
    domain: str
    description: str = ""
    metrics: tuple[MetricSpec, ...] = ()
    base_candidate: Candidate = field(default_factory=lambda: Candidate("base"))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience alias used by callers that prefer ``id``.
    @property
    def id(self) -> str:
        return self.task_id

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("OptimizationTask.task_id must not be empty")
        if not self.domain.strip():
            raise ValueError("OptimizationTask.domain must not be empty")
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate metric names: {names}")

    def ordered_metrics(self) -> tuple[MetricSpec, ...]:
        return tuple(sorted(self.metrics, key=lambda m: (m.priority, m.name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "description": self.description,
            "metrics": [m.to_dict() for m in self.metrics],
            "base_candidate": self.base_candidate.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationTask":
        return cls(
            task_id=str(data.get("task_id", data.get("id", ""))),
            domain=str(data.get("domain", "")),
            description=str(data.get("description", "")),
            metrics=tuple(MetricSpec.from_dict(m) for m in data.get("metrics", [])),
            base_candidate=Candidate.from_dict(data["base_candidate"])
            if data.get("base_candidate")
            else Candidate("base"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ProbeResult:
    """Outcome of executing one candidate once."""

    label: str
    candidate_name: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    exact: bool = False
    delivered: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    trajectory: str = ""
    run_dir: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.delivered and self.exact and not self.errors)

    @property
    def error(self) -> str:
        return "; ".join(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "candidate_name": self.candidate_name,
            "metrics": dict(self.metrics),
            "exact": bool(self.exact),
            "delivered": bool(self.delivered),
            "artifacts": dict(self.artifacts),
            "errors": list(self.errors),
            "trajectory": self.trajectory,
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeResult":
        errors = data.get("errors") or []
        if isinstance(errors, str):
            errors = [errors] if errors else []
        metrics = {
            str(k): float(v)
            for k, v in dict(data.get("metrics") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return cls(
            label=str(data["label"]),
            candidate_name=str(data.get("candidate_name", "")),
            metrics=metrics,
            exact=bool(data.get("exact", False)),
            delivered=bool(data.get("delivered", False)),
            artifacts={str(k): str(v) for k, v in dict(data.get("artifacts") or {}).items()},
            errors=[str(e) for e in errors],
            trajectory=str(data.get("trajectory", "")),
            run_dir=str(data.get("run_dir", "")),
        )


__all__ = ["Candidate", "MetricSpec", "OptimizationTask", "ProbeResult"]
