"""Proposers: where candidate JSON comes from.

Adapted in concept from Harness² (Google, Apache-2.0) ``core/proposer.py``
and ``demo.py``: a proposer is called with a label and returns a candidate;
failures raise instead of becoming empty proposals. In this adaptation the
external producers are Muse sub-agents that drop JSON artefacts on disk
(:class:`FileProposer`); :class:`ScriptedProposer` is the deterministic
stand-in for tests and the demo. Original code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .library import Experience
from .spec import Candidate, OptimizationTask, ProbeResult


class ProposerError(RuntimeError):
    """A proposer call produced no usable candidate. Never swallowed."""


@dataclass
class ProposalContext:
    task: OptimizationTask
    kind: str  # "invariant" | "domain" | "composition"
    label: str
    index: int = 1
    fallback: Candidate | None = None
    base_results: list[ProbeResult] = field(default_factory=list)
    pool: list[tuple[str, Candidate, ProbeResult]] = field(default_factory=list)
    library_entries: list[str] = field(default_factory=list)


@runtime_checkable
class Proposer(Protocol):
    def propose(self, context: ProposalContext) -> Candidate: ...

    def reflect(
        self,
        task: OptimizationTask,
        winner_label: str,
        winner: Candidate,
        result: ProbeResult | None,
    ) -> Experience | None: ...


def _candidate_from_payload(payload: Any, *, source: str) -> Candidate:
    if isinstance(payload, dict) and "candidate" in payload and isinstance(
        payload["candidate"], dict
    ):
        payload = payload["candidate"]
    try:
        return Candidate.from_dict(payload)  # type: ignore[arg-type]
    except Exception as exc:
        raise ProposerError(f"{source}: invalid candidate artefact: {exc}") from exc


class FileProposer:
    """Loads ``<label>.json`` (and ``reflection.json``) from a directory.

    Artefacts are produced externally, e.g. by Muse sub-agents. A missing
    or invalid file raises :class:`ProposerError` visibly.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _load_json(self, path: Path) -> Any:
        if not path.is_file():
            raise ProposerError(f"FileProposer: missing artefact {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProposerError(f"FileProposer: invalid JSON in {path}: {exc}") from exc

    def propose(self, context: ProposalContext) -> Candidate:
        path = self.directory / f"{context.label}.json"
        payload = self._load_json(path)
        return _candidate_from_payload(payload, source=str(path))

    def reflect(
        self,
        task: OptimizationTask,
        winner_label: str,
        winner: Candidate,
        result: ProbeResult | None,
    ) -> Experience | None:
        path = self.directory / "reflection.json"
        payload = self._load_json(path)
        if isinstance(payload, dict) and "experience" in payload:
            payload = payload["experience"]
        if not isinstance(payload, dict):
            raise ProposerError(f"FileProposer: invalid reflection artefact {path}")
        return Experience.from_dict(payload)


class ScriptedProposer:
    """Deterministic proposer for tests and the GPU-free demo.

    Default behaviour derives a ``gain`` setting from the stage kind/index;
    a composition takes the best pool gain and adds a margin, so with the
    matching fake executor the composition deterministically wins.
    ``overrides`` maps a stage label to an explicit candidate.
    """

    def __init__(
        self,
        overrides: dict[str, Candidate] | None = None,
        *,
        reflection: Experience | None = None,
    ):
        self.overrides = dict(overrides or {})
        self._reflection = reflection
        self.calls: list[str] = []

    def propose(self, context: ProposalContext) -> Candidate:
        self.calls.append(context.label)
        if context.label in self.overrides:
            return self.overrides[context.label]
        i = context.index
        if context.kind == "invariant":
            return Candidate(
                name=f"inv{i}",
                settings={"gain": float(i)},
                component_edits={"attention": f"invariant tweak {i}"},
                provenance="scripted",
                hypothesis=f"invariant change {i} improves throughput",
            )
        if context.kind == "domain":
            return Candidate(
                name=f"dom{i}",
                settings={"gain": float(2 * i)},
                component_edits={"mlp": f"domain tweak {i}"},
                provenance="scripted",
                hypothesis=f"domain change {i} improves throughput",
            )
        if context.kind == "composition":
            best = 0.0
            for _lbl, cand, _res in context.pool:
                try:
                    best = max(best, float(cand.settings.get("gain", 0.0)))
                except (TypeError, ValueError):
                    continue
            if context.fallback is not None:
                try:
                    best = max(best, float(context.fallback.settings.get("gain", 0.0)))
                except (TypeError, ValueError):
                    pass
            return Candidate(
                name=context.label,
                settings={"gain": best + 4.0 + float(i)},
                component_edits={"composition": f"combine best pool gains (label {context.label})"},
                provenance="scripted",
                hypothesis="composition of the best observed gains wins",
            )
        raise ProposerError(f"ScriptedProposer: unknown kind {context.kind!r}")

    def reflect(
        self,
        task: OptimizationTask,
        winner_label: str,
        winner: Candidate,
        result: ProbeResult | None,
    ) -> Experience | None:
        if self._reflection is not None:
            return self._reflection
        gain = winner.settings.get("gain", "?")
        return Experience(
            cue=f"When optimising {task.domain} tasks like {task.task_id}",
            lesson=f"Candidate {winner.name} (gain={gain}) was the exact, delivered winner",
            evidence=(
                f"winner label {winner_label} metrics "
                f"{result.metrics if result else {}} exact={result.exact if result else False}"
            ),
            confidence="high (deterministic fake probe)",
            edits=json.dumps(winner.component_edits, sort_keys=True),
        )


__all__ = [
    "FileProposer",
    "ProposalContext",
    "Proposer",
    "ProposerError",
    "ScriptedProposer",
]
