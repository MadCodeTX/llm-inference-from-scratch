"""Parallel and sequential recursion for k >= 1.

Adapted in concept from Harness² (Google, Apache-2.0) ``core/recursion.py``:
both modes spend exactly ``R = 4k`` rollouts and ``P = 3k`` sessions
(k base, 2k candidate probes, k finals; k invariant + k domain + k
compositions), then assert the ledger identity. This adaptation probes
inference candidates through an :class:`~.executor.Executor` instead of
rolling out harness trees. Original code.

The public runner is serial by default (``max_workers=1``) because GPU
probes must not collide; concurrency is an explicit opt-in parameter.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .executor import Executor
from .ledger import Ledger
from .library import Library
from .proposer import ProposalContext, Proposer, ProposerError
from .selection import SelectionResult, select_winner
from .spec import Candidate, OptimizationTask, ProbeResult

MODES = ("parallel", "sequential")
_MODE_TOKENS = {"parallel": "agg", "sequential": "seq", "agg": "agg", "seq": "seq"}


def mode_token(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in _MODE_TOKENS:
        raise ValueError(f"unknown mode {name!r}; expected parallel | sequential")
    return _MODE_TOKENS[key]


@dataclass
class StageRecord:
    label: str
    kind: str
    candidate: Candidate
    fell_back: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "candidate": self.candidate.to_dict(),
            "fell_back": self.fell_back,
            "error": self.error,
        }


@dataclass
class ModeResult:
    mode: str
    task_id: str
    k: int
    stages: list[StageRecord] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    results: dict[str, ProbeResult] = field(default_factory=dict)
    finals: list[str] = field(default_factory=list)
    ledger: Ledger | None = None
    selection: SelectionResult | None = None
    errors: list[str] = field(default_factory=list)
    library_entry: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.finals) and not self.errors and bool(self.selection and self.selection.ok)

    @property
    def winner_label(self) -> str:
        return self.selection.chosen_label if self.selection else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "finals": list(self.finals),
            "ok": self.ok,
            "winner_label": self.winner_label,
            "stages": [s.to_dict() for s in self.stages],
            "results": {lbl: r.to_dict() for lbl, r in self.results.items()},
            "selection": self.selection.to_dict() if self.selection else None,
            "ledger": self.ledger.to_dict() if self.ledger else None,
            "library_entry": self.library_entry,
            "errors": list(self.errors),
        }


class _Runner:
    def __init__(
        self,
        task: OptimizationTask,
        k: int,
        proposer: Proposer,
        executor: Executor,
        work_dir: Path,
        *,
        mode: str,
        library: Library | None,
        max_workers: int,
    ):
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.task = task
        self.k = int(k)
        self.proposer = proposer
        self.executor = executor
        self.work_dir = Path(work_dir)
        self.mode_token = mode_token(mode)
        self.mode = "parallel" if self.mode_token == "agg" else "sequential"
        self.library = library
        self.max_workers = max(1, int(max_workers))
        self.ledger = Ledger(mode=self.mode_token, task_id=task.task_id, k=self.k)
        self.result = ModeResult(mode=self.mode, task_id=task.task_id, k=self.k, ledger=self.ledger)

    # -- primitives -----------------------------------------------------
    def _probe(self, label: str, candidate: Candidate) -> ProbeResult:
        run_dir = self.work_dir / "runs" / label
        try:
            res = self.executor.execute(self.task, candidate, label, run_dir)
        except Exception as exc:  # an executor bug is recorded like a failed probe
            res = ProbeResult(
                label=label,
                candidate_name=candidate.name,
                errors=[f"{type(exc).__name__}: {exc}"],
                run_dir=str(run_dir),
            )
        self.result.candidates[label] = candidate
        self.result.results[label] = res
        return res

    def _probe_many(self, items: list[tuple[str, Candidate]]) -> dict[str, ProbeResult]:
        out: dict[str, ProbeResult] = {}
        if self.max_workers <= 1 or len(items) <= 1:
            for label, cand in items:
                out[label] = self._probe(label, cand)
                self.ledger.rollout(label)
            return out
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            probed = list(pool.map(lambda it: self._probe(it[0], it[1]), items))
        for (label, _cand), res in zip(items, probed):
            self.ledger.rollout(label)
            out[label] = res
        return out

    def _propose(
        self, *, kind: str, label: str, index: int, fallback: Candidate,
        pool: list[tuple[str, Candidate, ProbeResult]],
    ) -> Candidate:
        ctx = ProposalContext(
            task=self.task,
            kind=kind,
            label=label,
            index=index,
            fallback=fallback,
            base_results=[self.result.results[l] for l in sorted(self.result.results) if l.startswith("base")],
            pool=list(pool),
            library_entries=self.library.entries() if self.library else [],
        )
        try:
            candidate = self.proposer.propose(ctx)
            if candidate is None:
                raise ProposerError(f"proposer returned None for {label}")
        except Exception as exc:
            stage = StageRecord(label=label, kind=kind, candidate=fallback, fell_back=True,
                                error=f"{type(exc).__name__}: {exc}"[:400])
            self.result.stages.append(stage)
            self.result.errors.append(f"{label}: {stage.error}")
            self.ledger.session(label)
            return fallback
        self.result.stages.append(StageRecord(label=label, kind=kind, candidate=candidate))
        self.ledger.session(label)
        return candidate

    def _base_rollouts(self) -> list[tuple[str, Candidate, ProbeResult]]:
        items = [(f"base{i}", self.task.base_candidate) for i in range(1, self.k + 1)]
        probed = self._probe_many(items)
        return [(lbl, cand, probed[lbl]) for lbl, cand in items]

    def _finish(self, finals: list[str]) -> ModeResult:
        self.result.finals = list(finals)
        final_results = [self.result.results[l] for l in finals if l in self.result.results]
        self.result.selection = select_winner(
            final_results,
            self.task.ordered_metrics(),
            task_id=self.task.task_id,
            k=self.k,
            mode=self.mode_token,
        )
        if not self.result.selection.ok:
            self.result.errors.append(f"selection: {self.result.selection.error}")
        self.ledger.assert_identity()
        self._reflect()
        return self.result

    def _reflect(self) -> None:
        if self.library is None or self.result.selection is None:
            return
        if not self.result.selection.ok:
            return
        winner_label = self.result.selection.chosen_label
        winner = self.result.candidates.get(winner_label)
        if winner is None:
            return
        reflect = getattr(self.proposer, "reflect", None)
        if not callable(reflect):
            return
        try:
            experience = reflect(
                self.task, winner_label, winner, self.result.results.get(winner_label)
            )
        except Exception as exc:
            self.result.errors.append(f"reflection: {type(exc).__name__}: {exc}")
            return
        if experience is None:
            return
        entry = self.library.add(
            self.task.task_id,
            winner,
            experience=experience,
            label=winner_label,
            why=f"selected winner {winner_label} in {self.mode} mode (k={self.k})",
        )
        if entry is not None:
            self.result.library_entry = str(entry)


def run_parallel(
    task: OptimizationTask,
    k: int,
    *,
    proposer: Proposer,
    executor: Executor,
    work_dir: str | Path,
    library: Library | None = None,
    max_workers: int = 1,
) -> ModeResult:
    """k base, 2k independent candidates, k independent compositions."""
    r = _Runner(task, k, proposer, executor, Path(work_dir), mode="parallel",
                library=library, max_workers=max_workers)
    base = r._base_rollouts()
    specs: list[tuple[str, str, int]] = []
    for i in range(1, r.k + 1):
        specs.append((f"inv{i}", "invariant", i))
        specs.append((f"dom{i}", "domain", i))
    cand_items: list[tuple[str, Candidate]] = []
    for label, kind, idx in specs:
        cand = r._propose(kind=kind, label=label, index=idx,
                          fallback=task.base_candidate, pool=list(base))
        cand_items.append((label, cand))
    probed = r._probe_many(cand_items)
    pool = [(lbl, cand, probed[lbl]) for lbl, cand in cand_items]
    agg_items: list[tuple[str, Candidate]] = []
    for i in range(1, r.k + 1):
        label = f"agg{i}"
        cand = r._propose(kind="composition", label=label, index=i,
                          fallback=task.base_candidate, pool=pool)
        agg_items.append((label, cand))
    r._probe_many(agg_items)
    return r._finish([lbl for lbl, _ in agg_items])


def run_sequential(
    task: OptimizationTask,
    k: int,
    *,
    proposer: Proposer,
    executor: Executor,
    work_dir: str | Path,
    library: Library | None = None,
    max_workers: int = 1,
) -> ModeResult:
    """k nested rounds: each composes on top of the previous composition."""
    r = _Runner(task, k, proposer, executor, Path(work_dir), mode="sequential",
                library=library, max_workers=max_workers)
    base = r._base_rollouts()
    current = task.base_candidate
    history: list[tuple[str, Candidate, ProbeResult]] = list(base)
    finals: list[str] = []
    for t in range(1, r.k + 1):
        round_items: list[tuple[str, Candidate]] = []
        for kind, label in (("invariant", f"inv{t}"), ("domain", f"dom{t}")):
            cand = r._propose(kind=kind, label=label, index=t, fallback=current, pool=list(history))
            round_items.append((label, cand))
        probed = r._probe_many(round_items)
        round_pool = history + [(lbl, cand, probed[lbl]) for lbl, cand in round_items]
        clabel = f"c{t}"
        comp = r._propose(kind="composition", label=clabel, index=t, fallback=current, pool=round_pool)
        cres = r._probe_many([(clabel, comp)])[clabel]
        finals.append(clabel)
        history = round_pool + [(clabel, comp, cres)]
        current = comp
    return r._finish(finals)


def run_mode(
    mode: str,
    task: OptimizationTask,
    k: int,
    *,
    proposer: Proposer,
    executor: Executor,
    work_dir: str | Path,
    library: Library | None = None,
    max_workers: int = 1,
) -> ModeResult:
    token = mode_token(mode)
    if token == "agg":
        return run_parallel(task, k, proposer=proposer, executor=executor,
                            work_dir=work_dir, library=library, max_workers=max_workers)
    return run_sequential(task, k, proposer=proposer, executor=executor,
                          work_dir=work_dir, library=library, max_workers=max_workers)


__all__ = ["MODES", "ModeResult", "StageRecord", "mode_token", "run_mode", "run_parallel", "run_sequential"]
