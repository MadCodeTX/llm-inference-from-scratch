"""Selection: hard exactness gate, Pareto frontier, priority choice.

Adapted in concept from Harness² (Google, Apache-2.0) ``core/selection.py``:
anonymised candidate IDs and a delivered-only pool. This adaptation replaces
an LLM selector with a deterministic metric selector: a candidate that is
inexact or undelivered can never win, however fast it is. Original code.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .spec import MetricSpec, ProbeResult


def _anon_id(index: int) -> str:
    out = ""
    n = int(index) + 1
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def assign_ids(
    labels: Sequence[str], *, task_id: str = "", k: int = 1, mode: str = ""
) -> dict[str, str]:
    """Anonymised id -> label, stable across reruns for the same seed inputs."""
    seed = int(hashlib.sha1(f"{task_id}|{k}|{mode}".encode()).hexdigest()[:12], 16)
    ordered = sorted(labels)
    random.Random(seed).shuffle(ordered)
    return {_anon_id(i): lbl for i, lbl in enumerate(ordered)}


def _dominates(a: ProbeResult, b: ProbeResult, metrics: Sequence[MetricSpec]) -> bool:
    """True if ``a`` is at least as good as ``b`` everywhere and better somewhere."""
    at_least = True
    strictly = False
    for m in metrics:
        if m.name not in a.metrics or m.name not in b.metrics:
            return False
        av, bv = float(a.metrics[m.name]), float(b.metrics[m.name])
        if m.direction == "higher":
            if av < bv:
                at_least = False
            if av > bv:
                strictly = True
        else:
            if av > bv:
                at_least = False
            if av < bv:
                strictly = True
    return at_least and strictly


def pareto_frontier(
    results: Sequence[ProbeResult], metrics: Sequence[MetricSpec]
) -> list[ProbeResult]:
    """Non-dominated subset of ``results`` (input order preserved)."""
    pool = list(results)
    frontier: list[ProbeResult] = []
    for cand in pool:
        dominated = any(
            other is not cand and _dominates(other, cand, metrics) for other in pool
        )
        if not dominated:
            frontier.append(cand)
    return frontier


@dataclass
class SelectionResult:
    chosen_label: str = ""
    chosen_id: str = ""
    chosen_candidate: str = ""
    id_map: dict[str, str] = field(default_factory=dict)
    frontier_labels: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_label": self.chosen_label,
            "chosen_id": self.chosen_id,
            "chosen_candidate": self.chosen_candidate,
            "id_map": dict(self.id_map),
            "frontier_labels": list(self.frontier_labels),
            "dropped": list(self.dropped),
            "drop_reasons": dict(self.drop_reasons),
            "ok": self.ok,
            "error": self.error,
        }


def select_winner(
    results: Sequence[ProbeResult],
    metrics: Sequence[MetricSpec],
    *,
    task_id: str = "",
    k: int = 1,
    mode: str = "",
) -> SelectionResult:
    """Pick a winner. Exactness/delivered is a hard gate applied first."""
    ordered_metrics = sorted(metrics, key=lambda m: (m.priority, m.name))
    sel = SelectionResult()
    eligible: list[ProbeResult] = []
    for r in results:
        reasons: list[str] = []
        if not r.delivered:
            reasons.append("not delivered")
        if not r.exact:
            reasons.append("not exact")
        if r.errors:
            reasons.append(f"errors: {'; '.join(r.errors)}")
        missing = [m.name for m in ordered_metrics if m.name not in r.metrics]
        if missing and ordered_metrics:
            reasons.append(f"missing metrics: {', '.join(missing)}")
        if reasons:
            sel.dropped.append(r.label)
            sel.drop_reasons[r.label] = "; ".join(reasons)
        else:
            eligible.append(r)

    if not eligible:
        sel.error = "no candidate passed the exactness/delivered gate"
        # Still publish an anonymised map over the input for reporting.
        sel.id_map = assign_ids([r.label for r in results], task_id=task_id, k=k, mode=mode)
        return sel

    frontier = pareto_frontier(eligible, ordered_metrics) if ordered_metrics else eligible
    sel.frontier_labels = [r.label for r in frontier]

    def sort_key(r: ProbeResult) -> tuple:
        keys: list[float] = [m.better_key(r.metrics[m.name]) for m in ordered_metrics if m.name in r.metrics]
        return (*keys, r.label)

    winner = sorted(frontier, key=sort_key)[0]
    sel.id_map = assign_ids(
        [r.label for r in results], task_id=task_id, k=k, mode=mode
    )
    # Re-map so the map covers at least the eligible pool deterministically;
    # assign_ids already covers every input label.
    sel.chosen_label = winner.label
    sel.chosen_candidate = winner.candidate_name
    sel.chosen_id = next((a for a, lbl in sel.id_map.items() if lbl == winner.label), "")
    sel.ok = True
    return sel


__all__ = ["SelectionResult", "assign_ids", "pareto_frontier", "select_winner"]
