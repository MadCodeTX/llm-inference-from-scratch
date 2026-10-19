"""Budget accounting for one optimisation round.

The round shape is borrowed as a *concept* from Harness² (Google,
Apache-2.0; see ``harness/UPSTREAM.md``): a width-``k`` round spends
``4k`` probe rollouts and ``3k`` proposer sessions.  The code below is
written for this repository and shares no text with upstream.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _check_width(k: int) -> int:
    width = int(k)
    if width < 1:
        raise ValueError(f"round width k must be at least 1, got {k!r}")
    return width


def expected_rollouts(k: int) -> int:
    """Probes in a width-``k`` round: base, two candidate waves, finals."""
    return 4 * _check_width(k)


def expected_sessions(k: int) -> int:
    """Proposer sessions in a width-``k`` round: three waves of ``k``."""
    return 3 * _check_width(k)


class BudgetError(AssertionError):
    """The recorded round does not add up to its declared width."""


@dataclass
class LedgerEntry:
    kind: str
    label: str
    cost: float = 0.0
    cached: bool = False
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "cost": self.cost,
            "cached": self.cached,
            "attempts": self.attempts,
        }


@dataclass
class Ledger:
    mode: str
    task_id: str
    k: int
    entries: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_width(self.k)

    # -- recording -----------------------------------------------------
    def rollout(self, label: str, cost: float = 0.0, *, cached: bool = False) -> None:
        self.entries.append(LedgerEntry("rollout", label, float(cost), bool(cached)))

    def session(self, label: str, cost: float = 0.0, *, cached: bool = False, attempts: int = 1) -> None:
        entry = LedgerEntry("session", label, float(cost), bool(cached), int(attempts))
        self.entries.append(entry)

    # -- summaries -----------------------------------------------------
    def _of_kind(self, kind: str) -> list[LedgerEntry]:
        return [entry for entry in self.entries if entry.kind == kind]

    @property
    def R(self) -> int:
        return len(self._of_kind("rollout"))

    @property
    def P(self) -> int:
        return len(self._of_kind("session"))

    @property
    def cost(self) -> float:
        return round(sum(entry.cost for entry in self.entries), 6)

    def labels(self, kind: str) -> list[str]:
        return [entry.label for entry in self._of_kind(kind)]

    def duplicates(self) -> list[str]:
        repeated: set[str] = set()
        for kind in ("rollout", "session"):
            counts = Counter(self.labels(kind))
            repeated.update(label for label, n in counts.items() if n > 1)
        return sorted(repeated)

    # -- validation ----------------------------------------------------
    def assert_identity(self) -> None:
        want = (expected_rollouts(self.k), expected_sessions(self.k))
        got = (self.R, self.P)
        repeated = self.duplicates()
        if got == want and not repeated:
            return
        lines = [
            f"budget mismatch for task {self.task_id!r} (mode={self.mode}, k={self.k}): "
            f"recorded R={got[0]}/P={got[1]}, round requires R={want[0]}/P={want[1]}",
            "rollout labels: " + (", ".join(self.labels("rollout")) or "none recorded"),
            "session labels: " + (", ".join(self.labels("session")) or "none recorded"),
        ]
        if repeated:
            lines.append(
                "DUPLICATE labels: " + ", ".join(repeated)
                + " (one label must name one stage; a repeat hides an overwrite)"
            )
        raise BudgetError("\n".join(lines))

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "R": self.R,
            "expected_R": expected_rollouts(self.k),
            "P": self.P,
            "expected_P": expected_sessions(self.k),
            "session_attempts": sum(e.attempts for e in self._of_kind("session")),
            "cost": self.cost,
            "rollouts": self.labels("rollout"),
            "sessions": self.labels("session"),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ledger":
        ledger = cls(mode=str(data["mode"]), task_id=str(data["task_id"]), k=int(data["k"]))
        for raw in data.get("entries", []):
            ledger.entries.append(
                LedgerEntry(
                    kind=str(raw["kind"]),
                    label=str(raw["label"]),
                    cost=float(raw.get("cost", 0.0)),
                    cached=bool(raw.get("cached", False)),
                    attempts=int(raw.get("attempts", 1)),
                )
            )
        return ledger


__all__ = ["BudgetError", "Ledger", "LedgerEntry", "expected_rollouts", "expected_sessions"]
