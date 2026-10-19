"""Recursive optimisation harness for inference settings.

An original adaptation of the Harness² recursive-optimisation idea
(Google, Apache-2.0) to this repository: typed task/candidate/probe
specs, a ``R=4k / P=3k`` budget ledger, a per-domain experience library,
exactness-gated Pareto selection, file/scripted proposers, command/fake
executors, and parallel/sequential recursion for ``k >= 1``.
"""

from .demo import demo_task, load_metrics_file, run_demo
from .executor import CommandExecutor, Executor, FakeExecutor
from .ledger import BudgetError, Ledger, expected_rollouts, expected_sessions
from .library import Experience, Library, has_experience
from .proposer import FileProposer, ProposalContext, Proposer, ProposerError, ScriptedProposer
from .recursion import ModeResult, run_mode, run_parallel, run_sequential
from .selection import SelectionResult, assign_ids, pareto_frontier, select_winner
from .spec import Candidate, MetricSpec, OptimizationTask, ProbeResult

__all__ = [
    "BudgetError",
    "Candidate",
    "CommandExecutor",
    "Executor",
    "Experience",
    "FakeExecutor",
    "FileProposer",
    "Ledger",
    "Library",
    "MetricSpec",
    "ModeResult",
    "OptimizationTask",
    "ProbeResult",
    "ProposalContext",
    "Proposer",
    "ProposerError",
    "ScriptedProposer",
    "SelectionResult",
    "assign_ids",
    "demo_task",
    "expected_rollouts",
    "expected_sessions",
    "has_experience",
    "load_metrics_file",
    "pareto_frontier",
    "run_demo",
    "run_mode",
    "run_parallel",
    "run_sequential",
    "select_winner",
]
