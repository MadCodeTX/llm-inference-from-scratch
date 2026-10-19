# Upstream Harness² — pin, attribution, and deliberate differences

## Pin

- Upstream project: Harness² — *Recursive Agent Harnessing for an Open World*
  (Xu et al., arXiv:2610.14224).
- Pinned commit: `1cfe254e8d25b15f37048c3730134686abe60335`
  (`Fix arXiv link`, verified as `HEAD` of the `harness2-upstream` checkout
  read while authoring this tree).
- License: Apache License 2.0 (upstream `LICENSE`, Copyright 2026 Google LLC).
  This adaptation is a conceptual reuse of the method and component taxonomy;
  no upstream code is vendored into this repo. Where this tree's wording
  follows the upstream component model (c1–c8, Propose–Probe–Compose,
  `R = 4k` / `P = 3k` accounting, per-domain experience library), it does so
  under that Apache-2.0 attribution.
- Upstream references read: upstream `README.md`, `docs/SETUP.md`,
  `docs/EXPERIMENTS.md`, and the `harness2/core/` modules
  (`prompts.py`, `tree.py`, `library.py`, `ledger.py`, `spec.py`).

## What we reuse conceptually

- **Propose–Probe–Compose**: propose harness edits, probe their effects on the
  same task, compose a refined harness from behavioural contrasts — without
  needing ground-truth feedback at proposal time.
- **The c1–c8 component tree**: system prompt, guardrails, memory, tool
  guidance, sub-agents, skills, scripts, plugins — each with a different
  runtime cost and reach (always-on vs on-demand vs in-the-tool-loop).
- **Invariant vs domain-specific candidates**: every step probes both scopes.
- **Parallel vs sequential recursion**: width (more candidates per step) vs
  depth (more successive steps carrying refined state forward).
- **Budget accounting**: `R = 4k` rollouts, `P = 3k` improvement stages.
- **Per-domain experience library**: adopted harnesses + evidence-supported
  lessons (`cue / lesson / evidence / confidence`), read only within a domain,
  appended after adoption.

## Deliberate differences (and why)

| Aspect | Upstream Harness² | This tree |
|---|---|---|
| Executor substrate | OpenCode / Codex agent benchmarks (JobBench, LAB, WorkBuddy-Bench) | Muse workflows (`muse-wonder-1.0` coordinator + sub-agents) plus deterministic engine probes |
| What a probe is | A full agent rollout producing a deliverable | A fixed runner command (`scripts/run_qwen38_tp2.py`, `benchmarks/run_bench.py`, …) producing a JSON artifact |
| Selection | Blind deliverable selection: an anonymous pool, judged without scores | Exact-token gate first, then metric / Pareto selection on the artifact's measured fields — no blind judging |
| Correctness oracle | None at probe time (judging is a separate stage) | Token-exactness is a hard gate *inside* the step: inexact ⇒ rejected before any speed comparison |
| Resource discipline | Domains concurrent, tasks ordered within a domain | GPU probes strictly serial (one probe owns both RTX 4090s); only CPU-side proposing/composing is parallel |
| Patch surface | Editor edits the harness tree freely; scripts in a proposal may ship with it | The probe runner never applies an arbitrary patch: it executes the task card's named command with named flags only. Engine edits happen in a separate, human-reviewed change |

The differences all push the same way: upstream optimises *judged quality of
open-ended deliverables*, where no exact oracle exists, so blind selection is
the honest option. Inference optimisation has an exact oracle (greedy token
ids) and a numeric objective (TPOT, TTFT, tokens/min), so selection here is
*stricter*, not looser: exactness first, numbers second, judgement last.
