# Harness² for inference optimization — operating loop

This tree adapts Harness² (Apache-2.0, see [UPSTREAM.md](UPSTREAM.md)) to one
job: making the from-scratch Qwen3.8-27B-FP8 engine in this repo faster than
the incumbent vLLM arms, **without ever claiming a speed number that was not
measured**. The upstream method is Propose–Probe–Compose over an eight-component
harness; here the "harness" being edited is the *optimization programme* itself
— which kernel, tile, graph, or batching lever to try next, under what probe —
and the "deliverable" being selected is a measured engine configuration, not a
document.

## The loop (one task card = one Propose–Probe–Compose step)

1. **Pick a task card** from [`tasks/`](tasks/) (e.g. `decode-tpot.md`). The card
   fixes the domain, the current best measured result with its artifact path,
   the Arm A / Arm B bars where comparable, the objective metrics and their
   directions, the exactness gate, and the stop condition. Nothing outside the
   card's stop condition ends the step early; nothing inside it may be skipped.
2. **Read the domain library first.** Open `library/<domain>/INDEX.md` and the
   `EXPERIENCE.md` entries it lists. A cue that matches means the lesson's
   applicability limits must be checked before proposing anything the library
   already settled (e.g. re-tuning FP8 tiles before a CUDA graph exists).
3. **Propose `k` invariant + `k` domain candidates.** Invariant candidates change
   something true of every domain (exactness discipline, ABBA pairing, host-sync
   accounting). Domain candidates change the card's lever (a tile, a chunk size,
   a snapshot interval). Each candidate names the file(s) it would touch — file
   ownership is disjoint across parallel candidates (see
   [guardrails.md](guardrails.md)).
4. **Probe deterministically.** Probes are the repo's own runners, never an
   agent-written timing loop: `scripts/run_qwen38_tp2.py`,
   `scripts/prefill_bench_tp2.py`, `scripts/prefix_ttft_bench.py`,
   `scripts/batched_decode_bench.py`, `benchmarks/run_bench.py`. Probes run
   **serially** on the bigbox GPUs (one probe owns both RTX 4090s at a time) and
   only inside an approved GPU window. The probe runner executes a fixed command
   from the task card; it never applies an arbitrary patch.
5. **Gate on exactness before speed.** A candidate whose greedy token ids are
   not identical to the reference (bridge / Arm A ids in the artifact, checked
   by the runner's own `exact_match` fields) is rejected no matter how fast it
   looks. Only exact candidates enter metric comparison.
6. **Compose by metric / Pareto selection.** Among exact candidates, select on
   the card's objective vector (e.g. TPOT ↓ at fixed shape; tokens/min ↑ at
   fixed concurrency). A candidate that wins one metric and loses another is
   Pareto-kept, not averaged away; a candidate dominated on every metric is
   discarded. There is no blind "pick the nicest output" step — see
   [UPSTREAM.md](UPSTREAM.md) for why this differs from upstream.
7. **Reflect.** The coordinator appends one `EXPERIENCE.md` entry to the domain
   library (cue, lesson, evidence with artifact path + values, confidence,
   applicability limits) and updates `INDEX.md`. Unmeasured impressions are not
   lessons and are not written.

Budget accounting for a step follows upstream: with width `k`, expect
`R = 4k` probe rollouts (`k` shared base + `2k` candidates + `k` finals) and
`P = 3k` improvement sessions (`k` invariant + `k` domain + `k` compositions).
See [docs/29_harness2_method.md](../docs/29_harness2_method.md) for the full
accounting and the parallel-width vs sequential-depth trade-off.

## How a Muse coordinator runs this (muse-wonder-1.0)

The coordinator is a Muse session running `muse-wonder-1.0`. It does not
edit engine code itself during a step; it delegates:

- **Proposers** (sub-agents, role briefs in [`sub_agents/`](sub_agents/)) each
  receive one task card + the domain library and return *one* candidate each:
  a named lever, the exact files it touches, and the probe command from the
  card with only that lever's flag changed. Proposers never touch the GPU.
- **The probe runner** is not an agent free-for-all: the coordinator runs the
  card's probe command itself, serially, in the approved window, and records
  the artifact path the runner writes. Sub-agents receive the artifact JSON as
  data, never as instructions.
- **Composers** (sub-agents) receive the anonymised exact-passing candidate
  artifacts and the card's objective vector, and return a selection with the
  metric table that justifies it. Composers cannot see which proposer wrote
  which candidate.
- **The reflector** (sub-agent) receives the adopted candidate + its artifact
  and drafts the library entry; the coordinator checks every number in the
  draft against the artifact before writing it.

Parallel width comes from proposers/composers (cheap, CPU-only reasoning).
Sequential depth comes from repeating the loop with the updated library.
GPU probes are the serial bottleneck by design — see the method chapter.

## Always-on vs on-demand

`systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`, and
`tool_descriptions/tool_guidance.md` are always-on: keep them short, because
every word is paid on every sub-agent run. Reusable procedures live in
[`skills/`](skills/) and are loaded only when their trigger line (in the
always-on text) fires. Deterministic helpers are described in
[`scripts/README.md`](scripts/README.md); tool-loop enforcement, if ever
needed, belongs in [`plugin/README.md`](plugin/README.md) — prose cannot
enforce the exactness gate, a check script can.

## Start here

1. Read [guardrails.md](guardrails.md) — the six non-negotiables.
2. Pick a card in [`tasks/`](tasks/); read its domain library.
3. Run one `k = 1` step before widening: one invariant + one domain candidate,
   `R = 4`, `P = 3`.
