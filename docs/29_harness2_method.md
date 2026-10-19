# 29 — Harness² method for inference optimization: Propose–Probe–Compose against vLLM

> **Attribution.** The method adapted here is Harness² (Xu et al.,
> *Harness²: Recursive Agent Harnessing for an Open World*, arXiv:2610.14224;
> pinned upstream commit `1cfe254e8d25b15f37048c3730134686abe60335`,
> Apache-2.0). The harness tree that operationalises this chapter lives in
> [`harness/`](../harness/README.md); upstream pinning and the deliberate
> differences from the original framework are recorded in
> [`harness/UPSTREAM.md`](../harness/UPSTREAM.md). All prose below is
> original to this repo. Framing for the engine side follows Kiely,
> *Inference Engineering* (2026), sec. 4.5 (honest benchmarking) and
> sec. 5.2/5.4 (speculation, tensor parallelism) — see
> [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Treat "how we search for speed" as itself an editable system: propose small,
named changes to the optimization programme, probe each one with a
deterministic runner on the same task, and compose the next programme only
from measured contrasts — so every adopted lever carries the artifact that
justifies it, and every rejected lever stays rejected for a recorded reason.

## Why inference needs this

Chapters 21–28 built the engine piece by piece: gated-delta linear layers
(ch 21), the hybrid stack (ch 22), tensor parallelism (ch 23), the native
MTP head (ch 24), static decode state (ch 25), chunkwise prefill (ch 26),
block prefix caching (ch 27), and batched decode (ch 28). Each chapter
ended with a measurement, and the measurements do not compose themselves.
The decode story alone required four levers in a specific order — fused
FP8, fused RMSNorm, CUDA-graph capture, *then* a tile retune that had
measured zero before the graph existed. A search process that tries levers
in the wrong order, or that trusts a single unpaired timing run, will
confidently adopt noise. This chapter is about the search discipline, not
a new kernel.

The endgame makes the discipline load-bearing. The goal stated in
[benchmarks/WIN.md](../benchmarks/WIN.md) is to beat vLLM serving
`Qwen/Qwen3.8-27B-FP8` on this box (2× RTX 4090): Arm A at TPOT p50
20.3 ms (short, c1), Arm B (vLLM + MTP K=1, MEASURED-WITH-DEVIATION at
`--max-model-len 256608`) at 4,272–4,291 tok/min c1 short. Our best
measured decode is 23.67 ms (42.25 tok/s, graphed TP2 + fused gated-delta,
`benchmarks/baseline/ours_tp2_graph_v2.json`) — close enough that sloppy
measurement could fake a win, far enough that only real levers close it.
The method below exists to keep those two failure modes apart.

## Propose–Probe–Compose, translated

Upstream Harness² refines an *agent harness* (prompts, tools, memory,
skills, orchestration) for open-ended professional tasks, where no exact
oracle exists and candidate deliverables must be judged blind. Our
translation keeps the loop and replaces the oracle:

1. **Propose.** For one task card (a domain goal with a fixed shape, e.g.
   decode TPOT on the capital probe), draft `k` *invariant* candidates —
   levers true of every domain, such as paired ABBA timing or host-sync
   accounting — and `k` *domain-specific* candidates — the tile, chunk
   size, snapshot interval, or speculative mode the card is about. Each
   candidate names one lever, the disjoint files it touches, and the exact
   probe command. A candidate that cannot name its probe is not a
   candidate.
2. **Probe.** Run the card's deterministic runner
   (`scripts/run_qwen38_tp2.py`, `scripts/prefill_bench_tp2.py`,
   `scripts/prefix_ttft_bench.py`, `scripts/batched_decode_bench.py`, or
   `benchmarks/run_bench.py` for serving claims). Probes run serially —
   one probe owns both GPUs — inside an approved GPU window, and the
   runner writes a JSON artifact recording its full configuration. The
   probe step never applies an arbitrary patch: engine edits are a
   separate, reviewable change; the probe only executes named flags.
3. **Gate.** Before any timing is compared, greedy token ids must match
   the reference recorded in the artifact (`exact_match`, e.g. 32/32 ×3
   against both the pipeline bridge and Arm A for decode). Inexact
   candidates are rejected outright. This is the sharpest difference from
   upstream: where Harness² must judge deliverables blind because no exact
   answer exists, inference has one, so judgement is demoted to last place.
4. **Compose.** Among exact candidates, select on the card's measured
   objective vector — TPOT (lower) at fixed probe shape, prefill tok/s
   (higher) at fixed length, warm TTFT (lower) at fixed `n_reused`. A
   candidate that wins one metric and loses another is Pareto-kept with
   both numbers attached; a candidate dominated on all metrics is
   discarded. Composition produces the next step's baseline configuration,
   not a narrative about which candidate "seemed best".
5. **Reflect.** Append one library entry per adopted (or decisively
   rejected) lever: cue, lesson, evidence (artifact path + values),
   confidence, applicability limits. The library is per-domain and is read
   only within that domain, so a decode lesson cannot silently bless a
   prefill proposal.

Two worked contrasts show why each stage earns its keep. The FP8 tile
retune (BLOCK_N=32 at M=1) measured ~0 ms wall change before CUDA graphs
(48.48 → 48.80 ms paired, kernel gain absorbed by host pacing) and
−15.2% after them (28.88 → 24.48 ms ABBA,
`benchmarks/baseline/ours_tp2_tile_ab.json`): proposing the same lever at
the wrong point in the sequence produces a *measured rejection* of a good
idea, and only the library entry recording the order-dependence prevents
the idea from being discarded permanently. Conversely, native MTP
speculation shows a beautiful secondary metric — acceptance 0.88–0.94,
exact everywhere — and a worse objective: capital ×128 effective TPOT
32.51 ms (graphed) against 23.67 ms nonspec
(`benchmarks/baseline/ours_tp2_mtp_matrix.json`). A compose step keyed on
acceptance would have shipped a slowdown; keyed on effective TPOT, it
kept `--speculative off` and recorded why.

## Parallel width vs sequential depth

A step's budget is parameterised by `k`, and there are two ways to spend
more of it:

- **Parallel recursion** probes `k` invariant + `k` domain candidates
  against the *same* baseline and composes from the full pool. Width
  explores: it answers "which of these levers matters at all?" in one
  step, and its candidates are independent, so CPU-side proposing and
  composing parallelise freely.
- **Sequential recursion** runs `k` successive Propose–Probe–Compose
  steps, each starting from the previous step's adopted configuration and
  accumulated library. Depth refines: it answers "what does the second
  lever buy once the first is in?", which is exactly the question the
  tile-retune story poses — no parallel step from the pre-graph baseline
  could have discovered the post-graph tile win, because the win did not
  exist until the graph step landed.

The accounting is inherited from upstream and enforced as an identity,
not an estimate: **R = 4k rollouts** (`k` shared base executions + `2k`
candidate probes + `k` final confirmations) and **P = 3k improvement
sessions** (`k` invariant + `k` domain proposals + `k` compositions). At
`k = 1` a step is 4 probe runs and 3 agent sessions — cheap enough to run
per card before widening. The asymmetry that governs this repo: agent
sessions are cheap and parallel, GPU probes are serial and window-bound.
Width therefore scales with proposer count almost for free until the
probe queue saturates; depth is what actually spends GPU windows. Spend
width to avoid wasting depth on dead levers, and depth to exploit
order-dependent ones. Selection and reflection sit outside the R/P
ledger, as upstream records them.

## The eight components (c1–c8) in this repo

Upstream exposes eight editable components around a fixed model. Ours map
onto the `harness/` tree; the mapping matters because each component has a
different cost — always-on text is paid on every sub-agent run, on-demand
skills cost nothing until invoked, and only the tool-loop component can
enforce anything.

| # | Component | Here | Cost |
|---|---|---|---|
| c1 | System prompt | [`harness/systemprompt.md`](../harness/systemprompt.md): coordinator procedure + skill trigger lines | always on |
| c2 | Guardrails | [`harness/guardrails.md`](../harness/guardrails.md): measured-only, exactness-first, GPU-window and ownership rules | always on |
| c3 | Memory | [`harness/LongTermMEMORY.md`](../harness/LongTermMEMORY.md): the measured scoreboard summary | always on |
| c4 | Tool guidance | [`harness/tool_descriptions/tool_guidance.md`](../harness/tool_descriptions/tool_guidance.md): which runner, which flags, ABBA discipline | always on |
| c5 | Sub-agents | [`harness/sub_agents/`](../harness/sub_agents/): invariant/domain proposers, composer, reflector, probe runner — each sees only its brief + inputs, returns text | own context |
| c6 | Skills | [`harness/skills/`](../harness/skills/): `propose-probe-compose`, `profile-first`, `exactness-gate`, `library-reflect` | on demand |
| c7 | Scripts | [`harness/scripts/README.md`](../harness/scripts/README.md): no new scripts — the repo's runners are the deterministic helpers | on demand |
| c8 | Plugin | [`harness/plugin/README.md`](../harness/plugin/README.md): deliberately empty until a mechanical failure recurs | in the tool loop |

The form rules carry over from upstream because the failure modes do.
Prefer a skill for anything with a recognisable trigger ("a probe result
looks too good" → `profile-first`), a script for a deterministic operation
agents keep hand-rolling (timing loops are the local example), a sub-agent
for a bounded question whose input is large and answer small (which lever
does this profile support?), and always-on text only for rules that bind
every run — the six guardrails. A skill that no always-on line names is
dead weight, so `systemprompt.md` carries the trigger lines. c8 stays
empty on purpose: the exactness gate is already enforced mechanically by
the runners' own `exact_match` fields, and machinery without a recurring
failure to justify it is just surface area. The first plugin, when a
failure earns it, should be the exactness tripwire — block composition
until inexact candidates are marked rejected.

In the Muse deployment, a coordinator session (muse-wonder-1.0)
runs the loop: proposers and composers are sub-agents working CPU-only in
parallel; the coordinator itself executes probe commands serially and
checks every reflected number against its artifact. Parallel agents get
disjoint file ownership, declared per candidate — the guardrail that makes
width safe.

## Reflection and the experience library

After adoption, upstream stores the adopted harness plus a lesson with
cue, lesson, evidence, and confidence in a per-domain library; later tasks
in the domain read it, other domains never see it. Ours does the same at
[`harness/library/`](../harness/library/decode/INDEX.md), seeded only from
completed measured work, for example:

- *Decode*: reference dequant bound the v1 bridge (243 ms/GPU/token of
  dequant vs 25 ms of matmul at 1.83 tok/s), fused FP8 removed that wall
  (19.6–19.7 tok/s), TP2 proved host-launch-bound (clean 44.37 vs GPU-busy
  31.28 ms/token) so CUDA graphs were the structural fix (28.9 ms), tiles
  were retuned only after the graph (28.88 → 24.48 ms), and fusing the
  gated-delta step set the current best (23.67 ms).
- *Speculative*: acceptance 0.88–0.94 with effective TPOT worse than
  nonspec in every mode (eager 38.68, graph 32.51, merged graph2
  32.90 ms at capital ×128) — the lesson is that the M=2 verify chunk
  (34.6 ms vs the 24.5 ms M=1 step), not the draft, is the blocker.
- *Prefix*: snapshot interval 256 took warm TTFT from 1.31 s to 0.18 s
  (GPU, 2048+256 tokens, `n_reused=2048`, argmax cold==warm) — and the
  lesson includes its own tripwire: a warm number without the reuse count
  is not evidence.
- *Batching*: eager aggregate throughput scales (B=16: 20,001 tok/min)
  while per-seq latency does not move — the caveat is the lesson.

Two properties make the library worth its upkeep. First, entries carry
*applicability limits*, which is where negative knowledge lives: "tile
rankings invert after graphs" is a limit on every pre-graph tile entry,
and it is the single most valuable sentence in the decode library.
Second, retrieval is by construction — a domain directory only ever holds
its own entries — so cross-domain contamination needs no classifier, just
directory discipline. Confidence is graded by measurement design: paired
ABBA with exactness is high; a single unpaired run, however exact, is not.

## Beating vLLM without fooling ourselves

The win definition is external and fixed in advance: `run_bench.py` on
`benchmarks/suite.json`, same box, quality gate (greedy short-shape ids
token-identical, `failed == 0`, full completion budget), and thresholds of
≥1.25× tokens/min with ≥20% lower p50 TPOT at c1–c4 short/medium, ≥30%
lower prefix TTFT, ≥15% lower cold-long TTFT. The method chapter adds
three standing temptations and how the loop refuses each:

1. **Overfitting one prompt.** The local TP2 runner probes three quality
   prompts (capital/water/counting ×32) because they carry recorded
   bridge + Arm A references. A lever tuned until capital is fast may
   merely fit capital. The refusal: exactness is required on all three
   probes, the 128-token capital run checks length-generalisation against
   a same-run nonspec reference, and no *serving* claim is made until the
   suite's four shapes (2048→512 up to 49152→1024, plus the shared-prefix
   shape) are measured through `run_bench.py`. A candidate that only wins
   on the probe it was tuned on stays a local-runner result.
2. **Cross-shape numbers dressed as wins.** Our eager batched runner
   posts 20,001 tok/min at B=16, numerically above Arm B's c4 bar — on
   6-token prompts, eager, ignore-EOS, in a different harness. The
   throughput ladder in WIN.md records this as "passes numerically only",
   and the guardrail generalises it: any comparison across shapes,
   concurrencies, or harnesses is context, labelled as such when written,
   never a verdict. The same rule protects the other direction — Arm B's
   chunk TPOT (~24.7 ms per multi-token chunk) is never compared against
   Arm A's per-token 20.3 ms.
3. **Secondary-metric adoption.** Acceptance (speculative), kernel-time
   savings absorbed by host pacing (pre-graph tiles), and warm TTFT
   without reuse counts are all real numbers that measure the wrong
   thing. The compose step adopts only on the card's objective vector,
   with the secondary metric attached as evidence — which is how the MTP
   work ended in a *recorded rejection* (`--speculative off` kept) rather
   than a shipped slowdown, and how the profile's overlapping "memo"
   categories are never summed into a phantom total.

What remains open, stated plainly: per-stream decode still loses to both
arms (23.67 vs 20.3 ms; Arm B ~71 tok/s), prefill rate still loses
(1,731.6 tok/s @2048 vs Arm A's ~2.9–3.4k effective), and prefix TTFT has
no comparable-shape claim at all. The next depth steps are visible from
the artifacts — the FP8 kernel's 14.57 ms/token and the 6.32 ms/token
elementwise residue under the graph, the M=2 verify chunk, a graphed
batched step — and each is a task card, not a hope. When one of them
lands, it will land as an artifact, an exactness record, and a library
entry; when the serving card finally runs, it will be judged by the
thresholds above and by nothing friendlier.

## See also

- The operational tree: [`harness/README.md`](../harness/README.md) and
  the six task cards under [`harness/tasks/`](../harness/tasks/decode-tpot.md).
- The bars and the protocol: [benchmarks/WIN.md](../benchmarks/WIN.md).
- Engine chapters this method searches over:
  [21](21_linear_attention.md) · [22](22_hybrid_qwen38.md) ·
  [23](23_tensor_parallel.md) · [24](24_mtp.md) ·
  [25](25_static_decode.md) · [26](26_prefill.md) ·
  [27](27_prefix_cache.md) · [28](28_batching.md).
