# Harness round 3 — decode TPOT (GEMV kernel candidate)

Source artifact: [decode_round3.json](decode_round3.json) (live GPU round at
git HEAD `71995f2e37edf24ba210cc24c793df0fa462c1fc`; the GEMV candidate is
uncommitted working-tree code — 5 modified files listed in the artifact —
and this round edits no source and commits nothing). Round 2
([decode_round2.json](decode_round2.json)) confirmed `--scale-cache on`
(C1, 22.850 ms aggregate median) as the retained winner. The kernel lane
then added an opt-in M=1 GEMV kernel (`--fp8-config gemv`,
`M1_GEMV_CONFIG={BLOCK_N:16, BLOCK_K:256, num_warps:4, num_stages:3}`,
dispatched only at exactly M=1; M>1 falls back to the `new` dot config).

**Question:** does candidate `R3G_candidate_gemv` (retained winner config
plus `--fp8-config gemv`, single variable) beat base `R3B_base_new`
(`--decode-mode graph --scale-cache on --fp8-config new`) exactly and
faster?

Design: ABBA order `B,G,G,B,B,G` per batch, 3 measured runs per candidate
per shape, 12 torchrun processes total, all return code 0, no runs
discarded. Batch A runs all three recorded probes (capital, water,
counting) ×32 greedy in one process; batch B runs capital ×128 greedy.
GPUs were idle before (1 MiB, no compute apps) and after; no probe process
remains. Kernel smoke before the round:
`.venv-gpu/bin/python -m pytest tests/test_fp8_triton_gpu.py -k gemv -q`
→ 5 passed. Raw logs/JSON under `/tmp/harness_r3/`.

| Shape | Base (new) median TPOT (min–max) | GEMV median TPOT (min–max) | Δ GEMV vs base | Exact |
|---|---|---|---|---|
| capital ×32 | 22.819 ms (22.812–22.820) | 22.269 ms (22.268–22.270) | −2.41% | 32/32 vs bridge + Arm A, every run |
| water ×32 | 22.822 ms (22.815–22.822) | 22.274 ms (22.269–22.274) | −2.40% | 32/32 vs bridge + Arm A, every run |
| counting ×32 | 22.815 ms (22.809–22.818) | 22.268 ms (22.267–22.270) | −2.40% | 32/32 vs bridge + Arm A, every run |
| capital ×128 | 22.853 ms (22.846–22.858) | 22.307 ms (22.301–22.308) | −2.39% | 32/32 prefix vs both refs; 128/128 vs base reference, every run |
| **Aggregate (pooled, n=12/candidate)** | **22.820 ms, 43.82 tok/s** | **22.270 ms, 44.90 tok/s** | **−2.41%** | exact everywhere |

Peak memory (runner-reported, per rank) is identical for base and GEMV:
16,367,239,168 B at ×32 and 16,346,798,592 B at ×128. Logits finite on
every run. Within-run spreads are ≤0.01 ms, so the ~2.4% gain is uniform
across prompts and lengths rather than a single-shape effect; it is also
smaller than the kernel lane's synthetic per-shape gains (up to −33%),
consistent with its warned weighted-mix expectation.

Selection rule (pre-stated, zero tolerance): any inexact candidate run
rejects the candidate regardless of speed; otherwise the candidate
becomes default only if exact everywhere, aggregate-median-faster, and
no shape median regresses by more than 1%. All conditions hold (exact
everywhere; −2.41% aggregate; worst shape −2.39%, i.e. no regression),
so the recommendation is **BECOME DEFAULT**: promote the GEMV kernel to
the default M=1 FP8 path. This round itself changes no default (no
source edited); the promotion is a separate commit of the already-present
uncommitted candidate plus a default flip.

Deterministic selection on CPU reproduces the pick (capital ×32 is the
task-card shape the importer scores):

```bash
.venv/bin/python scripts/opt_harness.py import-metrics \
  benchmarks/harness/decode_round3.json --task-id decode-tpot
```

Frontier/chosen: `R3G_candidate_gemv` (recorded in the artifact's
`import_metrics` block; raw stdout at `/tmp/harness_r3/import_stdout.json`).

**No vLLM win.** The GEMV aggregate median 22.270 ms is above the Arm A
20.3 ms bar (and above the 19.17 ms roofline floor), so no beating-vLLM
claim is made. Even a sub-20.3 ms local result would be labelled
local-runner only, not a same-harness suite win.
